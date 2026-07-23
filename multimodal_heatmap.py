"""
多模态热力图生成器 - 支持 RGB + IR 输入
"""
import warnings
warnings.filterwarnings('ignore')
import torch
import cv2
import numpy as np
import os
from ultralytics.nn.tasks import attempt_load_weights
from PIL import Image
from scipy.ndimage import gaussian_filter

def letterbox(im, new_shape=(640, 640), color=(114, 114, 114), auto=False, scaleFill=True, scaleup=True, stride=32):
    """对图像进行 letterbox 处理"""
    shape = im.shape[:2]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:
        r = min(r, 1.0)
    ratio = r, r
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]
    if auto:
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)
    elif scaleFill:
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[1], new_shape[0])
        ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]
    dw /= 2
    dh /= 2
    if shape[::-1] != new_unpad:
        im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    return im, ratio, (dw, dh)

class MultimodalHeatmap:
    def __init__(self, weight_path, device='cuda:0'):
        """
        初始化多模态热力图生成器
        :param weight_path: 模型权重路径
        :param device: 计算设备
        """
        self.device = torch.device(device)
        self.model = attempt_load_weights(weight_path, self.device)
        self.model.eval()
        self.features = {}
        self.hooks = []
        
    def register_hook(self, layer_idx, feature_part='low'):
        """为指定层注册 hook 捕获特征图。

        fusehd_fourth 返回 [low, high]，其中 high 形状通常为 (B, C, 3, H, W)。
        feature_part 可选: low / high / both。
        """
        valid_parts = {'low', 'high', 'both'}
        if feature_part not in valid_parts:
            raise ValueError(f"feature_part must be one of {valid_parts}, got {feature_part}")

        def hook_fn(module, input, output):
            if isinstance(output, (list, tuple)):
                if feature_part == 'both':
                    self.features[(layer_idx, 'low')] = output[0].detach().cpu()
                    self.features[(layer_idx, 'high')] = output[1].detach().cpu()
                    return
                part_idx = 0 if feature_part == 'low' else 1
                feat = output[part_idx]
            else:
                if feature_part in {'high', 'both'}:
                    print(f"⚠️ 层 {layer_idx} 不是高低频列表输出，改为捕获普通输出")
                feat = output
            feature_key = 'low' if feature_part == 'both' else feature_part
            self.features[(layer_idx, feature_key)] = feat.detach().cpu()
        
        target_layer = self.model.model[layer_idx]
        h = target_layer.register_forward_hook(hook_fn)
        self.hooks.append(h)
        
    def remove_hooks(self):
        """移除所有 hooks"""
        for h in self.hooks:
            h.remove()
        self.hooks.clear()
        self.features.clear()
    
    def _smooth_heatmap(self, heatmap, sigma=3.0):
        """高斯平滑热力图"""
        return gaussian_filter(heatmap.astype(np.float32), sigma=sigma)
    
    def _normalize_heatmap(self, heatmap):
        """将热力图归一化到 [0, 255]"""
        h_min = heatmap.min()
        h_max = heatmap.max()
        if h_max > h_min:
            heatmap = (heatmap - h_min) / (h_max - h_min) * 255
        else:
            heatmap = np.zeros_like(heatmap)
        return heatmap.astype(np.uint8)
    
    def _upsample_heatmap(self, heatmap, target_h, target_w):
        """双线性插值上采样热力图"""
        return cv2.resize(heatmap, (target_w, target_h), interpolation=cv2.INTER_LINEAR)

    def _feature_to_mask(self, feat, target_h, target_w, smooth_sigma=3.0, use_abs=False):
        """将单层特征转换为归一化的热力图 mask（0~1）"""
        if feat.ndim == 5:
            # 高频输出为 3 个子带，先把子带维度合并为强度图。
            feat = feat.abs().mean(dim=2)
            use_abs = False
        if use_abs:
            feat = feat.abs()
        heatmap = feat[0].mean(dim=0).numpy()
        heatmap_up = self._upsample_heatmap(heatmap, target_h, target_w)
        heatmap_smooth = self._smooth_heatmap(heatmap_up, sigma=smooth_sigma)
        heatmap_smooth = np.clip(heatmap_smooth, 0, None)
        h_min, h_max = heatmap_smooth.min(), heatmap_smooth.max()
        if h_max > h_min:
            return (heatmap_smooth - h_min) / (h_max - h_min)
        return np.zeros_like(heatmap_smooth, dtype=np.float32)
    
    def generate_heatmaps(self, rgb_path, ir_path, layer_indices, output_dir='./heatmap_output', smooth_sigma=3.0, weights=None, generate_average=False, feature_part='low'):
        """
        生成多层热力图、可选加权平均热力图
        :param rgb_path: RGB 图像路径
        :param ir_path: IR 图像路径
        :param layer_indices: 要可视化的层索引列表
        :param output_dir: 输出目录
        :param smooth_sigma: 高斯平滑参数
        :param weights: 各层权重，None 则等权平均
        :param generate_average: 是否生成平均热力图，默认生成
        :param feature_part: fusehd_fourth 输出选择，low / high / both
        """
        os.makedirs(output_dir, exist_ok=True)
        
        # 读取并预处理图像
        img_rgb = cv2.imread(rgb_path)
        img_ir = cv2.imread(ir_path)
        
        if img_rgb is None or img_ir is None:
            print(f"❌ 无法加载图像")
            return
        
        # Letterbox 处理
        img_rgb_lb, _, _ = letterbox(img_rgb)
        img_ir_lb, _, _ = letterbox(img_ir)
        
        original_h, original_w = img_rgb.shape[:2]
        img_h, img_w = img_rgb_lb.shape[:2]
        
        # 转为 RGB 并归一化
        img_rgb_norm = cv2.cvtColor(img_rgb_lb, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        img_ir_norm = cv2.cvtColor(img_ir_lb, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        
        # 拼接为 6 通道
        img_concat = np.concatenate((img_rgb_norm, img_ir_norm), axis=2)  # (H, W, 6)
        tensor = torch.from_numpy(np.transpose(img_concat, (2, 0, 1))).unsqueeze(0).to(self.device)  # (1, 6, H, W)
        
        # 注册 hooks
        for layer_idx in layer_indices:
            self.register_hook(layer_idx, feature_part=feature_part)
        
        # 前向传播
        with torch.no_grad():
            _ = self.model(tensor)
        
        selected_parts = ['low', 'high'] if feature_part == 'both' else [feature_part]
        layer_masks_by_part = {part: [] for part in selected_parts}
        for layer_idx in layer_indices:
            layer_name = self.model.model[layer_idx].__class__.__name__
            for part in selected_parts:
                feature_key = (layer_idx, part)
                if feature_key not in self.features:
                    print(f"❌ 层 {layer_idx} 没有捕获到 {part} 特征")
                    continue
                feat = self.features[feature_key]
                print(f"✓ 层 {layer_idx} ({layer_name}) {part}: {feat.shape}")
                mask = self._feature_to_mask(feat, img_h, img_w, smooth_sigma=smooth_sigma, use_abs=(part == 'high'))
                layer_masks_by_part[part].append(mask)

                heatmap_only = self._normalize_heatmap(mask)
                heatmap_only = cv2.resize(heatmap_only, (original_w, original_h), interpolation=cv2.INTER_LINEAR)
                heatmap_color = cv2.applyColorMap(heatmap_only, cv2.COLORMAP_JET)
                single_name = f'layer_{layer_idx}_{layer_name}_{part}_heatmap.png'
                single_path = os.path.join(output_dir, single_name)
                single_rgb = cv2.addWeighted(img_rgb, 0.4, heatmap_color, 0.6, 0)
                cv2.imwrite(single_path, single_rgb)
                print(f"✓ 保存: {single_path}")

        if not any(layer_masks_by_part.values()):
            self.remove_hooks()
            return

        if not generate_average:
            self.remove_hooks()
            return

        for part, layer_masks in layer_masks_by_part.items():
            if len(layer_masks) == 0:
                continue

            if weights is None:
                part_weights = np.ones(len(layer_masks), dtype=np.float32) / len(layer_masks)
            else:
                part_weights = np.asarray(weights, dtype=np.float32)
                part_weights = part_weights[:len(layer_masks)]
                part_weights = part_weights / (part_weights.sum() + 1e-8)

            avg_mask = np.zeros_like(layer_masks[0], dtype=np.float32)
            for w, mask in zip(part_weights, layer_masks):
                avg_mask += w * mask

            avg_mask = self._smooth_heatmap(avg_mask, sigma=max(1.0, smooth_sigma * 0.5))
            avg_mask = np.clip(avg_mask, 0, None)
            avg_mask = self._normalize_heatmap(avg_mask)
            avg_mask = cv2.resize(avg_mask, (original_w, original_h), interpolation=cv2.INTER_LINEAR)
            avg_color = cv2.applyColorMap(avg_mask, cv2.COLORMAP_JET)
            avg_rgb = cv2.addWeighted(img_rgb, 0.4, avg_color, 0.6, 0)

            avg_name = f'average_{part}_heatmap.png'
            avg_only_name = f'average_{part}_heatmap_only.png'
            cv2.imwrite(os.path.join(output_dir, avg_name), avg_rgb)
            cv2.imwrite(os.path.join(output_dir, avg_only_name), avg_color)
            print(f"✓ 保存: {os.path.join(output_dir, avg_name)}")
            print(f"✓ 保存: {os.path.join(output_dir, avg_only_name)}")

        self.remove_hooks()

    def generate_heatmaps_on_target_image(self, rgb_path, ir_path, target_image_path, layer_indices, output_dir='./heatmap_output', smooth_sigma=3.0, alpha=0.6, feature_part='low'):
        """
        使用 RGB + IR 输入生成指定层 mask，并叠加到另一张目标图上
        :param rgb_path: RGB 图像路径，用于模型输入
        :param ir_path: IR 图像路径，用于模型输入
        :param target_image_path: 被叠加热图的目标图像路径
        :param layer_indices: 要可视化的层索引列表
        :param output_dir: 输出目录
        :param smooth_sigma: 高斯平滑参数
        :param alpha: 热力图透明度
        :param feature_part: fusehd_fourth 输出选择，low / high / both
        """
        os.makedirs(output_dir, exist_ok=True)

        img_rgb = cv2.imread(rgb_path)
        img_ir = cv2.imread(ir_path)
        target_img = cv2.imread(target_image_path)

        if img_rgb is None or img_ir is None:
            print(f"❌ 无法加载模型输入图像")
            return
        if target_img is None:
            print(f"❌ 无法加载目标叠加图像: {target_image_path}")
            return

        img_rgb_lb, _, _ = letterbox(img_rgb)
        img_ir_lb, _, _ = letterbox(img_ir)
        img_h, img_w = img_rgb_lb.shape[:2]
        target_h, target_w = target_img.shape[:2]

        img_rgb_norm = cv2.cvtColor(img_rgb_lb, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        img_ir_norm = cv2.cvtColor(img_ir_lb, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        img_concat = np.concatenate((img_rgb_norm, img_ir_norm), axis=2)
        tensor = torch.from_numpy(np.transpose(img_concat, (2, 0, 1))).unsqueeze(0).to(self.device)

        for layer_idx in layer_indices:
            self.register_hook(layer_idx, feature_part=feature_part)

        with torch.no_grad():
            _ = self.model(tensor)

        selected_parts = ['low', 'high'] if feature_part == 'both' else [feature_part]
        for layer_idx in layer_indices:
            layer_name = self.model.model[layer_idx].__class__.__name__
            for part in selected_parts:
                feature_key = (layer_idx, part)
                if feature_key not in self.features:
                    print(f"❌ 层 {layer_idx} 没有捕获到 {part} 特征")
                    continue

                feat = self.features[feature_key]
                print(f"✓ 层 {layer_idx} ({layer_name}) {part}: {feat.shape}")

                mask = self._feature_to_mask(feat, img_h, img_w, smooth_sigma=smooth_sigma, use_abs=(part == 'high'))
                heatmap_only = self._normalize_heatmap(mask)
                heatmap_only = cv2.resize(heatmap_only, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
                heatmap_color = cv2.applyColorMap(heatmap_only, cv2.COLORMAP_JET)
                overlay = cv2.addWeighted(target_img, 1.0 - alpha, heatmap_color, alpha, 0)

                output_name = f'layer_{layer_idx}_{layer_name}_{part}_heatmap_on_target.png'
                output_path = os.path.join(output_dir, output_name)
                cv2.imwrite(output_path, overlay)
                print(f"✓ 保存: {output_path}")

        self.remove_hooks()

def main():
    generator = MultimodalHeatmap(
        weight_path="/home/asefu/project/WaveMamba-main/runs/M3FD_latest/midlle_p1_p2_p33_p4/weights/best.pt",
        device='cuda:0'
    )
    
    # 生成多个层的热力图，可通过 generate_average 控制是否输出总平均热力图
    layers_to_visualize = [49]
    weights = [0.3, 0.3, 0.4]
    generate_average = False
    overlay_on_target = False
    feature_part = 'low'  # 可选: low / high / both
    target_image_path = r"/home/asefu/data/M3FD_YOLO/image/train/00116.png"
    
    try:
        print(f"\n生成层 {layers_to_visualize} 的热力图，并计算加权平均...")
        generator.generate_heatmaps(
            rgb_path=r"/home/asefu/data/M3FD_YOLO/images/train/00116.png",
            ir_path=r"/home/asefu/data/M3FD_YOLO/image/train/00116.png",
            layer_indices=layers_to_visualize,
            output_dir='/home/asefu/project/WaveMamba-main/runs/gradcam/M3FD/00116',
            smooth_sigma=3.0,
            weights=weights,
            generate_average=generate_average,
            feature_part=feature_part,
        )

        if overlay_on_target:
            generator.generate_heatmaps_on_target_image(
                rgb_path=r"/home/asefu/data/M3FD_YOLO/images/train/00116.png",
                ir_path=r"/home/asefu/data/M3FD_YOLO/image/train/00116.png",
                target_image_path=target_image_path,
                layer_indices=layers_to_visualize,
                output_dir='/home/asefu/project/WaveMamba-main/runs/gradcam/M3FD/train_00116/target_overlay',
                smooth_sigma=3.0,
                alpha=0.3,
                feature_part=feature_part,
            )
    except Exception as e:
        print(f"❌ 生成失败: {e}")
    
    print("\n✅ 完成！热力图已保存到输出目录")

if __name__ == '__main__':
    main()
