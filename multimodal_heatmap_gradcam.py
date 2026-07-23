import warnings
warnings.filterwarnings("ignore")

import argparse
import os

import cv2
import numpy as np
import torch
from PIL import Image
from tqdm import trange

from ultralytics.nn.tasks import attempt_load_weights
from ultralytics.utils.ops import non_max_suppression, xywh2xyxy

from pytorch_grad_cam import (
    EigenCAM,
    EigenGradCAM,
    GradCAM,
    GradCAMPlusPlus,
    LayerCAM,
    RandomCAM,
    XGradCAM,
)
from pytorch_grad_cam.activations_and_gradients import ActivationsAndGradients
from pytorch_grad_cam.utils.image import scale_cam_image, show_cam_on_image


CAM_METHODS = {
    "GradCAMPlusPlus": GradCAMPlusPlus,
    "GradCAM": GradCAM,
    "XGradCAM": XGradCAM,
    "EigenCAM": EigenCAM,
    "LayerCAM": LayerCAM,
    "RandomCAM": RandomCAM,
    "EigenGradCAM": EigenGradCAM,
}


def letterbox(im, new_shape=(640, 640), color=(114, 114, 114), auto=False, scaleFill=True, scaleup=True, stride=32):
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


def _unwrap_tensor(output):
    if isinstance(output, torch.Tensor):
        return output
    if isinstance(output, (list, tuple)) and len(output):
        return _unwrap_tensor(output[0])
    return None


class SafeActivationsAndGradients(ActivationsAndGradients):
    def save_activation(self, module, input, output):
        activation = _unwrap_tensor(output)
        if activation is None:
            return
        if self.reshape_transform is not None:
            activation = self.reshape_transform(activation)
        self.activations.append(activation.cpu().detach())

    def save_gradient(self, module, input, output):
        out = _unwrap_tensor(output)
        if out is None or not hasattr(out, "requires_grad") or not out.requires_grad:
            return

        def _store_grad(grad):
            if self.reshape_transform is not None:
                grad = self.reshape_transform(grad)
            self.gradients = [grad.cpu().detach()] + self.gradients

        out.register_hook(_store_grad)

    def post_process(self, result):
        logits_ = result[:, 4:]
        boxes_ = result[:, :4]
        _, indices = torch.sort(logits_.max(1)[0], descending=True)
        sorted_logits = torch.transpose(logits_[0], dim0=0, dim1=1)[indices[0]]
        sorted_boxes = torch.transpose(boxes_[0], dim0=0, dim1=1)[indices[0]]
        _ = xywh2xyxy(sorted_boxes).cpu().detach().numpy()
        return sorted_logits, sorted_boxes

    def __call__(self, x):
        self.gradients = []
        self.activations = []
        model_output = self.model(x)
        post_result, pre_post_boxes = self.post_process(model_output[0])
        return [[post_result, pre_post_boxes]]


class YoloDetectionTarget(torch.nn.Module):
    def __init__(self, output_type, conf, ratio):
        super().__init__()
        self.output_type = output_type
        self.conf = conf
        self.ratio = ratio

    def forward(self, data):
        post_result, pre_post_boxes = data
        result = []
        for i in trange(int(post_result.size(0) * self.ratio)):
            if float(post_result[i].max()) < self.conf:
                break
            if self.output_type in ("class", "all"):
                result.append(post_result[i].max())
            if self.output_type in ("box", "all"):
                for j in range(4):
                    result.append(pre_post_boxes[i, j])
        return sum(result)


class MultimodalGradCAMHeatmap:
    def __init__(
        self,
        weight,
        device,
        method,
        layers,
        backward_type,
        conf_threshold,
        ratio,
        show_box,
        renormalize,
    ):
        self.device = torch.device(device)
        self.model = attempt_load_weights(weight, self.device)
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad_(True)

        self.layers = layers
        self.method_name = method
        self.backward_type = backward_type
        self.conf_threshold = conf_threshold
        self.ratio = ratio
        self.show_box = show_box
        self.renormalize = renormalize

        self.target = YoloDetectionTarget(backward_type, conf_threshold, ratio)
        names = self.model.names if hasattr(self.model, "names") else {0: "obj"}
        self.model_names = names
        self.colors = [(0, 0, 255) for _ in range(len(names))]

    def _build_cam(self, layer_idx):
        target_layer = self.model.model[layer_idx]
        cam_cls = CAM_METHODS[self.method_name]
        cam = cam_cls(self.model, [target_layer])
        cam.activations_and_grads = SafeActivationsAndGradients(self.model, [target_layer], None)
        return cam

    def _prepare_tensor(self, rgb_path, ir_path):
        img_rgb = cv2.imread(rgb_path)
        img_ir = cv2.imread(ir_path)
        if img_rgb is None or img_ir is None:
            raise FileNotFoundError(f"Failed to read RGB/IR image: {rgb_path} | {ir_path}")

        img_rgb = letterbox(img_rgb)[0]
        img_ir = letterbox(img_ir)[0]

        rgb_norm = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        ir_norm = cv2.cvtColor(img_ir, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        mm = np.concatenate((rgb_norm, ir_norm), axis=2)

        tensor = torch.from_numpy(np.transpose(mm, (2, 0, 1))).unsqueeze(0).to(self.device)
        return tensor, rgb_norm, img_rgb

    def _predict_boxes(self, tensor):
        pred = self.model(tensor)[0]
        pred = non_max_suppression(pred, conf_thres=self.conf_threshold, iou_thres=0.65)[0]
        return pred

    def _draw_boxes(self, pred, cam_image):
        for data in pred:
            data = data.cpu().detach().numpy()
            x1, y1, x2, y2 = list(map(int, list(data[:4])))
            cls_id = int(data[4:].argmax())
            conf = float(data[4:].max())
            cv2.rectangle(cam_image, (x1, y1), (x2, y2), tuple(int(x) for x in self.colors[cls_id]), 2)
            cv2.putText(
                cam_image,
                f"{self.model_names[cls_id]} {conf:.2f}",
                (x1 - 10, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                tuple(int(x) for x in self.colors[cls_id]),
                1,
                lineType=cv2.LINE_AA,
            )
        return cam_image

    def _renormalize_cam_in_bounding_boxes(self, boxes, image_float_np, grayscale_cam):
        if boxes is None or len(boxes) == 0:
            renorm = scale_cam_image(grayscale_cam.copy())
            return show_cam_on_image(image_float_np, renorm, use_rgb=True)

        # 仅作为可选项：默认不截断全图响应
        renorm = np.clip(grayscale_cam.copy(), 0, None)
        renorm = scale_cam_image(renorm)
        return show_cam_on_image(image_float_np, renorm, use_rgb=True)

    def process_one(self, rgb_path, ir_path, save_dir):
        os.makedirs(save_dir, exist_ok=True)
        tensor, rgb_norm, rgb_bgr = self._prepare_tensor(rgb_path, ir_path)
        pred = self._predict_boxes(tensor)
        boxes = pred[:, :4].cpu().detach().numpy().astype(np.int32) if len(pred) else np.zeros((0, 4), dtype=np.int32)

        for layer_idx in self.layers:
            cam = self._build_cam(layer_idx)
            try:
                grayscale_cam = cam(tensor, [self.target])[0, :]
                grayscale_cam = cv2.GaussianBlur(grayscale_cam, (0, 0), sigmaX=1.2, sigmaY=1.2)
                grayscale_cam = cv2.resize(grayscale_cam, (rgb_norm.shape[1], rgb_norm.shape[0]), interpolation=cv2.INTER_CUBIC)
                grayscale_cam = np.clip(grayscale_cam, 0, None)
                grayscale_cam = grayscale_cam / (grayscale_cam.max() + 1e-8)
                grayscale_cam = grayscale_cam ** 1.4

                cam_image = show_cam_on_image(rgb_norm, grayscale_cam, use_rgb=True)

                if self.renormalize:
                    cam_image = self._renormalize_cam_in_bounding_boxes(boxes, rgb_norm, grayscale_cam)
                else:
                    cam_image = show_cam_on_image(rgb_norm, grayscale_cam, use_rgb=True)

                if self.show_box and len(pred):
                    cam_image = self._draw_boxes(pred, cam_image)

                layer_name = self.model.model[layer_idx].__class__.__name__
                out_name = f"layer_{layer_idx}_{layer_name}_gradcam.png"
                Image.fromarray(cam_image).save(os.path.join(save_dir, out_name))
                print(f"Saved: {out_name}")
            finally:
                if hasattr(cam, "activations_and_grads") and cam.activations_and_grads is not None:
                    cam.activations_and_grads.release()

    def __call__(self, rgb_path, ir_path, save_dir):
        if os.path.exists(save_dir):
            for f in os.listdir(save_dir):
                fp = os.path.join(save_dir, f)
                if os.path.isfile(fp):
                    os.remove(fp)

        os.makedirs(save_dir, exist_ok=True)

        if os.path.isdir(rgb_path):
            if not os.path.isdir(ir_path):
                raise ValueError("When --img-rgb is a directory, --img-ir must also be a directory.")

            valid_ext = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")
            rgb_files = sorted([f for f in os.listdir(rgb_path) if f.lower().endswith(valid_ext)])

            for rgb_name in rgb_files:
                rgb_file = os.path.join(rgb_path, rgb_name)
                ir_file = os.path.join(ir_path, rgb_name)
                if not os.path.exists(ir_file):
                    print(f"Skip {rgb_name}: IR pair not found")
                    continue
                sample_name = os.path.splitext(rgb_name)[0]
                sample_dir = os.path.join(save_dir, sample_name)
                self.process_one(rgb_file, ir_file, sample_dir)
        else:
            self.process_one(rgb_path, ir_path, save_dir)


def parse_args():
    parser = argparse.ArgumentParser(description="Multimodal detection Grad-CAM heatmap generator")
    parser.add_argument("--img-rgb", type=str, default='/home/asefu/data/M3FD_YOLO/images/train/00116.png', help="RGB image path or folder")
    parser.add_argument("--img-ir", type=str, default='/home/asefu/data/M3FD_YOLO/image/train/00116.png', help="IR image path or folder")
    parser.add_argument("--save-dir", type=str, default="/home/asefu/project/WaveMamba-main/runs/gradcam/M3FD/train_00116", help="Output directory")
    parser.add_argument(
        "--weight",
        type=str,
        default="/home/asefu/project/WaveMamba-main/runs/M3FD_latest/midlle_p1_p2_p33_p4/weights/best.pt",
        help="Model checkpoint path",
    )
    parser.add_argument("--device", type=str, default="cuda:0", help="cuda:0 or cpu")
    parser.add_argument(
        "--method",
        type=str,
        default="GradCAM",
        choices=list(CAM_METHODS.keys()),
        help="CAM method",
    )
    parser.add_argument(
        "--layers",
        type=str,
        default="0, 1, 10, 13, 20, 25, 32, 38, 33, 39, 49, 52, 55",
        help="Comma-separated layer indices (prefer higher-resolution fusion layers to reduce blocky CAMs)",
    )
    parser.add_argument("--backward-type", type=str, default="all", choices=["class", "box", "all"])
    parser.add_argument("--conf-threshold", type=float, default=0.5)
    parser.add_argument("--ratio", type=float, default=0.02)
    parser.add_argument("--show-box", action="store_true", help="Draw predicted boxes")
    parser.add_argument("--renormalize", action="store_true", help="Enable bbox-wise renormalize (off by default for full-image response)")
    return parser.parse_args()


def main():
    args = parse_args()
    layers = [int(x.strip()) for x in args.layers.split(",") if x.strip()]

    generator = MultimodalGradCAMHeatmap(
        weight=args.weight,
        device=args.device,
        method=args.method,
        layers=layers,
        backward_type=args.backward_type,
        conf_threshold=args.conf_threshold,
        ratio=args.ratio,
        show_box=args.show_box,
        renormalize=args.renormalize,
    )
    generator(args.img_rgb, args.img_ir, args.save_dir)


if __name__ == "__main__":
    main()
