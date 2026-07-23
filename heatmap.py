
import warnings
warnings.filterwarnings('ignore')
warnings.simplefilter('ignore')
import torch, yaml, cv2, os, shutil, sys
import argparse
import numpy as np
np.random.seed(0)
import matplotlib.pyplot as plt
from tqdm import trange
from PIL import Image
from ultralytics.nn.tasks import attempt_load_weights
from ultralytics.utils.torch_utils import intersect_dicts
from ultralytics.utils.ops import xywh2xyxy, non_max_suppression
from pytorch_grad_cam import GradCAMPlusPlus, GradCAM, XGradCAM, EigenCAM, LayerCAM, RandomCAM, EigenGradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image, scale_cam_image
from pytorch_grad_cam.activations_and_gradients import ActivationsAndGradients

def letterbox(im, new_shape=(640, 640), color=(114, 114, 114), auto=False, scaleFill=True, scaleup=True, stride=32):
    # Resize and pad image while meeting stride-multiple constraints
    shape = im.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:  # only scale down, do not scale up (for better val mAP)
        r = min(r, 1.0)

    # Compute padding
    ratio = r, r  # width, height ratios
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
    if auto:  # minimum rectangle
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding
    elif scaleFill:  # stretch
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[1], new_shape[0])
        ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # width, height ratios

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
    return im, ratio, (dw, dh)

class ActivationsAndGradients:
    """ Class for extracting activations and
    registering gradients from targetted intermediate layers """

    def __init__(self, model, target_layers, reshape_transform):
        self.model = model
        self.gradients = []
        self.activations = []
        self.reshape_transform = reshape_transform
        self.handles = []
        for target_layer in target_layers:
            self.handles.append(
                target_layer.register_forward_hook(self.save_activation))
            # Because of https://github.com/pytorch/pytorch/issues/61519,
            # we don't use backward hook to record gradients.
            self.handles.append(
                target_layer.register_forward_hook(self.save_gradient))

    def save_activation(self, module, input, output):
        activation = output

        if self.reshape_transform is not None:
            activation = self.reshape_transform(activation)
        self.activations.append(activation.cpu().detach())

    def save_gradient(self, module, input, output):
        if not hasattr(output, "requires_grad") or not output.requires_grad:
            # You can only register hooks on tensor requires grad.
            return

        # Gradients are computed in reverse order
        def _store_grad(grad):
            if self.reshape_transform is not None:
                grad = self.reshape_transform(grad)
            self.gradients = [grad.cpu().detach()] + self.gradients

        output.register_hook(_store_grad)

    def post_process(self, result):
        logits_ = result[:, 4:]
        boxes_ = result[:, :4]
        sorted, indices = torch.sort(logits_.max(1)[0], descending=True)
        return torch.transpose(logits_[0], dim0=0, dim1=1)[indices[0]], torch.transpose(boxes_[0], dim0=0, dim1=1)[indices[0]], xywh2xyxy(torch.transpose(boxes_[0], dim0=0, dim1=1)[indices[0]]).cpu().detach().numpy()
  
    def __call__(self, x):
        self.gradients = []
        self.activations = []
        model_output = self.model(x)
        post_result, pre_post_boxes, post_boxes = self.post_process(model_output[0])
        return [[post_result, pre_post_boxes]]

    def release(self):
        for handle in self.handles:
            handle.remove()

class yolov8_target(torch.nn.Module):
    def __init__(self, ouput_type, conf, ratio) -> None:
        super().__init__()
        self.ouput_type = ouput_type
        self.conf = conf
        self.ratio = ratio
    
    def forward(self, data):
        post_result, pre_post_boxes = data
        result = []
        for i in trange(int(post_result.size(0) * self.ratio)):
            if float(post_result[i].max()) < self.conf:
                break
            if self.ouput_type == 'class' or self.ouput_type == 'all':
                result.append(post_result[i].max())
            elif self.ouput_type == 'box' or self.ouput_type == 'all':
                for j in range(4):
                    result.append(pre_post_boxes[i, j])
        return sum(result)

class yolov8_heatmap:
    def __init__(self, weight, device, method, layer, backward_type, conf_threshold, ratio, show_box, renormalize):
        device = torch.device(device)
        ckpt = torch.load(weight)
        #print(ckpt)
        model_names = ckpt['model'].names
        model = attempt_load_weights(weight, device)
        model.info()
        for p in model.parameters():
            p.requires_grad_(True)
        model.eval()
        
        target = yolov8_target(backward_type, conf_threshold, ratio)
        
        # colors = np.random.uniform(0, 255, size=(len(model_names), 3)).astype(np.int64)
        colors=[(0,0,255)for i in range(len(model_names))]
        self.weight = weight
        self.device = device
        self.model_names = model_names
        self.model = model
        self.method_name = method
        self.layer = layer
        self.backward_type = backward_type
        self.conf_threshold = conf_threshold
        self.ratio = ratio
        self.show_box = show_box
        self.renormalize = renormalize
        self.target = target
        self.colors = colors

    def _build_cam_for_layer(self, layer_idx):
        target_layer = self.model.model[layer_idx]
        cam = eval(self.method_name)(self.model, [target_layer])
        cam.activations_and_grads = ActivationsAndGradients(self.model, [target_layer], None)
        return cam
    
    def post_process(self, result):
        result = non_max_suppression(result, conf_thres=self.conf_threshold, iou_thres=0.65)[0]
        return result

    def draw_detections(self, box, color, name, img):
        
        xmin, ymin, xmax, ymax = list(map(int, list(box)))
        cv2.rectangle(img, (xmin, ymin), (xmax, ymax), tuple(int(x) for x in color), 2)
        cv2.putText(img, str(name), (xmin-10, ymin - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, tuple(int(x) for x in color), 1, lineType=cv2.LINE_AA)
        return img

    def renormalize_cam_in_bounding_boxes(self, boxes, image_float_np, grayscale_cam):
        """Normalize the CAM to be in the range [0, 1] 
        inside every bounding boxes, and zero outside of the bounding boxes. """
        renormalized_cam = np.zeros(grayscale_cam.shape, dtype=np.float32)
        for x1, y1, x2, y2 in boxes:
            x1, y1 = max(x1, 0), max(y1, 0)
            x2, y2 = min(grayscale_cam.shape[1] - 1, x2), min(grayscale_cam.shape[0] - 1, y2)
            renormalized_cam[y1:y2, x1:x2] = scale_cam_image(grayscale_cam[y1:y2, x1:x2].copy())    
        renormalized_cam = scale_cam_image(grayscale_cam.copy())    
        renormalized_cam = scale_cam_image(renormalized_cam)
        eigencam_image_renormalized = show_cam_on_image(image_float_np, renormalized_cam, use_rgb=True)
        return eigencam_image_renormalized
    
    def process(self, img_path,imgir_path, save_dir):
        # img process
        img = cv2.imread(img_path)
        img_rgb=img
        img = letterbox(img)[0]
        #print(img.shape)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = np.float32(img) / 255.0
        
        
        
        
        #print(img.shape)

        imgir = cv2.imread(imgir_path)
        img_ir=imgir
        imgir = letterbox(imgir)[0]
        imgir = cv2.cvtColor(imgir, cv2.COLOR_BGR2RGB)
        imgir = np.float32(imgir) / 255.0
 

        img=np.concatenate((img,imgir),axis=2)

        tensor = torch.from_numpy(np.transpose(img, axes=[2, 0, 1])).unsqueeze(0).to(self.device)

        
        pred = self.model(tensor)[0]
        pred = self.post_process(pred)

   
        
        img=img[...,0:3]
        os.makedirs(save_dir, exist_ok=True)

        pred_boxes = pred[:, :4].cpu().detach().numpy().astype(np.int32) if len(pred) else np.zeros((0, 4), dtype=np.int32)

        for layer_idx in self.layer:
            cam = self._build_cam_for_layer(layer_idx)
            try:
                grayscale_cam = cam(tensor, [self.target])
                grayscale_cam = grayscale_cam[0, :]
                cam_image = show_cam_on_image(img[...,:3], grayscale_cam, use_rgb=True)

                if self.renormalize and len(pred_boxes):
                    cam_image = self.renormalize_cam_in_bounding_boxes(pred_boxes, img, grayscale_cam)
                if self.show_box:
                    for data in pred:
                        data = data.cpu().detach().numpy()
                        cam_image = self.draw_detections(
                            data[:4],
                            self.colors[int(data[4:].argmax())],
                            f'{self.model_names[int(data[4:].argmax())]} {float(data[4:].max()):.2f}',
                            cam_image,
                        )

                layer_name = self.model.model[layer_idx].__class__.__name__
                out_name = f'layer_{layer_idx}_{layer_name}.png'
                Image.fromarray(cam_image).save(os.path.join(save_dir, out_name))
            finally:
                if hasattr(cam, 'activations_and_grads') and cam.activations_and_grads is not None:
                    cam.activations_and_grads.release()
    
    def __call__(self, img_path, imgir_path,save_path):
        # remove dir if exist
        if os.path.exists(save_path):
            shutil.rmtree(save_path)
        # make dir if not exist
        os.makedirs(save_path, exist_ok=True)

        if os.path.isdir(img_path):
            if not os.path.isdir(imgir_path):
                raise ValueError('When --img-rgb is a directory, --img-ir must also be a directory with matched filenames.')

            valid_ext = ('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff')
            rgb_files = sorted([f for f in os.listdir(img_path) if f.lower().endswith(valid_ext)])

            for rgb_name in rgb_files:
                rgb_file = os.path.join(img_path, rgb_name)
                ir_file = os.path.join(imgir_path, rgb_name)
                if not os.path.exists(ir_file):
                    print(f'[WARN] Skip {rgb_name}: matching IR file not found in {imgir_path}')
                    continue

                stem = os.path.splitext(rgb_name)[0]
                sample_save_dir = os.path.join(save_path, stem)
                self.process(rgb_file, ir_file, sample_save_dir)
        else:
            self.process(img_path, imgir_path, save_path)
        
def parse_args():
    parser = argparse.ArgumentParser(description='Generate heatmaps for selected YOLO layers.')
    parser.add_argument('--img-rgb', type=str, required=True, help='Path to RGB image')
    parser.add_argument('--img-ir', type=str, required=True, help='Path to IR image')
    parser.add_argument('--save-dir', type=str, default='heatmap_fusehd_fouthn', help='Output directory')
    parser.add_argument('--weight', type=str,
                        default='/home/asefu/project/WaveMamba-main/runs/M3FD_large111/nano_p1_p2_p33_p45/weights/best.pt',
                        help='Model checkpoint path')
    parser.add_argument('--device', type=str, default='cuda:0', help='Device, e.g. cuda:0 or cpu')
    parser.add_argument('--method', type=str, default='GradCAM',
                        choices=['GradCAMPlusPlus', 'GradCAM', 'XGradCAM', 'EigenCAM', 'LayerCAM', 'RandomCAM', 'EigenGradCAM'],
                        help='CAM method')
    parser.add_argument('--layers', type=str,
                        default='3,4,6,10,13,15,20,25,27,33,39,41,49,52,55',
                        help='Comma-separated layer indices')
    parser.add_argument('--conf-threshold', type=float, default=0.5, help='Confidence threshold')
    parser.add_argument('--ratio', type=float, default=0.02, help='Top-k ratio for backward targets')
    parser.add_argument('--show-box', action='store_true', help='Draw detection boxes on heatmap')
    parser.add_argument('--no-renormalize', action='store_true', help='Disable bbox-region renormalization')
    return parser.parse_args()


def get_params(args=None):
    default_layers = [3, 4, 6, 10, 13, 15, 20, 25, 27, 33, 39, 41, 49, 52, 55]

    if args is None:
        selected_layers = default_layers
        weight = '/home/asefu/project/WaveMamba-main/runs/M3FD_large111/nano_p1_p2_p33_p45/weights/best.pt'
        device = 'cuda:0'
        method = 'GradCAM'
        conf_threshold = 0.5
        ratio = 0.02
        show_box = False
        renormalize = True
    else:
        selected_layers = [int(x.strip()) for x in args.layers.split(',') if x.strip()]
        weight = args.weight
        device = args.device
        method = args.method
        conf_threshold = args.conf_threshold
        ratio = args.ratio
        show_box = args.show_box
        renormalize = not args.no_renormalize

    params = {
        'weight': weight,
        'device': device,
        'method': method, # GradCAMPlusPlus, GradCAM, XGradCAM, EigenCAM, LayerCAM, RandomCAM, EigenGradCAM
        'layer': selected_layers,
        'backward_type': 'all', # class, box, all
        'conf_threshold': conf_threshold, # 0.2
        'ratio': ratio, # 0.02-0.1
        'show_box': show_box,
        'renormalize': renormalize
    }
    return params

if __name__ == '__main__':
    cli_args = parse_args()
    model = yolov8_heatmap(**get_params(cli_args))
    model(cli_args.img_rgb, cli_args.img_ir, cli_args.save_dir)