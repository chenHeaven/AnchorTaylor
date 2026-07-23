import argparse
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO


# 颜色映射（按照 detect_one_image.py 的标准）
CLASS_COLORS = {
    0: (255, 0, 0),          # car - 红色
    1: (0, 255, 255),        # truck - 青蓝
    2: (0, 255, 0),          # people - 绿色
    3: (255, 165, 0),        # bus - 橙色
    4: (255, 255, 0),        # lamp - 黄色
    5: (255, 0, 255),        # motorcycle - 品红/洋红
}

COLORS = CLASS_COLORS


def letterbox(im, new_shape=(640, 640), color=(114, 114, 114), auto=False, scaleFill=True, scaleup=True, stride=32):
    """Resize and pad image while meeting stride-multiple constraints."""
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Detect on two-stream images (RGB + Thermal/IR) using YOLOv8.")
    parser.add_argument("--rgb-image", default="/home/asefu/data/M3FD_YOLO/images/val/02656.png", help="Input RGB image path.")
    parser.add_argument("--thermal-image", default="/home/asefu/data/M3FD_YOLO/image/val/02656.png", help="Input thermal/IR image path.")
    parser.add_argument("--weights", default="/home/asefu/project/WaveMamba-main/runs/M3FD_latest/midlle_p1_p2_p33_p4/weights/best.pt", help="YOLOv8 weights path (trained on two-stream input).")
    parser.add_argument("--output-rgb", default="/home/asefu/project/WaveMamba-main/runs/detect/M3FD", help="Output RGB image path with detections.")
    parser.add_argument("--output-thermal", default="/home/asefu/project/WaveMamba-main/runs/detect/M3FD", help="Output thermal image path with detections.")
    parser.add_argument(
        "--labels",
        action="store_true",
        help="Draw class labels and confidence text on the images.",
    )
    parser.add_argument(
        "--class-names",
        action="store_true",
        default=True,
        help="Whether to display class names in labels (default: True).",
    )
    parser.add_argument(
        "--line-width",
        type=int,
        default=2,
        help="Bounding box line width.",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.5,
        help="Confidence threshold.",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
        help="Image size for inference.",
    )
    return parser.parse_args()


def scale_coords(coords, ratio, pad):
    """Convert coordinates from letterbox space back to original image space.
    
    Args:
        coords: (x1, y1, x2, y2) in letterbox space
        ratio: (ratio_x, ratio_y) from letterbox
        pad: (dw, dh) padding from letterbox
    
    Returns:
        (x1, y1, x2, y2) in original image space
    """
    x1, y1, x2, y2 = coords
    ratio_x, ratio_y = ratio
    dw, dh = pad
    
    # 反向转换：先减去 padding，再除以 ratio
    x1 = max(0, int((x1 - dw) / ratio_x))
    y1 = max(0, int((y1 - dh) / ratio_y))
    x2 = int((x2 - dw) / ratio_x)
    y2 = int((y2 - dh) / ratio_y)
    
    return x1, y1, x2, y2


def draw_detections(image, boxes, names, labels_enabled: bool, line_width: int, show_class_names: bool = True, ratio=None, pad=None):
    """Draw bounding boxes and labels on image using CLASS_COLORS.
    
    Args:
        image: Image to draw on
        boxes: Detection boxes from YOLO
        names: Class names dictionary
        labels_enabled: Whether to draw labels
        line_width: Line width for boxes
        show_class_names: Whether to display class names in the label text
        ratio: Optional (ratio_x, ratio_y) for coordinate scaling
        pad: Optional (dw, dh) padding for coordinate scaling
    """
    for box in boxes:
        cls_id = int(box.cls[0].item())
        # 使用 CLASS_COLORS，注意 cv2 使用 BGR 而不是 RGB，所以需要反转
        color_rgb = COLORS.get(cls_id, (255, 255, 255))
        color_bgr = tuple(reversed(color_rgb))
        
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
        
        # 如果提供了 ratio 和 pad，进行坐标转换
        if ratio is not None and pad is not None:
            x1, y1, x2, y2 = scale_coords((x1, y1, x2, y2), ratio, pad)

        cv2.rectangle(image, (x1, y1), (x2, y2), color_bgr, line_width)

        if labels_enabled:
            confidence = float(box.conf[0].item())
            if show_class_names:
                class_name = names.get(cls_id, str(cls_id)) if names else str(cls_id)
                label = f"{cls_id}:{class_name} {confidence:.2f}"
            else:
                label = f"{cls_id} {confidence:.2f}"
            (text_width, text_height), baseline = cv2.getTextSize(
                label,
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                1,
            )

            text_top = max(y1 - text_height - baseline - 4, 0)
            text_bottom = max(y1, text_height + baseline + 4)
            cv2.rectangle(
                image,
                (x1, text_top),
                (x1 + text_width + 4, text_bottom),
                color_bgr,
                thickness=-1,
            )
            cv2.putText(
                image,
                label,
                (x1 + 2, text_bottom - baseline - 2),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )


def resolve_output_path(image_path: Path, output_path: Path, suffix: str = "", conf: float = 0.5) -> Path:
    if output_path.suffix:
        return output_path

    output_path.mkdir(parents=True, exist_ok=True)
    img_suffix = image_path.suffix or ".png"
    return output_path / f"{image_path.stem}{suffix}_det_conf{conf}{img_suffix}"


def main() -> int:
    args = parse_args()

    rgb_path = Path(args.rgb_image)
    thermal_path = Path(args.thermal_image)
    weights_path = Path(args.weights)

    if not rgb_path.exists():
        raise FileNotFoundError(f"RGB image not found: {rgb_path}")
    if not thermal_path.exists():
        raise FileNotFoundError(f"Thermal image not found: {thermal_path}")
    if not weights_path.exists():
        raise FileNotFoundError(f"Weights not found: {weights_path}")

    # 读取两个图像
    rgb_img = cv2.imread(str(rgb_path))
    thermal_img = cv2.imread(str(thermal_path))
    
    if rgb_img is None:
        raise RuntimeError(f"Failed to read RGB image: {rgb_path}")
    if thermal_img is None:
        raise RuntimeError(f"Failed to read thermal image: {thermal_path}")

    # 保存原始尺寸用于后续恢复
    orig_rgb_shape = rgb_img.shape[:2]
    orig_thermal_shape = thermal_img.shape[:2]

    # Letterbox 处理
    rgb_img_padded, rgb_ratio, rgb_pad = letterbox(rgb_img, new_shape=args.imgsz)
    thermal_img_padded, thermal_ratio, thermal_pad = letterbox(thermal_img, new_shape=args.imgsz)

    # 拼接两个图像沿着通道维度（双分支输入）
    combined_img = np.concatenate((rgb_img_padded, thermal_img_padded), axis=2)

    # 加载模型
    model = YOLO(str(weights_path))
    
    # 推理
    results = model.predict(source=combined_img, verbose=False, conf=args.conf, imgsz=args.imgsz)
    if not results:
        raise RuntimeError("No prediction result returned by YOLO.")

    result = results[0]
    names = getattr(result, "names", None) or getattr(model, "names", None) or {}

    # 重新读取原始图像用于绘制（不带 padding）
    rgb_display = cv2.imread(str(rgb_path))
    thermal_display = cv2.imread(str(thermal_path))

    # 在两个原始图像上绘制检测结果（传入 ratio 和 pad 进行坐标转换）
    draw_detections(rgb_display, result.boxes, names, labels_enabled=args.labels, line_width=max(1, args.line_width), show_class_names=args.class_names, ratio=rgb_ratio, pad=rgb_pad)
    draw_detections(thermal_display, result.boxes, names, labels_enabled=args.labels, line_width=max(1, args.line_width), show_class_names=args.class_names, ratio=thermal_ratio, pad=thermal_pad)

    # 确定输出路径
    output_rgb_path = Path(args.output_rgb) if args.output_rgb else rgb_path.parent
    if output_rgb_path.suffix:
        # 如果有文件扩展名，作为文件路径
        output_rgb_file = output_rgb_path
    else:
        # 如果没有扩展名，作为文件夹路径
        output_rgb_file = resolve_output_path(rgb_path, output_rgb_path, "_rgb", args.conf)
    
    output_thermal_path = Path(args.output_thermal) if args.output_thermal else thermal_path.parent
    if output_thermal_path.suffix:
        # 如果有文件扩展名，作为文件路径
        output_thermal_file = output_thermal_path
    else:
        # 如果没有扩展名，作为文件夹路径
        output_thermal_file = resolve_output_path(thermal_path, output_thermal_path, "_thermal", args.conf)

    # 保存结果
    output_rgb_file.parent.mkdir(parents=True, exist_ok=True)
    output_thermal_file.parent.mkdir(parents=True, exist_ok=True)

    success_rgb = cv2.imwrite(str(output_rgb_file), rgb_display)
    success_thermal = cv2.imwrite(str(output_thermal_file), thermal_display)

    if not success_rgb:
        raise RuntimeError(f"Failed to write RGB output image: {output_rgb_file}")
    if not success_thermal:
        raise RuntimeError(f"Failed to write thermal output image: {output_thermal_file}")

    print(f"Saved RGB detection image to: {output_rgb_file}")
    print(f"Saved thermal detection image to: {output_thermal_file}")
    print(f"Total detections: {len(result.boxes)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
