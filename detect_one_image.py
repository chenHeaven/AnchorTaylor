import argparse
from pathlib import Path

import cv2
from ultralytics import YOLO


CLASS_COLORS = {
    0: (255, 0, 0),
    1: (0, 180, 255),
    2: (0, 255, 0),
    3: (255, 128, 0),
    4: (255, 255, 0),
    5: (255, 0, 255),
}

COLORS = CLASS_COLORS


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Draw YOLOv8 detections on a single image.")
    parser.add_argument("--image", default="/root/shared-nvme/flir_align/image/val/FLIR_09264.jpeg", help="Input image path.")
    parser.add_argument("--weights", default="/root/shared-nvme/project/yolo/runs/flir/yolov8_ir/weights/best.pt", help="YOLOv8 weights path, for example runs/detect/rgb/weights/best.pt.")
    parser.add_argument("--output", default="/root/shared-nvme/project/yolo/detect/flir/IR", help="Output image path.")
    parser.add_argument(
        "--labels",
        action="store_true",
        help="Draw class labels and confidence text on the image.",
    )
    parser.add_argument(
        "--line-width",
        type=int,
        default=2,
        help="Bounding box line width.",
    )
    return parser.parse_args()


def draw_detections(image, boxes, names, labels_enabled: bool, line_width: int):
    for box in boxes:
        cls_id = int(box.cls[0].item())
        color = tuple(reversed(COLORS.get(cls_id, (255, 255, 255))))
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())

        cv2.rectangle(image, (x1, y1), (x2, y2), color, line_width)

        if labels_enabled:
            class_name = names.get(cls_id, str(cls_id)) if names else str(cls_id)
            confidence = float(box.conf[0].item())
            label = f"{cls_id}:{class_name} {confidence:.2f}"
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
                color,
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


def resolve_output_path(image_path: Path, output_path: Path) -> Path:
    if output_path.suffix:
        return output_path

    output_path.mkdir(parents=True, exist_ok=True)
    suffix = image_path.suffix or ".png"
    return output_path / f"{image_path.stem}_det{suffix}"


def main() -> int:
    args = parse_args()

    image_path = Path(args.image)
    weights_path = Path(args.weights)
    output_path = Path(args.output)

    if not image_path.exists():
        raise FileNotFoundError(f"Input image not found: {image_path}")
    if not weights_path.exists():
        raise FileNotFoundError(f"Weights not found: {weights_path}")

    model = YOLO(str(weights_path))
    results = model.predict(source=str(image_path), verbose=False)
    if not results:
        raise RuntimeError("No prediction result returned by YOLO.")

    result = results[0]
    names = getattr(result, "names", None) or getattr(model, "names", None) or {}
    image = cv2.imread(str(image_path))
    if image is None:
        raise RuntimeError(f"Failed to read image: {image_path}")

    draw_detections(image, result.boxes, names, labels_enabled=args.labels, line_width=max(1, args.line_width))

    output_file = resolve_output_path(image_path, output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    success = cv2.imwrite(str(output_file), image)
    if not success:
        raise RuntimeError(f"Failed to write output image: {output_file}")

    print(f"Saved detection image to: {output_file}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())