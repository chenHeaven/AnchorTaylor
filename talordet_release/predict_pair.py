#!/usr/bin/env python3
"""Run AnchorTaylor inference on one aligned RGB/infrared image pair."""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

from ultralytics import YOLO


def letterbox(image: np.ndarray, size: int) -> np.ndarray:
    return cv2.resize(image, (size, size), interpolation=cv2.INTER_LINEAR)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--rgb", type=Path, required=True)
    parser.add_argument("--ir", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("runs/predict_pair"))
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.65)
    parser.add_argument("--device", default="0")
    return parser.parse_args()


def read_image(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    return image


def main() -> int:
    args = parse_args()
    if not args.weights.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {args.weights}")
    if args.imgsz <= 0 or args.imgsz % 32:
        raise ValueError("--imgsz must be a positive multiple of 32")

    rgb = read_image(args.rgb)
    ir = read_image(args.ir)
    if rgb.shape[:2] != ir.shape[:2]:
        raise ValueError(f"Aligned pair has different shapes: RGB={rgb.shape}, IR={ir.shape}")

    original_h, original_w = rgb.shape[:2]
    paired = np.concatenate((letterbox(rgb, args.imgsz), letterbox(ir, args.imgsz)), axis=2)
    model = YOLO(str(args.weights))
    result = model.predict(
        source=paired,
        imgsz=args.imgsz,
        conf=args.conf,
        iou=args.iou,
        device=args.device,
        verbose=False,
    )[0]

    rendered = rgb.copy()
    scale_x, scale_y = original_w / args.imgsz, original_h / args.imgsz
    names = result.names or {}
    for box in result.boxes:
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        p1 = (round(x1 * scale_x), round(y1 * scale_y))
        p2 = (round(x2 * scale_x), round(y2 * scale_y))
        class_id = int(box.cls.item())
        confidence = float(box.conf.item())
        label = f"{names.get(class_id, class_id)} {confidence:.2f}"
        cv2.rectangle(rendered, p1, p2, (0, 255, 0), 2)
        cv2.putText(rendered, label, (p1[0], max(16, p1[1] - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (0, 255, 0), 1, cv2.LINE_AA)

    args.output.mkdir(parents=True, exist_ok=True)
    output_path = args.output / f"{args.rgb.stem}_detections.png"
    if not cv2.imwrite(str(output_path), rendered):
        raise RuntimeError(f"Failed to write: {output_path}")
    print(f"detections={len(result.boxes)} output={output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
