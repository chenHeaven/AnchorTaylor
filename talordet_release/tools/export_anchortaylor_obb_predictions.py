#!/usr/bin/env python3
"""Export AnchorTaylor OBB predictions with exact image IDs and stretch inversion."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import torch

from ultralytics.models.yolo.obb import OBBValidator
from ultralytics.utils import ops


FIELDS = [
    "image_id",
    "class_name",
    "confidence",
    *(f"{axis}{index}" for index in range(1, 5) for axis in "xy"),
]


class PolygonExportValidator(OBBValidator):
    """Capture post-NMS polygons before the legacy xywh scale-back path."""

    def __init__(self, output_csv: Path, *args, **kwargs):
        self.output_csv = output_csv
        self.rows: list[dict[str, object]] = []
        self.seen_images: set[str] = set()
        super().__init__(*args, **kwargs)

    def update_metrics(self, preds, batch):
        input_height, input_width = batch["img"].shape[2:]
        if (input_height, input_width) != (640, 640):
            raise RuntimeError(f"stretch export expects 640x640 input, got {input_height}x{input_width}")
        for index, pred in enumerate(preds):
            image_id = Path(batch["im_file"][index]).stem
            if image_id in self.seen_images:
                raise ValueError(f"duplicate validation image ID: {image_id}")
            self.seen_images.add(image_id)
            if not len(pred):
                continue

            rboxes = torch.cat((pred[:, :4], pred[:, -1:]), dim=-1)
            polygons = ops.xywhr2xyxyxyxy(rboxes).detach().cpu().numpy()
            original_height, original_width = map(float, batch["ori_shape"][index])
            if (original_height, original_width) != (512, 640):
                raise RuntimeError(f"unexpected cropped source shape: {original_height}x{original_width}")
            polygons[..., 0] *= original_width / float(input_width)
            polygons[..., 1] *= original_height / float(input_height)
            confidences = pred[:, 4].detach().cpu().numpy()
            classes = pred[:, 5].int().detach().cpu().numpy()

            for polygon, confidence, class_id in zip(polygons, confidences, classes):
                flat = polygon.reshape(-1).tolist()
                row = {
                    "image_id": image_id,
                    "class_name": self.names[int(class_id)],
                    "confidence": float(confidence),
                }
                row.update(
                    {
                        f"{axis}{point_index}": float(flat[(point_index - 1) * 2 + offset])
                        for point_index in range(1, 5)
                        for offset, axis in enumerate("xy")
                    }
                )
                self.rows.append(row)

        super().update_metrics(preds, batch)

    def write_predictions(self) -> None:
        self.output_csv.parent.mkdir(parents=True, exist_ok=True)
        with self.output_csv.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=FIELDS)
            writer.writeheader()
            writer.writerows(self.rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="0")
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--image-ids", type=Path, help="Expected validation manifest, including empty images")
    args = parser.parse_args()
    torch.manual_seed(0)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False

    validator_args = {
        "model": str(args.weights),
        "data": str(args.data),
        "device": args.device,
        "batch": args.batch,
        "imgsz": args.imgsz,
        "conf": 0.001,
        "iou": 0.7,
        "max_det": 300,
        "workers": 8,
        "plots": False,
        "save_txt": False,
        "save_json": False,
        "verbose": False,
        "half": False,
        "augment": False,
        "project": str(args.output.parent),
        "name": "native_validation",
    }
    validator = PolygonExportValidator(output_csv=args.output, args=validator_args)
    native_stats = validator(model=str(args.weights))
    if len(validator.seen_images) != 1469:
        raise RuntimeError(f"expected 1469 images, exported {len(validator.seen_images)}")
    if args.image_ids:
        expected_ids = {line.strip() for line in args.image_ids.read_text().splitlines() if line.strip()}
        if validator.seen_images != expected_ids:
            raise RuntimeError("exported image IDs differ from the common validation manifest")
    validator.write_predictions()
    args.output.with_name("exported_image_ids.txt").write_text(
        "\n".join(sorted(validator.seen_images)) + "\n"
    )
    native_path = args.output.with_name("native_metrics.json")
    native_path.write_text(json.dumps({key: float(value) for key, value in native_stats.items()}, indent=2) + "\n")
    print(f"exported {len(validator.rows)} predictions for {len(validator.seen_images)} images to {args.output}")


if __name__ == "__main__":
    main()
