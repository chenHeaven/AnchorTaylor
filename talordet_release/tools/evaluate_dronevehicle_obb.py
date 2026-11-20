#!/usr/bin/env python3
"""Evaluate DroneVehicle OBB predictions with one framework-independent metric.

Both inputs are CSV files. Ground-truth rows contain::

    image_id,class_name,x1,y1,x2,y2,x3,y3,x4,y4[,ignore]

Prediction rows contain::

    image_id,class_name,confidence,x1,y1,x2,y2,x3,y3,x4,y4

Coordinates are absolute pixels in the common cropped 640x512 image view.
AP follows the interpolation used by Ultralytics: a monotonic precision
envelope integrated on 101 evenly-spaced recall samples.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


CLASSES = ("Car", "Bus", "Truck", "Freight car", "Van")
CLASS_ALIASES = {
    "car": "Car",
    "bus": "Bus",
    "truck": "Truck",
    "freightcar": "Freight car",
    "freight": "Freight car",
    "cargo": "Freight car",
    "van": "Van",
}
IOU_THRESHOLDS = np.linspace(0.50, 0.95, 10)


@dataclass(frozen=True)
class ObjectRecord:
    image_id: str
    class_name: str
    points: np.ndarray
    confidence: float = 1.0
    ignore: bool = False


def canonical_class(name: str) -> str:
    key = "".join(character for character in name.lower() if character.isalnum())
    try:
        return CLASS_ALIASES[key]
    except KeyError as exc:
        raise ValueError(f"unknown DroneVehicle class: {name!r}") from exc


def polygon(values: list[str], source: Path, line_number: int) -> np.ndarray:
    if len(values) != 8:
        raise ValueError(f"{source}:{line_number}: expected eight polygon coordinates")
    points = np.asarray([float(value) for value in values], dtype=np.float32).reshape(4, 2)
    if not np.all(np.isfinite(points)):
        raise ValueError(f"{source}:{line_number}: non-finite polygon")
    hull = cv2.convexHull(points).reshape(-1, 2)
    if len(hull) < 3 or cv2.contourArea(hull) <= 0:
        raise ValueError(f"{source}:{line_number}: degenerate polygon")
    return hull


def read_ground_truth(path: Path) -> list[ObjectRecord]:
    rows: list[ObjectRecord] = []
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"image_id", "class_name", *(f"{axis}{index}" for index in range(1, 5) for axis in "xy")}
        if reader.fieldnames is None or not required.issubset(reader.fieldnames):
            raise ValueError(f"{path}: missing columns {sorted(required - set(reader.fieldnames or []))}")
        for line_number, row in enumerate(reader, 2):
            points = polygon(
                [row[f"{axis}{index}"] for index in range(1, 5) for axis in "xy"], path, line_number
            )
            ignored = str(row.get("ignore", "0")).strip().lower() in {"1", "true", "yes"}
            rows.append(ObjectRecord(str(row["image_id"]), canonical_class(row["class_name"]), points, ignore=ignored))
    return rows


def read_predictions(path: Path) -> list[ObjectRecord]:
    rows: list[ObjectRecord] = []
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        required = {
            "image_id", "class_name", "confidence",
            *(f"{axis}{index}" for index in range(1, 5) for axis in "xy"),
        }
        if reader.fieldnames is None or not required.issubset(reader.fieldnames):
            raise ValueError(f"{path}: missing columns {sorted(required - set(reader.fieldnames or []))}")
        for line_number, row in enumerate(reader, 2):
            confidence = float(row["confidence"])
            if not np.isfinite(confidence) or confidence < 0 or confidence > 1:
                raise ValueError(f"{path}:{line_number}: invalid confidence {confidence}")
            points = polygon(
                [row[f"{axis}{index}"] for index in range(1, 5) for axis in "xy"], path, line_number
            )
            rows.append(ObjectRecord(str(row["image_id"]), canonical_class(row["class_name"]), points, confidence))
    return rows


def rotated_iou(first: np.ndarray, second: np.ndarray) -> float:
    area_first = float(cv2.contourArea(first))
    area_second = float(cv2.contourArea(second))
    intersection, _ = cv2.intersectConvexConvex(first, second)
    union = area_first + area_second - float(intersection)
    return float(intersection) / union if union > 0 else 0.0


def compute_ap(recall: np.ndarray, precision: np.ndarray) -> float:
    """Match Ultralytics' 101-sample interpolated AP implementation."""
    modified_recall = np.concatenate(([0.0], recall, [1.0]))
    modified_precision = np.concatenate(([1.0], precision, [0.0]))
    modified_precision = np.flip(np.maximum.accumulate(np.flip(modified_precision)))
    samples = np.linspace(0.0, 1.0, 101)
    return float(np.trapz(np.interp(samples, modified_recall, modified_precision), samples))


def evaluate_class(
    ground_truth: list[ObjectRecord], predictions: list[ObjectRecord], thresholds: np.ndarray
) -> tuple[np.ndarray, int, int]:
    usable = [record for record in ground_truth if not record.ignore]
    ignored = [record for record in ground_truth if record.ignore]
    ground_truth_by_image: dict[str, list[ObjectRecord]] = defaultdict(list)
    ignored_by_image: dict[str, list[ObjectRecord]] = defaultdict(list)
    for record in usable:
        ground_truth_by_image[record.image_id].append(record)
    for record in ignored:
        ignored_by_image[record.image_id].append(record)

    ordered_predictions = sorted(predictions, key=lambda record: record.confidence, reverse=True)
    true_positive = np.zeros((len(ordered_predictions), len(thresholds)), dtype=np.float64)
    false_positive = np.zeros_like(true_positive)
    matched = [defaultdict(set) for _ in thresholds]

    for prediction_index, prediction in enumerate(ordered_predictions):
        candidates = ground_truth_by_image.get(prediction.image_id, [])
        overlaps = np.asarray([rotated_iou(prediction.points, item.points) for item in candidates])
        ignored_overlaps = np.asarray([
            rotated_iou(prediction.points, item.points) for item in ignored_by_image.get(prediction.image_id, [])
        ])
        order = np.argsort(-overlaps) if len(overlaps) else np.empty(0, dtype=np.int64)
        for threshold_index, threshold in enumerate(thresholds):
            selected = next(
                (
                    int(index) for index in order
                    if overlaps[index] >= threshold and int(index) not in matched[threshold_index][prediction.image_id]
                ),
                None,
            )
            if selected is not None:
                matched[threshold_index][prediction.image_id].add(selected)
                true_positive[prediction_index, threshold_index] = 1.0
            elif len(ignored_overlaps) and float(ignored_overlaps.max()) >= threshold:
                continue
            else:
                false_positive[prediction_index, threshold_index] = 1.0

    ap = np.zeros(len(thresholds), dtype=np.float64)
    if usable:
        for threshold_index in range(len(thresholds)):
            accumulated_tp = np.cumsum(true_positive[:, threshold_index])
            accumulated_fp = np.cumsum(false_positive[:, threshold_index])
            recall = accumulated_tp / len(usable)
            precision = accumulated_tp / np.maximum(accumulated_tp + accumulated_fp, 1e-16)
            ap[threshold_index] = compute_ap(recall, precision)
    return ap, len(usable), len(ordered_predictions)


def evaluate(
    ground_truth: list[ObjectRecord], predictions: list[ObjectRecord], image_ids: set[str] | None = None
) -> dict[str, object]:
    evaluation_images = image_ids or {item.image_id for item in ground_truth}
    missing_ground_truth_images = {item.image_id for item in ground_truth} - evaluation_images
    if missing_ground_truth_images:
        raise ValueError(f"ground truth contains {len(missing_ground_truth_images)} IDs absent from image list")
    unknown_images = sorted({item.image_id for item in predictions} - evaluation_images)
    if unknown_images:
        raise ValueError(f"predictions contain {len(unknown_images)} image IDs absent from ground truth")

    per_class: dict[str, dict[str, object]] = {}
    all_ap = []
    for class_name in CLASSES:
        ap, ground_truth_count, prediction_count = evaluate_class(
            [item for item in ground_truth if item.class_name == class_name],
            [item for item in predictions if item.class_name == class_name],
            IOU_THRESHOLDS,
        )
        all_ap.append(ap)
        per_class[class_name] = {
            "ground_truths": ground_truth_count,
            "predictions": prediction_count,
            "AP50": float(ap[0]),
            "AP50_95": float(ap.mean()),
            "AP_by_IoU": {f"{threshold:.2f}": float(value) for threshold, value in zip(IOU_THRESHOLDS, ap)},
        }
    matrix = np.stack(all_ap)
    return {
        "protocol": "rotated IoU, Ultralytics-style 101-point interpolated AP",
        "iou_thresholds": [float(value) for value in IOU_THRESHOLDS],
        "class_order": list(CLASSES),
        "images": len(evaluation_images),
        "ground_truths": sum(not item.ignore for item in ground_truth),
        "predictions": len(predictions),
        "mAP50": float(matrix[:, 0].mean()),
        "mAP50_95": float(matrix.mean()),
        "classes": per_class,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ground-truth", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--image-ids", type=Path, help="One evaluated image ID per line, including empty images.")
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    image_ids = None
    if args.image_ids:
        image_ids = {line.strip() for line in args.image_ids.read_text().splitlines() if line.strip()}
        if not image_ids:
            raise ValueError(f"{args.image_ids}: no image IDs")
    report = evaluate(read_ground_truth(args.ground_truth), read_predictions(args.predictions), image_ids)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
