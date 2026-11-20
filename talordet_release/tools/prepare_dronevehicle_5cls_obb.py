#!/usr/bin/env python3
"""Convert canonical DroneVehicle IR polygons to five-class Ultralytics OBB labels.

The paired 840x712 RGB/IR images are cropped by 100 pixels on every side,
matching the established 640x512 DroneVehicle protocol.  Source files are
never modified.  By default only labels and an audit manifest are written;
``--image-mode crop`` additionally materializes both cropped image streams.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import xml.etree.ElementTree as ET
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, JpegImagePlugin


CLASS_NAMES = ["bus", "car", "freight_car", "truck", "van"]
CLASS_TO_ID = {name: index for index, name in enumerate(CLASS_NAMES)}
ALIASES = {
    "bus": "bus",
    "car": "car",
    "freight car": "freight_car",
    "freight_car": "freight_car",
    "feright car": "freight_car",
    "feright_car": "freight_car",
    "feright": "freight_car",
    "truck": "truck",
    "van": "van",
}
SOURCE_SIZE = (840, 712)
CROP_XYXY = (100.0, 100.0, 740.0, 612.0)
OUTPUT_SIZE = (640, 512)
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
SPLITS = {
    "train": ("images/trainimg", "image/trainimgr", "labels11111/trainlabelr"),
    "val": ("images/valimg", "image/valimgr", "labels11111/vallabelr"),
    "test": ("test_1111111/testimg", "test_1111111/testimgr", "test_1111111/testlabelr"),
}
COLORS = [(230, 90, 70), (52, 152, 219), (241, 196, 15), (46, 204, 113), (155, 89, 182)]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--splits", nargs="+", choices=tuple(SPLITS), default=("train", "val", "test"))
    parser.add_argument("--image-mode", choices=("none", "crop"), default="none")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--visualize", type=int, default=12,
                        help="Number of deterministic overlay samples per split; 0 disables overlays.")
    return parser.parse_args()


def files_by_stem(directory: Path, suffixes: set[str]) -> dict[str, Path]:
    if not directory.is_dir():
        raise FileNotFoundError(directory)
    result = {p.stem: p for p in directory.iterdir() if p.is_file() and p.suffix.lower() in suffixes}
    if not result:
        raise RuntimeError(f"no matching files in {directory}")
    return result


def canonical_class(raw_name: str | None, source: Path, object_index: int) -> int | None:
    name = " ".join((raw_name or "").strip().lower().replace("-", " ").split())
    if name == "*":
        return None
    canonical = ALIASES.get(name)
    if canonical is None:
        raise ValueError(f"{source}: object {object_index}: unknown class {raw_name!r}")
    return CLASS_TO_ID[canonical]


def object_points(obj: ET.Element) -> tuple[np.ndarray | None, str]:
    polygon = obj.find("polygon")
    if polygon is not None:
        points = []
        for index in range(1, 5):
            x, y = polygon.findtext(f"x{index}"), polygon.findtext(f"y{index}")
            if x is None or y is None:
                return None, "incomplete_polygon"
            points.append((float(x), float(y)))
        return np.asarray(points, dtype=np.float32), "polygon"
    box = obj.find("bndbox")
    if box is not None:
        x1, y1 = float(box.findtext("xmin")), float(box.findtext("ymin"))
        x2, y2 = float(box.findtext("xmax")), float(box.findtext("ymax"))
        return np.asarray(((x1, y1), (x2, y1), (x2, y2), (x1, y2)), dtype=np.float32), "bndbox"
    if obj.find("point") is not None:
        return None, "point_only"
    return None, "missing_geometry"


def clip_edge(points: list[tuple[float, float]], inside, intersect) -> list[tuple[float, float]]:
    if not points:
        return []
    output = []
    previous = points[-1]
    previous_inside = inside(previous)
    for current in points:
        current_inside = inside(current)
        if current_inside:
            if not previous_inside:
                output.append(intersect(previous, current))
            output.append(current)
        elif previous_inside:
            output.append(intersect(previous, current))
        previous, previous_inside = current, current_inside
    return output


def clip_polygon(points: np.ndarray) -> np.ndarray:
    """Clip a convex quadrilateral to the established crop rectangle."""
    left, top, right, bottom = CROP_XYXY
    poly = [tuple(map(float, point)) for point in points]

    def vertical(x_value):
        def intersection(a, b):
            dx = b[0] - a[0]
            t = 0.0 if abs(dx) < 1e-12 else (x_value - a[0]) / dx
            return x_value, a[1] + t * (b[1] - a[1])
        return intersection

    def horizontal(y_value):
        def intersection(a, b):
            dy = b[1] - a[1]
            t = 0.0 if abs(dy) < 1e-12 else (y_value - a[1]) / dy
            return a[0] + t * (b[0] - a[0]), y_value
        return intersection

    poly = clip_edge(poly, lambda p: p[0] >= left, vertical(left))
    poly = clip_edge(poly, lambda p: p[0] <= right, vertical(right))
    poly = clip_edge(poly, lambda p: p[1] >= top, horizontal(top))
    poly = clip_edge(poly, lambda p: p[1] <= bottom, horizontal(bottom))
    if len(poly) < 3:
        return np.empty((0, 2), dtype=np.float32)
    shifted = np.asarray(poly, dtype=np.float32) - np.asarray((left, top), dtype=np.float32)
    shifted[:, 0] = np.clip(shifted[:, 0], 0.0, OUTPUT_SIZE[0])
    shifted[:, 1] = np.clip(shifted[:, 1], 0.0, OUTPUT_SIZE[1])
    return shifted


def four_point_obb(clipped: np.ndarray) -> np.ndarray | None:
    if len(clipped) < 3 or abs(cv2.contourArea(clipped)) <= 1e-4:
        return None
    rectangle = cv2.boxPoints(cv2.minAreaRect(clipped.astype(np.float32)))
    rectangle[:, 0] = np.clip(rectangle[:, 0], 0.0, OUTPUT_SIZE[0])
    rectangle[:, 1] = np.clip(rectangle[:, 1], 0.0, OUTPUT_SIZE[1])
    if abs(cv2.contourArea(rectangle)) <= 1e-4:
        return None
    return rectangle


def parse_xml(path: Path) -> tuple[list[tuple[int, np.ndarray]], Counter, list[dict[str, object]]]:
    root = ET.parse(path).getroot()
    size = (int(root.findtext("size/width")), int(root.findtext("size/height")))
    if size != SOURCE_SIZE:
        raise ValueError(f"{path}: image size {size}, expected {SOURCE_SIZE}")
    records: list[tuple[int, np.ndarray]] = []
    events: list[dict[str, object]] = []
    counts: Counter = Counter()
    for object_index, obj in enumerate(root.findall("object")):
        class_id = canonical_class(obj.findtext("name"), path, object_index)
        if class_id is None:
            events.append({"file": path.name, "object_index": object_index, "reason": "invalid_class_*"})
            continue
        points, geometry = object_points(obj)
        if points is None:
            events.append({"file": path.name, "object_index": object_index, "class": CLASS_NAMES[class_id],
                           "reason": geometry})
            continue
        counts[("source", class_id)] += 1
        clipped = clip_polygon(points)
        obb = four_point_obb(clipped)
        if obb is None:
            counts[("outside", class_id)] += 1
            continue
        if np.any(points[:, 0] < CROP_XYXY[0]) or np.any(points[:, 0] > CROP_XYXY[2]) or \
                np.any(points[:, 1] < CROP_XYXY[1]) or np.any(points[:, 1] > CROP_XYXY[3]):
            counts[("boundary", class_id)] += 1
        counts[("kept", class_id)] += 1
        records.append((class_id, obb))
    return records, counts, events


def label_text(records: list[tuple[int, np.ndarray]]) -> str:
    width, height = OUTPUT_SIZE
    lines = []
    for class_id, points in records:
        normalized = points / np.asarray((width, height), dtype=np.float32)
        if not np.all(np.isfinite(normalized)) or np.any(normalized < -1e-6) or np.any(normalized > 1.0 + 1e-6):
            raise ValueError(f"invalid normalized OBB for class {class_id}: {normalized.tolist()}")
        normalized = np.clip(normalized, 0.0, 1.0)
        lines.append(str(class_id) + " " + " ".join(f"{float(v):.8f}" for v in normalized.reshape(-1)))
    return "\n".join(lines) + ("\n" if lines else "")


def crop_jpeg(source: Path, destination: Path) -> None:
    with Image.open(source) as image:
        if image.size != SOURCE_SIZE:
            raise ValueError(f"{source}: image size {image.size}, expected {SOURCE_SIZE}")
        cropped = image.crop(tuple(map(int, CROP_XYXY)))
        destination.parent.mkdir(parents=True, exist_ok=True)
        if source.suffix.lower() in {".jpg", ".jpeg"} and getattr(image, "quantization", None):
            cropped.save(destination.with_suffix(".jpg"), format="JPEG", qtables=image.quantization,
                         subsampling=JpegImagePlugin.get_sampling(image))
        else:
            cropped.convert("RGB").save(destination.with_suffix(".jpg"), format="JPEG", quality=95)


def draw_overlay(rgb_path: Path, records: list[tuple[int, np.ndarray]], destination: Path) -> None:
    with Image.open(rgb_path) as image:
        canvas = image.crop(tuple(map(int, CROP_XYXY))).convert("RGB")
    draw = ImageDraw.Draw(canvas)
    for class_id, points in records:
        polygon = [tuple(map(float, point)) for point in points]
        draw.line(polygon + [polygon[0]], fill=COLORS[class_id], width=2)
        draw.text(polygon[0], CLASS_NAMES[class_id], fill=COLORS[class_id])
    destination.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(destination, quality=92)


def process_split(source_root: Path, output_root: Path, split: str, args: argparse.Namespace) -> dict[str, object]:
    rgb_rel, ir_rel, xml_rel = SPLITS[split]
    rgb = files_by_stem(source_root / rgb_rel, IMAGE_SUFFIXES)
    ir = files_by_stem(source_root / ir_rel, IMAGE_SUFFIXES)
    xml = files_by_stem(source_root / xml_rel, {".xml"})
    stems = set(rgb)
    if stems != set(ir) or stems != set(xml):
        raise RuntimeError(f"{split} pairing mismatch: rgb={len(rgb)}, ir={len(ir)}, xml={len(xml)}")

    label_dir = output_root / "labels" / split
    label_dir.mkdir(parents=True)
    aggregate: Counter = Counter()
    skipped: list[dict[str, object]] = []
    parsed: dict[str, list[tuple[int, np.ndarray]]] = {}
    labels_digest = hashlib.sha256()
    xml_digest = hashlib.sha256()
    empty_images = 0
    for stem in sorted(stems):
        records, counts, events = parse_xml(xml[stem])
        parsed[stem] = records
        aggregate.update(counts)
        skipped.extend(events)
        content = label_text(records)
        (label_dir / f"{stem}.txt").write_text(content)
        labels_digest.update(stem.encode() + b"\0" + content.encode())
        xml_digest.update(stem.encode() + b"\0" + xml[stem].read_bytes())
        empty_images += not records

    if args.image_mode == "crop":
        def materialize(stem: str) -> None:
            crop_jpeg(rgb[stem], output_root / "images" / split / f"{stem}.jpg")
            crop_jpeg(ir[stem], output_root / "image" / split / f"{stem}.jpg")

        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            for index, _ in enumerate(pool.map(materialize, sorted(stems)), 1):
                if index % 2000 == 0 or index == len(stems):
                    print(f"{split}: cropped {index}/{len(stems)} paired images", flush=True)

    if args.visualize:
        ordered = sorted(stems)
        boundary = [stem for stem in ordered if any(
            np.any(points[:, 0] <= 0.01) or np.any(points[:, 0] >= OUTPUT_SIZE[0] - 0.01)
            or np.any(points[:, 1] <= 0.01) or np.any(points[:, 1] >= OUTPUT_SIZE[1] - 0.01)
            for _, points in parsed[stem]
        )]
        stride = max(len(ordered) // max(args.visualize, 1), 1)
        selected = list(dict.fromkeys(boundary[: args.visualize // 2] + ordered[::stride]))[: args.visualize]
        for stem in selected:
            draw_overlay(rgb[stem], parsed[stem], output_root / "audit_visualizations" / split / f"{stem}.jpg")

    def class_counts(kind: str) -> dict[str, int]:
        return {CLASS_NAMES[index]: aggregate[(kind, index)] for index in range(len(CLASS_NAMES))}

    return {
        "pairs": len(stems),
        "empty_images_after_crop": empty_images,
        "source_objects": class_counts("source"),
        "kept_objects": class_counts("kept"),
        "outside_crop_dropped": class_counts("outside"),
        "boundary_adjusted": class_counts("boundary"),
        "non_box_events": skipped,
        "labels_sha256": labels_digest.hexdigest(),
        "xml_sha256": xml_digest.hexdigest(),
        "source_rgb": str((source_root / rgb_rel).resolve()),
        "source_ir": str((source_root / ir_rel).resolve()),
        "source_xml": str((source_root / xml_rel).resolve()),
    }


def audit_written_labels(output_root: Path, splits: tuple[str, ...] | list[str]) -> None:
    for split in splits:
        for path in sorted((output_root / "labels" / split).glob("*.txt")):
            for line_number, line in enumerate(path.read_text().splitlines(), 1):
                fields = line.split()
                if len(fields) != 9:
                    raise RuntimeError(f"{path}:{line_number}: expected class plus four points")
                class_id = int(fields[0])
                values = tuple(map(float, fields[1:]))
                if class_id not in range(len(CLASS_NAMES)) or not all(math.isfinite(v) and 0 <= v <= 1 for v in values):
                    raise RuntimeError(f"{path}:{line_number}: invalid OBB label")


def main() -> None:
    args = parse_args()
    source_root = args.source.resolve()
    output_root = args.output.resolve()
    if output_root.exists():
        raise FileExistsError(f"refusing to overwrite existing output: {output_root}")
    if args.workers <= 0 or args.visualize < 0:
        raise ValueError("workers must be positive and visualize must be non-negative")
    output_root.mkdir(parents=True)
    manifest = {
        "dataset": "DroneVehicle",
        "protocol": "five-class OBB from canonical infrared polygons",
        "classes": {str(index): name for index, name in enumerate(CLASS_NAMES)},
        "source_size": list(SOURCE_SIZE),
        "crop_xyxy": list(map(int, CROP_XYXY)),
        "output_size": list(OUTPUT_SIZE),
        "image_mode": args.image_mode,
        "splits": {},
    }
    try:
        for split in args.splits:
            manifest["splits"][split] = process_split(source_root, output_root, split, args)
        audit_written_labels(output_root, args.splits)
        manifest["status"] = "complete"
        (output_root / ".PREPROCESS_COMPLETE.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n"
        )
    except Exception:
        (output_root / ".PREPROCESS_FAILED").write_text("Conversion did not complete; do not use this directory.\n")
        raise
    print(json.dumps(manifest, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
