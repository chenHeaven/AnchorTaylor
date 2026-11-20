#!/usr/bin/env python3
"""Export the canonical YOLO-OBB validation labels to the common CSV schema."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


NAMES = {0: "Bus", 1: "Car", 2: "Freight car", 3: "Truck", 4: "Van"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--image-ids-output", type=Path, help="Optional manifest including empty label files.")
    parser.add_argument("--width", type=float, default=640.0)
    parser.add_argument("--height", type=float, default=512.0)
    args = parser.parse_args()

    if not args.labels.is_dir():
        raise FileNotFoundError(args.labels)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fields = ["image_id", "class_name", *(f"{axis}{index}" for index in range(1, 5) for axis in "xy"), "ignore"]
    rows = 0
    images = 0
    with args.output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for label_path in sorted(args.labels.glob("*.txt")):
            images += 1
            for line_number, line in enumerate(label_path.read_text().splitlines(), 1):
                if not line.strip():
                    continue
                values = line.split()
                if len(values) != 9:
                    raise ValueError(f"{label_path}:{line_number}: expected class plus eight coordinates")
                class_id = int(values[0])
                if class_id not in NAMES:
                    raise ValueError(f"{label_path}:{line_number}: invalid class {class_id}")
                normalized = list(map(float, values[1:]))
                coordinates = [
                    value * (args.width if index % 2 == 0 else args.height)
                    for index, value in enumerate(normalized)
                ]
                row = {"image_id": label_path.stem, "class_name": NAMES[class_id], "ignore": 0}
                row.update({
                    f"{axis}{index}": coordinates[(index - 1) * 2 + offset]
                    for index in range(1, 5) for offset, axis in enumerate("xy")
                })
                writer.writerow(row)
                rows += 1
    print(f"exported {rows} objects from {images} label files to {args.output}")
    if args.image_ids_output:
        args.image_ids_output.parent.mkdir(parents=True, exist_ok=True)
        image_ids = [path.stem for path in sorted(args.labels.glob("*.txt"))]
        args.image_ids_output.write_text("\n".join(image_ids) + "\n")
        print(f"exported {len(image_ids)} image IDs to {args.image_ids_output}")


if __name__ == "__main__":
    main()
