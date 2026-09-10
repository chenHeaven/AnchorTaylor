#!/usr/bin/env python3
"""Validate paired RGB/infrared paths declared by an AnchorTaylor YAML file."""

from __future__ import annotations

import argparse
from pathlib import Path

import yaml


IMAGE_EXTENSIONS = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}


def resolve(root: Path, value: str) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else root / path


def inventory(directory: Path) -> set[str]:
    if not directory.is_dir():
        raise FileNotFoundError(f"Directory not found: {directory}")
    return {
        str(path.relative_to(directory).with_suffix(""))
        for path in directory.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("yaml", type=Path)
    args = parser.parse_args()
    config = yaml.safe_load(args.yaml.read_text(encoding="utf-8"))
    root = Path(config.get("path", args.yaml.parent)).expanduser()
    if not root.is_absolute():
        root = (args.yaml.parent / root).resolve()

    for split in ("train", "val"):
        rgb = inventory(resolve(root, config[split]))
        ir = inventory(resolve(root, config[f"{split}_ir"]))
        missing_ir, missing_rgb = sorted(rgb - ir), sorted(ir - rgb)
        if missing_ir or missing_rgb:
            raise RuntimeError(
                f"{split}: unpaired files; missing_ir={missing_ir[:5]}, missing_rgb={missing_rgb[:5]}"
            )
        print(f"{split}: {len(rgb)} aligned RGB/IR pairs")

    names = config.get("names", {})
    if int(config.get("nc", 0)) != len(names):
        raise ValueError("nc does not match the number of class names")
    print(f"classes: {len(names)}; dataset root: {root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
