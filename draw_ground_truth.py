import argparse
from pathlib import Path
from typing import List, Tuple

from PIL import Image, ImageDraw, ImageFont


CLASS_NAMES = {
    0: "car",
    1: "person",
    2: "bicycle"
}

COLORS = {
    # 0: (255, 0, 0),
    0: (0, 255, 0),
    # 2: (255, 128, 0)
}

Label = Tuple[int, float, float, float, float]


def resolve_output_dir(output_dir: Path) -> Path:
    """Return a writable output directory, falling back to local runs/ if needed."""
    try:
        output_dir.mkdir(parents=True, exist_ok=True)
        probe = output_dir / ".write_test"
        probe.write_text("ok")
        probe.unlink()
        return output_dir
    except PermissionError:
        fallback = Path.cwd() / "runs" / "detect_ground_truth"
        fallback.mkdir(parents=True, exist_ok=True)
        print(
            f"Warning: cannot write to {output_dir}. "
            f"Fallback to {fallback}. You can also set --output-dir explicitly."
        )
        return fallback


def read_yolo_label(label_path: Path) -> List[Label]:
    labels: List[Label] = []
    if not label_path.exists():
        return labels

    for line_no, line in enumerate(label_path.read_text().splitlines(), start=1):
        line = line.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) < 5:
            print(f"Skip invalid label line {label_path}:{line_no}: {line}")
            continue
        cls = int(float(parts[0]))
        x, y, w, h = map(float, parts[1:5])
        labels.append((cls, x, y, w, h))
    return labels


def yolo_to_xyxy(label: Label, img_w: int, img_h: int) -> Tuple[int, int, int, int, int]:
    cls, x, y, w, h = label
    x1 = int(round((x - w / 2) * img_w))
    y1 = int(round((y - h / 2) * img_h))
    x2 = int(round((x + w / 2) * img_w))
    y2 = int(round((y + h / 2) * img_h))

    x1 = max(0, min(img_w - 1, x1))
    y1 = max(0, min(img_h - 1, y1))
    x2 = max(0, min(img_w - 1, x2))
    y2 = max(0, min(img_h - 1, y2))
    return cls, x1, y1, x2, y2


def draw_one(
    image_path: Path,
    label_path: Path,
    save_path: Path,
    line_width: int = 2,
    show_label_name: bool = True,
) -> None:
    image = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(image)
    img_w, img_h = image.size
    labels = read_yolo_label(label_path)

    try:
        font = ImageFont.truetype("DejaVuSans.ttf", size=max(12, img_w // 80))
    except OSError:
        font = ImageFont.load_default()

    for label in labels:
        cls, x1, y1, x2, y2 = yolo_to_xyxy(label, img_w, img_h)
        color = COLORS.get(cls, (255, 255, 255))

        for offset in range(line_width):
            draw.rectangle((x1 - offset, y1 - offset, x2 + offset, y2 + offset), outline=color)

        if show_label_name:
            text = CLASS_NAMES.get(cls, str(cls))
            bbox = draw.textbbox((x1, y1), text, font=font)
            text_w = bbox[2] - bbox[0]
            text_h = bbox[3] - bbox[1]
            text_y1 = max(0, y1 - text_h - 4)
            text_y2 = text_y1 + text_h + 4
            draw.rectangle((x1, text_y1, x1 + text_w + 6, text_y2), fill=color)
            draw.text((x1 + 3, text_y1 + 2), text, fill=(0, 0, 0), font=font)

    save_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(save_path)


def main():
    parser = argparse.ArgumentParser(description="Draw ground-truth YOLO labels on validation images.")
    parser.add_argument("--image-dir", type=Path, default=Path("/home/asefu/data/LLVIP/image/val/190009.jpg"))
    parser.add_argument("--label-dir", type=Path, default=Path("/home/asefu/data/LLVIP/labels/val/190009.txt"))
    parser.add_argument("--output-dir", type=Path, default=Path("/home/asefu/data/LLVIP/detect_ground_truth_ir"))
    parser.add_argument("--line-width", type=int, default=2)
    parser.add_argument("--show-label-name", type=int, choices=[0, 1], default=0)
    args = parser.parse_args()
    output_dir = resolve_output_dir(args.output_dir)

    valid_exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
    show_label_name = bool(args.show_label_name)

    if args.image_dir.is_file():
        if args.image_dir.suffix.lower() not in valid_exts:
            raise ValueError(f"Unsupported image extension: {args.image_dir}")

        if args.label_dir.is_file():
            label_path = args.label_dir
        else:
            label_path = args.label_dir / f"{args.image_dir.stem}.txt"

        save_path = output_dir if output_dir.suffix.lower() in valid_exts else output_dir / args.image_dir.name
        draw_one(
            args.image_dir,
            label_path,
            save_path,
            line_width=args.line_width,
            show_label_name=show_label_name,
        )
        print(f"Done. Saved 1 image to {save_path}")
        return

    image_paths = sorted(p for p in args.image_dir.iterdir() if p.suffix.lower() in valid_exts)

    if not image_paths:
        raise FileNotFoundError(f"No images found in {args.image_dir}")

    count = 0
    for image_path in image_paths:
        label_path = args.label_dir / f"{image_path.stem}.txt"
        save_path = output_dir / image_path.name
        draw_one(
            image_path,
            label_path,
            save_path,
            line_width=args.line_width,
            show_label_name=show_label_name,
        )
        count += 1
        if count % 100 == 0:
            print(f"Processed {count}/{len(image_paths)}")

    print(f"Done. Saved {count} images to {output_dir}")


if __name__ == "__main__":
    main()
