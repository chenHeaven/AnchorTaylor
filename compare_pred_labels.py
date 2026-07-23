import argparse
import csv
from pathlib import Path
from typing import Dict, List, Tuple


Box = Tuple[int, float, float, float, float]
XYXYBox = Tuple[int, float, float, float, float]


def read_yolo_labels(path: Path) -> List[Box]:
    if not path.exists():
        return []

    boxes: List[Box] = []
    for line_no, line in enumerate(path.read_text().splitlines(), start=1):
        line = line.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) < 5:
            print(f"Skip invalid line {path}:{line_no}: {line}")
            continue
        cls = int(float(parts[0]))
        x, y, w, h = map(float, parts[1:5])
        boxes.append((cls, x, y, w, h))
    return boxes


def xywh_to_xyxy(boxes: List[Box]) -> List[XYXYBox]:
    converted: List[XYXYBox] = []
    for cls, x, y, w, h in boxes:
        x1 = x - w / 2
        y1 = y - h / 2
        x2 = x + w / 2
        y2 = y + h / 2
        converted.append((cls, x1, y1, x2, y2))
    return converted


def single_iou(a: XYXYBox, b: XYXYBox) -> float:
    _, ax1, ay1, ax2, ay2 = a
    _, bx1, by1, bx2, by2 = b

    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)

    inter_w = max(inter_x2 - inter_x1, 0.0)
    inter_h = max(inter_y2 - inter_y1, 0.0)
    inter = inter_w * inter_h

    area_a = max(ax2 - ax1, 0.0) * max(ay2 - ay1, 0.0)
    area_b = max(bx2 - bx1, 0.0) * max(by2 - by1, 0.0)
    union = area_a + area_b - inter
    return inter / union if union > 1e-12 else 0.0


def match_image(gt_boxes: List[Box], pred_boxes: List[Box], iou_thr: float) -> Dict[str, int]:
    gt = xywh_to_xyxy(gt_boxes)
    pred = xywh_to_xyxy(pred_boxes)

    candidates = []
    for pred_idx, pred_box in enumerate(pred):
        for gt_idx, gt_box in enumerate(gt):
            candidates.append((single_iou(pred_box, gt_box), pred_idx, gt_idx))
    candidates.sort(reverse=True, key=lambda x: x[0])

    matched_pred = set()
    matched_gt = set()
    wrong_class = 0
    true_positive = 0

    for iou, pred_idx, gt_idx in candidates:
        if iou < iou_thr:
            break
        if pred_idx in matched_pred or gt_idx in matched_gt:
            continue
        matched_pred.add(pred_idx)
        matched_gt.add(gt_idx)
        if pred[pred_idx][0] == gt[gt_idx][0]:
            true_positive += 1
        else:
            wrong_class += 1

    missed = len(gt) - len(matched_gt)
    low_iou_or_extra = len(pred) - len(matched_pred)
    false_positive = wrong_class + low_iou_or_extra

    return {
        "gt": len(gt),
        "pred": len(pred),
        "tp": true_positive,
        "missed": missed,
        "false_positive": false_positive,
        "wrong_class": wrong_class,
        "low_iou_or_extra": low_iou_or_extra,
    }


def collect_ids(gt_dir: Path, pred_dir: Path) -> List[str]:
    ids = {p.stem for p in gt_dir.glob("*.txt")}
    ids.update(p.stem for p in pred_dir.glob("*.txt"))
    return sorted(ids)


def main():
    parser = argparse.ArgumentParser(description="Compare YOLO prediction labels with ground truth labels.")
    parser.add_argument("--pred-dir", type=Path, default=Path("/home/asefu/project/TalorDet/runs/predict/ICAFusion/flir/labels"))
    parser.add_argument("--gt-dir", type=Path, default=Path("/home/asefu/data/flir_align/labels/val"))
    parser.add_argument("--iou-thr", type=float, default=0.5, help="IoU threshold for matching predictions to ground truth")
    parser.add_argument("--topk", type=int, default=10)
    parser.add_argument("--csv", type=Path, default=Path("/home/asefu/project/TalorDet/runs/predict/ICAFusion/flir/label_compare_summary.csv"))
    args = parser.parse_args()

    pred_dir = args.pred_dir.resolve()
    gt_dir = args.gt_dir.resolve()
    rows = []

    for image_id in collect_ids(gt_dir, pred_dir):
        gt_boxes = read_yolo_labels(gt_dir / f"{image_id}.txt")
        pred_boxes = read_yolo_labels(pred_dir / f"{image_id}.txt")
        stat = match_image(gt_boxes, pred_boxes, args.iou_thr)
        rows.append({"image_id": image_id, **stat, "total_error": stat["missed"] + stat["false_positive"]})

    rows.sort(key=lambda r: (-r["total_error"], -r["missed"], -r["false_positive"], r["image_id"]))

    args.csv.parent.mkdir(parents=True, exist_ok=True)
    with args.csv.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "image_id",
                "gt",
                "pred",
                "tp",
                "missed",
                "false_positive",
                "wrong_class",
                "low_iou_or_extra",
                "total_error",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"IoU threshold: {args.iou_thr}")
    print(f"Compared {len(rows)} images")
    print(f"Saved full summary to: {args.csv}")

    print(f"\nTop {args.topk} images by missed labels:")
    for r in sorted(rows, key=lambda r: (-r["missed"], -r["total_error"], r["image_id"]))[: args.topk]:
        print(
            f"{r['image_id']}  missed={r['missed']}  fp={r['false_positive']}  "
            f"wrong_cls={r['wrong_class']}  low_iou_or_extra={r['low_iou_or_extra']}  "
            f"gt={r['gt']}  pred={r['pred']}"
        )

    print(f"\nTop {args.topk} images by false positives:")
    for r in sorted(rows, key=lambda r: (-r["false_positive"], -r["total_error"], r["image_id"]))[: args.topk]:
        print(
            f"{r['image_id']}  fp={r['false_positive']}  missed={r['missed']}  "
            f"wrong_cls={r['wrong_class']}  low_iou_or_extra={r['low_iou_or_extra']}  "
            f"gt={r['gt']}  pred={r['pred']}"
        )

    print(f"\nTop {args.topk} images by total errors:")
    for r in rows[: args.topk]:
        print(
            f"{r['image_id']}  total={r['total_error']}  missed={r['missed']}  "
            f"fp={r['false_positive']}  wrong_cls={r['wrong_class']}  "
            f"low_iou_or_extra={r['low_iou_or_extra']}  gt={r['gt']}  pred={r['pred']}"
        )


if __name__ == "__main__":
    main()
