#!/usr/bin/env bash
set -Eeuo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

if [[ $# -ne 3 ]]; then
    echo "Usage: bash scripts/run_three_seeds.sh DATA_YAML SOURCE_PT PLATFORM_TAG" >&2
    exit 2
fi

DATA="$1"
SOURCE="$2"
PLATFORM_TAG="$3"
DEVICE="${DEVICE:-0}"
BATCH="${BATCH:-128}"
NBS="${NBS:-128}"
VAL_BATCH="${VAL_BATCH:-64}"
WORKERS="${WORKERS:-8}"
IMGSZ="${IMGSZ:-640}"
EPOCHS="${EPOCHS:-120}"
PATIENCE="${PATIENCE:-40}"
LR0="${LR0:-0.001}"
LRF="${LRF:-0.05}"
CLOSE_MOSAIC="${CLOSE_MOSAIC:-20}"

[[ -f "$DATA" ]] || { echo "Dataset YAML not found: $DATA" >&2; exit 1; }
[[ -f "$SOURCE" ]] || { echo "Checkpoint not found: $SOURCE" >&2; exit 1; }
[[ "$PLATFORM_TAG" =~ ^[A-Za-z0-9._-]+$ ]] || {
    echo "PLATFORM_TAG may contain only letters, digits, dot, underscore, and hyphen" >&2
    exit 1
}

python tools/check_dataset.py "$DATA"
mkdir -p runs/platform
python tools/capture_platform.py --output "runs/platform/${PLATFORM_TAG}.json"

for seed in 0 1 2; do
    name="${PLATFORM_TAG}_seed${seed}"
    output="runs/detect/${name}"
    [[ ! -e "$output" ]] || {
        echo "Refusing to overwrite existing output: $output" >&2
        exit 1
    }
    python train.py \
        --name "$name" --data "$DATA" --source "$SOURCE" \
        --model-family yolo8 --model-scale n \
        --mode p3_residual --p3-anchor rgb \
        --taylor-order 3 --recurrence additive \
        --fusion-hidden 16 --gate-max 0.25 \
        --p2-bidirectional-main --p2-main-hidden 16 --p2-main-gate-max 0.10 \
        --device "$DEVICE" --batch "$BATCH" --nbs "$NBS" --val-batch "$VAL_BATCH" \
        --workers "$WORKERS" --imgsz "$IMGSZ" --amp \
        --epochs "$EPOCHS" --patience "$PATIENCE" \
        --init-seed "$seed" --train-seed "$seed" \
        --lr0 "$LR0" --lrf "$LRF" --warmup-epochs 5 --weight-decay 0.0005 \
        --mosaic 1.0 --close-mosaic "$CLOSE_MOSAIC" --correct-border \
        --anchor-freeze-epochs 0 --module-diagnostics --diagnostic-interval 50
done

python tools/collect_seed_results.py \
    --runs "runs/detect/${PLATFORM_TAG}_seed0" \
           "runs/detect/${PLATFORM_TAG}_seed1" \
           "runs/detect/${PLATFORM_TAG}_seed2"
