# AnchorTaylor reproducible release

AnchorTaylor is a compact paired RGB/infrared detector built on YOLOv8n. It
preserves an RGB-pretrained main path, exchanges bounded local detail at P2,
and applies an RGB-anchored third-order recursive correction at P3. Only the
corrected RGB representation continues through the upper backbone and FPN-PAN.

This directory is self-contained except for datasets and official YOLO
checkpoints. It intentionally excludes local runs, private data, and trained
weights.

## 1. Environment

The audited environment used Python 3.10, PyTorch 2.1.2, torchvision 0.16.2,
and CUDA 12.1.

```bash
cd talordet_release
conda env create -f environment.yml
conda activate talordet
python tools/smoke_modules.py
```

If the PyTorch CUDA package is unavailable on your platform, install the
matching PyTorch build first, then install `requirements.txt`. The AnchorTaylor
path does not require MMCV or Mamba; those dependencies belong only to legacy
modules retained in the research fork.

## 2. Dataset layout and where paths are changed

Edit the top-level `path:` field in one of these files:

- `data/M3FD.yaml`
- `data/LLVIP.yaml`
- `data/DroneVehicle_5cls_HBB.yaml`

Expected layout:

```text
DATASET_ROOT/
├── images/
│   ├── train/
│   └── val/
├── image/
│   ├── train/
│   └── val/
└── labels/
    ├── train/
    └── val/
```

`images/` stores RGB images and `image/` stores infrared images. Paired files
must have the same relative filename and stem. Labels use standard YOLO
normalized `class x_center y_center width height` rows and correspond to the
RGB path. Verify a configuration before training:

```bash
python tools/check_dataset.py data/M3FD.yaml
```

The DroneVehicle template describes the paper's five-class horizontal-box
protocol after removing the 100-pixel image border. Dataset conversion is not
performed automatically because the source annotation license and local source
layout may differ.

## 3. Official pretrained checkpoint

Place the official `yolov8n.pt` checkpoint in this directory or pass its full
path through `--source`. The checkpoint is not redistributed.

## 4. Initialization audit

Run the CPU audit before occupying a GPU. It verifies dataset metadata,
transferred pretrained tensors, paired branch initialization, parameter count,
and exact RGB-path identity at initialization.

```bash
python train.py \
  --name m3fd_audit \
  --data data/M3FD.yaml \
  --source yolov8n.pt \
  --model-family yolo8 --model-scale n \
  --mode p3_residual --p3-anchor rgb \
  --taylor-order 3 --recurrence additive \
  --fusion-hidden 16 --gate-max 0.25 \
  --p2-bidirectional-main --p2-main-hidden 16 --p2-main-gate-max 0.10 \
  --device cpu --batch 2 --nbs 128 --workers 0 \
  --epochs 300 --patience 120 --init-seed 0 --train-seed 0 \
  --correct-border --audit-only
```

## 5. Three-seed training

The wrapper records software/hardware metadata and trains seeds 0, 1, and 2
without overwriting existing runs.

M3FD paper recipe:

```bash
BATCH=128 NBS=128 EPOCHS=300 PATIENCE=120 LR0=0.0007 CLOSE_MOSAIC=30 \
DEVICE=0,1 bash scripts/run_three_seeds.sh \
  data/M3FD.yaml yolov8n.pt m3fd_workstation_a
```

DroneVehicle cross-platform recipe, run independently on each machine:

```bash
BATCH=128 NBS=128 EPOCHS=120 PATIENCE=40 LR0=0.001 CLOSE_MOSAIC=20 \
DEVICE=0,1 bash scripts/run_three_seeds.sh \
  data/DroneVehicle_5cls_HBB.yaml yolov8n.pt drone_workstation_a
```

Use a different platform tag on the second machine, for example
`drone_workstation_b`. Keep the data split, checkpoint checksum, batch size,
optimizer, and stopping rule identical. The script writes platform metadata to
`runs/platform/` and training outputs to `runs/detect/`.

Summarize the three seeds:

```bash
python tools/collect_seed_results.py \
  --runs runs/detect/drone_workstation_a_seed0 \
         runs/detect/drone_workstation_a_seed1 \
         runs/detect/drone_workstation_a_seed2
```

The paper reports every seed plus mean and sample standard deviation. Do not
substitute a best-seed score for the mean.

## 6. Pair inference

```bash
python predict_pair.py \
  --weights runs/detect/EXPERIMENT/weights/best.pt \
  --rgb /path/to/rgb.png \
  --ir /path/to/infrared.png \
  --device 0
```

## Pretraining result and reproducibility boundary

Under the matched M3FD recipe, random initialization obtains `53.41+/-0.21`
mAP, whereas RGB-pretrained initialization obtains `59.15+/-0.45`. The paired
gains for seeds 0, 1, and 2 are `+5.679`, `+5.356`, and `+6.186` points. The
effect is therefore consistent across the tested seeds and much larger than
the within-condition variation. This demonstrates the value of external RGB
pretraining under the fixed budget; it is not an upper bound on a separately
tuned from-scratch recipe. Cross-platform repeatability must still be measured
directly.

## License

The modified Ultralytics code is AGPL-3.0. See `LICENSE` and
`THIRD_PARTY_NOTICES.md`.
