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
- `data/DroneVehicle_5cls_OBB.yaml` (paper protocol)

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
must have the same relative filename and stem. M3FD and LLVIP use normalized horizontal
YOLO boxes (`class cx cy w h`). DroneVehicle uses five-class oriented boxes
(`class x1 y1 x2 y2 x3 y3 x4 y4`, four normalized corner coordinates per axis).
The historical `DroneVehicle_5cls_HBB.yaml` template is for horizontal-box
experiments. The paper uses the infrared polygon annotations for both streams.

Run `python tools/check_dataset.py data/M3FD.yaml` to check the paired file
inventories and class count.

### Prepare DroneVehicle

Use an empty output path. The converter expects RGB, IR and IR annotation XML
directories under the source root as follows. Arrange the source folders or
symlink them to these paths before conversion.

| Split | RGB | IR | IR XML |
| --- | --- | --- | --- |
| train | `images/trainimg` | `image/trainimgr` | `labels11111/trainlabelr` |
| val | `images/valimg` | `image/valimgr` | `labels11111/vallabelr` |

```bash
DV_ROOT=/absolute/path/to/DroneVehicle_5cls_OBB
python tools/prepare_dronevehicle_5cls_obb.py --source /path/to/raw/DroneVehicle \
  --output "$DV_ROOT" --splits train val --image-mode crop
```

The converter removes the 100-pixel border from 840×712 pairs, adjusts the
corners to the 640×512 crop and writes labels in class-ID order
`bus, car, freight_car, truck, van`. Set `path:` in
`data/DroneVehicle_5cls_OBB.yaml` to the absolute output directory.

## 3. Official pretrained checkpoints

Use the official `yolov8n.pt` for M3FD and LLVIP and `yolov8n-obb.pt` for
the DroneVehicle n model. Pass the local checkpoint path via `--source`.

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

## 5. M3FD horizontal-box training

The wrapper records software/hardware metadata and trains seeds 0, 1, and 2
without overwriting existing runs.

M3FD paper recipe:

```bash
BATCH=128 NBS=128 EPOCHS=300 PATIENCE=120 LR0=0.0007 CLOSE_MOSAIC=30 \
DEVICE=0,1 bash scripts/run_three_seeds.sh \
  data/M3FD.yaml yolov8n.pt m3fd_workstation_a
```

The wrapper summarizes the three training logs at their best-fitness epochs.
The M3FD main comparison separately re-evaluates the saved best checkpoints.

## DroneVehicle oriented-box training and evaluation

The paper's n-model result uses seed 0. Run the following command from
`talordet_release/`. To audit initialization first, append `--audit-only`.
Choose the two visible GPU IDs for your machine with `--device`.

```bash
python train.py --name drone_yolov8n_obb_seed0 --data data/DroneVehicle_5cls_OBB.yaml \
  --source yolov8n-obb.pt --model-family yolo8 --model-scale n --task obb \
  --mode p3_residual --p3-anchor rgb --taylor-order 3 --recurrence additive \
  --fusion-hidden 16 --gate-max 0.25 --p2-bidirectional-main \
  --p2-main-hidden 16 --p2-main-gate-max 0.10 \
  --device 0,1 --batch 128 --nbs 128 --val-batch 64 --workers 8 --imgsz 640 --amp \
  --epochs 120 --patience 40 --init-seed 0 --train-seed 0 --lr0 0.001 --lrf 0.05 \
  --warmup-epochs 5 --weight-decay 0.0005 --mosaic 1.0 --close-mosaic 20 \
  --correct-border --anchor-freeze-epochs 0 --module-diagnostics --diagnostic-interval 50
```

Use the selected `best.pt` for the common evaluation. The training validator
uses ProbIoU for matching. The paper scores exported corners with exact
polygon IoU and 101-point interpolated AP. The evaluation split comprises the
1,469 validation pairs with 24,490 annotated objects. The official test split
has 8,980 pairs and is a separate benchmark.

```bash
EVAL=runs/drone_polygon_eval
python tools/export_dronevehicle_obb_ground_truth.py --labels "$DV_ROOT/labels/val" \
  --output "$EVAL/gt.csv" --image-ids-output "$EVAL/image_ids.txt"
python -m tools.export_anchortaylor_obb_predictions \
  --weights runs/detect/drone_yolov8n_obb_seed0/weights/best.pt \
  --data data/DroneVehicle_5cls_OBB.yaml --device 0 --batch 8 \
  --image-ids "$EVAL/image_ids.txt" --output "$EVAL/predictions.csv"
python tools/evaluate_dronevehicle_obb.py --ground-truth "$EVAL/gt.csv" \
  --predictions "$EVAL/predictions.csv" --image-ids "$EVAL/image_ids.txt" \
  --output "$EVAL/polygon_metrics.json"
```

The n-model reference is **81.8% OBB mAP50 and 52.2% OBB mAP50–95**.
Read these metrics from `polygon_metrics.json` (fractions, multiply by 100).
The exporter also writes the native validator scores to `native_metrics.json`.

## 6. Pair inference

A horizontal detector draws rectangles. An OBB checkpoint draws the four predicted corners on the RGB image. For DroneVehicle, supply the cropped pair.

```bash
python predict_pair.py \
  --weights runs/detect/EXPERIMENT/weights/best.pt \
  --rgb /path/to/rgb.png \
  --ir /path/to/infrared.png \
  --device 0
```

## Pretraining result and reproducibility boundary

M3FD training-time mAP50–95 (%) under the matched budget, at the best-fitness epoch:

| Initialization | Seed 0 | Seed 1 | Seed 2 | Mean ± sample SD |
| --- | ---: | ---: | ---: | ---: |
| Random | 53.656 | 53.276 | 53.292 | 53.41 ± 0.21 |
| RGB-pretrained | 59.335 | 59.618 | 59.478 | 59.48 ± 0.14 |

The paper uses the repeat run for pretrained seed 1. The mean gain is 6.07
percentage points. These values describe the fixed training budget.

## License

The modified Ultralytics code is AGPL-3.0. See `LICENSE` and
`THIRD_PARTY_NOTICES.md`.
