"""Train official-weight-anchored RGB/IR baselines and P3 fusion models."""

import argparse
import hashlib
import json
import os
import random
import sys
from pathlib import Path

import numpy as np
import torch
import yaml


ROOT = Path(__file__).resolve().parent
os.environ["PYTHONPATH"] = str(ROOT) + os.pathsep + os.environ.get("PYTHONPATH", "")
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ultralytics import YOLO
from ultralytics.utils import SETTINGS
from ultralytics.utils.torch_utils import init_seeds


MODEL_TEMPLATES = {
    "yolo8": ROOT / "ultralytics/cfg/models/v8/yolov8.yaml",
    "yolo11": ROOT / "ultralytics/cfg/models/11/yolo11.yaml",
}
TRAIN_CFG = ROOT / "ultralytics/cfg/defaultm3fd.yaml"
PROJECT = ROOT / "runs/detect"


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def dataset_identity(path):
    path = Path(path).resolve()
    with open(path, encoding="utf-8") as file:
        data = yaml.safe_load(file)
    names = data.get("names")
    count = len(names) if isinstance(names, (list, dict)) else None
    nc = int(data.get("nc", count or 0))
    if nc <= 0 or (count is not None and count != nc):
        raise ValueError(f"Invalid nc/names in dataset YAML: {path}")
    return path, nc, names


def matching_state(target, source):
    target_state, source_state = target.state_dict(), source.state_dict()
    matching = {
        key: value for key, value in source_state.items()
        if key in target_state and target_state[key].shape == value.shape
    }
    return matching, sum(value.numel() for value in matching.values())


def nested_tensors_equal(left, right):
    """Return True when two nested detector outputs are bitwise identical."""
    if torch.is_tensor(left) and torch.is_tensor(right):
        return torch.equal(left, right)
    if isinstance(left, (list, tuple)) and isinstance(right, type(left)):
        return len(left) == len(right) and all(nested_tensors_equal(a, b) for a, b in zip(left, right))
    if isinstance(left, dict) and isinstance(right, dict):
        return left.keys() == right.keys() and all(nested_tensors_equal(left[key], right[key]) for key in left)
    return left == right


def build_yaml(args, nc):
    with open(MODEL_TEMPLATES[args.model_family], encoding="utf-8") as file:
        config = yaml.safe_load(file)
    config.update(nc=nc, ch=6, scale=args.model_scale)
    internal_mode = {"p3_mean": "mean", "p3_residual": "residual"}.get(args.mode, args.mode)
    config["pretrained_anchor_fusion"] = {
        "mode": internal_mode,
        "anchor_modality": args.p3_anchor,
        "fusion_layer": 4,
        "hidden": args.fusion_hidden,
        "gate_max": args.gate_max,
        "order": args.taylor_order,
        "recurrence": args.recurrence,
    }
    if args.p2_bidirectional_main:
        config["anchor_p2_bidirectional_fusion"] = {
            "mode": "bidirectional_adaptive",
            "fusion_layer": 2,
            "hidden": args.p2_main_hidden,
            "gate_max": args.p2_main_gate_max,
        }
    if args.mode == "p2_only":
        # Consume the same initialization stream as the removed main-model P3
        # O3 block before constructing P2. This keeps every retained P2 tensor
        # paired with the corresponding full-model seed.
        config["p2_only_reference_p3_init"] = {
            "hidden": args.fusion_hidden,
            "gate_max": args.gate_max,
            "order": 3,
            "recurrence": "additive",
        }
    if args.scale_assignment:
        config["anchor_scale_assignment"] = {"assignment": args.scale_assignment, "init_seed": args.init_seed}
    if args.single_taylor_scale:
        config["anchor_single_taylor_scale"] = {"scale": args.single_taylor_scale, "init_seed": args.init_seed}
    if args.deep_taylor_scales:
        layer_by_scale = {"p4": 6, "p5": 9}
        config["anchor_deep_taylor_fusions"] = {
            "layers": [layer_by_scale[scale] for scale in args.deep_taylor_scales],
            "hidden": args.fusion_hidden,
            "gate_max": args.gate_max,
            "order": args.taylor_order,
            "recurrence": args.recurrence,
        }
    if args.p2_rgb_anchor_protected:
        config["anchor_p2_rgb_protected_fusion"] = {
            "mode": "rgb_anchor_protected",
            "fusion_layer": 2,
            "hidden": args.p2_main_hidden,
            "gate_max": args.p2_main_gate_max,
        }
    if args.p2_aux_multifilter:
        config["anchor_p2_aux_fusion"] = {
            "mode": "multifilter",
            "fusion_layer": 2,
            "hidden": args.p2_aux_hidden,
            "gate_max": args.p2_aux_gate_max,
            "loss_weight": args.p2_aux_loss_weight,
            "decay_start_epoch": args.p2_aux_decay_start,
            "decay_end_epoch": args.p2_aux_decay_end,
        }
    config["experiment"] = {
        "family": "pretrained_anchor_p3_fusion",
        "source": str(args.source) if args.source else None,
        "initialization": "scratch" if args.no_pretrained else "pretrained",
        "official_pretrained_origin": None if args.no_pretrained else f"{args.model_family}{args.model_scale}.pt",
        "architecture_scope": (
            "YOLO11 backbone/neck with legacy Detect" if args.model_family == "yolo11" else "YOLOv8"
        ),
        "model_scale": args.model_scale,
    }
    directory = ROOT / "weights/model_configs"
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{args.name}_nc{nc}_anchor_{args.model_family}{args.model_scale}.yaml"
    with open(path, "w", encoding="utf-8") as file:
        yaml.safe_dump(config, file, allow_unicode=True, sort_keys=False)
    return path


def initialize_ir_stem(detector):
    if getattr(detector, "anchor_ir_stem", None) is None:
        return
    last_dual_layer = len(detector.anchor_ir_stem) - 1
    detector.anchor_ir_stem.load_state_dict(detector.model[: last_dual_layer + 1].state_dict(), strict=True)
    # Keep official convolutional filters and affine BN parameters, but learn
    # thermal running statistics independently from a neutral state.
    for module in detector.anchor_ir_stem.modules():
        if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
            module.reset_running_stats()


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--name", required=True)
    parser.add_argument("--data", type=Path, required=True)
    initialization = parser.add_mutually_exclusive_group(required=True)
    initialization.add_argument("--source", type=Path)
    initialization.add_argument("--no-pretrained", action="store_true",
                                help="Build from YAML without loading any external initialization weights.")
    parser.add_argument("--model-family", choices=("yolo8", "yolo11"), default="yolo8")
    parser.add_argument("--model-scale", choices=("n", "s", "m", "l", "x"), default="n")
    parser.add_argument(
        "--mode", choices=("rgb_only", "ir_only", "p2_only", "p3_mean", "p3_residual"), required=True
    )
    parser.add_argument("--fusion-hidden", type=int, default=16)
    parser.add_argument("--p3-anchor", choices=("rgb", "ir"), default="rgb")
    parser.add_argument("--gate-max", type=float, default=0.25)
    parser.add_argument("--taylor-order", type=int, choices=(1, 2, 3), default=1)
    parser.add_argument(
        "--recurrence", choices=("additive", "bounded_mul", "rms_capped"), default="additive"
    )
    parser.add_argument("--anchor-freeze-epochs", type=int, default=0)
    parser.add_argument("--p2-bidirectional-main", action="store_true")
    parser.add_argument("--scale-assignment", choices=("DT", "DD", "TT", "TD"), default=None,
                        help="Cross P2 detail/Taylor exchange with P3 detail/Taylor RGB correction.")
    parser.add_argument("--single-taylor-scale", choices=("p3", "p4", "p5"), default=None,
                        help="Fix bidirectional P2; keep exactly one O3 module at the selected scale.")
    parser.add_argument("--p2-rgb-anchor-protected", action="store_true")
    parser.add_argument("--p2-main-hidden", type=int, default=16)
    parser.add_argument("--p2-main-gate-max", type=float, default=0.10)
    parser.add_argument(
        "--deep-taylor-scales",
        nargs="*",
        choices=("p4", "p5"),
        default=(),
        help="Continue the same RGB-anchored Taylor correction at selected deep backbone scales.",
    )
    parser.add_argument("--p2-aux-multifilter", action="store_true")
    parser.add_argument("--p2-aux-hidden", type=int, default=16)
    parser.add_argument("--p2-aux-gate-max", type=float, default=0.10)
    parser.add_argument("--p2-aux-loss-weight", type=float, default=0.10)
    parser.add_argument("--p2-aux-decay-start", type=int, default=-1)
    parser.add_argument("--p2-aux-decay-end", type=int, default=-1)
    parser.add_argument("--device", default="0")
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--nbs", type=int, default=64)
    parser.add_argument("--val-batch", type=int, default=0, help="Validation batch per rank; 0 uses trainer default")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--patience", type=int, default=120)
    parser.add_argument("--init-seed", type=int, default=0)
    parser.add_argument("--train-seed", type=int, default=0)
    parser.add_argument("--lr0", type=float, default=7e-4)
    parser.add_argument("--lrf", type=float, default=0.05)
    parser.add_argument("--warmup-epochs", type=float, default=5.0)
    parser.add_argument("--weight-decay", type=float, default=5e-4)
    parser.add_argument("--mosaic", type=float, default=1.0)
    parser.add_argument("--close-mosaic", type=int, default=30)
    parser.add_argument("--correct-border", action="store_true")
    parser.add_argument("--module-diagnostics", action="store_true")
    parser.add_argument("--diagnostic-interval", type=int, default=50)
    amp_group = parser.add_mutually_exclusive_group()
    amp_group.add_argument("--amp", dest="amp", action="store_true")
    amp_group.add_argument("--no-amp", dest="amp", action="store_false")
    parser.set_defaults(amp=False)
    parser.add_argument("--audit-only", action="store_true")
    return parser.parse_args()


def validate_args(args):
    if not args.data.is_file() or (not args.no_pretrained and not args.source.is_file()):
        raise FileNotFoundError(f"Missing data/source: data={args.data}, source={args.source}")
    if min(args.batch, args.nbs, args.workers + 1, args.epochs, args.patience) <= 0:
        raise ValueError("batch, nbs, epochs and patience must be positive; workers must be non-negative")
    if args.val_batch < 0:
        raise ValueError("--val-batch must be non-negative")
    devices = [item.strip() for item in str(args.device).split(",") if item.strip()]
    if len(devices) > 1 and args.batch % len(devices):
        raise ValueError(f"Global batch {args.batch} must be divisible by DDP world size {len(devices)}")
    if args.imgsz % 32:
        raise ValueError("--imgsz must be divisible by 32")
    if not 0 <= args.anchor_freeze_epochs < args.epochs:
        raise ValueError("--anchor-freeze-epochs must satisfy 0 <= value < epochs")
    if args.mode in {"rgb_only", "ir_only"} and args.anchor_freeze_epochs:
        raise ValueError("Single-modality baselines do not use anchor freezing")
    if args.fusion_hidden <= 0 or args.gate_max <= 0.0 or args.diagnostic_interval <= 0:
        raise ValueError("fusion-hidden, gate-max and diagnostic-interval must be positive")
    if args.mode != "p3_residual" and args.taylor_order != 1:
        raise ValueError("--taylor-order 2/3 is only valid with --mode p3_residual")
    if args.mode != "p3_residual" and args.p3_anchor != "rgb":
        raise ValueError("--p3-anchor ir is only valid with --mode p3_residual")
    if args.recurrence != "additive" and (args.mode != "p3_residual" or args.taylor_order < 2):
        raise ValueError("Constrained --recurrence requires p3_residual with --taylor-order 2 or 3")
    if args.single_taylor_scale:
        if (not args.p2_bidirectional_main or args.mode != "p3_residual" or args.p3_anchor != "rgb"
                or args.taylor_order != 3 or args.recurrence != "additive" or args.deep_taylor_scales
                or args.scale_assignment or args.p2_rgb_anchor_protected or args.p2_aux_multifilter
                or args.model_family != "yolo8"):
            raise ValueError("Single-scale O3 requires YOLOv8, fixed bidirectional P2 and no other fusion variants")
    if args.scale_assignment:
        if (not args.p2_bidirectional_main or args.mode != "p3_residual" or args.p3_anchor != "rgb"
                or args.taylor_order != 3 or args.recurrence != "additive" or args.deep_taylor_scales
                or args.p2_rgb_anchor_protected or args.p2_aux_multifilter):
            raise ValueError("Scale assignment requires bidirectional P2, RGB P3, additive O3 and no other fusions")
    if args.p2_bidirectional_main:
        if args.mode not in {"p2_only", "p3_residual"}:
            raise ValueError("--p2-bidirectional-main requires --mode p2_only or p3_residual")
    if args.mode == "p2_only":
        if (not args.p2_bidirectional_main or args.scale_assignment or args.single_taylor_scale
                or args.deep_taylor_scales or args.p2_rgb_anchor_protected or args.p2_aux_multifilter):
            raise ValueError("P2-only requires exactly the main-path P2 exchange and no P3/P4/P5 fusion variant")
    if args.deep_taylor_scales:
        if args.mode != "p3_residual" or args.p3_anchor != "rgb":
            raise ValueError("--deep-taylor-scales requires RGB-anchored --mode p3_residual")
        if tuple(dict.fromkeys(args.deep_taylor_scales)) != tuple(args.deep_taylor_scales):
            raise ValueError("--deep-taylor-scales must not contain duplicates")
        if tuple(sorted(args.deep_taylor_scales)) != tuple(args.deep_taylor_scales):
            raise ValueError("--deep-taylor-scales must be ordered as p4 p5")
        if min(args.p2_main_hidden, args.p2_main_gate_max) <= 0:
            raise ValueError("P2 main-path hidden and gate max must be positive")
    if sum((args.p2_bidirectional_main, args.p2_rgb_anchor_protected, args.p2_aux_multifilter)) > 1:
        raise ValueError("P2 bidirectional, RGB-protected and auxiliary modes are separate experiments")
    if args.p2_rgb_anchor_protected:
        if args.mode != "p3_residual" or args.p3_anchor != "rgb":
            raise ValueError("P2 RGB-anchor protection requires RGB-anchored residual P3 fusion")
        if min(args.p2_main_hidden, args.p2_main_gate_max) <= 0:
            raise ValueError("P2 RGB-protected hidden and gate max must be positive")
    if args.p2_aux_multifilter:
        if args.mode != "p3_residual":
            raise ValueError("--p2-aux-multifilter requires --mode p3_residual")
        if min(args.p2_aux_hidden, args.p2_aux_gate_max, args.p2_aux_loss_weight) <= 0:
            raise ValueError("P2 auxiliary hidden, gate max and loss weight must be positive")
        decay = (args.p2_aux_decay_start, args.p2_aux_decay_end)
        if decay != (-1, -1) and not (0 <= decay[0] < decay[1] <= args.epochs):
            raise ValueError("P2 auxiliary decay must be -1/-1 or 0 <= start < end <= epochs")


def main():
    args = parse_args()
    validate_args(args)
    SETTINGS["wandb"] = False
    run_dir = PROJECT / args.name
    if run_dir.exists() and not args.audit_only:
        raise FileExistsError(f"Run directory already exists: {run_dir}")
    data_path, nc, names = dataset_identity(args.data)
    init_seeds(args.init_seed, deterministic=True)
    seed_audit = {
        "torch": int(torch.initial_seed()),
        "python": repr(random.getstate()[1][:5]),
        "numpy": np.random.get_state()[1][:5].tolist(),
    }
    model_yaml = build_yaml(args, nc)
    # Build and audit the initialization checkpoint on CPU.  Otherwise the
    # parent process establishes a CUDA context on the first visible device
    # before spawning DDP ranks, which can deadlock their first NCCL setup.
    previous_probe_device = os.environ.get("ULTRALYTICS_MODEL_PROBE_DEVICE")
    os.environ["ULTRALYTICS_MODEL_PROBE_DEVICE"] = "cpu"
    try:
        model = YOLO(str(model_yaml))
        detector = model.model
        source = None if args.no_pretrained else YOLO(str(args.source)).model
    finally:
        if previous_probe_device is None:
            os.environ.pop("ULTRALYTICS_MODEL_PROBE_DEVICE", None)
        else:
            os.environ["ULTRALYTICS_MODEL_PROBE_DEVICE"] = previous_probe_device
    source_nc, matching, matching_parameters, incompatible = None, {}, 0, None
    if source is not None:
        source_nc = int(source.yaml.get("nc", 0))
        matching, matching_parameters = matching_state(detector, source)
        incompatible = detector.load_state_dict(matching, strict=False)
    del source
    initialize_ir_stem(detector)

    fusion = getattr(detector, "anchor_p3_fusion", None)
    exact_rgb_identity_at_initialization = False
    if args.mode == "p3_residual":
        with torch.no_grad():
            device = next(detector.parameters()).device
            rgb = torch.randn(2, detector.model[4].cv2.conv.out_channels, 8, 8, device=device)
            ir = torch.randn_like(rgb)
            expected = rgb if args.p3_anchor == "rgb" else ir
            if not torch.equal(fusion(rgb, ir), expected):
                raise RuntimeError(
                    f"Residual P3 fusion is not an exact {args.p3_anchor.upper()} identity at initialization"
                )
    p2_fusion = getattr(detector, "anchor_p2_detail_fusion", None)
    if p2_fusion is not None:
        with torch.no_grad():
            device = next(p2_fusion.parameters()).device
            channels = detector.model[2].cv2.conv.out_channels
            rgb = torch.randn(2, channels, 16, 16, device=device)
            ir = torch.randn_like(rgb)
            if not torch.equal(p2_fusion(rgb, ir), rgb):
                raise RuntimeError("P2 frequency fusion is not an exact RGB identity at initialization")
    p2_bidirectional = getattr(detector, "anchor_p2_bidirectional_injection", None)
    if p2_bidirectional is not None:
        with torch.no_grad():
            device = next(p2_bidirectional.parameters()).device
            channels = detector.model[2].cv2.conv.out_channels
            rgb = torch.randn(2, channels, 16, 16, device=device)
            ir = torch.randn_like(rgb)
            rgb_out, ir_out = p2_bidirectional(rgb, ir)
            if not torch.equal(rgb_out, rgb) or not torch.equal(ir_out, ir):
                raise RuntimeError("P2 bidirectional injection is not an exact two-stream identity at initialization")
    p2_rgb_only = getattr(detector, "anchor_p2_rgb_only_injection", None)
    if p2_rgb_only is not None:
        with torch.no_grad():
            device = next(p2_rgb_only.parameters()).device
            channels = detector.model[2].cv2.conv.out_channels
            rgb = torch.randn(2, channels, 16, 16, device=device)
            ir = torch.randn_like(rgb)
            if not torch.equal(p2_rgb_only(rgb, ir), rgb):
                raise RuntimeError("P2-only RGB correction is not an exact RGB identity at initialization")
    deep_taylor_fusions = getattr(detector, "anchor_deep_taylor_fusions", None)
    if deep_taylor_fusions is not None:
        with torch.no_grad():
            for layer, fusion_module in deep_taylor_fusions.items():
                channels = fusion_module.gates[0][-1].out_channels
                rgb = torch.randn(2, channels, 8, 8)
                ir = torch.randn_like(rgb)
                if not torch.equal(fusion_module(rgb, ir), rgb):
                    raise RuntimeError(f"Deep Taylor fusion at layer {layer} is not an exact RGB identity")
    if args.mode in {"p2_only", "p3_residual"}:
        # Verify the complete graph, not only isolated correctors.  With all
        # zero-initialized gates, P2/P3/P4/P5 paths must reproduce the same
        # detector output as the RGB-only route bit for bit.
        was_training = detector.training
        detector.eval()
        probe = torch.randn(1, 6, 128, 128)
        configured_mode = detector.pretrained_anchor_config["mode"]
        with torch.no_grad():
            detector.pretrained_anchor_config["mode"] = "rgb_only"
            rgb_reference = detector(probe)
            detector.pretrained_anchor_config["mode"] = configured_mode
            anchored_output = detector(probe)
        detector.pretrained_anchor_config["mode"] = configured_mode
        detector.train(was_training)
        if not nested_tensors_equal(anchored_output, rgb_reference):
            raise RuntimeError("Pretrained-anchor detector is not an exact RGB-only graph at initialization")
        exact_rgb_identity_at_initialization = True
    p2_rgb_protected = getattr(detector, "anchor_p2_rgb_protected_injection", None)
    if p2_rgb_protected is not None:
        with torch.no_grad():
            device = next(p2_rgb_protected.parameters()).device
            channels = detector.model[2].cv2.conv.out_channels
            rgb = torch.randn(2, channels, 16, 16, device=device)
            ir = torch.randn_like(rgb)
            rgb_out, ir_out = p2_rgb_protected(rgb, ir)
            if not torch.equal(rgb_out, rgb) or not torch.equal(ir_out, ir):
                raise RuntimeError("P2 RGB-anchor protection is not an exact two-stream identity at initialization")
            p2_rgb_protected.shared_gate[-1].bias.fill_(0.5)
            rgb_out, ir_out = p2_rgb_protected(rgb, ir)
            if not torch.equal(rgb_out, rgb) or torch.equal(ir_out, ir):
                raise RuntimeError("P2 RGB-anchor protection does not preserve RGB while allowing IR correction")
            p2_rgb_protected.shared_gate[-1].bias.zero_()

    audit = {
        "experiment": "pretrained_anchor_p3_fusion",
        "name": args.name,
        "mode": args.mode,
        "model_scale": args.model_scale,
        "model_family": args.model_family,
        "taylor_order": args.taylor_order,
        "scale_assignment": args.scale_assignment,
        "single_taylor_scale": args.single_taylor_scale,
        "single_taylor_config": getattr(detector, "anchor_single_taylor_config", None),
        "ir_stem_layers": len(detector.anchor_ir_stem) if hasattr(detector, "anchor_ir_stem") else 0,
        "scale_assignment_config": getattr(detector, "anchor_scale_assignment_config", None),
        "p2_module_class": type(getattr(detector, "anchor_p2_bidirectional_injection", None)).__name__,
        "p3_module_class": type(getattr(detector, "anchor_p3_fusion", None)).__name__,
        "p3_anchor_modality": args.p3_anchor,
        "p2_bidirectional_main": args.p2_bidirectional_main,
        "p2_bidirectional_config": getattr(detector, "anchor_p2_bidirectional_config", None),
        "p2_rgb_only": getattr(detector, "anchor_p2_rgb_only_injection", None) is not None,
        "p2_rgb_only_module_class": type(getattr(detector, "anchor_p2_rgb_only_injection", None)).__name__,
        "p2_only_reference_p3_init": detector.yaml.get("p2_only_reference_p3_init"),
        "deep_taylor_scales": list(args.deep_taylor_scales),
        "deep_taylor_config": getattr(detector, "anchor_deep_taylor_config", None),
        "p2_rgb_anchor_protected": args.p2_rgb_anchor_protected,
        "p2_rgb_anchor_protected_config": getattr(detector, "anchor_p2_rgb_protected_config", None),
        "p2_aux_multifilter": args.p2_aux_multifilter,
        "p2_aux_config": getattr(detector, "anchor_p2_aux_config", None),
        "dataset_yaml": str(data_path),
        "dataset_yaml_sha256": sha256(data_path),
        "dataset_nc": nc,
        "dataset_names": names,
        "initialization": "scratch" if args.no_pretrained else "pretrained",
        "source_checkpoint": str(args.source.resolve()) if args.source else None,
        "source_checkpoint_sha256": sha256(args.source) if args.source else None,
        "model_yaml": str(model_yaml.resolve()),
        "model_yaml_sha256": sha256(model_yaml),
        "pretrained_parameters": matching_parameters,
        "pretrained_tensors": len(matching),
        "source_nc": source_nc,
        "missing_after_source_load": incompatible.missing_keys if incompatible else [],
        "unexpected_after_source_load": incompatible.unexpected_keys if incompatible else [],
        "ir_stem_initialized_from_loaded_rgb": not args.no_pretrained and getattr(detector, "anchor_ir_stem", None) is not None,
        "ir_stem_initialized_from_random_rgb": args.no_pretrained and getattr(detector, "anchor_ir_stem", None) is not None,
        "exact_rgb_identity_at_initialization": exact_rgb_identity_at_initialization,
        "total_parameters": sum(p.numel() for p in detector.parameters()),
        "train_args": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        "seed_state_before_model": seed_audit,
    }
    audit_dir = ROOT / "weights"
    audit_dir.mkdir(parents=True, exist_ok=True)
    with open(audit_dir / f"{args.name}_anchor_audit.json", "w", encoding="utf-8") as file:
        json.dump(audit, file, ensure_ascii=False, indent=2)
    print(json.dumps(audit, ensure_ascii=False, indent=2))
    model.ckpt = {"model": detector}
    devices = [item.strip() for item in str(args.device).split(",") if item.strip()]
    ddp_init_path = None
    if len(devices) > 1:
        # Ultralytics DDP reconstructs the trainer in subprocesses from args.
        # Persist the already-loaded official initialization so every rank
        # starts from exactly the audited detector instead of a random YAML model.
        ddp_init_path = audit_dir / f"{args.name}_ddp_init.pt"
        model.save(ddp_init_path)
        reloaded = YOLO(str(ddp_init_path))
        original_state = detector.state_dict()
        reloaded_state = reloaded.model.state_dict()
        if original_state.keys() != reloaded_state.keys() or any(
            not torch.equal(original_state[key].cpu(), reloaded_state[key].cpu()) for key in original_state
        ):
            raise RuntimeError("DDP initialization checkpoint does not exactly reproduce the audited model")
        model = reloaded
        print(f"DDP initialization checkpoint verified: {ddp_init_path}")

    if args.audit_only:
        if ddp_init_path is not None:
            ddp_init_path.unlink(missing_ok=True)
        return

    try:
        model.train(
            cfg=str(TRAIN_CFG), data=str(data_path), project=str(PROJECT), name=args.name,
            pretrained=not args.no_pretrained,
            device=args.device, epochs=args.epochs, patience=args.patience,
            batch=args.batch, nbs=args.nbs, val_batch=args.val_batch, imgsz=args.imgsz,
            optimizer="AdamW", lr0=args.lr0, lrf=args.lrf, cos_lr=True,
            warmup_epochs=args.warmup_epochs, warmup_momentum=0.8, warmup_bias_lr=0.0,
            momentum=0.937, weight_decay=args.weight_decay,
            box=7.5, cls=0.5, dfl=1.5,
            mosaic=args.mosaic, close_mosaic=args.close_mosaic,
            hsv_h=0.015, hsv_s=0.7, hsv_v=0.4, translate=0.1, scale=0.5,
            ir_aug_p=0.0, ir_gain=0.0, ir_contrast=0.0, ir_noise=0.0, ir_blur_p=0.0,
            legacy_multimodal_border=not args.correct_border,
            seed=args.train_seed, deterministic=True, amp=args.amp, workers=args.workers,
            anchor_freeze_epochs=args.anchor_freeze_epochs,
            module_diagnostics=args.module_diagnostics,
            diagnostic_interval=args.diagnostic_interval,
            save_period=-1,
        )
    finally:
        if ddp_init_path is not None:
            ddp_init_path.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
