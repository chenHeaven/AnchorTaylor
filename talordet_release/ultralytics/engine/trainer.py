# Ultralytics YOLO 🚀, AGPL-3.0 license
"""
Train a model on a dataset.

Usage:
    $ yolo mode=train model=yolov8n.pt data=coco128.yaml imgsz=640 epochs=100 batch=16
"""

import csv
import math
import os
import subprocess
import time
import warnings
from copy import deepcopy
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import torch
from torch import distributed as dist
from torch import nn, optim

from ultralytics.cfg import get_cfg, get_save_dir
from ultralytics.data.utils import check_cls_dataset, check_det_dataset
from ultralytics.nn.tasks import attempt_load_one_weight, attempt_load_weights
from ultralytics.utils import (
    DEFAULT_CFG,
    LOGGER,
    RANK,
    TQDM,
    __version__,
    callbacks,
    clean_url,
    colorstr,
    emojis,
    yaml_save,
)
from ultralytics.utils.autobatch import check_train_batch_size
from ultralytics.utils.checks import check_amp, check_file, check_imgsz, check_model_file_from_stem, print_args
from ultralytics.utils.dist import ddp_cleanup, generate_ddp_command
from ultralytics.utils.files import get_latest_run
from ultralytics.utils.torch_utils import (
    EarlyStopping,
    ModelEMA,
    convert_optimizer_state_dict_to_fp16,
    init_seeds,
    one_cycle,
    select_device,
    strip_optimizer,
)


class BaseTrainer:
    """
    BaseTrainer.

    A base class for creating trainers.

    Attributes:
        args (SimpleNamespace): Configuration for the trainer.
        validator (BaseValidator): Validator instance.
        model (nn.Module): Model instance.
        callbacks (defaultdict): Dictionary of callbacks.
        save_dir (Path): Directory to save results.
        wdir (Path): Directory to save weights.
        last (Path): Path to the last checkpoint.
        best (Path): Path to the best checkpoint.
        save_period (int): Save checkpoint every x epochs (disabled if < 1).
        batch_size (int): Batch size for training.
        epochs (int): Number of epochs to train for.
        start_epoch (int): Starting epoch for training.
        device (torch.device): Device to use for training.
        amp (bool): Flag to enable AMP (Automatic Mixed Precision).
        scaler (amp.GradScaler): Gradient scaler for AMP.
        data (str): Path to data.
        trainset (torch.utils.data.Dataset): Training dataset.
        testset (torch.utils.data.Dataset): Testing dataset.
        ema (nn.Module): EMA (Exponential Moving Average) of the model.
        resume (bool): Resume training from a checkpoint.
        lf (nn.Module): Loss function.
        scheduler (torch.optim.lr_scheduler._LRScheduler): Learning rate scheduler.
        best_fitness (float): The best fitness value achieved.
        fitness (float): Current fitness value.
        loss (float): Current loss value.
        tloss (float): Total loss value.
        loss_names (list): List of loss names.
        csv (Path): Path to results CSV file.
    """

    def __init__(self, cfg=DEFAULT_CFG, overrides=None, _callbacks=None):
        """
        Initializes the BaseTrainer class.

        Args:
            cfg (str, optional): Path to a configuration file. Defaults to DEFAULT_CFG.
            overrides (dict, optional): Configuration overrides. Defaults to None.
        """
        self.args = get_cfg(cfg, overrides)
        self.check_resume(overrides)
        self.device = select_device(self.args.device, self.args.batch)
        self.validator = None
        self.metrics = None
        self.plots = {}
        init_seeds(self.args.seed + 1 + RANK, deterministic=self.args.deterministic)

        # Dirs
        self.save_dir = get_save_dir(self.args)
        self.args.name = self.save_dir.name  # update name for loggers
        self.wdir = self.save_dir / "weights"  # weights dir
        if RANK in {-1, 0}:
            self.wdir.mkdir(parents=True, exist_ok=True)  # make dir
            self.args.save_dir = str(self.save_dir)
            yaml_save(self.save_dir / "args.yaml", vars(self.args))  # save run args
        self.last, self.best = self.wdir / "last.pt", self.wdir / "best.pt"  # checkpoint paths
        self.save_period = self.args.save_period

        self.batch_size = self.args.batch
        self.epochs = self.args.epochs
        self.start_epoch = 0
        if RANK == -1:
            #print_args(vars(self.args))
            print("1")

        # Device
        if self.device.type in {"cpu", "mps"}:
            self.args.workers = 0  # faster CPU training as time dominated by inference, not dataloading

        # Model and Dataset
        self.model = check_model_file_from_stem(self.args.model)  # add suffix, i.e. yolov8n -> yolov8n.pt
        self.trainset, self.testset,self.trainirset,self.testirset = self.get_dataset()
        self.ema = None

        # Optimization utils init
        self.lf = None
        self.scheduler = None

        # Epoch level metrics
        self.best_fitness = None
        self.fitness = None
        self.loss = None
        self.tloss = None
        self.loss_names = ["Loss"]
        self.csv = self.save_dir / "results.csv"
        self.module_diagnostics_csv = self.save_dir / "module_diagnostics.csv"
        self.class_metrics_csv = self.save_dir / "class_metrics.csv"
        self._diagnostic_sums = {}
        self._diagnostic_counts = {}
        self.plot_idx = [0, 1, 2]

        # Callbacks
        self.callbacks = _callbacks or callbacks.get_default_callbacks()
        if RANK in {-1, 0}:
            callbacks.add_integration_callbacks(self)

    def add_callback(self, event: str, callback):
        """Appends the given callback."""
        self.callbacks[event].append(callback)

    def set_callback(self, event: str, callback):
        """Overrides the existing callbacks with the given callback."""
        self.callbacks[event] = [callback]

    def run_callbacks(self, event: str):
        """Run all existing callbacks associated with a particular event."""
        for callback in self.callbacks.get(event, []):
            callback(self)

    def _diagnostics_enabled(self):
        return bool(getattr(self.args, "module_diagnostics", False))

    def _diagnostic_model(self):
        return self.model.module if hasattr(self.model, "module") else self.model

    def _set_diagnostic_capture(self, enabled):
        if not self._diagnostics_enabled():
            return
        for module in self._diagnostic_model().modules():
            if callable(getattr(module, "diagnostic_values", None)):
                module.capture_diagnostics = bool(enabled)

    def _reset_diagnostics(self):
        self._diagnostic_sums.clear()
        self._diagnostic_counts.clear()

    def _apply_pixel_taylor_schedule(self, epoch):
        """Apply one exact curriculum value to the train model and its EMA."""
        delay = int(getattr(self.args, "pixel_taylor_delay_epochs", 0))
        ramp = int(getattr(self.args, "pixel_taylor_ramp_epochs", 0))
        if delay < 0 or ramp < 0:
            raise ValueError("Pixel Taylor delay/ramp epochs must be non-negative")
        if epoch < delay:
            scale = 0.0
        elif ramp > 0:
            scale = min(1.0, (epoch - delay + 1) / ramp)
        else:
            scale = 1.0

        train_model = self._diagnostic_model()
        train_fusion = getattr(train_model, "pixel_fusion", None)
        setter = getattr(train_fusion, "set_taylor_schedule_scale", None)
        if callable(setter):
            setter(scale)
            # Optimizer groups are built before epoch zero while these
            # parameters are trainable. Disabling gradients during a zero
            # schedule prevents decoupled AdamW decay from drifting them.
            branch_enabled = scale > 0.0
            for name, parameter in train_fusion.named_parameters():
                if (
                    name.startswith("derivatives.")
                    or name == "order_gate_logits"
                    or name.startswith("target_gates.")
                ):
                    parameter.requires_grad_(branch_enabled)
            head_injector = getattr(train_model, "pixel_taylor_head_injector", None)
            if head_injector is not None:
                for parameter in head_injector.parameters():
                    parameter.requires_grad_(branch_enabled)

        ema_model = getattr(getattr(self, "ema", None), "ema", None)
        ema_fusion = getattr(ema_model, "pixel_fusion", None)
        ema_setter = getattr(ema_fusion, "set_taylor_schedule_scale", None)
        if callable(ema_setter):
            ema_setter(scale)

        if train_fusion is not None and epoch in {0, delay, delay + max(ramp - 1, 0)}:
            LOGGER.info(
                f"Pixel Taylor curriculum: epoch {epoch + 1}, scale={scale:.4f}, "
                f"delay={delay}, ramp={ramp}"
            )

    def _apply_fusion_aux_schedule(self, epoch):
        """Update the training-only fusion detector weight before the epoch forward passes."""
        train_model = self._diagnostic_model()
        aux_head = getattr(train_model, "fusion_aux_head", None)
        setter = getattr(aux_head, "set_loss_weight_for_epoch", None)
        if not callable(setter):
            return
        weight = setter(epoch)
        ema_model = getattr(getattr(self, "ema", None), "ema", None)
        ema_head = getattr(ema_model, "fusion_aux_head", None)
        ema_setter = getattr(ema_head, "set_loss_weight_for_epoch", None)
        if callable(ema_setter):
            ema_setter(epoch)
        if self._diagnostics_enabled():
            self._accumulate_diagnostic("fusion_aux_head/scheduled_weight", weight)
        boundary_epochs = {
            0,
            int(getattr(aux_head, "decay_start_epoch", -1)),
            int(getattr(aux_head, "decay_end_epoch", -1)),
        }
        if epoch in boundary_epochs:
            LOGGER.info(
                f"Fusion auxiliary schedule: epoch {epoch + 1}, weight={weight:.6f}, "
                f"decay=[{aux_head.decay_start_epoch}, {aux_head.decay_end_epoch}]"
            )

    def _apply_pretrained_anchor_schedule(self, epoch):
        """Initially optimize only the new IR stem/fusion, then unfreeze the official detector."""
        train_model = self._diagnostic_model()
        if getattr(train_model, "pretrained_anchor_config", None) is None:
            return
        freeze_epochs = int(getattr(self.args, "anchor_freeze_epochs", 0))
        if freeze_epochs <= 0:
            return
        detector_trainable = epoch >= freeze_epochs
        for name, parameter in train_model.named_parameters():
            if name.startswith("model."):
                parameter.requires_grad_(detector_trainable and ".dfl." not in name)
        if not detector_trainable:
            # requires_grad=False does not freeze running statistics. Keep the
            # pretrained RGB normalization anchor exact during adapter warmup.
            for module in train_model.model.modules():
                if isinstance(module, nn.modules.batchnorm._BatchNorm):
                    module.eval()
        if epoch in {0, freeze_epochs}:
            LOGGER.info(
                f"Pretrained-anchor schedule: epoch {epoch + 1}, "
                f"official detector trainable={detector_trainable}, freeze_epochs={freeze_epochs}"
            )

    def _accumulate_diagnostic(self, key, value):
        if torch.is_tensor(value):
            value = value.detach().float().mean().item()
        try:
            value = float(value)
        except (TypeError, ValueError):
            return
        if not math.isfinite(value):
            return
        self._diagnostic_sums[key] = self._diagnostic_sums.get(key, 0.0) + value
        self._diagnostic_counts[key] = self._diagnostic_counts.get(key, 0) + 1

    def _collect_forward_diagnostics(self):
        if not self._diagnostics_enabled():
            return
        for name, module in self._diagnostic_model().named_modules():
            getter = getattr(module, "diagnostic_values", None)
            if not callable(getter):
                continue
            for key, value in getter().items():
                self._accumulate_diagnostic(f"{name or 'model'}/{key}", value)

    def _collect_gradient_diagnostics(self, total_norm):
        if not self._diagnostics_enabled():
            return
        self._accumulate_diagnostic("grad/global_norm_before_clip", total_norm)
        diagnostic_names = (
            "pixel_fusion",
            "pixel_taylor_head_injector",
            "fusion_aux_encoder",
            "fusion_aux_head",
            "anchor_ir_stem",
            "anchor_p3_fusion",
            "anchor_deep_taylor_fusions",
            "anchor_p2_detail_fusion",
            "anchor_p2_bidirectional_injection",
            "anchor_p2_rgb_only_injection",
            "anchor_p2_rgb_protected_injection",
            "correction_gates",
            "quality_residual_gates",
            "quality_predictor",
            "quality_mixer_logits",
            "derivative_residual",
            "rgb_detail_filter",
            "derivative.global_gate",
            "positive_wavelet_taylor_residual",
            "model.0.conv",
        )
        for name, parameter in self._diagnostic_model().named_parameters():
            if parameter.grad is not None and any(token in name for token in diagnostic_names):
                self._accumulate_diagnostic(f"grad/{name}", parameter.grad.detach().float().norm())

    def _save_module_diagnostics(self):
        if not self._diagnostics_enabled() or RANK not in {-1, 0}:
            return
        values = {
            key: self._diagnostic_sums[key] / self._diagnostic_counts[key]
            for key in sorted(self._diagnostic_sums)
        }
        if not values:
            return
        forward_counts = [count for key, count in self._diagnostic_counts.items() if not key.startswith("grad/")]
        row = {
            "epoch": self.epoch + 1,
            "forward_samples": max(forward_counts, default=0),
            "optimizer_steps": self._diagnostic_counts.get("grad/global_norm_before_clip", 0),
            **values,
        }
        fieldnames = list(row)
        if self.module_diagnostics_csv.exists():
            with open(self.module_diagnostics_csv, newline="", encoding="utf-8") as file:
                reader = csv.DictReader(file)
                existing = list(reader.fieldnames or [])
                existing_rows = list(reader)
            # Diagnostics legitimately gain columns when a delayed branch is
            # unfrozen. Keep a stable union schema and backfill earlier rows
            # instead of aborting training at the curriculum boundary.
            merged = existing + [name for name in fieldnames if name not in existing]
            if merged != existing:
                temporary = self.module_diagnostics_csv.with_suffix(".csv.tmp")
                with open(temporary, "w", newline="", encoding="utf-8") as file:
                    writer = csv.DictWriter(file, fieldnames=merged, restval="")
                    writer.writeheader()
                    writer.writerows(existing_rows)
                os.replace(temporary, self.module_diagnostics_csv)
                LOGGER.info(
                    f"Expanded module diagnostics schema with {len(merged) - len(existing)} new columns."
                )
            fieldnames = merged
        with open(self.module_diagnostics_csv, "a", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=fieldnames, restval="")
            if file.tell() == 0:
                writer.writeheader()
            writer.writerow(row)

    def _save_class_metrics(self):
        """Write per-class and validation-completeness metrics outside results.csv."""
        if RANK not in {-1, 0}:
            return
        class_values = getattr(self.validator, "class_diagnostics", None)
        if not class_values:
            return
        validation_values = getattr(self.validator, "validation_diagnostics", {}) or {}
        row = {
            "epoch": self.epoch + 1,
            **{key: class_values[key] for key in sorted(class_values)},
            **{f"validation/{key}": validation_values[key] for key in sorted(validation_values)},
        }
        fieldnames = list(row)
        if self.class_metrics_csv.exists():
            with open(self.class_metrics_csv, newline="", encoding="utf-8") as file:
                reader = csv.DictReader(file)
                existing = list(reader.fieldnames or [])
                existing_rows = list(reader)
            merged = existing + [name for name in fieldnames if name not in existing]
            if merged != existing:
                temporary = self.class_metrics_csv.with_suffix(".csv.tmp")
                with open(temporary, "w", newline="", encoding="utf-8") as file:
                    writer = csv.DictWriter(file, fieldnames=merged, restval="")
                    writer.writeheader()
                    writer.writerows(existing_rows)
                os.replace(temporary, self.class_metrics_csv)
            fieldnames = merged
        with open(self.class_metrics_csv, "a", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=fieldnames, restval="")
            if file.tell() == 0:
                writer.writeheader()
            writer.writerow(row)

    def train(self):
        """Allow device='', device=None on Multi-GPU systems to default to device=0."""
        if isinstance(self.args.device, str) and len(self.args.device):  # i.e. device='0' or device='0,1,2,3'
            world_size = len(self.args.device.split(","))
        elif isinstance(self.args.device, (tuple, list)):  # i.e. device=[0, 1, 2, 3] (multi-GPU from CLI is list)
            world_size = len(self.args.device)
        elif torch.cuda.is_available():  # i.e. device=None or device='' or device=number
            world_size = 1  # default to device 0
        else:  # i.e. device='cpu' or 'mps'
            world_size = 0

        # Run subprocess if DDP training, else train normally
        if world_size > 1 and "LOCAL_RANK" not in os.environ:
            # Argument checks
            if self.args.rect:
                LOGGER.warning("WARNING ⚠️ 'rect=True' is incompatible with Multi-GPU training, setting 'rect=False'")
                self.args.rect = False
            if self.args.batch == -1:
                LOGGER.warning(
                    "WARNING ⚠️ 'batch=-1' for AutoBatch is incompatible with Multi-GPU training, setting "
                    "default 'batch=16'"
                )
                self.args.batch = 16

            # Command
            cmd, file = generate_ddp_command(world_size, self)
            try:
                LOGGER.info(f'{colorstr("DDP:")} debug command {" ".join(cmd)}')
                subprocess.run(cmd, check=True)
            except Exception as e:
                raise e
            finally:
                ddp_cleanup(self, str(file))

        else:
            self._do_train(world_size)

    def _setup_scheduler(self):
        """Initialize training learning rate scheduler."""
        if self.args.cos_lr:
            self.lf = one_cycle(1, self.args.lrf, self.epochs)  # cosine 1->hyp['lrf']
        else:
            self.lf = lambda x: max(1 - x / self.epochs, 0) * (1.0 - self.args.lrf) + self.args.lrf  # linear
        self.scheduler = optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=self.lf)

    def _setup_ddp(self, world_size):
        """Initializes and sets the DistributedDataParallel parameters for training."""
        torch.cuda.set_device(RANK)
        self.device = torch.device("cuda", RANK)
        # LOGGER.info(f'DDP info: RANK {RANK}, WORLD_SIZE {world_size}, DEVICE {self.device}')
        os.environ["TORCH_NCCL_BLOCKING_WAIT"] = "1"  # set to enforce timeout
        dist.init_process_group(
            backend="nccl" if dist.is_nccl_available() else "gloo",
            timeout=timedelta(seconds=10800),  # 3 hours
            rank=RANK,
            world_size=world_size,
        )

    def _setup_train(self, world_size):
        """Builds dataloaders and optimizer on correct rank process."""

        # Model
        self.run_callbacks("on_pretrain_routine_start")
        ckpt = self.setup_model()
        self.model = self.model.to(self.device)
        self.set_model_attributes()

        # Freeze layers
        freeze_list = (
            self.args.freeze
            if isinstance(self.args.freeze, list)
            else range(self.args.freeze)
            if isinstance(self.args.freeze, int)
            else []
        )
        always_freeze_names = [".dfl", ".wt_filter", ".wt_filters", ".iwt_filter", ".iwt_filters"]  # always freeze these layers
        freeze_layer_names = [f"model.{x}." for x in freeze_list] + always_freeze_names
        adapter_only = bool(getattr(self.args, "adapter_only", False))
        requested_adapter_layers = getattr(self.args, "adapter_train_layers", None)
        if isinstance(requested_adapter_layers, int):
            requested_adapter_layers = [requested_adapter_layers]
        adapter_trainable_names = tuple(
            pattern
            for layer in (requested_adapter_layers or [])
            for pattern in (
                f"model.{int(layer)}.derivative.global_context.",
                f"model.{int(layer)}.derivative.global_gate",
            )
        )
        if adapter_only and not adapter_trainable_names:
            raise ValueError("adapter_only=True requires at least one adapter_train_layers entry")
        wavelet_adapter_only = bool(getattr(self.args, "wavelet_adapter_only", False))
        pixel_taylor_adapter_only = bool(getattr(self.args, "pixel_taylor_adapter_only", False))
        pixel_taylor_joint = bool(getattr(self.args, "pixel_taylor_joint", False))
        pixel_taylor_joint_scope = str(getattr(self.args, "pixel_taylor_joint_scope", "fusion_stem"))
        requested_wavelet_layers = getattr(self.args, "wavelet_adapter_train_layers", None)
        if isinstance(requested_wavelet_layers, int):
            requested_wavelet_layers = [requested_wavelet_layers]
        wavelet_trainable_names = tuple(
            f"model.{int(layer)}.positive_wavelet_taylor_residual."
            for layer in (requested_wavelet_layers or [])
        )
        if wavelet_adapter_only and not wavelet_trainable_names:
            raise ValueError("wavelet_adapter_only=True requires wavelet_adapter_train_layers")
        if sum((adapter_only, wavelet_adapter_only, pixel_taylor_adapter_only)) > 1:
            raise ValueError("adapter_only, wavelet_adapter_only and pixel_taylor_adapter_only are mutually exclusive")
        if pixel_taylor_joint and not pixel_taylor_adapter_only:
            raise ValueError("pixel_taylor_joint=True requires pixel_taylor_adapter_only=True")
        if pixel_taylor_joint_scope not in {"fusion", "fusion_stem", "fusion_stem_conv"}:
            raise ValueError("pixel_taylor_joint_scope must be 'fusion', 'fusion_stem' or 'fusion_stem_conv'")
        pixel_taylor_trainable_names = (
            "pixel_fusion.derivatives.",
            "pixel_fusion.order_gate_logits",
            "pixel_fusion.target_gates.",
        )
        if pixel_taylor_joint:
            pixel_taylor_trainable_names += ("pixel_fusion.concat_projection.",)
            if pixel_taylor_joint_scope == "fusion_stem":
                pixel_taylor_trainable_names += ("model.0.",)
            elif pixel_taylor_joint_scope == "fusion_stem_conv":
                pixel_taylor_trainable_names += ("model.0.conv.",)
        if pixel_taylor_adapter_only and not any(
            any(token in name for token in pixel_taylor_trainable_names)
            for name, _ in self.model.named_parameters()
        ):
            raise ValueError("pixel_taylor_adapter_only=True requires a concat_taylor model")
        for k, v in self.model.named_parameters():
            # v.register_hook(lambda x: torch.nan_to_num(x))  # NaN to 0 (commented for erratic training results)
            if adapter_only and not any(x in k for x in adapter_trainable_names):
                v.requires_grad = False
            elif wavelet_adapter_only and not any(x in k for x in wavelet_trainable_names):
                v.requires_grad = False
            elif pixel_taylor_adapter_only and not any(x in k for x in pixel_taylor_trainable_names):
                v.requires_grad = False
            elif any(x in k for x in freeze_layer_names):
                LOGGER.info(f"Freezing layer '{k}'")
                v.requires_grad = False
            elif not v.requires_grad and v.dtype.is_floating_point:  # only floating point Tensor can require gradients
                LOGGER.info(
                    f"WARNING ⚠️ setting 'requires_grad=True' for frozen layer '{k}'. "
                    "See ultralytics.engine.trainer for customization of frozen layers."
                )
                v.requires_grad = True

        # Check AMP
        self.amp = torch.tensor(self.args.amp).to(self.device)  # True or False
        # Run the auxiliary YOLOv8 AMP probe only in single-process training.
        # In DDP it performs asymmetric model construction on rank 0 before the
        # first collective, which can leave the other ranks blocked in the AMP
        # broadcast. The requested AMP flag is already identical on all ranks.
        if self.amp and RANK == -1:
            callbacks_backup = callbacks.default_callbacks.copy()  # backup callbacks as check_amp() resets them
            self.amp = torch.tensor(check_amp(self.model), device=self.device)
            callbacks.default_callbacks = callbacks_backup  # restore callbacks
        # Every DDP rank is reconstructed from the same generated overrides,
        # so the AMP flag is already identical.  Avoid a redundant scalar
        # collective before DDP construction; this legacy broadcast can hang
        # indefinitely when the parent has already initialized CUDA.
        self.amp = bool(self.amp)  # as boolean
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.amp)
        if world_size > 1:
            self.model = nn.parallel.DistributedDataParallel(self.model, device_ids=[RANK])

        # Check imgsz
        gs = max(int(self.model.stride.max() if hasattr(self.model, "stride") else 32), 32)  # grid size (max stride)
        self.args.imgsz = check_imgsz(self.args.imgsz, stride=gs, floor=gs, max_dim=1)
        self.stride = gs  # for multiscale training

        # Batch size
        if self.batch_size == -1 and RANK == -1:  # single-GPU only, estimate best batch size
            self.args.batch = self.batch_size = check_train_batch_size(self.model, self.args.imgsz, self.amp)

        # Dataloaders
        batch_size = self.batch_size // max(world_size, 1)
        self.train_loader = self.get_dataloader(self.trainset, self.trainirset,batch_size=batch_size, rank=RANK, mode="train")
        if RANK in {-1, 0}:
            # Note: When training DOTA dataset, double batch size could get OOM on images with >2000 objects.
            requested_val_batch = int(getattr(self.args, "val_batch", 0))
            val_batch_size = requested_val_batch if requested_val_batch > 0 else batch_size * 2
            self.test_loader = self.get_dataloader(
                self.testset,
                self.testirset,
                batch_size=batch_size if self.args.task == "obb" else val_batch_size,
                rank=-1,
                mode="val",
            )
            self.validator = self.get_validator()
            metric_keys = self.validator.metrics.keys + self.label_loss_items(prefix="val")
            self.metrics = dict(zip(metric_keys, [0] * len(metric_keys)))
            self.ema = ModelEMA(self.model)
            if self.args.plots:
                self.plot_training_labels()

        # Optimizer
        self.accumulate = max(round(self.args.nbs / self.batch_size), 1)  # accumulate loss before optimizing
        weight_decay = self.args.weight_decay * self.batch_size * self.accumulate / self.args.nbs  # scale weight_decay
        iterations = math.ceil(len(self.train_loader.dataset) / max(self.batch_size, self.args.nbs)) * self.epochs
        self.optimizer = self.build_optimizer(
            model=self.model,
            name=self.args.optimizer,
            lr=self.args.lr0,
            momentum=self.args.momentum,
            decay=weight_decay,
            iterations=iterations,
        )
        # Scheduler
        self._setup_scheduler()
        self.stopper, self.stop = EarlyStopping(patience=self.args.patience), False
        self.resume_training(ckpt)
        self.scheduler.last_epoch = self.start_epoch - 1  # do not move
        self.run_callbacks("on_pretrain_routine_end")

    def _do_train(self, world_size=1):
        """Train completed, evaluate and plot if specified by arguments."""
        if world_size > 1:
            self._setup_ddp(world_size)
        self._setup_train(world_size)

        nb = len(self.train_loader)  # number of batches
        nw = max(round(self.args.warmup_epochs * nb), 100) if self.args.warmup_epochs > 0 else -1  # warmup iterations
        last_opt_step = -1
        self.epoch_time = None
        self.epoch_time_start = time.time()
        self.train_time_start = time.time()
        self.run_callbacks("on_train_start")
        LOGGER.info(
            f'Image sizes {self.args.imgsz} train, {self.args.imgsz} val\n'
            f'Using {self.train_loader.num_workers * (world_size or 1)} dataloader workers\n'
            f"Logging results to {colorstr('bold', self.save_dir)}\n"
            f'Starting training for ' + (f"{self.args.time} hours..." if self.args.time else f"{self.epochs} epochs...")
        )
        if self.args.close_mosaic:
            base_idx = (self.epochs - self.args.close_mosaic) * nb
            self.plot_idx.extend([base_idx, base_idx + 1, base_idx + 2])
        epoch = self.start_epoch
        while True:
            self.epoch = epoch
            self._reset_diagnostics()
            self.run_callbacks("on_train_epoch_start")
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")  # suppress 'Detected lr_scheduler.step() before optimizer.step()'
                self.scheduler.step()

            self.model.train()
            self._apply_pixel_taylor_schedule(epoch)
            self._apply_fusion_aux_schedule(epoch)
            self._apply_pretrained_anchor_schedule(epoch)
            adapter_only_epoch = bool(getattr(self.args, "adapter_only", False))
            wavelet_adapter_only_epoch = bool(getattr(self.args, "wavelet_adapter_only", False))
            pixel_taylor_adapter_only_epoch = bool(getattr(self.args, "pixel_taylor_adapter_only", False))
            if adapter_only_epoch or wavelet_adapter_only_epoch or pixel_taylor_adapter_only_epoch:
                requested_adapter_layers = getattr(self.args, "adapter_train_layers", [])
                if isinstance(requested_adapter_layers, int):
                    requested_adapter_layers = [requested_adapter_layers]
                trainable_context_names = [
                    f"model.{int(layer)}.derivative.global_context"
                    for layer in (requested_adapter_layers or [])
                ]
                requested_wavelet_layers = getattr(self.args, "wavelet_adapter_train_layers", [])
                if isinstance(requested_wavelet_layers, int):
                    requested_wavelet_layers = [requested_wavelet_layers]
                trainable_context_names.extend(
                    f"model.{int(layer)}.positive_wavelet_taylor_residual"
                    for layer in (requested_wavelet_layers or [])
                )
                if pixel_taylor_adapter_only_epoch:
                    trainable_context_names.extend(
                        ("pixel_fusion.derivatives", "pixel_fusion.target_gates")
                    )
                    if bool(getattr(self.args, "pixel_taylor_joint", False)):
                        trainable_context_names.append("pixel_fusion.concat_projection")
                        joint_scope = str(getattr(self.args, "pixel_taylor_joint_scope", "fusion_stem"))
                        if joint_scope == "fusion_stem":
                            trainable_context_names.append("model.0")
                        elif joint_scope == "fusion_stem_conv":
                            trainable_context_names.append("model.0.conv")
                trainable_context_names = tuple(trainable_context_names)
                # Frozen BN statistics and stochastic layers belong to the
                # clean anchor and must not drift during branch-only training.
                for module_name, module in self.model.named_modules():
                    inside_adapter = any(name in module_name for name in trainable_context_names)
                    frozen_stochastic = isinstance(module, (nn.modules.batchnorm._BatchNorm, nn.Dropout)) or (
                        module.__class__.__name__ == "DropPath"
                    )
                    if frozen_stochastic and not inside_adapter:
                        module.eval()
            if RANK != -1:
                self.train_loader.sampler.set_epoch(epoch)
            pbar = enumerate(self.train_loader)
            # Update dataloader attributes (optional)
            if epoch == (self.epochs - self.args.close_mosaic):
                self._close_dataloader_mosaic()
                self.train_loader.reset()

            if RANK in {-1, 0}:
                LOGGER.info(self.progress_string())
                pbar = TQDM(enumerate(self.train_loader), total=nb)
            self.tloss = None
            self.optimizer.zero_grad()
            for i, batch in pbar:
                self.run_callbacks("on_train_batch_start")
                # Warmup
                # img1= batch['img'][0][1:,...]
                # img2= batch['img'][0][:1,...]
                # img1=img1.permute(1, 2, 0)
                # img2=img2.permute(1,2,0)


                # import matplotlib.pyplot as plt

                # plt.imshow(img1)
                # plt.show()
                # plt.savefig('/home/mjy/ultralytics/images/'+str(i)+'rgb.jpg')
                #
                # plt.imshow(img)
                # plt.show()
                # plt.savefig('/home/mjy/ultralytics/images/'+str(i)+'ir.jpg')

                # import matplotlib.pyplot as plt
                # import cv2
                # import numpy as np
                # img2=np.array(img2)
                # plt.imshow(img1)
                # plt.savefig('/home/mjy/ultralytics/images/'+str(i)+'rgb.jpg')
                # plt.close()

                # cv2.imwrite('/home/mjy/ultralytics/images/'+str(i)+'ir.jpg', img2)


                # import cv2
                # for i in len(batch['img']):
                #     img1= batch['img'][i][:3,...]
                #     img2=batch['img'][i][3:,...]
                #     img1=img1.permute(1, 2, 0)
                #     gray_image_array = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
                #     img1 = gray_image_array[..., np.newaxis]

                #     img1=img1.permute(2, 0, 1)
                #     batch['img'][i]=np.concatenate((img1,img2),axis=1)


                ni = i + nb * epoch
                if ni <= nw:
                    xi = [0, nw]  # x interp
                    self.accumulate = max(1, int(np.interp(ni, xi, [1, self.args.nbs / self.batch_size]).round()))
                    for j, x in enumerate(self.optimizer.param_groups):
                        # Bias lr falls from 0.1 to lr0, all other lrs rise from 0.0 to lr0
                        x["lr"] = np.interp(
                            ni, xi, [self.args.warmup_bias_lr if j == 0 else 0.0, x["initial_lr"] * self.lf(epoch)]
                        )
                        if "momentum" in x:
                            x["momentum"] = np.interp(ni, xi, [self.args.warmup_momentum, self.args.momentum])

                # Forward. Diagnostic capture is sampled to avoid material
                # synchronization or memory overhead on every batch.
                diagnostic_interval = int(getattr(self.args, "diagnostic_interval", 50))
                capture_diagnostics = self._diagnostics_enabled() and i % diagnostic_interval == 0
                self._set_diagnostic_capture(capture_diagnostics)
                with torch.cuda.amp.autocast(self.amp):
                    batch = self.preprocess_batch(batch)
                    self.loss, self.loss_items = self.model(batch)
                    if RANK != -1:
                        self.loss *= world_size
                    self.tloss = (
                        (self.tloss * i + self.loss_items) / (i + 1) if self.tloss is not None else self.loss_items
                    )
                if capture_diagnostics:
                    self._collect_forward_diagnostics()
                    self._set_diagnostic_capture(False)

                # Backward
                torch.use_deterministic_algorithms(False)
                self.scaler.scale(self.loss).backward()

                # Optimize - https://pytorch.org/docs/master/notes/amp_examples.html
                if ni - last_opt_step >= self.accumulate:
                    self.optimizer_step()
                    last_opt_step = ni

                    # Timed stopping
                    if self.args.time:
                        self.stop = (time.time() - self.train_time_start) > (self.args.time * 3600)
                        if RANK != -1:  # if DDP training
                            broadcast_list = [self.stop if RANK == 0 else None]
                            dist.broadcast_object_list(broadcast_list, 0)  # broadcast 'stop' to all ranks
                            self.stop = broadcast_list[0]
                        if self.stop:  # training time exceeded
                            break

                # Log
                mem = f"{torch.cuda.memory_reserved() / 1E9 if torch.cuda.is_available() else 0:.3g}G"  # (GB)
                loss_len = self.tloss.shape[0] if len(self.tloss.shape) else 1
                losses = self.tloss if loss_len > 1 else torch.unsqueeze(self.tloss, 0)
                if RANK in {-1, 0}:
                    pbar.set_description(
                        ("%11s" * 2 + "%11.4g" * (2 + loss_len))
                        % (f"{epoch + 1}/{self.epochs}", mem, *losses, batch["cls"].shape[0], batch["img"].shape[-1])
                    )
                    self.run_callbacks("on_batch_end")
                    if self.args.plots and ni in self.plot_idx:
                        self.plot_training_samples(batch, ni)

                self.run_callbacks("on_train_batch_end")

            self.lr = {f"lr/pg{ir}": x["lr"] for ir, x in enumerate(self.optimizer.param_groups)}  # for loggers
            self.run_callbacks("on_train_epoch_end")
            if RANK in {-1, 0}:
                final_epoch = epoch + 1 >= self.epochs
                self.ema.update_attr(self.model, include=["yaml", "nc", "args", "names", "stride", "class_weights"])

                # Validation
                if self.args.val or final_epoch or self.stopper.possible_stop or self.stop:
                    self.metrics, self.fitness = self.validate()
                self.save_metrics(metrics={**self.label_loss_items(self.tloss), **self.metrics, **self.lr})
                self._save_module_diagnostics()
                self.stop |= self.stopper(epoch + 1, self.fitness) or final_epoch
                if self.args.time:
                    self.stop |= (time.time() - self.train_time_start) > (self.args.time * 3600)

                # Save model
                if self.args.save or final_epoch:
                    self.save_model()
                    self.run_callbacks("on_model_save")

            # Scheduler
            t = time.time()
            self.epoch_time = t - self.epoch_time_start
            self.epoch_time_start = t
            if self.args.time:
                mean_epoch_time = (t - self.train_time_start) / (epoch - self.start_epoch + 1)
                self.epochs = self.args.epochs = math.ceil(self.args.time * 3600 / mean_epoch_time)
                self._setup_scheduler()
                self.scheduler.last_epoch = self.epoch  # do not move
                self.stop |= epoch >= self.epochs  # stop if exceeded epochs
            self.run_callbacks("on_fit_epoch_end")
            torch.cuda.empty_cache()  # clear GPU memory at end of epoch, may help reduce CUDA out of memory errors

            # Early Stopping
            if RANK != -1:  # if DDP training
                broadcast_list = [self.stop if RANK == 0 else None]
                dist.broadcast_object_list(broadcast_list, 0)  # broadcast 'stop' to all ranks
                self.stop = broadcast_list[0]
            if self.stop:
                break  # must break all DDP ranks
            epoch += 1

        if RANK in {-1, 0}:
            # Do final val with best.pt
            LOGGER.info(
                f"\n{epoch - self.start_epoch + 1} epochs completed in "
                f"{(time.time() - self.train_time_start) / 3600:.3f} hours."
            )
            self.final_eval()
            if self.args.plots:
                self.plot_metrics()
            self.run_callbacks("on_train_end")
        torch.cuda.empty_cache()
        self.run_callbacks("teardown")

    def save_model(self):
        """Save model training checkpoints with additional metadata."""
        import io

        import pandas as pd  # scope for faster 'import ultralytics'

        # Serialize ckpt to a byte buffer once (faster than repeated torch.save() calls)
        buffer = io.BytesIO()
        torch.save(
            {
                "epoch": self.epoch,
                "best_fitness": self.best_fitness,
                "model": None,  # resume and final checkpoints derive from EMA
                "ema": deepcopy(self.ema.ema).half(),
                "updates": self.ema.updates,
                "optimizer": convert_optimizer_state_dict_to_fp16(deepcopy(self.optimizer.state_dict())),
                "train_args": vars(self.args),  # save as dict
                "train_metrics": {**self.metrics, **{"fitness": self.fitness}},
                "train_results": {k.strip(): v for k, v in pd.read_csv(self.csv).to_dict(orient="list").items()},
                "date": datetime.now().isoformat(),
                "version": __version__,
                "license": "AGPL-3.0 (https://ultralytics.com/license)",
                "docs": "https://docs.ultralytics.com",
            },
            buffer,
        )
        serialized_ckpt = buffer.getvalue()  # get the serialized content to save

        # Save checkpoints
        self.last.write_bytes(serialized_ckpt)  # save last.pt
        if self.best_fitness == self.fitness:
            self.best.write_bytes(serialized_ckpt)  # save best.pt
        if (self.save_period > 0) and (self.epoch > 0) and (self.epoch % self.save_period == 0):
            (self.wdir / f"epoch{self.epoch}.pt").write_bytes(serialized_ckpt)  # save epoch, i.e. 'epoch3.pt'

    def get_dataset(self):
        """
        Get train, val path from data dict if it exists.

        Returns None if data format is not recognized.
        """
        try:
            if self.args.task == "classify":
                data = check_cls_dataset(self.args.data)
            elif self.args.data.split(".")[-1] in {"yaml", "yml"} or self.args.task in {
                "detect",
                "segment",
                "pose",
                "obb",
            }:
                data = check_det_dataset(self.args.data)
                if "yaml_file" in data:
                    self.args.data = data["yaml_file"]  # for validating 'yolo train data=url.zip' usage
        except Exception as e:
            raise RuntimeError(emojis(f"Dataset '{clean_url(self.args.data)}' error {e}")) from e
        self.data = data
        return data["train"], data.get("val"),data["train_ir"],data.get("val_ir") or data.get("test")

    def setup_model(self):
        """Load/create/download model for any task."""
        if isinstance(self.model, torch.nn.Module):  # if model is loaded beforehand. No setup needed
            return

        model, weights = self.model, None
        ckpt = None
        if str(model).endswith(".pt"):
            weights, ckpt = attempt_load_one_weight(model)
            cfg = weights.yaml
        else:
            cfg = model
        self.model = self.get_model(cfg=cfg, weights=weights, verbose=RANK == -1)  # calls Model(cfg, weights)
        return ckpt

    def optimizer_step(self):
        """Perform a single step of the training optimizer with gradient clipping and EMA update."""
        self.scaler.unscale_(self.optimizer)  # unscale gradients
        total_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=10.0)  # clip gradients
        self._collect_gradient_diagnostics(total_norm)
        self.scaler.step(self.optimizer)
        self.scaler.update()
        self.optimizer.zero_grad()
        if self.ema:
            self.ema.update(self.model)

    def preprocess_batch(self, batch):
        """Allows custom preprocessing model inputs and ground truths depending on task type."""
        return batch

    def validate(self):
        """
        Runs validation on test set using self.validator.

        The returned dict is expected to contain "fitness" key.
        """
        metrics = self.validator(self)
        self._save_class_metrics()
        fitness = metrics.pop("fitness", -self.loss.detach().cpu().numpy())  # use loss as fitness measure if not found
        if not self.best_fitness or self.best_fitness < fitness:
            self.best_fitness = fitness
        return metrics, fitness

    def get_model(self, cfg=None, weights=None, verbose=True):
        """Get model and raise NotImplementedError for loading cfg files."""
        raise NotImplementedError("This task trainer doesn't support loading cfg files")

    def get_validator(self):
        """Returns a NotImplementedError when the get_validator function is called."""
        raise NotImplementedError("get_validator function not implemented in trainer")

    def get_dataloader(self, dataset_path, batch_size=16, rank=0, mode="train"):
        """Returns dataloader derived from torch.data.Dataloader."""
        raise NotImplementedError("get_dataloader function not implemented in trainer")

    def build_dataset(self, img_path, mode="train", batch=None):
        """Build dataset."""
        raise NotImplementedError("build_dataset function not implemented in trainer")

    def label_loss_items(self, loss_items=None, prefix="train"):
        """
        Returns a loss dict with labelled training loss items tensor.

        Note:
            This is not needed for classification but necessary for segmentation & detection
        """
        return {"loss": loss_items} if loss_items is not None else ["loss"]

    def set_model_attributes(self):
        """To set or update model parameters before training."""
        self.model.names = self.data["names"]

    def build_targets(self, preds, targets):
        """Builds target tensors for training YOLO model."""
        pass

    def progress_string(self):
        """Returns a string describing training progress."""
        return ""

    # TODO: may need to put these following functions into callback
    def plot_training_samples(self, batch, ni):
        """Plots training samples during YOLO training."""
        pass

    def plot_training_labels(self):
        """Plots training labels for YOLO model."""
        pass

    def save_metrics(self, metrics):
        """Saves training metrics to a CSV file."""
        keys, vals = list(metrics.keys()), list(metrics.values())
        n = len(metrics) + 1  # number of cols
        s = "" if self.csv.exists() else (("%23s," * n % tuple(["epoch"] + keys)).rstrip(",") + "\n")  # header
        with open(self.csv, "a") as f:
            f.write(s + ("%23.5g," * n % tuple([self.epoch + 1] + vals)).rstrip(",") + "\n")

    def plot_metrics(self):
        """Plot and display metrics visually."""
        pass

    def on_plot(self, name, data=None):
        """Registers plots (e.g. to be consumed in callbacks)"""
        path = Path(name)
        self.plots[path] = {"data": data, "timestamp": time.time()}

    def final_eval(self):
        """Performs final evaluation and validation for object detection YOLO model."""
        for f in self.last, self.best:
            if f.exists():
                strip_optimizer(f)  # strip optimizers
                if f is self.best:
                    LOGGER.info(f"\nValidating {f}...")
                    self.validator.args.plots = self.args.plots
                    self.metrics = self.validator(model=f)
                    self.metrics.pop("fitness", None)
                    self.run_callbacks("on_fit_epoch_end")

    def check_resume(self, overrides):
        """Check if resume checkpoint exists and update arguments accordingly."""
        resume = self.args.resume
        if resume:
            try:
                exists = isinstance(resume, (str, Path)) and Path(resume).exists()
                last = Path(check_file(resume) if exists else get_latest_run())

                # Check that resume data YAML exists, otherwise strip to force re-download of dataset
                ckpt_args = attempt_load_weights(last).args
                if not Path(ckpt_args["data"]).exists():
                    ckpt_args["data"] = self.args.data

                resume = True
                self.args = get_cfg(ckpt_args)
                self.args.model = self.args.resume = str(last)  # reinstate model
                for k in "imgsz", "batch", "device":  # allow arg updates to reduce memory or update device on resume
                    if k in overrides:
                        setattr(self.args, k, overrides[k])

            except Exception as e:
                raise FileNotFoundError(
                    "Resume checkpoint not found. Please pass a valid checkpoint to resume from, "
                    "i.e. 'yolo train resume model=path/to/last.pt'"
                ) from e
        self.resume = resume

    def resume_training(self, ckpt):

        #ckpt = torch.load('runs/obb/train/weights/last.pt')

        """Resume YOLO training from given epoch and best fitness."""
        if ckpt is None or not self.resume:
            return
        best_fitness = 0.0
        start_epoch = ckpt.get("epoch", -1) + 1
        if ckpt.get("optimizer", None) is not None:
            self.optimizer.load_state_dict(ckpt["optimizer"])  # optimizer
            best_fitness = ckpt["best_fitness"]
        if self.ema and ckpt.get("ema"):
            self.ema.ema.load_state_dict(ckpt["ema"].float().state_dict())  # EMA
            self.ema.updates = ckpt["updates"]
        assert start_epoch > 0, (
            f"{self.args.model} training to {self.epochs} epochs is finished, nothing to resume.\n"
            f"Start a new training without resuming, i.e. 'yolo train model={self.args.model}'"
        )
        LOGGER.info(f"Resuming training {self.args.model} from epoch {start_epoch + 1} to {self.epochs} total epochs")
        if self.epochs < start_epoch:
            LOGGER.info(
                f"{self.model} has been trained for {ckpt['epoch']} epochs. Fine-tuning for {self.epochs} more epochs."
            )
            self.epochs += ckpt["epoch"]  # finetune additional epochs
        self.best_fitness = best_fitness
        self.start_epoch = start_epoch
        if start_epoch > (self.epochs - self.args.close_mosaic):
            self._close_dataloader_mosaic()

    def _close_dataloader_mosaic(self):
        """Update dataloaders to stop using mosaic augmentation."""
        if hasattr(self.train_loader.dataset, "mosaic"):
            self.train_loader.dataset.mosaic = False
        if hasattr(self.train_loader.dataset, "close_mosaic"):
            LOGGER.info("Closing dataloader mosaic")
            self.train_loader.dataset.close_mosaic(hyp=self.args)

    def build_optimizer(self, model, name="auto", lr=0.001, momentum=0.9, decay=1e-5, iterations=1e5):
        """
        Constructs an optimizer for the given model, based on the specified optimizer name, learning rate, momentum,
        weight decay, and number of iterations.

        Args:
            model (torch.nn.Module): The model for which to build an optimizer.
            name (str, optional): The name of the optimizer to use. If 'auto', the optimizer is selected
                based on the number of iterations. Default: 'auto'.
            lr (float, optional): The learning rate for the optimizer. Default: 0.001.
            momentum (float, optional): The momentum factor for the optimizer. Default: 0.9.
            decay (float, optional): The weight decay for the optimizer. Default: 1e-5.
            iterations (float, optional): The number of iterations, which determines the optimizer if
                name is 'auto'. Default: 1e5.

        Returns:
            (torch.optim.Optimizer): The constructed optimizer.
        """

        g = [], [], []  # optimizer parameter groups
        adapter_gates = []
        architecture_decay = []
        architecture_no_decay = []
        positive_wavelet_decay = []
        positive_wavelet_no_decay = []
        positive_wavelet_gates = []
        pixel_joint_fusion_decay = []
        pixel_joint_fusion_no_decay = []
        pixel_joint_stem_decay = []
        pixel_joint_stem_no_decay = []
        pixel_target_gate_decay = []
        pixel_target_gate_no_decay = []
        bn = tuple(v for k, v in nn.__dict__.items() if "Norm" in k)  # normalization layers, i.e. BatchNorm2d()
        adapter_gate_lr_mult = float(getattr(self.args, "adapter_gate_lr_mult", 1.0))
        if adapter_gate_lr_mult <= 0.0:
            raise ValueError("adapter_gate_lr_mult must be positive")
        wavelet_branch_lr_mult = float(getattr(self.args, "wavelet_branch_lr_mult", 0.5))
        wavelet_gate_lr_mult = float(getattr(self.args, "wavelet_gate_lr_mult", 0.1))
        if wavelet_branch_lr_mult <= 0.0 or wavelet_gate_lr_mult <= 0.0:
            raise ValueError("Wavelet learning-rate multipliers must be positive")
        pixel_taylor_joint = bool(getattr(self.args, "pixel_taylor_joint", False))
        pixel_taylor_joint_scope = str(getattr(self.args, "pixel_taylor_joint_scope", "fusion_stem"))
        pixel_fusion_lr_mult = float(getattr(self.args, "pixel_taylor_fusion_lr_mult", 0.25))
        pixel_stem_lr_mult = float(getattr(self.args, "pixel_taylor_stem_lr_mult", 0.1))
        pixel_target_gate_lr_mult = float(getattr(self.args, "pixel_taylor_target_gate_lr_mult", 1.0))
        if pixel_fusion_lr_mult <= 0.0 or pixel_stem_lr_mult <= 0.0 or pixel_target_gate_lr_mult <= 0.0:
            raise ValueError("Pixel Taylor joint learning-rate multipliers must be positive")
        if name == "auto":
            LOGGER.info(
                f"{colorstr('optimizer:')} 'optimizer=auto' found, "
                f"ignoring 'lr0={self.args.lr0}' and 'momentum={self.args.momentum}' and "
                f"determining best 'optimizer', 'lr0' and 'momentum' automatically... "
            )
            nc = getattr(model, "nc", 10)  # number of classes
            lr_fit = round(0.002 * 5 / (4 + nc), 6)  # lr0 fit equation to 6 decimal places
            name, lr, momentum = ("SGD", 0.01, 0.9) if iterations > 10000 else ("AdamW", lr_fit, 0.9)
            self.args.warmup_bias_lr = 0.0  # no higher than 0.01 for Adam

        for module_name, module in model.named_modules():
            for param_name, param in module.named_parameters(recurse=False):
                if not param.requires_grad:
                    continue
                fullname = f"{module_name}.{param_name}" if module_name else param_name
                pixel_fusion_branch = fullname.startswith(
                    ("pixel_fusion.", "pixel_taylor_head_injector.")
                )
                architecture_gate = fullname.endswith(
                    (
                        "quality_residual_gates",
                        "quality_mixer_logits",
                        "derivative_residual_gate",
                        "rgb_detail_filter.filter_gate",
                    )
                )
                architecture_branch = pixel_fusion_branch or architecture_gate or any(
                    token in fullname
                    for token in (
                        "quality_predictor.",
                        "derivative_residual.",
                        "rgb_detail_filter.",
                    )
                )
                positive_wavelet_branch = "positive_wavelet_taylor_residual." in fullname
                positive_wavelet_gate = positive_wavelet_branch and fullname.endswith(".output_gate")
                pixel_target_gate = fullname.startswith("pixel_fusion.target_gates.")
                pixel_joint_fusion = pixel_taylor_joint and fullname.startswith("pixel_fusion.concat_projection.")
                pixel_joint_stem = (
                    pixel_taylor_joint
                    and pixel_taylor_joint_scope in {"fusion_stem", "fusion_stem_conv"}
                    and fullname.startswith("model.0.")
                )
                if pixel_target_gate:
                    target_group = (
                        pixel_target_gate_no_decay
                        if "bias" in param_name
                        or isinstance(module, bn)
                        or bool(getattr(param, "_no_weight_decay", False))
                        else pixel_target_gate_decay
                    )
                    target_group.append(param)
                elif pixel_joint_fusion:
                    target_group = (
                        pixel_joint_fusion_no_decay
                        if "bias" in param_name or isinstance(module, bn)
                        else pixel_joint_fusion_decay
                    )
                    target_group.append(param)
                elif pixel_joint_stem:
                    target_group = (
                        pixel_joint_stem_no_decay
                        if "bias" in param_name or isinstance(module, bn)
                        else pixel_joint_stem_decay
                    )
                    target_group.append(param)
                elif positive_wavelet_gate:
                    positive_wavelet_gates.append(param)
                elif positive_wavelet_branch:
                    # Keep every safe-branch bias out of optimizer group 0:
                    # group 0 receives warmup_bias_lr=0.1 in this codebase.
                    if "bias" in param_name or "gate" in param_name or isinstance(module, bn):
                        positive_wavelet_no_decay.append(param)
                    else:
                        positive_wavelet_decay.append(param)
                elif fullname.endswith(".derivative.global_gate"):
                    adapter_gates.append(param)
                elif architecture_branch:
                    # Every parameter in a newly introduced P2/P3 branch must
                    # stay out of optimizer group 0. Group 0 receives
                    # warmup_bias_lr (often 0.1), which saturated both the
                    # quality predictor and filter-mask biases in 2-3 epochs.
                    if architecture_gate or "bias" in param_name or (
                        pixel_fusion_branch and "gate" in param_name
                    ) or isinstance(module, bn):
                        architecture_no_decay.append(param)
                    else:
                        architecture_decay.append(param)
                elif "bias" in fullname:  # bias (no decay)
                    g[2].append(param)
                elif isinstance(module, bn):  # weight (no decay)
                    g[1].append(param)
                else:  # weight (with decay)
                    g[0].append(param)

        group_specs = [
            (g[2], lr, 0.0),
            (g[0], lr, decay),
            (g[1], lr, 0.0),
            (architecture_decay, lr, decay),
            (architecture_no_decay, lr, 0.0),
            (adapter_gates, lr * adapter_gate_lr_mult, 0.0),
            (positive_wavelet_decay, lr * wavelet_branch_lr_mult, decay),
            (positive_wavelet_no_decay, lr * wavelet_branch_lr_mult, 0.0),
            (positive_wavelet_gates, lr * wavelet_gate_lr_mult, 0.0),
            (pixel_joint_fusion_decay, lr * pixel_fusion_lr_mult, decay),
            (pixel_joint_fusion_no_decay, lr * pixel_fusion_lr_mult, 0.0),
            (pixel_joint_stem_decay, lr * pixel_stem_lr_mult, decay),
            (pixel_joint_stem_no_decay, lr * pixel_stem_lr_mult, 0.0),
            (pixel_target_gate_decay, lr * pixel_target_gate_lr_mult, decay),
            (pixel_target_gate_no_decay, lr * pixel_target_gate_lr_mult, 0.0),
        ]
        active_groups = [spec for spec in group_specs if spec[0]]
        if not active_groups:
            raise ValueError("No trainable parameters are available to build the optimizer")
        seed_params, seed_lr, seed_decay = active_groups[0]
        if name in {"Adam", "Adamax", "AdamW", "NAdam", "RAdam"}:
            optimizer = getattr(optim, name, optim.Adam)(
                seed_params, lr=seed_lr, betas=(momentum, 0.999), weight_decay=seed_decay
            )
        elif name == "RMSProp":
            optimizer = optim.RMSprop(seed_params, lr=seed_lr, momentum=momentum, weight_decay=seed_decay)
        elif name == "SGD":
            optimizer = optim.SGD(
                seed_params, lr=seed_lr, momentum=momentum, nesterov=True, weight_decay=seed_decay
            )
        else:
            raise NotImplementedError(
                f"Optimizer '{name}' not found in list of available optimizers "
                f"[Adam, AdamW, NAdam, RAdam, RMSProp, SGD, auto]."
                "To request support for addition optimizers please visit https://github.com/ultralytics/ultralytics."
            )

        for params, group_lr, group_decay in active_groups[1:]:
            optimizer.add_param_group(
                {"params": params, "lr": group_lr, "weight_decay": group_decay}
            )
        LOGGER.info(
            f"{colorstr('optimizer:')} {type(optimizer).__name__}(lr={lr}, momentum={momentum}) with parameter groups "
            f'{len(g[1])} weight(decay=0.0), {len(g[0])} weight(decay={decay}), '
            f'{len(g[2])} bias(decay=0.0), {len(architecture_decay)} architecture weight(decay={decay}), '
            f'{len(architecture_no_decay)} architecture bias/norm/gate(decay=0.0), '
            f'{len(adapter_gates)} adapter gate(decay=0.0), '
            f'{len(positive_wavelet_decay)} positive-wavelet weight(lr={wavelet_branch_lr_mult}x), '
            f'{len(positive_wavelet_no_decay)} positive-wavelet bias/norm(lr={wavelet_branch_lr_mult}x), '
            f'{len(positive_wavelet_gates)} positive-wavelet gate(lr={wavelet_gate_lr_mult}x), '
            f'{len(pixel_joint_fusion_decay) + len(pixel_joint_fusion_no_decay)} pixel-fusion joint(lr={pixel_fusion_lr_mult}x), '
            f'{len(pixel_joint_stem_decay) + len(pixel_joint_stem_no_decay)} pixel-stem joint(lr={pixel_stem_lr_mult}x), '
            f'{len(pixel_target_gate_decay) + len(pixel_target_gate_no_decay)} pixel-target-gate(lr={pixel_target_gate_lr_mult}x)'
        )
        return optimizer
