"""Task-driven pixel-level RGB/IR fusion front ends.

The module deliberately lives outside the YOLO layer list: it converts the
aligned six-channel loader output into a three-channel task image before a
standard YOLO backbone.  This keeps the detector topology and official weight
mapping unchanged while making the fusion hypothesis independently testable.
"""

import math
import contextlib

import torch
import torch.nn as nn
import torch.nn.functional as F

from ultralytics.nn.modules.vmamba2_simplified import HAS_MAMBA_SSM, VMAMBA2Block
from ultralytics.nn.modules.wavelet import (
    create_wavelet_filter,
    inverse_wavelet_transform,
    wavelet_transform,
)


def _gray(image):
    """Return a differentiable single-channel intensity image."""
    return image.mean(dim=1, keepdim=True)


def _gradient_magnitude(image):
    """Cheap, shape-preserving L1 spatial gradient magnitude."""
    dx = F.pad(image[..., 1:] - image[..., :-1], (0, 1, 0, 0))
    dy = F.pad(image[..., 1:, :] - image[..., :-1, :], (0, 0, 0, 1))
    return dx.abs() + dy.abs()


def _rms(value):
    return value.detach().float().square().mean().sqrt()


class PixelModalityQualityEncoder(nn.Module):
    """Predict positive global RGB/IR mixture weights from image statistics."""

    def __init__(self, hidden=16):
        super().__init__()
        self.predictor = nn.Sequential(
            nn.Linear(8, hidden),
            nn.SiLU(inplace=True),
            nn.Linear(hidden, 2),
        )
        # Equal RGB/IR weighting is deterministic and content-neutral at epoch 0.
        nn.init.zeros_(self.predictor[-1].weight)
        nn.init.zeros_(self.predictor[-1].bias)

    @staticmethod
    def _statistics(image):
        gray = _gray(image)
        mean = gray.mean(dim=(2, 3))
        std = gray.std(dim=(2, 3), unbiased=False)
        gradient = _gradient_magnitude(gray)
        gradient_mean = gradient.mean(dim=(2, 3))
        gradient_std = gradient.std(dim=(2, 3), unbiased=False)
        return torch.cat((mean, std, gradient_mean, gradient_std), dim=1)

    def forward(self, rgb, ir):
        logits = self.predictor(torch.cat((self._statistics(rgb), self._statistics(ir)), dim=1))
        return torch.softmax(logits, dim=1)


class _LightPixelBlock(nn.Module):
    """Memory-conscious full-resolution residual block with a zero output."""

    def __init__(self, in_channels, hidden=8, out_channels=3):
        super().__init__()
        self.input_projection = nn.Conv2d(in_channels, hidden, 1, bias=True)
        self.depthwise = nn.Conv2d(hidden, hidden, 3, padding=1, groups=hidden, bias=True)
        self.output_projection = nn.Conv2d(hidden, out_channels, 1, bias=True)
        self.activation = nn.SiLU(inplace=True)
        nn.init.zeros_(self.output_projection.weight)
        nn.init.zeros_(self.output_projection.bias)

    def forward(self, value):
        value = self.activation(self.input_projection(value))
        value = self.activation(self.depthwise(value))
        return self.output_projection(value)


class _PixelContextGate(nn.Module):
    """Predict a spatial target gate from local and dilated pixel context."""

    def __init__(self, in_channels=9, hidden=8):
        super().__init__()
        self.input_projection = nn.Conv2d(in_channels, hidden, 1, bias=True)
        self.local = nn.Conv2d(hidden, hidden, 3, padding=1, groups=hidden, bias=True)
        self.dilated = nn.Conv2d(hidden, hidden, 3, padding=2, dilation=2, groups=hidden, bias=True)
        self.output_projection = nn.Conv2d(hidden, 1, 1, bias=True)
        self.activation = nn.SiLU(inplace=True)
        # A zero logit gives a neutral 0.5 spatial selector. The Taylor
        # derivative itself is zero-initialized, so epoch-0 remains an exact
        # E03 no-op despite the nonzero selector.
        nn.init.zeros_(self.output_projection.weight)
        nn.init.zeros_(self.output_projection.bias)

    def forward(self, value):
        value = self.activation(self.input_projection(value))
        value = self.activation(self.local(value))
        value = self.activation(self.dilated(value))
        return self.output_projection(value)


class _WaveletLowContextBlock(nn.Module):
    """Fuse RGB/IR LL bands with local and dilated context and a zero residual."""

    def __init__(self, in_channels=9, hidden=8, out_channels=3):
        super().__init__()
        self.input_projection = nn.Conv2d(in_channels, hidden, 1, bias=True)
        self.local = nn.Conv2d(hidden, hidden, 3, padding=1, groups=hidden, bias=True)
        self.dilated = nn.Conv2d(hidden, hidden, 3, padding=2, dilation=2, groups=hidden, bias=True)
        self.output_projection = nn.Conv2d(hidden, out_channels, 1, bias=True)
        self.activation = nn.SiLU(inplace=True)
        nn.init.zeros_(self.output_projection.weight)
        nn.init.zeros_(self.output_projection.bias)

    def forward(self, value):
        value = self.activation(self.input_projection(value))
        value = self.activation(self.local(value))
        value = self.activation(self.dilated(value))
        return self.output_projection(value)


class _ExplicitTaylorContextBlock(nn.Module):
    """Predict a Taylor coefficient from local pixels and pooled global context."""

    def __init__(self, in_channels=9, hidden=8, out_channels=3, backend="cnn", downsample=8):
        super().__init__()
        if backend not in {"cnn", "vmamba"}:
            raise ValueError(f"Unknown explicit Taylor context backend {backend!r}")
        if backend == "vmamba" and not HAS_MAMBA_SSM:
            raise RuntimeError("Explicit Taylor VMamba context requires mamba_ssm")
        self.backend = backend
        self.downsample = int(downsample)
        self.local = _LightPixelBlock(in_channels, hidden=hidden, out_channels=out_channels)

        global_hidden = max(16, hidden * 2)
        self.global_input = nn.Conv2d(in_channels, global_hidden, 1, bias=True)
        if backend == "cnn":
            self.global_processor = nn.Sequential(
                nn.Conv2d(
                    global_hidden, global_hidden, 3, padding=1,
                    groups=global_hidden, bias=True,
                ),
                nn.SiLU(inplace=True),
                nn.Conv2d(global_hidden, global_hidden, 1, bias=True),
                nn.SiLU(inplace=True),
                nn.Conv2d(
                    global_hidden, global_hidden, 3, padding=2, dilation=2,
                    groups=global_hidden, bias=True,
                ),
            )
        else:
            self.global_processor = VMAMBA2Block(
                hidden_dim=global_hidden,
                d_state=8,
                expand=2,
                headdim=16,
                ngroups=1,
                chunk_size=128,
                use_cpe=True,
                drop_path=0.0,
            )
        self.global_output = nn.Conv2d(global_hidden, out_channels, 1, bias=True)
        self.activation = nn.SiLU(inplace=True)
        nn.init.zeros_(self.global_output.weight)
        nn.init.zeros_(self.global_output.bias)
        self.capture_diagnostics = False
        self._last_diagnostics = {}

    def forward(self, value):
        local = self.local(value)
        pooled = F.avg_pool2d(value, self.downsample, self.downsample)
        global_value = self.activation(self.global_input(pooled))
        if self.backend == "vmamba":
            global_value = self.global_processor(
                global_value.permute(0, 2, 3, 1).contiguous()
            ).permute(0, 3, 1, 2).contiguous()
        else:
            global_value = self.global_processor(global_value)
        global_value = self.global_output(global_value)
        global_value = F.interpolate(
            global_value, size=value.shape[-2:], mode="bilinear", align_corners=False
        )
        if self.capture_diagnostics:
            combined_rms = _rms(local + global_value).clamp_min(1e-12)
            self._last_diagnostics = {
                "local_coefficient_rms": _rms(local),
                "global_coefficient_rms": _rms(global_value),
                "global_fraction": _rms(global_value) / combined_rms,
            }
        return local + global_value

    def diagnostic_values(self):
        values = self._last_diagnostics
        self._last_diagnostics = {}
        return values


class HeadTaylorDetailInjector(nn.Module):
    """Inject a pixel Taylor detail map into the highest-resolution detection feature."""

    def __init__(self, out_channels, hidden=16, gate_max=0.1):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, hidden, 3, padding=1, bias=False),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden, out_channels, 1, bias=False),
        )
        self.gate_logit = nn.Parameter(torch.tensor(math.log(0.2 / 0.8)))
        self.gate_max = float(gate_max)
        self.capture_diagnostics = False
        self._last_diagnostics = {}

    def forward(self, features, detail):
        if not isinstance(features, list) or not features:
            raise TypeError("Head Taylor injector expects a non-empty detection feature list")
        target = features[0]
        detail = F.interpolate(detail, size=target.shape[-2:], mode="bilinear", align_corners=False)
        encoded = self.encoder(detail)
        gate = self.gate_max * torch.sigmoid(self.gate_logit)
        injected = gate * encoded
        output = list(features)
        output[0] = target + injected
        if self.capture_diagnostics:
            eps = torch.finfo(target.dtype).eps
            self._last_diagnostics = {
                "gate": gate.detach(),
                "detail_rms": _rms(detail),
                "injected_to_p3_rms": _rms(injected) / _rms(target).clamp_min(eps),
            }
        return output

    def diagnostic_values(self):
        values = self._last_diagnostics
        self._last_diagnostics = {}
        return values


class _PixelGlobalGate(nn.Module):
    """Combine full-resolution local evidence with a 1/8 global gate branch."""

    def __init__(self, in_channels=9, local_hidden=8, global_hidden=32, downsample=8, backend="cnn"):
        super().__init__()
        if backend not in {"cnn", "vmamba"}:
            raise ValueError(f"Unknown global gate backend {backend!r}")
        if downsample < 2:
            raise ValueError("Global gate downsample must be at least 2")
        if backend == "vmamba" and not HAS_MAMBA_SSM:
            raise RuntimeError("VMamba global gate requires mamba_ssm; refusing to use the identity fallback")
        self.backend = backend
        self.downsample = int(downsample)

        self.local_input = nn.Conv2d(in_channels, local_hidden, 1, bias=True)
        self.local_context = nn.Conv2d(
            local_hidden, local_hidden, 3, padding=1, groups=local_hidden, bias=True
        )
        self.local_output = nn.Conv2d(local_hidden, 1, 1, bias=True)

        self.global_input = nn.Conv2d(in_channels, global_hidden, 1, bias=True)
        if backend == "cnn":
            self.global_processor = nn.Sequential(
                nn.Conv2d(global_hidden, global_hidden, 3, padding=1, groups=global_hidden, bias=True),
                nn.SiLU(inplace=True),
                nn.Conv2d(global_hidden, global_hidden, 1, bias=True),
                nn.SiLU(inplace=True),
                nn.Conv2d(
                    global_hidden,
                    global_hidden,
                    3,
                    padding=2,
                    dilation=2,
                    groups=global_hidden,
                    bias=True,
                ),
            )
        else:
            self.global_processor = VMAMBA2Block(
                hidden_dim=global_hidden,
                d_state=8,
                expand=2,
                headdim=16,
                ngroups=1,
                chunk_size=128,
                use_cpe=True,
                drop_path=0.0,
            )
        self.global_output = nn.Conv2d(global_hidden, 1, 1, bias=True)
        self.activation = nn.SiLU(inplace=True)

        # Unlike the Taylor derivative, the selector need not be a constant at
        # epoch 0: its derivative term is exactly zero. Small random logits
        # break spatial symmetry while preserving exact E03 outputs.
        nn.init.normal_(self.local_output.weight, mean=0.0, std=1e-3)
        nn.init.zeros_(self.local_output.bias)
        nn.init.normal_(self.global_output.weight, mean=0.0, std=1e-3)
        nn.init.zeros_(self.global_output.bias)

    def forward(self, value):
        local = self.activation(self.local_input(value))
        local = self.activation(self.local_context(local))
        local_logit = self.local_output(local)

        pooled = F.avg_pool2d(value, self.downsample, self.downsample)
        global_features = self.activation(self.global_input(pooled))
        if self.backend == "vmamba":
            global_features = self.global_processor(global_features.permute(0, 2, 3, 1).contiguous())
            global_features = global_features.permute(0, 3, 1, 2).contiguous()
        else:
            global_features = self.global_processor(global_features)
        global_logit = self.global_output(global_features)
        global_logit = F.interpolate(global_logit, size=value.shape[-2:], mode="bilinear", align_corners=False)
        return local_logit + global_logit


class PixelTaylorFusion(nn.Module):
    """Fuse aligned RGB/IR pixels before a conventional three-channel detector.

    Modes form controlled ablations:
      - concat: initialized 1x1 six-to-three channel projection (E03)
      - residual: RGB identity plus a zero-initialized shallow residual (E04)
      - quality: positive quality-aware zero-order mixture only (E05/T00)
      - taylor: quality-aware base plus recursively learned ordered corrections
      - concat_taylor: E03 learned 6-to-3 base plus the same corrections
      - wavelet_context_taylor: contextual LL fusion plus Taylor corrections predicted
        directly for the three Haar high-frequency subbands
      - explicit_taylor: E03 base plus explicit learned coefficients times delta powers
      - head_taylor: E03 base image with a high-frequency Taylor detail side output

    Taylor terms are learned approximations used as an architectural blueprint;
    no claim is made that the blocks calculate analytic image derivatives.
    """

    MODES = {
        "concat", "residual", "quality", "taylor", "concat_taylor",
        "wavelet_context_taylor", "explicit_taylor", "head_taylor",
    }

    def __init__(
        self,
        mode="taylor",
        order=2,
        hidden=8,
        gate_max=0.5,
        correction_limit=0.0,
        taylor_frequency_mode="full",
        explicit_delta_tau=0.5,
        explicit_context="local",
        explicit_context_downsample=8,
        residual_limit=0.25,
        target_gate=False,
        target_gate_context=False,
        target_gate_normalize=False,
        target_gate_global=None,
        target_gate_downsample=8,
        target_gate_hidden=32,
        low_loss_weight=0.0,
        edge_loss_weight=0.0,
        background_loss_weight=0.0,
        gate_loss_weight=0.0,
        aux_downsample=4,
    ):
        super().__init__()
        if mode not in self.MODES:
            raise ValueError(f"Unknown pixel fusion mode {mode!r}; choose from {sorted(self.MODES)}")
        if order not in {0, 1, 2, 3}:
            raise ValueError(f"Pixel Taylor order must be 0, 1, 2, or 3, got {order}")
        taylor_modes = {
            "taylor", "concat_taylor", "wavelet_context_taylor", "explicit_taylor", "head_taylor"
        }
        if mode not in taylor_modes and order != 0:
            raise ValueError(f"Pixel fusion mode {mode!r} requires order=0, got {order}")
        if not 0.0 < gate_max <= 1.0:
            raise ValueError(f"gate_max must be in (0, 1], got {gate_max}")
        if not 0.0 <= correction_limit <= 1.0:
            raise ValueError(f"correction_limit must be in [0, 1], got {correction_limit}")
        if mode not in taylor_modes and correction_limit != 0.0:
            raise ValueError("correction_limit is only valid for Taylor fusion modes")
        frequency_modes = {"full", "high_conditioned", "high_projected"}
        if taylor_frequency_mode not in frequency_modes:
            raise ValueError(
                f"Unknown Taylor frequency mode {taylor_frequency_mode!r}; "
                f"choose from {sorted(frequency_modes)}"
            )
        if mode not in taylor_modes and taylor_frequency_mode != "full":
            raise ValueError("A non-full Taylor frequency mode requires a Taylor fusion mode")
        if mode == "wavelet_context_taylor" and taylor_frequency_mode != "full":
            raise ValueError("wavelet_context_taylor defines its own subband path and requires frequency mode 'full'")
        if mode == "explicit_taylor" and taylor_frequency_mode != "full":
            raise ValueError("explicit_taylor currently requires frequency mode 'full'")
        if mode == "head_taylor" and taylor_frequency_mode != "high_projected":
            raise ValueError("head_taylor requires frequency mode 'high_projected'")
        if explicit_delta_tau <= 0.0:
            raise ValueError("explicit_delta_tau must be positive")
        if explicit_context not in {"local", "pooled_cnn", "pooled_vmamba"}:
            raise ValueError(
                "explicit_context must be 'local', 'pooled_cnn' or 'pooled_vmamba'"
            )
        if mode != "explicit_taylor" and explicit_context != "local":
            raise ValueError("A non-local explicit_context requires explicit_taylor mode")
        if explicit_context_downsample < 2:
            raise ValueError("explicit_context_downsample must be at least 2")
        if not 0.0 < residual_limit <= 1.0:
            raise ValueError(f"residual_limit must be in (0, 1], got {residual_limit}")
        if aux_downsample < 1:
            raise ValueError(f"aux_downsample must be >= 1, got {aux_downsample}")
        if target_gate_context and not target_gate:
            raise ValueError("target_gate_context=True requires target_gate=True")
        if target_gate_normalize and not target_gate:
            raise ValueError("target_gate_normalize=True requires target_gate=True")
        if target_gate_global not in {None, "cnn", "vmamba"}:
            raise ValueError("target_gate_global must be null, 'cnn' or 'vmamba'")
        if target_gate_global is not None and not target_gate:
            raise ValueError("target_gate_global requires target_gate=True")
        if target_gate_global is not None and target_gate_context:
            raise ValueError("target_gate_global and target_gate_context are mutually exclusive")
        if target_gate_hidden <= 0 or target_gate_downsample < 2:
            raise ValueError("target_gate_hidden must be positive and target_gate_downsample must be >= 2")
        if mode == "wavelet_context_taylor" and target_gate:
            raise ValueError("wavelet_context_taylor does not use the legacy full-resolution target gate")
        for name, value in {
            "low_loss_weight": low_loss_weight,
            "edge_loss_weight": edge_loss_weight,
            "background_loss_weight": background_loss_weight,
            "gate_loss_weight": gate_loss_weight,
        }.items():
            if value < 0.0:
                raise ValueError(f"{name} must be non-negative, got {value}")

        self.mode = mode
        self.order = int(order)
        self.gate_max = float(gate_max)
        # A zero limit preserves the original unbounded Taylor correction.
        # A positive limit bounds each pixel/channel correction while keeping
        # the local slope at zero equal to one: L * tanh(raw / L).
        self.correction_limit = float(correction_limit)
        self.taylor_frequency_mode = taylor_frequency_mode
        self.explicit_delta_tau = float(explicit_delta_tau)
        self.explicit_context = explicit_context
        self.explicit_context_downsample = int(explicit_context_downsample)
        self.residual_limit = float(residual_limit)
        self.target_gate = bool(target_gate)
        self.target_gate_context = bool(target_gate_context)
        self.target_gate_normalize = bool(target_gate_normalize)
        self.target_gate_global = target_gate_global
        self.target_gate_downsample = int(target_gate_downsample)
        self.target_gate_hidden = int(target_gate_hidden)
        self.low_loss_weight = float(low_loss_weight)
        self.edge_loss_weight = float(edge_loss_weight)
        self.background_loss_weight = float(background_loss_weight)
        self.gate_loss_weight = float(gate_loss_weight)
        self.aux_downsample = int(aux_downsample)
        self.capture_diagnostics = False
        # Runtime-only curriculum multiplier. It is intentionally a Python
        # float rather than a state-dict buffer: Trainer applies the exact
        # same value to the live model and EMA at every epoch, avoiding EMA
        # smoothing of the schedule itself.
        self.taylor_schedule_scale = 1.0
        # Enabled transiently by DetectionModel.loss only. Model construction,
        # TensorBoard tracing and plain prediction must never retain a graph on
        # the module because ModelEMA deep-copies the complete model.
        self.collect_auxiliary_state = False
        self._last_diagnostics = {}
        self._loss_state = None
        self._head_taylor_detail = None

        if self.mode in {"concat", "concat_taylor", "explicit_taylor", "head_taylor"}:
            self.concat_projection = nn.Conv2d(6, 3, 1, bias=True)
            with torch.no_grad():
                self.concat_projection.weight.zero_()
                self.concat_projection.bias.zero_()
                for channel in range(3):
                    self.concat_projection.weight[channel, channel, 0, 0] = 0.5
                    self.concat_projection.weight[channel, channel + 3, 0, 0] = 0.5
        elif self.mode == "residual":
            self.residual = _LightPixelBlock(6, hidden=hidden)
        elif self.mode == "wavelet_context_taylor":
            self.low_context = _WaveletLowContextBlock(9, hidden=hidden, out_channels=3)
        else:
            self.quality_encoder = PixelModalityQualityEncoder(hidden=max(8, hidden * 2))

        if self.mode in taylor_modes:
            # concat_taylor must leave the global CPU RNG exactly where E03's
            # concat-only module leaves it. Otherwise the subsequently built
            # randomly initialized class head differs across the ablation even
            # though the epoch-0 fusion image is identical.
            rng_context = (
                torch.random.fork_rng(devices=[])
                if self.mode in {"concat_taylor", "wavelet_context_taylor", "explicit_taylor", "head_taylor"}
                else contextlib.nullcontext()
            )
            with rng_context:
                derivative_in = 27 if self.mode == "wavelet_context_taylor" else 9
                derivative_out = 9 if self.mode == "wavelet_context_taylor" else 3
                if self.mode == "explicit_taylor" and self.explicit_context != "local":
                    backend = self.explicit_context.removeprefix("pooled_")
                    self.derivatives = nn.ModuleList(
                        _ExplicitTaylorContextBlock(
                            derivative_in,
                            hidden=hidden,
                            out_channels=derivative_out,
                            backend=backend,
                            downsample=self.explicit_context_downsample,
                        )
                        for _ in range(self.order)
                    )
                else:
                    self.derivatives = nn.ModuleList(
                        _LightPixelBlock(derivative_in, hidden=hidden, out_channels=derivative_out)
                        for _ in range(self.order)
                    )
                initial_logit = math.log(0.2 / 0.8)
                self.order_gate_logits = nn.Parameter(torch.full((self.order,), initial_logit))
                if self.target_gate:
                    if self.target_gate_global is not None:
                        self.target_gates = nn.ModuleList(
                            _PixelGlobalGate(
                                9,
                                local_hidden=hidden,
                                global_hidden=self.target_gate_hidden,
                                downsample=self.target_gate_downsample,
                                backend=self.target_gate_global,
                            )
                            for _ in range(self.order)
                        )
                    elif self.target_gate_context:
                        self.target_gates = nn.ModuleList(
                            _PixelContextGate(9, hidden=hidden) for _ in range(self.order)
                        )
                    else:
                        self.target_gates = nn.ModuleList(nn.Conv2d(9, 1, 1, bias=True) for _ in range(self.order))
                        for gate in self.target_gates:
                            nn.init.zeros_(gate.weight)
                            nn.init.constant_(gate.bias, initial_logit)

            if self.taylor_frequency_mode != "full" or self.mode == "wavelet_context_taylor":
                analysis_filter, synthesis_filter = create_wavelet_filter("haar", 3, 3, torch.float32)
                # Haar filters are fixed implementation details rather than learned state. Keeping
                # them non-persistent lets full-frequency and wavelet ablations share strict state
                # dictionaries, while .to()/.half() still moves and casts the buffers correctly.
                self.register_buffer("pixel_wavelet_analysis", analysis_filter, persistent=False)
                self.register_buffer("pixel_wavelet_synthesis", synthesis_filter, persistent=False)

    @property
    def has_auxiliary_loss(self):
        return any(
            value > 0.0
            for value in (
                self.low_loss_weight,
                self.edge_loss_weight,
                self.background_loss_weight,
                self.gate_loss_weight,
            )
        )

    def set_taylor_schedule_scale(self, value):
        """Set the externally scheduled Taylor strength in [0, 1]."""
        value = float(value)
        if not 0.0 <= value <= 1.0 or not math.isfinite(value):
            raise ValueError(f"Taylor schedule scale must be finite and in [0, 1], got {value}")
        self.taylor_schedule_scale = value

    def _quality_base(self, rgb, ir):
        weights = self.quality_encoder(rgb, ir)
        base = weights[:, 0, None, None, None] * rgb + weights[:, 1, None, None, None] * ir
        return base, weights

    def _wavelet_components(self, value):
        """Return shape-preserving Haar low/high components, including odd-sized inputs."""
        height, width = value.shape[-2:]
        pad_right, pad_bottom = width % 2, height % 2
        if pad_right or pad_bottom:
            value = F.pad(value, (0, pad_right, 0, pad_bottom), mode="replicate")
        bands = wavelet_transform(value, self.pixel_wavelet_analysis)
        low_bands = torch.zeros_like(bands)
        low_bands[:, :, 0] = bands[:, :, 0]
        low = inverse_wavelet_transform(low_bands, self.pixel_wavelet_synthesis)
        low = low[..., :height, :width]
        return low, value[..., :height, :width] - low

    def _wavelet_highpass(self, value):
        return self._wavelet_components(value)[1]

    def _wavelet_bands(self, value):
        """Decompose an image and return its original spatial shape for exact cropping."""
        height, width = value.shape[-2:]
        if width % 2 or height % 2:
            value = F.pad(value, (0, width % 2, 0, height % 2), mode="replicate")
        return wavelet_transform(value, self.pixel_wavelet_analysis), (height, width)

    def _wavelet_image(self, bands, shape):
        return inverse_wavelet_transform(bands, self.pixel_wavelet_synthesis)[..., : shape[0], : shape[1]]

    def _bound_high_frequency_correction(self, raw_correction):
        """Bound a correction while retaining an effectively zero Haar LL component."""
        raw_correction = self._wavelet_highpass(raw_correction)
        correction_limit = float(getattr(self, "correction_limit", 0.0))
        if correction_limit > 0.0:
            correction = correction_limit * torch.tanh(raw_correction / correction_limit)
        else:
            correction = raw_correction
        # The nonlinear bound can recreate LL energy. Reproject and rescale without
        # changing frequency support so the pixel-domain absolute limit still holds.
        correction = self._wavelet_highpass(correction)
        if correction_limit > 0.0:
            peak = correction.detach().abs().amax(dim=(1, 2, 3), keepdim=True)
            correction = correction * (correction_limit / peak.clamp_min(1e-12)).clamp(max=1.0)
        return raw_correction, correction

    def _apply_wavelet_context_taylor(self, rgb, ir):
        """Fuse LL context and predict ordered Taylor corrections in Haar high subbands."""
        rgb_bands, shape = self._wavelet_bands(rgb)
        ir_bands, _ = self._wavelet_bands(ir)
        rgb_low, ir_low = rgb_bands[:, :, 0], ir_bands[:, :, 0]
        mean_low = 0.5 * (rgb_low + ir_low)
        low_input = torch.cat((rgb_low, ir_low, ir_low - rgb_low), dim=1)
        raw_low_residual = self.low_context(low_input)
        low_residual = self.residual_limit * torch.tanh(raw_low_residual)
        fused_low = mean_low + low_residual

        rgb_high, ir_high = rgb_bands[:, :, 1:], ir_bands[:, :, 1:]
        base_high = 0.5 * (rgb_high + ir_high)
        base_bands = torch.cat((fused_low.unsqueeze(2), base_high), dim=2)
        base = self._wavelet_image(base_bands, shape)

        batch, channels, subbands, height, width = base_high.shape
        base_high_flat = base_high.reshape(batch, channels * subbands, height, width)
        delta = (ir_high - rgb_high).reshape(batch, channels * subbands, height, width)
        raw_high_correction = torch.zeros_like(base_high_flat)
        term = delta
        factorial = 1
        terms, gates = [], []
        schedule_scale = float(getattr(self, "taylor_schedule_scale", 1.0))
        for index, derivative in enumerate(self.derivatives):
            derivative_input = torch.cat((base_high_flat, delta, term), dim=1)
            derivative_term = derivative(derivative_input)
            term = derivative_term + index * term
            factorial *= index + 1
            gate = schedule_scale * self.gate_max * torch.sigmoid(self.order_gate_logits[index])
            raw_high_correction = raw_high_correction + gate * term / factorial
            terms.append(term)
            gates.append(gate)

        correction_bands = torch.cat(
            (
                torch.zeros_like(fused_low).unsqueeze(2),
                raw_high_correction.reshape(batch, channels, subbands, height, width),
            ),
            dim=2,
        )
        raw_correction = self._wavelet_image(correction_bands, shape)
        raw_correction, bounded_correction = self._bound_high_frequency_correction(raw_correction)
        return (
            base + bounded_correction,
            base,
            terms,
            gates,
            raw_correction,
            bounded_correction,
            mean_low,
            low_residual,
            base_high_flat,
        )

    def _apply_taylor(self, base, rgb, ir):
        """Apply recursively learned, factorial-normalized corrections."""
        raw_correction = torch.zeros_like(base)
        terms, gates, spatial_gates, spatial_multipliers = [], [], [], []
        frequency_mode = getattr(self, "taylor_frequency_mode", "full")
        if frequency_mode == "full":
            taylor_base, taylor_rgb, taylor_ir = base, rgb, ir
        else:
            # The E03 image remains the stable fusion base. Only its and both modalities'
            # Haar high-frequency components are visible to the learned Taylor derivative.
            taylor_base = self._wavelet_highpass(base)
            taylor_rgb = self._wavelet_highpass(rgb)
            taylor_ir = self._wavelet_highpass(ir)
        delta = taylor_ir - taylor_rgb
        term = delta
        factorial = 1
        for index, derivative in enumerate(self.derivatives):
            gate_input = torch.cat((taylor_base, delta, term), dim=1)
            derivative_term = derivative(gate_input)
            if self.mode == "explicit_taylor":
                # The network learns only a bounded coefficient field; the order itself
                # is the explicit modality-difference power delta^(k+1).
                normalized_delta = torch.tanh(delta / self.explicit_delta_tau)
                term = torch.tanh(derivative_term) * normalized_delta.pow(index + 1)
            else:
                term = derivative_term + index * term
            factorial *= index + 1
            # getattr keeps checkpoints serialized before curriculum support
            # fully backward compatible (their effective scale is 1.0).
            schedule_scale = float(getattr(self, "taylor_schedule_scale", 1.0))
            global_gate = (
                schedule_scale
                * self.gate_max
                * torch.sigmoid(self.order_gate_logits[index])
            )
            if self.target_gate:
                spatial_gate = torch.sigmoid(self.target_gates[index](gate_input))
                if bool(getattr(self, "target_gate_normalize", False)):
                    # Reallocate a fixed average Taylor budget over space.
                    # Per-sample normalization prevents the derivative branch
                    # from compensating for a globally shrunken spatial gate.
                    spatial_mean = spatial_gate.mean(dim=(2, 3), keepdim=True).clamp_min(1e-6)
                    spatial_multiplier = spatial_gate / spatial_mean
                else:
                    spatial_multiplier = spatial_gate
                gate = global_gate * spatial_multiplier
                spatial_gates.append(spatial_gate)
                spatial_multipliers.append(spatial_multiplier)
            else:
                gate = global_gate
            raw_correction = raw_correction + gate * term / factorial
            terms.append(term)
            gates.append(gate)
        if frequency_mode == "high_projected":
            raw_correction = self._wavelet_highpass(raw_correction)
        correction_limit = float(getattr(self, "correction_limit", 0.0))
        if correction_limit > 0.0:
            bounded_correction = correction_limit * torch.tanh(
                raw_correction / correction_limit
            )
        else:
            bounded_correction = raw_correction
        if frequency_mode == "high_projected":
            # tanh is nonlinear and can reintroduce an LL component, so project once more.
            # A per-sample rescale then preserves both the pure-high-frequency constraint
            # and the requested absolute correction bound.
            bounded_correction = self._wavelet_highpass(bounded_correction)
            if correction_limit > 0.0:
                peak = bounded_correction.detach().abs().amax(dim=(1, 2, 3), keepdim=True)
                scale = (correction_limit / peak.clamp_min(1e-12)).clamp(max=1.0)
                bounded_correction = bounded_correction * scale
        fused = base + bounded_correction
        return (
            fused,
            terms,
            gates,
            spatial_gates,
            spatial_multipliers,
            raw_correction,
            bounded_correction,
        )

    def forward(self, inputs):
        if inputs.ndim != 4 or inputs.shape[1] != 6:
            raise ValueError(f"Pixel fusion expects BCHW input with 6 channels, got {tuple(inputs.shape)}")
        rgb, ir = inputs[:, :3], inputs[:, 3:]
        eps = torch.finfo(inputs.dtype).eps if inputs.dtype.is_floating_point else 1e-8
        mean_low = low_residual = high_base = None

        if self.mode == "wavelet_context_taylor":
            (
                fused,
                base,
                terms,
                gates,
                raw_correction,
                bounded_correction,
                mean_low,
                low_residual,
                high_base,
            ) = self._apply_wavelet_context_taylor(rgb, ir)
            quality_weights = None
            spatial_gates, spatial_multipliers = [], []
        elif self.mode in {"concat", "concat_taylor", "explicit_taylor", "head_taylor"}:
            base = self.concat_projection(inputs)
            fused = base
            quality_weights = None
            terms, gates, spatial_gates, spatial_multipliers = [], [], [], []
            raw_correction = bounded_correction = None
            if self.mode in {"concat_taylor", "explicit_taylor", "head_taylor"}:
                (
                    fused,
                    terms,
                    gates,
                    spatial_gates,
                    spatial_multipliers,
                    raw_correction,
                    bounded_correction,
                ) = self._apply_taylor(base, rgb, ir)
                if self.mode == "head_taylor":
                    self._head_taylor_detail = bounded_correction
                    fused = base
        elif self.mode == "residual":
            base = rgb
            raw_residual = self.residual(inputs)
            fused = rgb + self.residual_limit * torch.tanh(raw_residual)
            quality_weights = None
            terms, gates, spatial_gates, spatial_multipliers = [raw_residual], [], [], []
            raw_correction = bounded_correction = None
        else:
            base, quality_weights = self._quality_base(rgb, ir)
            fused = base
            terms, gates, spatial_gates, spatial_multipliers = [], [], [], []
            raw_correction = bounded_correction = None
            if self.mode == "taylor":
                (
                    fused,
                    terms,
                    gates,
                    spatial_gates,
                    spatial_multipliers,
                    raw_correction,
                    bounded_correction,
                ) = self._apply_taylor(base, rgb, ir)

        fused = fused.clamp(0.0, 1.0)
        # Keep a graph-connected state only when the training loss will use it.
        self._loss_state = (
            {
                "rgb": rgb,
                "ir": ir,
                "base": base,
                "fused": fused,
                "gates": gates,
                "spatial_gates": spatial_gates,
            }
            if self.training and self.has_auxiliary_loss and self.collect_auxiliary_state
            else None
        )

        if self.capture_diagnostics:
            base_rms = _rms(base).clamp_min(eps)
            diagnostics = {
                "schedule_scale": float(getattr(self, "taylor_schedule_scale", 1.0)),
                "fused_to_base_rms": _rms(fused - base) / base_rms,
                "fused_to_rgb_rms": _rms(fused - rgb) / _rms(rgb).clamp_min(eps),
                "fused_to_ir_rms": _rms(fused - ir) / _rms(ir).clamp_min(eps),
            }
            if quality_weights is not None:
                diagnostics["quality/rgb_weight"] = quality_weights[:, 0].detach().mean()
                diagnostics["quality/ir_weight"] = quality_weights[:, 1].detach().mean()
                diagnostics["quality/rgb_weight_std"] = quality_weights[:, 0].detach().std(unbiased=False)
            for index, term in enumerate(terms):
                diagnostics[f"order{index + 1}/term_to_base_rms"] = _rms(term) / base_rms
                if index < len(gates):
                    diagnostics[f"order{index + 1}/actual_contribution_to_base_rms"] = (
                        _rms(gates[index] * term / math.factorial(index + 1)) / base_rms
                    )
            for index, gate in enumerate(gates):
                gate_value = gate.detach() if torch.is_tensor(gate) else torch.as_tensor(gate)
                diagnostics[f"order{index + 1}/gate_mean"] = gate_value.float().mean()
                diagnostics[f"order{index + 1}/gate_std"] = gate_value.float().std(unbiased=False)
            for index, spatial_gate in enumerate(spatial_gates):
                diagnostics[f"order{index + 1}/spatial_gate_mean"] = spatial_gate.detach().float().mean()
                diagnostics[f"order{index + 1}/spatial_gate_std"] = spatial_gate.detach().float().std(
                    unbiased=False
                )
            for index, spatial_multiplier in enumerate(spatial_multipliers):
                diagnostics[f"order{index + 1}/spatial_multiplier_mean"] = (
                    spatial_multiplier.detach().float().mean()
                )
                diagnostics[f"order{index + 1}/spatial_multiplier_std"] = (
                    spatial_multiplier.detach().float().std(unbiased=False)
                )
            if raw_correction is not None:
                raw_rms = _rms(raw_correction)
                bounded_rms = _rms(bounded_correction)
                diagnostics["correction/raw_to_base_rms"] = raw_rms / base_rms
                diagnostics["correction/applied_to_base_rms"] = bounded_rms / base_rms
                diagnostics["correction/compression_ratio"] = bounded_rms / raw_rms.clamp_min(eps)
                correction_limit = float(getattr(self, "correction_limit", 0.0))
                diagnostics["correction/limit"] = correction_limit
                if correction_limit > 0.0:
                    diagnostics["correction/raw_over_limit_fraction"] = (
                        raw_correction.detach().abs() >= correction_limit
                    ).float().mean()
                frequency_mode = getattr(self, "taylor_frequency_mode", "full")
                if frequency_mode != "full" or self.mode == "wavelet_context_taylor":
                    base_low, base_high = self._wavelet_components(base)
                    correction_low, correction_high = self._wavelet_components(bounded_correction)
                    diagnostics["frequency/base_low_to_base_rms"] = _rms(base_low) / base_rms
                    diagnostics["frequency/base_high_to_base_rms"] = _rms(base_high) / base_rms
                    correction_rms = _rms(bounded_correction).clamp_min(eps)
                    diagnostics["frequency/correction_low_fraction"] = _rms(correction_low) / correction_rms
                    diagnostics["frequency/correction_high_fraction"] = _rms(correction_high) / correction_rms
            if mean_low is not None:
                low_rms = _rms(mean_low).clamp_min(eps)
                diagnostics["wavelet_context/low_residual_to_mean_low_rms"] = _rms(low_residual) / low_rms
                diagnostics["wavelet_context/high_base_rms"] = _rms(high_base)
            self._last_diagnostics = diagnostics
        return fused

    def pop_head_taylor_detail(self):
        """Consume the graph-connected head detail within the same detector forward."""
        detail = self._head_taylor_detail
        self._head_taylor_detail = None
        return detail

    def _foreground_mask(self, batch, height, width, device, dtype):
        mask = torch.zeros((batch["img"].shape[0], 1, height, width), device=device, dtype=dtype)

        def add_boxes(boxes, indices):
            if boxes is None or indices is None or boxes.numel() == 0:
                return
            boxes = boxes.detach().to(device=device, dtype=torch.float32).view(-1, 4)
            indices = indices.detach().to(device=device, dtype=torch.long).view(-1)
            for box, batch_index in zip(boxes, indices):
                cx, cy, bw, bh = box.tolist()
                x1 = max(0, min(width - 1, int((cx - bw / 2) * width)))
                x2 = max(x1 + 1, min(width, int(math.ceil((cx + bw / 2) * width))))
                y1 = max(0, min(height - 1, int((cy - bh / 2) * height)))
                y2 = max(y1 + 1, min(height, int(math.ceil((cy + bh / 2) * height))))
                if 0 <= int(batch_index) < mask.shape[0]:
                    mask[int(batch_index), :, y1:y2, x1:x2] = 1.0

        add_boxes(batch.get("bboxes"), batch.get("batch_idx"))
        add_boxes(batch.get("bboxes_ir"), batch.get("batch_idx_ir"))
        if mask.any():
            mask = F.max_pool2d(mask, 3, stride=1, padding=1)
            mask = F.avg_pool2d(mask, 5, stride=1, padding=2)
        return mask

    def auxiliary_loss(self, batch):
        """Return weighted task-image regularizers for the most recent forward."""
        if not self.has_auxiliary_loss:
            reference = next(self.parameters())
            return reference.sum() * 0.0
        if self._loss_state is None:
            raise RuntimeError("Pixel fusion auxiliary_loss must follow its training forward pass")
        state = self._loss_state
        factor = self.aux_downsample
        rgb = F.avg_pool2d(_gray(state["rgb"]), factor, factor)
        ir = F.avg_pool2d(_gray(state["ir"]), factor, factor)
        base = F.avg_pool2d(_gray(state["base"]), factor, factor)
        fused = F.avg_pool2d(_gray(state["fused"]), factor, factor)
        height, width = fused.shape[-2:]
        foreground = self._foreground_mask(batch, height, width, fused.device, fused.dtype)
        background = 1.0 - foreground

        low_kernel = 5
        low_fused = F.avg_pool2d(fused, low_kernel, stride=1, padding=low_kernel // 2)
        low_base = F.avg_pool2d(base.detach(), low_kernel, stride=1, padding=low_kernel // 2)
        low_loss = F.l1_loss(low_fused, low_base)

        fused_gradient = _gradient_magnitude(fused)
        target_gradient = torch.maximum(_gradient_magnitude(rgb), _gradient_magnitude(ir)).detach()
        foreground_denominator = foreground.sum().clamp_min(1.0)
        edge_loss = ((fused_gradient - target_gradient).abs() * foreground).sum() / foreground_denominator

        correction_gradient = _gradient_magnitude(fused - base.detach())
        background_denominator = background.sum().clamp_min(1.0)
        background_loss = (correction_gradient * background).sum() / background_denominator

        gate_loss = fused.new_zeros(())
        if self.gate_loss_weight > 0.0 and state["spatial_gates"]:
            # Supervise only the spatial selector. The global Taylor magnitude
            # remains detection-driven. Balance foreground and background so
            # the sparse VEDAI boxes cannot be overwhelmed by background.
            foreground_denominator = foreground.sum().clamp_min(1.0)
            background_denominator = background.sum().clamp_min(1.0)
            balanced_losses = []
            foreground_gate_means = []
            background_gate_means = []
            for spatial_gate in state["spatial_gates"]:
                probability = F.interpolate(
                    spatial_gate, size=(height, width), mode="bilinear", align_corners=False
                ).clamp(1e-5, 1 - 1e-5)
                foreground_loss = -(foreground * probability.log()).sum() / foreground_denominator
                background_loss_gate = -(background * (1.0 - probability).log()).sum() / background_denominator
                balanced_losses.append(0.5 * (foreground_loss + background_loss_gate))
                foreground_gate_means.append((foreground * probability).sum() / foreground_denominator)
                background_gate_means.append((background * probability).sum() / background_denominator)
            gate_loss = sum(balanced_losses) / len(balanced_losses)
            foreground_gate_mean = sum(foreground_gate_means) / len(foreground_gate_means)
            background_gate_mean = sum(background_gate_means) / len(background_gate_means)

        total = (
            self.low_loss_weight * low_loss
            + self.edge_loss_weight * edge_loss
            + self.background_loss_weight * background_loss
            + self.gate_loss_weight * gate_loss
        )
        if self.capture_diagnostics:
            loss_diagnostics = {
                    "loss/low": low_loss.detach(),
                    "loss/foreground_edge": edge_loss.detach(),
                    "loss/background_high": background_loss.detach(),
                    "loss/gate": gate_loss.detach(),
                    "loss/weighted_total": total.detach(),
                    "mask/foreground_fraction": foreground.detach().mean(),
                }
            if self.gate_loss_weight > 0.0 and state["spatial_gates"]:
                loss_diagnostics.update(
                    {
                        "gate/spatial_foreground_mean": foreground_gate_mean.detach(),
                        "gate/spatial_background_mean": background_gate_mean.detach(),
                        "gate/spatial_separation": (foreground_gate_mean - background_gate_mean).detach(),
                    }
                )
            self._last_diagnostics.update(loss_diagnostics)
        self._loss_state = None
        return total

    def diagnostic_values(self):
        values = self._last_diagnostics
        self._last_diagnostics = {}
        return values
