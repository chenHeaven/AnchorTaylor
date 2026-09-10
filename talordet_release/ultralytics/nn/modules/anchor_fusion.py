"""Pretrained-feature-preserving RGB/IR fusion modules."""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class P2FrequencySelectiveFusion(nn.Module):
    """Select complementary RGB/IR P2 detail from a fixed multi-filter bank."""

    filter_names = (
        "rgb_laplacian",
        "rgb_gaussian_residual",
        "rgb_sobel",
        "ir_laplacian",
        "ir_gaussian_residual",
        "ir_sobel",
    )

    def __init__(self, channels, hidden=16, gate_max=0.10):
        super().__init__()
        channels, hidden = int(channels), int(hidden)
        if channels <= 0 or hidden <= 0 or gate_max <= 0.0:
            raise ValueError("channels, hidden and gate_max must be positive")
        self.channels = channels
        self.gate_max = float(gate_max)

        laplacian = torch.tensor(
            [[0.0, -1.0, 0.0], [-1.0, 4.0, -1.0], [0.0, -1.0, 0.0]]
        ).div_(4.0)
        gaussian_1d = torch.tensor([1.0, 4.0, 6.0, 4.0, 1.0])
        gaussian = torch.outer(gaussian_1d, gaussian_1d).div_(256.0)
        sobel_x = torch.tensor(
            [[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]]
        ).div_(8.0)
        sobel_y = sobel_x.t().contiguous()
        self.register_buffer("laplacian_kernel", laplacian.view(1, 1, 3, 3), persistent=True)
        self.register_buffer("gaussian_kernel", gaussian.view(1, 1, 5, 5), persistent=True)
        self.register_buffer("sobel_x_kernel", sobel_x.view(1, 1, 3, 3), persistent=True)
        self.register_buffer("sobel_y_kernel", sobel_y.view(1, 1, 3, 3), persistent=True)

        # Select one complementary response from 3 filters x 2 modalities for
        # every sample and P2 channel. Channel-wise global selection avoids a
        # prohibitively large 6C x 160 x 160 softmax while the residual gate
        # below still provides spatially varying target selection.
        self.filter_selector = nn.Sequential(
            nn.Conv2d(3 * channels, hidden, 1, bias=True),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden, 6 * channels, 1, bias=True),
        )
        nn.init.zeros_(self.filter_selector[-1].weight)
        nn.init.zeros_(self.filter_selector[-1].bias)
        self.detail_refine = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, groups=channels, bias=True),
            nn.SiLU(inplace=True),
            nn.Conv2d(channels, channels, 1, bias=True),
        )
        self.residual_gate = nn.Sequential(
            nn.Conv2d(3 * channels, hidden, 1, bias=True),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden, channels, 1, bias=True),
        )
        # The auxiliary P2 representation starts as the official RGB P2
        # feature. The gate may subsequently introduce only useful detail.
        nn.init.zeros_(self.residual_gate[-1].weight)
        nn.init.zeros_(self.residual_gate[-1].bias)
        self.capture_diagnostics = False
        self._last_diagnostics = {}

    @staticmethod
    def _depthwise_filter(x, kernel):
        padding = kernel.shape[-1] // 2
        padded = F.pad(x, (padding, padding, padding, padding), mode="replicate")
        weight = kernel.to(dtype=x.dtype).expand(x.shape[1], 1, -1, -1)
        return F.conv2d(padded, weight, groups=x.shape[1])

    def _filter_bank(self, x):
        laplacian = self._depthwise_filter(x, self.laplacian_kernel)
        gaussian_residual = x - self._depthwise_filter(x, self.gaussian_kernel)
        sobel_x = self._depthwise_filter(x, self.sobel_x_kernel)
        sobel_y = self._depthwise_filter(x, self.sobel_y_kernel)
        # L1 gradient magnitude is AMP-safe and orientation independent.
        sobel = 0.5 * (sobel_x.abs() + sobel_y.abs())
        return laplacian, gaussian_residual, sobel

    def forward(self, rgb, ir):
        if rgb.shape != ir.shape:
            raise ValueError(f"Aligned RGB/IR P2 shapes differ: {tuple(rgb.shape)} vs {tuple(ir.shape)}")
        if rgb.ndim != 4 or rgb.shape[1] != self.channels:
            raise ValueError(f"Expected BCHW P2 features with {self.channels} channels, got {tuple(rgb.shape)}")
        context = torch.cat((rgb, ir, (ir - rgb).abs()), dim=1)
        responses = torch.stack((*self._filter_bank(rgb), *self._filter_bank(ir)), dim=2)
        logits = self.filter_selector(F.adaptive_avg_pool2d(context, 1)).view(
            rgb.shape[0], self.channels, len(self.filter_names), 1, 1
        )
        filter_weights = logits.softmax(dim=2)
        selected_detail = (filter_weights * responses).sum(dim=2)
        correction = self.detail_refine(selected_detail)
        multiplier = self.gate_max * torch.tanh(self.residual_gate(context))
        fused = rgb + multiplier * correction

        if self.capture_diagnostics:
            dims, eps = (1, 2, 3), 1e-8
            rgb_rms = rgb.detach().float().square().mean(dims).sqrt().clamp_min(eps)
            values = {
                "gate_mean": multiplier.detach().float().mean(),
                "gate_abs_mean": multiplier.detach().float().abs().mean(),
                "gate_abs_max": multiplier.detach().float().abs().amax(),
                "detail_to_rgb_rms": (
                    selected_detail.detach().float().square().mean(dims).sqrt() / rgb_rms
                ).mean(),
                "correction_to_rgb_rms": (
                    (fused - rgb).detach().float().square().mean(dims).sqrt() / rgb_rms
                ).mean(),
            }
            for index, name in enumerate(self.filter_names):
                values[f"filter_weight/{name}"] = filter_weights[:, :, index].detach().float().mean()
                values[f"filter_rms/{name}"] = (
                    responses[:, :, index].detach().float().square().mean(dims).sqrt() / rgb_rms
                ).mean()
            self._last_diagnostics = values
        return fused

    def diagnostic_values(self):
        values = self._last_diagnostics
        self._last_diagnostics = {}
        return values


class P2BidirectionalAdaptiveInjection(nn.Module):
    """Inject complementary P2 detail into both streams with one bounded budget.

    The shared gate is exactly zero at initialization, so both output streams
    reproduce their official-weight inputs.  A channel-wise softmax allocates
    the available correction budget between RGB and IR instead of allowing two
    independent gates to grow without constraint.
    """

    filter_names = ("laplacian", "gaussian_residual", "sobel")

    def __init__(self, channels, hidden=16, gate_max=0.10):
        super().__init__()
        channels, hidden = int(channels), int(hidden)
        if channels <= 0 or hidden <= 0 or gate_max <= 0.0:
            raise ValueError("channels, hidden and gate_max must be positive")
        self.channels = channels
        self.gate_max = float(gate_max)

        laplacian = torch.tensor(
            [[0.0, -1.0, 0.0], [-1.0, 4.0, -1.0], [0.0, -1.0, 0.0]]
        ).div_(4.0)
        gaussian_1d = torch.tensor([1.0, 4.0, 6.0, 4.0, 1.0])
        gaussian = torch.outer(gaussian_1d, gaussian_1d).div_(256.0)
        sobel_x = torch.tensor(
            [[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]]
        ).div_(8.0)
        self.register_buffer("laplacian_kernel", laplacian.view(1, 1, 3, 3), persistent=True)
        self.register_buffer("gaussian_kernel", gaussian.view(1, 1, 5, 5), persistent=True)
        self.register_buffer("sobel_x_kernel", sobel_x.view(1, 1, 3, 3), persistent=True)
        self.register_buffer("sobel_y_kernel", sobel_x.t().contiguous().view(1, 1, 3, 3), persistent=True)

        # Select a useful filter independently for each modality and channel.
        self.filter_selector = nn.Sequential(
            nn.Conv2d(3 * channels, hidden, 1, bias=True),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden, 6 * channels, 1, bias=True),
        )
        nn.init.zeros_(self.filter_selector[-1].weight)
        nn.init.zeros_(self.filter_selector[-1].bias)

        # RGB detail corrects IR; IR detail/saliency corrects RGB.  Including
        # the inter-modal detail difference prevents identical corrections
        # from prematurely homogenizing the two streams before P3 Taylor.
        self.rgb_to_ir = self._make_refiner(channels)
        self.ir_to_rgb = self._make_refiner(channels)
        self.allocation = nn.Sequential(
            nn.Conv2d(3 * channels, hidden, 1, bias=True),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden, 2 * channels, 1, bias=True),
        )
        nn.init.zeros_(self.allocation[-1].weight)
        nn.init.zeros_(self.allocation[-1].bias)
        self.shared_gate = nn.Sequential(
            nn.Conv2d(3 * channels, hidden, 1, bias=True),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden, channels, 1, bias=True),
        )
        nn.init.zeros_(self.shared_gate[-1].weight)
        nn.init.zeros_(self.shared_gate[-1].bias)
        self.capture_diagnostics = False
        self._last_diagnostics = {}

    @staticmethod
    def _make_refiner(channels):
        return nn.Sequential(
            nn.Conv2d(2 * channels, channels, 1, bias=True),
            nn.SiLU(inplace=True),
            nn.Conv2d(channels, channels, 3, padding=1, groups=channels, bias=True),
            nn.SiLU(inplace=True),
            nn.Conv2d(channels, channels, 1, bias=True),
        )

    @staticmethod
    def _depthwise_filter(x, kernel):
        padding = kernel.shape[-1] // 2
        padded = F.pad(x, (padding, padding, padding, padding), mode="replicate")
        weight = kernel.to(dtype=x.dtype).expand(x.shape[1], 1, -1, -1)
        return F.conv2d(padded, weight, groups=x.shape[1])

    def _filter_bank(self, x):
        laplacian = self._depthwise_filter(x, self.laplacian_kernel)
        gaussian_residual = x - self._depthwise_filter(x, self.gaussian_kernel)
        sobel_x = self._depthwise_filter(x, self.sobel_x_kernel)
        sobel_y = self._depthwise_filter(x, self.sobel_y_kernel)
        return torch.stack((laplacian, gaussian_residual, 0.5 * (sobel_x.abs() + sobel_y.abs())), dim=2)

    def forward(self, rgb, ir):
        if rgb.shape != ir.shape:
            raise ValueError(f"Aligned RGB/IR P2 shapes differ: {tuple(rgb.shape)} vs {tuple(ir.shape)}")
        if rgb.ndim != 4 or rgb.shape[1] != self.channels:
            raise ValueError(f"Expected BCHW P2 features with {self.channels} channels, got {tuple(rgb.shape)}")
        context = torch.cat((rgb, ir, (ir - rgb).abs()), dim=1)
        pooled = F.adaptive_avg_pool2d(context, 1)
        filter_logits = self.filter_selector(pooled).view(rgb.shape[0], self.channels, 2, 3, 1, 1)
        filter_weights = filter_logits.softmax(dim=3)
        rgb_bank, ir_bank = self._filter_bank(rgb), self._filter_bank(ir)
        rgb_detail = (filter_weights[:, :, 0] * rgb_bank).sum(dim=2)
        ir_detail = (filter_weights[:, :, 1] * ir_bank).sum(dim=2)

        correction_to_ir = self.rgb_to_ir(torch.cat((rgb_detail, rgb_detail - ir_detail), dim=1))
        correction_to_rgb = self.ir_to_rgb(torch.cat((ir_detail, ir_detail - rgb_detail), dim=1))
        allocation = self.allocation(pooled).view(rgb.shape[0], self.channels, 2, 1, 1).softmax(dim=2)
        shared_gate = self.gate_max * torch.tanh(self.shared_gate(context))
        rgb_injection = shared_gate * allocation[:, :, 0] * correction_to_rgb
        ir_injection = shared_gate * allocation[:, :, 1] * correction_to_ir
        rgb_out, ir_out = rgb + rgb_injection, ir + ir_injection

        if self.capture_diagnostics:
            dims, eps = (1, 2, 3), 1e-8
            rgb_rms = rgb.detach().float().square().mean(dims).sqrt().clamp_min(eps)
            ir_rms = ir.detach().float().square().mean(dims).sqrt().clamp_min(eps)
            values = {
                "gate_mean": shared_gate.detach().float().mean(),
                "gate_abs_mean": shared_gate.detach().float().abs().mean(),
                "gate_abs_max": shared_gate.detach().float().abs().amax(),
                "allocation_rgb": allocation[:, :, 0].detach().float().mean(),
                "allocation_ir": allocation[:, :, 1].detach().float().mean(),
                "rgb_injection_to_rgb_rms": (
                    rgb_injection.detach().float().square().mean(dims).sqrt() / rgb_rms
                ).mean(),
                "ir_injection_to_ir_rms": (
                    ir_injection.detach().float().square().mean(dims).sqrt() / ir_rms
                ).mean(),
                "post_injection_delta_to_rgb_rms": (
                    (ir_out - rgb_out).detach().float().square().mean(dims).sqrt() / rgb_rms
                ).mean(),
            }
            for index, name in enumerate(self.filter_names):
                values[f"filter_weight/rgb_{name}"] = filter_weights[:, :, 0, index].detach().float().mean()
                values[f"filter_weight/ir_{name}"] = filter_weights[:, :, 1, index].detach().float().mean()
            self._last_diagnostics = values
        return rgb_out, ir_out

    def diagnostic_values(self):
        values = self._last_diagnostics
        self._last_diagnostics = {}
        return values


class P2RGBCorrectionOnly(P2BidirectionalAdaptiveInjection):
    """Return the exact RGB output of P2 exchange and omit its unused reverse refiner.

    This is the behaviorally equivalent P2-only ablation interface: both input
    modalities, filter banks, filter weights, allocation logits, and the
    IR-to-RGB correction are unchanged.  The RGB-to-IR refiner is removed
    because the infrared branch terminates immediately after P2.
    """

    def __init__(self, channels, hidden=16, gate_max=0.10):
        super().__init__(channels, hidden=hidden, gate_max=gate_max)
        del self.rgb_to_ir

    def forward(self, rgb, ir):
        if rgb.shape != ir.shape:
            raise ValueError(f"Aligned RGB/IR P2 shapes differ: {tuple(rgb.shape)} vs {tuple(ir.shape)}")
        if rgb.ndim != 4 or rgb.shape[1] != self.channels:
            raise ValueError(f"Expected BCHW P2 features with {self.channels} channels, got {tuple(rgb.shape)}")
        context = torch.cat((rgb, ir, (ir - rgb).abs()), dim=1)
        pooled = F.adaptive_avg_pool2d(context, 1)
        filter_logits = self.filter_selector(pooled).view(rgb.shape[0], self.channels, 2, 3, 1, 1)
        filter_weights = filter_logits.softmax(dim=3)
        rgb_bank, ir_bank = self._filter_bank(rgb), self._filter_bank(ir)
        rgb_detail = (filter_weights[:, :, 0] * rgb_bank).sum(dim=2)
        ir_detail = (filter_weights[:, :, 1] * ir_bank).sum(dim=2)

        correction_to_rgb = self.ir_to_rgb(torch.cat((ir_detail, ir_detail - rgb_detail), dim=1))
        allocation = self.allocation(pooled).view(rgb.shape[0], self.channels, 2, 1, 1).softmax(dim=2)
        shared_gate = self.gate_max * torch.tanh(self.shared_gate(context))
        rgb_injection = shared_gate * allocation[:, :, 0] * correction_to_rgb
        rgb_out = rgb + rgb_injection

        if self.capture_diagnostics:
            dims, eps = (1, 2, 3), 1e-8
            rgb_rms = rgb.detach().float().square().mean(dims).sqrt().clamp_min(eps)
            values = {
                "gate_mean": shared_gate.detach().float().mean(),
                "gate_abs_mean": shared_gate.detach().float().abs().mean(),
                "gate_abs_max": shared_gate.detach().float().abs().amax(),
                "allocation_rgb": allocation[:, :, 0].detach().float().mean(),
                "rgb_injection_to_rgb_rms": (
                    rgb_injection.detach().float().square().mean(dims).sqrt() / rgb_rms
                ).mean(),
            }
            for index, name in enumerate(self.filter_names):
                values[f"filter_weight/rgb_{name}"] = filter_weights[:, :, 0, index].detach().float().mean()
                values[f"filter_weight/ir_{name}"] = filter_weights[:, :, 1, index].detach().float().mean()
            self._last_diagnostics = values
        return rgb_out


class P2RGBAnchorProtectedInjection(P2BidirectionalAdaptiveInjection):
    """Correct only the IR stream while preserving the official RGB P2 anchor exactly."""

    def __init__(self, channels, hidden=16, gate_max=0.10):
        super().__init__(channels, hidden=hidden, gate_max=gate_max)
        # These bidirectional-only branches would otherwise become unused DDP
        # parameters.  The protected variant has one explicit RGB -> IR path.
        del self.ir_to_rgb
        del self.allocation

    def forward(self, rgb, ir):
        if rgb.shape != ir.shape:
            raise ValueError(f"Aligned RGB/IR P2 shapes differ: {tuple(rgb.shape)} vs {tuple(ir.shape)}")
        if rgb.ndim != 4 or rgb.shape[1] != self.channels:
            raise ValueError(f"Expected BCHW P2 features with {self.channels} channels, got {tuple(rgb.shape)}")
        context = torch.cat((rgb, ir, (ir - rgb).abs()), dim=1)
        pooled = F.adaptive_avg_pool2d(context, 1)
        filter_logits = self.filter_selector(pooled).view(rgb.shape[0], self.channels, 2, 3, 1, 1)
        filter_weights = filter_logits.softmax(dim=3)
        rgb_bank, ir_bank = self._filter_bank(rgb), self._filter_bank(ir)
        rgb_detail = (filter_weights[:, :, 0] * rgb_bank).sum(dim=2)
        ir_detail = (filter_weights[:, :, 1] * ir_bank).sum(dim=2)

        correction_to_ir = self.rgb_to_ir(torch.cat((rgb_detail, rgb_detail - ir_detail), dim=1))
        protected_gate = self.gate_max * torch.tanh(self.shared_gate(context))
        ir_injection = protected_gate * correction_to_ir
        rgb_out, ir_out = rgb, ir + ir_injection

        if self.capture_diagnostics:
            dims, eps = (1, 2, 3), 1e-8
            rgb_rms = rgb.detach().float().square().mean(dims).sqrt().clamp_min(eps)
            ir_rms = ir.detach().float().square().mean(dims).sqrt().clamp_min(eps)
            self._last_diagnostics = {
                "gate_mean": protected_gate.detach().float().mean(),
                "gate_abs_mean": protected_gate.detach().float().abs().mean(),
                "gate_abs_max": protected_gate.detach().float().abs().amax(),
                "rgb_anchor_delta_abs_max": (rgb_out - rgb).detach().float().abs().amax(),
                "ir_injection_to_ir_rms": (
                    ir_injection.detach().float().square().mean(dims).sqrt() / ir_rms
                ).mean(),
                "post_injection_delta_to_rgb_rms": (
                    (ir_out - rgb_out).detach().float().square().mean(dims).sqrt() / rgb_rms
                ).mean(),
                **{
                    f"filter_weight/rgb_{name}": filter_weights[:, :, 0, index].detach().float().mean()
                    for index, name in enumerate(self.filter_names)
                },
                **{
                    f"filter_weight/ir_{name}": filter_weights[:, :, 1, index].detach().float().mean()
                    for index, name in enumerate(self.filter_names)
                },
            }
        return rgb_out, ir_out


class PretrainedAnchorP3Fusion(nn.Module):
    """Fuse aligned P3 features while preserving the selected anchor at initialization."""

    def __init__(
        self,
        channels,
        mode="residual",
        hidden=16,
        gate_max=0.25,
        order=1,
        recurrence="additive",
        anchor_modality="rgb",
    ):
        super().__init__()
        channels, hidden = int(channels), int(hidden)
        if mode not in {"mean", "residual"}:
            raise ValueError(f"Unsupported pretrained-anchor fusion mode: {mode}")
        order = int(order)
        recurrence = str(recurrence)
        anchor_modality = str(anchor_modality).lower()
        if channels <= 0 or hidden <= 0 or gate_max <= 0.0 or order not in {1, 2, 3}:
            raise ValueError("channels, hidden and gate_max must be positive; order must be 1, 2, or 3")
        if recurrence not in {"additive", "bounded_mul", "rms_capped"}:
            raise ValueError(f"Unsupported pretrained-anchor Taylor recurrence: {recurrence}")
        if anchor_modality not in {"rgb", "ir"}:
            raise ValueError(f"Unsupported P3 anchor modality: {anchor_modality}")
        if mode != "residual" and recurrence != "additive":
            raise ValueError("Constrained Taylor recurrence requires residual fusion mode")
        self.mode = mode
        self.gate_max = float(gate_max)
        self.order = order
        self.recurrence = recurrence
        self.anchor_modality = anchor_modality
        if mode == "residual":
            self.derivatives = nn.ModuleList(
                nn.Sequential(
                    nn.Conv2d(3 * channels, hidden, 1, bias=True),
                    nn.SiLU(inplace=True),
                    nn.Conv2d(hidden, channels, 3, padding=1, bias=True),
                )
                for _ in range(order - 1)
            )
            self.gates = nn.ModuleList()
            for index in range(order):
                gate = nn.Sequential(
                    nn.Conv2d((2 if index == 0 else 3) * channels, hidden, 1, bias=True),
                    nn.SiLU(inplace=True),
                    nn.Conv2d(hidden, channels, 1, bias=True),
                )
                nn.init.zeros_(gate[-1].weight)
                nn.init.zeros_(gate[-1].bias)
                self.gates.append(gate)
        else:
            self.derivatives = nn.ModuleList()
            self.gates = nn.ModuleList()
        self.capture_diagnostics = False
        self._last_diagnostics = {}

    def forward(self, rgb, ir):
        if rgb.shape != ir.shape:
            raise ValueError(f"Aligned RGB/IR P3 shapes differ: {tuple(rgb.shape)} vs {tuple(ir.shape)}")
        # Checkpoints produced before anchor selection was explicit have no
        # attribute here; they are the historical RGB-anchor experiments.
        anchor_modality = getattr(self, "anchor_modality", "rgb")
        base, other = (rgb, ir) if anchor_modality == "rgb" else (ir, rgb)
        delta = other - base
        if self.mode == "mean":
            fused = base + 0.5 * delta
            multipliers = [base.new_full((base.shape[0], 1, 1, 1), 0.5)]
            contributions = [0.5 * delta]
        else:
            terms = [delta]
            for index, derivative in enumerate(self.derivatives, start=1):
                previous = terms[-1]
                derivative_value = derivative(torch.cat((previous, rgb, ir), dim=1))
                if self.recurrence == "bounded_mul":
                    next_term = torch.tanh(derivative_value) * previous
                else:
                    next_term = derivative_value + index * previous
                    if self.recurrence == "rms_capped":
                        # Preserve the additive-recursion direction while
                        # preventing any sample/channel from increasing its
                        # spatial RMS beyond the preceding Taylor term.
                        dims = (2, 3)
                        eps = torch.finfo(torch.float32).eps
                        previous_rms = previous.float().square().mean(dims, keepdim=True).sqrt()
                        next_rms = next_term.float().square().mean(dims, keepdim=True).sqrt()
                        scale = (previous_rms / next_rms.clamp_min(eps)).clamp(max=1.0)
                        next_term = next_term * scale.to(next_term.dtype)
                terms.append(next_term)
            multipliers, contributions = [], []
            for index, (term, gate) in enumerate(zip(terms, self.gates), start=1):
                gate_input = torch.cat((rgb, ir), dim=1) if index == 1 else torch.cat((rgb, ir, term), dim=1)
                multiplier = self.gate_max * torch.tanh(gate(gate_input))
                multipliers.append(multiplier)
                contributions.append(multiplier * term / math.factorial(index))
            fused = base + sum(contributions)
        if self.capture_diagnostics:
            dims, eps = (1, 2, 3), 1e-8
            base_rms = base.float().square().mean(dims).sqrt().clamp_min(eps)
            correction = (fused - base).float()
            all_multipliers = torch.cat([item.detach().float().flatten() for item in multipliers])
            values = {
                "anchor_is_ir": float(anchor_modality == "ir"),
                "gate_mean": all_multipliers.mean(),
                "gate_abs_mean": all_multipliers.abs().mean(),
                "gate_abs_max": all_multipliers.abs().amax(),
                f"delta_to_{anchor_modality}_rms": (
                    delta.detach().float().square().mean(dims).sqrt() / base_rms
                ).mean(),
                f"correction_to_{anchor_modality}_rms": (
                    correction.detach().square().mean(dims).sqrt() / base_rms
                ).mean(),
            }
            for index, contribution in enumerate(contributions, start=1):
                values[f"order{index}_to_{anchor_modality}_rms"] = (
                    contribution.detach().float().square().mean(dims).sqrt() / base_rms
                ).mean()
            if self.mode == "residual":
                for index, term in enumerate(terms, start=1):
                    values[f"term{index}_to_{anchor_modality}_rms"] = (
                        term.detach().float().square().mean(dims).sqrt() / base_rms
                    ).mean()
            self._last_diagnostics = values
        return fused

    def diagnostic_values(self):
        values = self._last_diagnostics
        self._last_diagnostics = {}
        return values
