"""Scale-assignment ablations with fixed P2 dual / P3 RGB-only interfaces."""
import torch
import torch.nn as nn
import torch.nn.functional as F

from .anchor_fusion import P2BidirectionalAdaptiveInjection, PretrainedAnchorP3Fusion


class P2BidirectionalTaylorExchange(nn.Module):
    """Two independent O3 corrections evaluated on the same unmodified input pair."""

    def __init__(self, channels, hidden=16, gate_max=0.10, order=3):
        super().__init__()
        self.rgb_correction = PretrainedAnchorP3Fusion(
            channels, hidden=hidden, gate_max=gate_max, order=order,
            mode="residual", recurrence="additive", anchor_modality="rgb",
        )
        self.ir_correction = PretrainedAnchorP3Fusion(
            channels, hidden=hidden, gate_max=gate_max, order=order,
            mode="residual", recurrence="additive", anchor_modality="ir",
        )

    def forward(self, rgb, ir):
        return self.rgb_correction(rgb, ir), self.ir_correction(rgb, ir)


class P3DetailCorrection(P2BidirectionalAdaptiveInjection):
    """Exact RGB output of D, with the unused RGB-to-IR refiner removed.

The two-way softmax allocation is retained: both logits affect the RGB weight.
Both detail banks also remain necessary for the cross-modal detail difference.
"""

    def __init__(self, channels, hidden=16, gate_max=0.25):
        super().__init__(channels, hidden=hidden, gate_max=gate_max)
        del self.rgb_to_ir

    def forward(self, rgb, ir):
        if rgb.shape != ir.shape or rgb.ndim != 4 or rgb.shape[1] != self.channels:
            raise ValueError("P3 detail correction requires aligned BCHW feature pairs")
        context = torch.cat((rgb, ir, (ir - rgb).abs()), dim=1)
        pooled = F.adaptive_avg_pool2d(context, 1)
        logits = self.filter_selector(pooled).view(rgb.shape[0], self.channels, 2, 3, 1, 1)
        weights = logits.softmax(dim=3)
        rgb_detail = (weights[:, :, 0] * self._filter_bank(rgb)).sum(dim=2)
        ir_detail = (weights[:, :, 1] * self._filter_bank(ir)).sum(dim=2)
        correction = self.ir_to_rgb(torch.cat((ir_detail, ir_detail - rgb_detail), dim=1))
        allocation = self.allocation(pooled).view(rgb.shape[0], self.channels, 2, 1, 1).softmax(dim=2)
        gate = self.gate_max * torch.tanh(self.shared_gate(context))
        injection = gate * allocation[:, :, 0] * correction
        if self.capture_diagnostics:
            dims = (1, 2, 3)
            base_rms = rgb.detach().float().square().mean(dims).sqrt().clamp_min(1e-8)
            self._last_diagnostics = {
                "gate_abs_mean": gate.detach().float().abs().mean(),
                "gate_abs_max": gate.detach().float().abs().amax(),
                "allocation_rgb": allocation[:, :, 0].detach().float().mean(),
                "correction_to_rgb_rms": (
                    injection.detach().float().square().mean(dims).sqrt() / base_rms
                ).mean(),
            }
        return rgb + injection
