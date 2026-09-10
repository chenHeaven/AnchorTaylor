#!/usr/bin/env python3
"""Fast CPU smoke test for the two AnchorTaylor fusion modules."""

import torch

from ultralytics.nn.modules.anchor_fusion import (
    P2BidirectionalAdaptiveInjection,
    PretrainedAnchorP3Fusion,
)


def main() -> None:
    torch.manual_seed(0)
    rgb_p2, ir_p2 = torch.randn(2, 32, 16, 16), torch.randn(2, 32, 16, 16)
    p2 = P2BidirectionalAdaptiveInjection(32, hidden=16, gate_max=0.10).eval()
    with torch.no_grad():
        rgb_out, ir_out = p2(rgb_p2, ir_p2)
    assert torch.equal(rgb_out, rgb_p2)
    assert torch.equal(ir_out, ir_p2)

    rgb_p3, ir_p3 = torch.randn(2, 64, 8, 8), torch.randn(2, 64, 8, 8)
    p3 = PretrainedAnchorP3Fusion(
        64, mode="residual", hidden=16, gate_max=0.25,
        order=3, recurrence="additive", anchor_modality="rgb"
    ).eval()
    with torch.no_grad():
        fused = p3(rgb_p3, ir_p3)
    assert torch.equal(fused, rgb_p3)
    print("PASS: P2 and P3 modules preserve their anchored inputs at initialization")


if __name__ == "__main__":
    main()
