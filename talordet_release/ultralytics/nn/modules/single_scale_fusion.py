"""Parameter-free RGB passthrough for single-location high-order fusion tests."""
from torch import nn


class RGBPassthrough(nn.Module):
    """Keep the RGB feature without cross-modal aggregation at an inactive scale."""

    def forward(self, rgb, ir):
        return rgb
