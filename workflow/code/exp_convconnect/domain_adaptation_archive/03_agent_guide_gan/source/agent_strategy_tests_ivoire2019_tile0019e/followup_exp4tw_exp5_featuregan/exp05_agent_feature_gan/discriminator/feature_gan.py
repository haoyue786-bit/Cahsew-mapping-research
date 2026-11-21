"""Fixed spatial Stage4 feature discriminator for Exp5."""
import torch
from torch import nn


class Stage4PatchDiscriminator(nn.Module):
    def __init__(self, in_channels=768, reduction_channels=256, hidden_channels=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, reduction_channels, 1), nn.LeakyReLU(.2, inplace=True),
            nn.Conv2d(reduction_channels, hidden_channels, 3, padding=1), nn.LeakyReLU(.2, inplace=True),
            nn.Conv2d(hidden_channels, hidden_channels // 2, 3, padding=1), nn.LeakyReLU(.2, inplace=True),
            nn.Conv2d(hidden_channels // 2, 1, 3, padding=1),
        )

    def forward(self, stage4_map):
        if stage4_map.ndim != 4 or stage4_map.shape[1] != 768:
            raise ValueError(f"expected B,768,H,W Stage4 map, got {tuple(stage4_map.shape)}")
        return self.net(stage4_map)
