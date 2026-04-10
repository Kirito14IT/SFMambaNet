import torch.nn as nn
from mamba_ssm.modules.mamba_simple import Mamba
from .mamba_block import Mamba_Block


class BiCSM(nn.Module):
    def __init__(self, channels, points, out_channels=None, d_state=8):
        super(BiCSM, self).__init__()
        if not out_channels:
            out_channels = channels
        self.shot_cut = None
        if out_channels != channels:
            self.shot_cut = nn.Conv2d(channels, out_channels, kernel_size=1)

        self.conv1 = nn.Sequential(
            nn.InstanceNorm2d(channels, eps=1e-3),
            nn.BatchNorm2d(channels),
            nn.ReLU(),
            nn.Conv2d(channels, out_channels, kernel_size=1),
        )
        self.norm = nn.LayerNorm(points)
        self.SAM = Mamba(points, d_state)
        self.conv3 = nn.Sequential(
            nn.InstanceNorm2d(out_channels, eps=1e-3),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=1)
        )

    def forward(self, x):
        out = self.conv1(x).squeeze(-1)
        out_n = self.norm(out)
        out = out + self.SAM(out_n)
        out = self.conv3(out.unsqueeze(-1))
        if self.shot_cut:
            out = out + self.shot_cut(x)
        else:
            out = out + x
        return out
