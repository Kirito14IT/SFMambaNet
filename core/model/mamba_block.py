import torch.nn as nn
from mamba_ssm.modules.mamba_simple import Mamba


class Mamba_Block(nn.Module):
    def __init__(self, channels, d_state):
        super(Mamba_Block, self).__init__()
        self.norm = nn.LayerNorm(channels)
        self.mamba = Mamba(d_model=channels, d_state=d_state)

    def forward(self, x):
        data = x
        x = self.norm(x.transpose(1, 2).squeeze(-1))
        out = self.mamba(x)
        out = self.norm(out + data.transpose(1, 2).squeeze(-1))
        return out.transpose(1, 2).unsqueeze(-1)
