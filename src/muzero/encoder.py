import copy
import math
import torch as t
import torch.nn as nn
from model_components import ResidualBlock


def _init_weights(module):
    if isinstance(module, (nn.Conv2d, nn.Linear)):
        nn.init.orthogonal_(module.weight, gain=math.sqrt(2))
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, nn.BatchNorm1d):
        nn.init.ones_(module.weight)
        nn.init.zeros_(module.bias)


class MuZeroEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        widths = config.encoder_widths

        layers = []
        c_in = 4
        for c_out in widths:
            layers += [
                nn.Conv2d(c_in, c_out, kernel_size=3, stride=2, padding=1),
                nn.GroupNorm(8, c_out),
                nn.SiLU(),
            ]
            c_in = c_out
        self.stem = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool2d(config.latent_spatial)

        if widths[-1] != config.latent_channels:
            self.channel_proj = nn.Conv2d(widths[-1], config.latent_channels, kernel_size=1)
        else:
            self.channel_proj = None

        self.res_blocks = nn.Sequential(*[
            ResidualBlock(config.latent_channels)
            for _ in range(config.encoder_res_blocks)
        ])

        self.apply(_init_weights)

    def forward(self, x):
        x = self.stem(x)
        x = self.pool(x)
        if self.channel_proj is not None:
            x = self.channel_proj(x)
        return self.res_blocks(x)


class EMAEncoder:
    def __init__(self, online, decay):
        self.online = online
        self.target = copy.deepcopy(online)
        self.decay = decay
        for p in self.target.parameters():
            p.requires_grad_(False)

    def to(self, device):
        self.target = self.target.to(device)
        return self

    @t.no_grad()
    def update(self):
        for tp, op in zip(self.target.parameters(), self.online.parameters()):
            tp.data.mul_(self.decay).add_(op.data, alpha=1 - self.decay)

    def encode_target(self, x):
        return self.target(x)
