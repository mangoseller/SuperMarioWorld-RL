import math
import torch as t
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


def _init_weights(module):
    if isinstance(module, (nn.Conv2d, nn.Linear)):
        nn.init.orthogonal_(module.weight, gain=math.sqrt(2))
        if module.bias is not None:
            nn.init.zeros_(module.bias)


def scalar_to_support(x, support_min, support_max):
    support_size = support_max - support_min + 1
    x = x.float().clamp(support_min, support_max)
    x_shifted = x - support_min
    x_floor = x_shifted.floor().long().clamp(0, support_size - 2)
    x_frac = x_shifted - x_floor.float()

    target = t.zeros(*x.shape, support_size, device=x.device, dtype=x.dtype)
    target.scatter_(-1, x_floor.unsqueeze(-1), (1 - x_frac).unsqueeze(-1))
    target.scatter_(-1, (x_floor + 1).unsqueeze(-1), x_frac.unsqueeze(-1))
    return target


def support_to_scalar(logits, support_min, support_max):
    support = t.arange(support_min, support_max + 1, dtype=logits.dtype, device=logits.device)
    return (F.softmax(logits, dim=-1) * support).sum(-1)


class PolicyHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        flat_dim = config.latent_channels * config.latent_spatial * config.latent_spatial
        self.net = nn.Sequential(
            nn.Linear(flat_dim, config.head_hidden_dim),
            nn.ReLU(),
        )
        self.out = nn.Linear(config.head_hidden_dim, config.num_actions)

        self.apply(_init_weights)
        nn.init.orthogonal_(self.out.weight, gain=0.01)
        nn.init.zeros_(self.out.bias)

    def forward(self, z):
        if z.dim() == 4:
            z = rearrange(z, 'b c h w -> b (c h w)')
        return self.out(self.net(z))

    def loss(self, logits, target_probs):
        log_probs = F.log_softmax(logits, dim=-1)
        return -(target_probs * log_probs).sum(-1).mean()


class ValueHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        flat_dim = config.latent_channels * config.latent_spatial * config.latent_spatial
        support_size = config.value_support_max - config.value_support_min + 1

        self.support_min = config.value_support_min
        self.support_max = config.value_support_max

        self.net = nn.Sequential(
            nn.Linear(flat_dim, config.head_hidden_dim),
            nn.ReLU(),
        )
        self.out = nn.Linear(config.head_hidden_dim, support_size)

        self.apply(_init_weights)
        nn.init.orthogonal_(self.out.weight, gain=1.0)
        nn.init.zeros_(self.out.bias)

    def forward(self, z):
        if z.dim() == 4:
            z = rearrange(z, 'b c h w -> b (c h w)')
        return self.out(self.net(z))

    def predict(self, logits):
        return support_to_scalar(logits, self.support_min, self.support_max)

    def loss(self, logits, target_scalars):
        target = scalar_to_support(target_scalars, self.support_min, self.support_max)
        return -(target * F.log_softmax(logits, dim=-1)).sum(-1).mean()


class RewardHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        flat_dim = config.latent_channels * config.latent_spatial * config.latent_spatial
        support_size = config.reward_support_max - config.reward_support_min + 1

        self.support_min = config.reward_support_min
        self.support_max = config.reward_support_max

        self.net = nn.Sequential(
            nn.Linear(flat_dim, 256),
            nn.ReLU(),
        )
        self.out = nn.Linear(256, support_size)

        self.apply(_init_weights)
        nn.init.orthogonal_(self.out.weight, gain=1.0)
        nn.init.zeros_(self.out.bias)

    def forward(self, z):
        if z.dim() == 4:
            z = rearrange(z, 'b c h w -> b (c h w)')
        return self.out(self.net(z))

    def predict(self, logits):
        return support_to_scalar(logits, self.support_min, self.support_max)

    def loss(self, logits, target_scalars):
        target = scalar_to_support(target_scalars, self.support_min, self.support_max)
        return -(target * F.log_softmax(logits, dim=-1)).sum(-1).mean()
