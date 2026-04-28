import math
import torch as t
import torch.nn as nn
import torch.nn.functional as F
from einops import repeat, rearrange
from model_components import ResidualBlock


def _init_weights(module):
    if isinstance(module, (nn.Conv2d, nn.Linear)):
        nn.init.orthogonal_(module.weight, gain=math.sqrt(2))
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, nn.BatchNorm1d):
        nn.init.ones_(module.weight)
        nn.init.zeros_(module.bias)


class DynamicsModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        lc = config.latent_channels
        dc = config.dynamics_channels
        na = config.num_actions

        self.num_actions = na

        self.input_proj = nn.Sequential(
            nn.Conv2d(lc + na, dc, kernel_size=1),
            nn.GroupNorm(8, dc),
            nn.SiLU(),
        )
        self.res_blocks = nn.Sequential(*[
            ResidualBlock(dc)
            for _ in range(config.dynamics_res_blocks)
        ])
        self.output_proj = nn.Sequential(
            nn.Conv2d(dc, lc, kernel_size=1),
            nn.GroupNorm(8, lc),
            nn.SiLU(),
        )

        self.apply(_init_weights)

    def forward(self, z, a):
        _, _, h, w = z.shape
        a_onehot = F.one_hot(a.long(), num_classes=self.num_actions).float()
        a_tiled = repeat(a_onehot, 'b c -> b c h w', h=h, w=w)
        x = t.cat([z, a_tiled], dim=1)
        x = self.input_proj(x)
        x = self.res_blocks(x)
        return self.output_proj(x)


class DynamicsHead(nn.Module):
    """BYOL-style projector + predictor for latent prediction loss.

    Shared projector is applied to both online (predicted) and target branches.
    Predictor is applied to the online branch only. The asymmetry plus stop-grad
    on the target side prevents representational collapse without a contrastive term.
    BatchNorm in the projector is load-bearing for BYOL-style collapse prevention;
    do not replace with GroupNorm here."""

    def __init__(self, config):
        super().__init__()
        flat_dim = config.latent_channels * config.latent_spatial * config.latent_spatial
        pd = config.proj_dim

        self.projector = nn.Sequential(
            nn.Linear(flat_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, pd),
        )
        self.predictor = nn.Sequential(
            nn.Linear(pd, pd),
            nn.BatchNorm1d(pd),
            nn.ReLU(),
            nn.Linear(pd, pd),
        )

        self.apply(_init_weights)

    def project(self, z):
        return self.projector(rearrange(z, 'b c h w -> b (c h w)'))

    def loss(self, z_pred, z_target):
        pred_out = self.predictor(self.project(z_pred))

        with t.no_grad():
            target_proj = self.project(z_target)

        pred_out = F.normalize(pred_out, dim=-1)
        target_proj = F.normalize(target_proj, dim=-1)
        return -(pred_out * target_proj).sum(dim=-1).mean()
