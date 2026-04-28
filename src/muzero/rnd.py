import math
import torch as t
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


class _RunningMeanStd:
    def __init__(self):
        self.mean = 0.0
        self.var = 1.0
        self.count = 1e-4

    def update(self, x):
        x = x.detach().cpu().float().flatten()
        if x.numel() == 0:
            return
        n = float(x.numel())
        batch_mean = float(x.mean())
        batch_var = float(x.var(unbiased=False)) if x.numel() > 1 else 0.0
        delta = batch_mean - self.mean
        total = self.count + n
        self.mean += delta * n / total
        self.var = (
            self.var * self.count + batch_var * n + delta ** 2 * self.count * n / total
        ) / total
        self.count = total

    def normalize(self, x, clip=5.0):
        std = max(self.var ** 0.5, 1e-8)
        return ((x.float() - self.mean) / std).clamp(-clip, clip)

    def state_dict(self):
        return {
            "mean": self.mean,
            "var": self.var,
            "count": self.count,
        }

    def load_state_dict(self, state):
        self.mean = float(state.get("mean", 0.0))
        self.var = float(state.get("var", 1.0))
        self.count = float(state.get("count", 1e-4))


def _init_weights(module):
    if isinstance(module, (nn.Conv2d, nn.Linear)):
        nn.init.orthogonal_(module.weight, gain=math.sqrt(2))
        if module.bias is not None:
            nn.init.zeros_(module.bias)


class _RNDNet(nn.Module):
    def __init__(self, config):
        super().__init__()
        c = config.rnd_channels
        self.cnn = nn.Sequential(
            nn.Conv2d(4, c, kernel_size=8, stride=4),
            nn.LeakyReLU(),
            nn.Conv2d(c, c * 2, kernel_size=4, stride=2),
            nn.LeakyReLU(),
            nn.Conv2d(c * 2, c * 2, kernel_size=3, stride=1),
            nn.LeakyReLU(),
        )
        flat_dim = (c * 2) * 7 * 7
        self.head = nn.Linear(flat_dim, config.rnd_embedding_dim)
        self.apply(_init_weights)

    def forward(self, x):
        return self.head(rearrange(self.cnn(x), 'b c h w -> b (c h w)'))


class RNDModule:
    def __init__(self, config, device):
        self.config = config
        self.device = t.device(device)
        self.target = _RNDNet(config).to(self.device)
        self.predictor = _RNDNet(config).to(self.device)
        for p in self.target.parameters():
            p.requires_grad_(False)
        self.target.eval()
        self.predictor.eval()
        self._optimizer = t.optim.Adam(self.predictor.parameters(), lr=1e-4)
        self._stats = _RunningMeanStd()

    def state_dict(self):
        return {
            "target": self.target.state_dict(),
            "predictor": self.predictor.state_dict(),
            "optimizer": self._optimizer.state_dict(),
            "stats": self._stats.state_dict(),
        }

    def load_state_dict(self, state):
        self.target.load_state_dict(state["target"])
        self.predictor.load_state_dict(state["predictor"])
        self._optimizer.load_state_dict(state["optimizer"])
        self._stats.load_state_dict(state.get("stats", {}))
        self.target.eval()
        self.predictor.eval()

    def current_coef(self, grad_step):
        if grad_step >= self.config.rnd_anneal_steps:
            return 0.0
        frac = grad_step / self.config.rnd_anneal_steps
        return self.config.rnd_coef_start * (1.0 - frac) + self.config.rnd_coef_end * frac

    @t.no_grad()
    def compute_intrinsic(self, obs_uint8, grad_step):
        coef = self.current_coef(grad_step)
        if coef == 0.0:
            return t.zeros(obs_uint8.shape[0])
        obs = obs_uint8.to(self.device).float() / 255.0
        pred = self.predictor(obs)
        tgt = self.target(obs)
        raw = (pred - tgt).pow(2).mean(-1)
        self._stats.update(raw)
        normalised = self._stats.normalize(raw, clip=self.config.rnd_clip)
        return (coef * normalised.clamp_min(0.0)).cpu()

    def train_step(self, obs_batch):
        if obs_batch.dtype == t.uint8:
            obs = obs_batch.to(self.device).float() / 255.0
        else:
            obs = obs_batch.to(self.device).float()
        self.predictor.train()
        self._optimizer.zero_grad(set_to_none=True)
        with t.no_grad():
            tgt = self.target(obs)
        pred = self.predictor(obs)
        loss = F.mse_loss(pred, tgt)
        loss.backward()
        self._optimizer.step()
        self.predictor.eval()
        return float(loss.item())
