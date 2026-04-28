"""Train 1-step dynamics on offline transitions and verify latent variance.

Usage:
    python scripts/verify_dynamics.py --data data/offline_transitions.pt

Phase 1: Train encoder + dynamics + BYOL head on next-latent prediction.
Phase 2: Variance check; confirm per-coordinate latent variance is roughly uniform.

Pass criteria:
    max/min variance ratio < 100
    > 80% of latent dims above 1% of max variance
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import argparse
import torch as t
import torch.nn as nn
from einops import rearrange
from muzero.config import MuZeroConfig
from muzero.encoder import MuZeroEncoder, EMAEncoder
from muzero.dynamics import DynamicsModel, DynamicsHead


def _build_eval_batch(obs, actions, dones, size, device):
    """Sample a fixed batch of non-terminal transitions for loss tracking."""
    valid = (~dones[:-1]).nonzero(as_tuple=True)[0]
    idx = valid[t.randperm(len(valid))[:size]]
    return (
        obs[idx].to(device).float() / 255.0,
        actions[idx].to(device),
        obs[idx + 1].to(device).float() / 255.0,
    )


@t.no_grad()
def _measure_losses(encoder, dynamics, ema, head, obs_b, acts_b, nobs_b):
    """Returns (dynamics_loss, identity_baseline_loss) on a fixed batch."""
    encoder.eval(); dynamics.eval(); head.eval()
    z = encoder(obs_b)
    z_pred = dynamics(z, acts_b)
    z_target = ema.encode_target(nobs_b)
    dyn = head.loss(z_pred, z_target).item()
    idn = head.loss(z, z_target).item()      # identity: z directly, no dynamics
    encoder.train(); dynamics.train(); head.train()
    return dyn, idn


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default='data/offline_transitions.pt')
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--lr', type=float, default=3e-4)
    args = parser.parse_args()

    device = t.device('cuda' if t.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    data = t.load(args.data, map_location='cpu', weights_only=False)
    obs     = data['obs']      # (N, 4, 84, 84) uint8
    actions = data['actions']  # (N,)
    dones   = data['dones']    # (N,) bool
    N = obs.shape[0]
    print(f"Dataset: {N} transitions  |  Episodes: {dones.sum().item()}  |  "
          f"Level: {data.get('level', '?')}  |  Skip: {data.get('frame_skip', '?')}")

    config = MuZeroConfig()
    encoder  = MuZeroEncoder(config).to(device)
    ema      = EMAEncoder(encoder, decay=config.ema_decay).to(device)
    dynamics = DynamicsModel(config).to(device)
    head     = DynamicsHead(config).to(device)

    params = list(encoder.parameters()) + list(dynamics.parameters()) + list(head.parameters())
    optimizer = t.optim.Adam(params, lr=args.lr)

    eval_obs, eval_acts, eval_nobs = _build_eval_batch(obs, actions, dones, size=512, device=device)

    # ---- Phase 1: Training ----
    print("\n--- Phase 1: Training ---")
    init_dyn, init_id = _measure_losses(encoder, dynamics, ema, head, eval_obs, eval_acts, eval_nobs)
    print(f"Initial: dynamics={init_dyn:.4f}  identity baseline={init_id:.4f}")

    step = 0
    for epoch in range(args.epochs):
        perm = t.randperm(N - 1)   # -1 because next_obs[i] = obs[i+1]
        epoch_losses = []

        for i in range(0, N - 1, args.batch_size):
            idx = perm[i:i + args.batch_size]

            obs_b   = obs[idx].to(device).float() / 255.0
            acts_b  = actions[idx].to(device)
            nobs_b  = obs[idx + 1].to(device).float() / 255.0
            dones_b = dones[idx].to(device)

            # Skip transitions that cross episode boundaries
            valid = ~dones_b
            if valid.sum() == 0:
                continue

            z      = encoder(obs_b[valid])
            z_pred = dynamics(z, acts_b[valid])
            z_target = ema.encode_target(nobs_b[valid])
            loss = head.loss(z_pred, z_target)

            optimizer.zero_grad()
            loss.backward()
            t.nn.utils.clip_grad_norm_(params, max_norm=0.5)
            optimizer.step()
            ema.update()

            epoch_losses.append(loss.item())
            step += 1
            if step % 500 == 0:
                print(f"  step {step}: loss={loss.item():.4f}")

        mean_loss = sum(epoch_losses) / len(epoch_losses) if epoch_losses else float('nan')
        print(f"Epoch {epoch + 1}/{args.epochs}: mean loss={mean_loss:.4f}")

    final_dyn, final_id = _measure_losses(encoder, dynamics, ema, head, eval_obs, eval_acts, eval_nobs)
    print(f"\nFinal: dynamics={final_dyn:.4f}  identity baseline={final_id:.4f}")
    print(f"Dynamics improvement over identity: {final_id - final_dyn:+.4f}")

    # ---- Phase 2: Variance check ----
    print("\n--- Phase 2: Latent variance check ---")
    encoder.eval()
    all_z = []

    with t.no_grad():
        for i in range(0, N, args.batch_size):
            z = encoder(obs[i:i + args.batch_size].to(device).float() / 255.0)
            all_z.append(rearrange(z, 'b c h w -> b (c h w)').cpu())

    Z = t.cat(all_z, dim=0).float()   # (N, latent_channels * latent_spatial^2)
    var = Z.var(dim=0)

    max_var  = var.max().item()
    min_var  = var.min().item()
    ratio    = max_var / (min_var + 1e-12)
    frac_pct = (var > 0.01 * max_var).float().mean().item() * 100

    print(f"Variance: max={max_var:.4f}  min={min_var:.6f}  ratio={ratio:.1f}")
    print(f"Dims above 1% of max: {frac_pct:.1f}%")

    criteria = {
        "max/min ratio < 100": ratio < 100,
        ">80% dims active":    frac_pct > 80.0,
        "dynamics beats identity": final_dyn < final_id,
    }
    print()
    for label, passed in criteria.items():
        print(f"  {'PASS' if passed else 'FAIL'}  {label}")

    if all(criteria.values()):
        print("\nAll checks passed.")
    else:
        print("\nSome checks failed; inspect the loss curves and latent distributions before proceeding.")


if __name__ == '__main__':
    main()
