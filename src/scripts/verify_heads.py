"""Verify MuZero policy, value, and reward heads on offline transitions.

Usage:
    python scripts/verify_heads.py --data data/offline_transitions.pt

Phase 1: Shape and entropy checks.
Phase 2: Value regression on n-step discounted returns.
Phase 3: Reward regression from resulting-state latents.
Phase 4: Policy entropy check after encoder updates.
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import argparse
import torch as t
import torch.nn.functional as F
from muzero.config import MuZeroConfig
from muzero.encoder import MuZeroEncoder
from muzero.heads import PolicyHead, ValueHead, RewardHead


def _as_float_obs(obs):
    if obs.dtype == t.uint8:
        return obs.float() / 255.0

    obs = obs.float()
    if obs.max() > 1.0:
        obs = obs / 255.0
    return obs


def _n_step_returns(rewards, dones, n, gamma):
    returns = t.zeros_like(rewards, dtype=t.float32)
    alive = t.ones_like(dones, dtype=t.bool)
    discount = t.ones_like(rewards, dtype=t.float32)

    for k in range(n):
        end = rewards.shape[0] - k
        if end <= 0:
            break

        valid = alive[:end]
        returns[:end] += valid.float() * discount[:end] * rewards[k:k + end]
        alive[:end] &= ~dones[k:k + end]
        discount[:end] *= gamma

    return returns


def _sample_indices(indices, batch_size):
    if indices.shape[0] <= batch_size:
        return indices
    return indices[t.randperm(indices.shape[0])[:batch_size]]


def _policy_entropy(encoder, policy, obs, batch_size, device):
    encoder.eval(); policy.eval()
    idx = _sample_indices(t.arange(obs.shape[0]), batch_size)
    obs_b = _as_float_obs(obs[idx].to(device))

    with t.no_grad():
        logits = policy(encoder(obs_b))
        probs = F.softmax(logits, dim=-1)
        entropy = -(probs * F.log_softmax(logits, dim=-1)).sum(-1).mean().item()

    encoder.train(); policy.train()
    return entropy


@t.no_grad()
def _measure_value(encoder, value_head, obs_b, targets_b):
    encoder.eval(); value_head.eval()
    logits = value_head(encoder(obs_b))
    loss = value_head.loss(logits, targets_b).item()
    mse = F.mse_loss(value_head.predict(logits), targets_b).item()
    encoder.train(); value_head.train()
    return loss, mse


@t.no_grad()
def _measure_reward(encoder, reward_head, next_obs_b, rewards_b):
    encoder.eval(); reward_head.eval()
    logits = reward_head(encoder(next_obs_b))
    loss = reward_head.loss(logits, rewards_b).item()
    mse = F.mse_loss(reward_head.predict(logits), rewards_b).item()
    encoder.train(); reward_head.train()
    return loss, mse


def _prepare_limit(obs, actions, rewards, dones, max_samples):
    if max_samples is None or max_samples >= obs.shape[0]:
        return obs, actions, rewards, dones

    obs = obs[:max_samples]
    actions = actions[:max_samples]
    rewards = rewards[:max_samples]
    dones = dones[:max_samples]
    return obs, actions, rewards, dones


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default='data/offline_transitions.pt')
    parser.add_argument('--value_epochs', type=int, default=10)
    parser.add_argument('--reward_epochs', type=int, default=5)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--eval_batch_size', type=int, default=512)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--max_samples', type=int, default=None)
    args = parser.parse_args()

    if not os.path.exists(args.data):
        raise FileNotFoundError(
            f"{args.data} not found. Generate it with scripts/collect_offline_data.py "
            "or pass --data to an existing offline transition file."
        )

    device = t.device('cuda' if t.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    data = t.load(args.data, map_location='cpu', weights_only=False)
    obs = data['obs']
    actions = data['actions']
    rewards = data['rewards'].float()
    dones = data['dones'].bool()
    obs, actions, rewards, dones = _prepare_limit(obs, actions, rewards, dones, args.max_samples)

    config = MuZeroConfig()
    returns = _n_step_returns(rewards, dones, config.td_steps, config.gamma)
    reward_indices = (~dones[:-1]).nonzero(as_tuple=True)[0]
    if reward_indices.shape[0] == 0:
        raise ValueError("Offline data has no non-terminal transitions for reward-head verification.")

    print(f"Dataset: {obs.shape[0]} transitions  |  Episodes: {dones.sum().item()}  |  "
          f"Level: {data.get('level', '?')}  |  Skip: {data.get('frame_skip', '?')}")
    print(f"Return targets: min={returns.min().item():.3f}  max={returns.max().item():.3f}")
    print(f"Reward targets: min={rewards.min().item():.3f}  max={rewards.max().item():.3f}")

    encoder = MuZeroEncoder(config).to(device)
    policy = PolicyHead(config).to(device)
    value_head = ValueHead(config).to(device)
    reward_head = RewardHead(config).to(device)

    # ---- Phase 1: Shape and entropy checks ----
    print("\n--- Phase 1: Shape and entropy checks ---")
    idx = _sample_indices(t.arange(obs.shape[0]), args.eval_batch_size)
    obs_b = _as_float_obs(obs[idx].to(device))

    with t.no_grad():
        z = encoder(obs_b)
        policy_logits = policy(z)
        value_logits = value_head(z)
        reward_logits = reward_head(z)
        value_pred = value_head.predict(value_logits)
        reward_pred = reward_head.predict(reward_logits)

    value_support_size = config.value_support_max - config.value_support_min + 1
    reward_support_size = config.reward_support_max - config.reward_support_min + 1

    assert policy_logits.shape == (idx.shape[0], config.num_actions)
    assert value_logits.shape == (idx.shape[0], value_support_size)
    assert reward_logits.shape == (idx.shape[0], reward_support_size)
    assert value_pred.shape == (idx.shape[0],)
    assert reward_pred.shape == (idx.shape[0],)

    init_entropy = _policy_entropy(encoder, policy, obs, args.eval_batch_size, device)
    print(f"Policy entropy at init: {init_entropy:.3f} nats")

    # ---- Phase 2: Value regression ----
    print("\n--- Phase 2: Value regression ---")
    eval_idx = _sample_indices(t.arange(obs.shape[0]), args.eval_batch_size)
    eval_obs = _as_float_obs(obs[eval_idx].to(device))
    eval_returns = returns[eval_idx].to(device)

    value_params = list(encoder.parameters()) + list(value_head.parameters())
    optimizer = t.optim.Adam(value_params, lr=args.lr)
    init_value_loss, init_value_mse = _measure_value(encoder, value_head, eval_obs, eval_returns)
    print(f"Initial value CE={init_value_loss:.4f}  scalar MSE={init_value_mse:.4f}")

    for epoch in range(args.value_epochs):
        perm = t.randperm(obs.shape[0])
        losses = []
        for i in range(0, obs.shape[0], args.batch_size):
            batch = perm[i:i + args.batch_size]
            obs_b = _as_float_obs(obs[batch].to(device))
            returns_b = returns[batch].to(device)

            logits = value_head(encoder(obs_b))
            loss = value_head.loss(logits, returns_b)

            optimizer.zero_grad()
            loss.backward()
            t.nn.utils.clip_grad_norm_(value_params, max_norm=0.5)
            optimizer.step()
            losses.append(loss.item())

        print(f"Epoch {epoch + 1}/{args.value_epochs}: mean CE={sum(losses) / len(losses):.4f}")

    final_value_loss, final_value_mse = _measure_value(encoder, value_head, eval_obs, eval_returns)
    print(f"Final value CE={final_value_loss:.4f}  scalar MSE={final_value_mse:.4f}")

    # ---- Phase 3: Reward regression ----
    print("\n--- Phase 3: Reward regression ---")
    eval_reward_idx = _sample_indices(reward_indices, args.eval_batch_size)
    eval_next_obs = _as_float_obs(obs[eval_reward_idx + 1].to(device))
    eval_rewards = rewards[eval_reward_idx].to(device)

    reward_params = list(encoder.parameters()) + list(reward_head.parameters())
    optimizer = t.optim.Adam(reward_params, lr=args.lr)
    init_reward_loss, init_reward_mse = _measure_reward(encoder, reward_head, eval_next_obs, eval_rewards)
    print(f"Initial reward CE={init_reward_loss:.4f}  scalar MSE={init_reward_mse:.4f}")

    for epoch in range(args.reward_epochs):
        perm = reward_indices[t.randperm(reward_indices.shape[0])]
        losses = []
        for i in range(0, perm.shape[0], args.batch_size):
            batch = perm[i:i + args.batch_size]
            next_obs_b = _as_float_obs(obs[batch + 1].to(device))
            rewards_b = rewards[batch].to(device)

            logits = reward_head(encoder(next_obs_b))
            loss = reward_head.loss(logits, rewards_b)

            optimizer.zero_grad()
            loss.backward()
            t.nn.utils.clip_grad_norm_(reward_params, max_norm=0.5)
            optimizer.step()
            losses.append(loss.item())

        print(f"Epoch {epoch + 1}/{args.reward_epochs}: mean CE={sum(losses) / len(losses):.4f}")

    final_reward_loss, final_reward_mse = _measure_reward(encoder, reward_head, eval_next_obs, eval_rewards)
    print(f"Final reward CE={final_reward_loss:.4f}  scalar MSE={final_reward_mse:.4f}")

    # ---- Phase 4: Policy entropy after encoder training ----
    print("\n--- Phase 4: Policy entropy post-training ---")
    final_entropy = _policy_entropy(encoder, policy, obs, args.eval_batch_size, device)
    print(f"Policy entropy post-training: {final_entropy:.3f} nats")

    criteria = {
        "policy entropy at init > 2.5 nats": init_entropy > 2.5,
        "value scalar MSE decreases >50%": final_value_mse < init_value_mse * 0.5,
        "reward scalar MSE decreases >50%": final_reward_mse < init_reward_mse * 0.5,
        "policy entropy post-training > 1.5 nats": final_entropy > 1.5,
    }

    print()
    for label, passed in criteria.items():
        print(f"  {'PASS' if passed else 'FAIL'}  {label}")

    if all(criteria.values()):
        print("\nAll checks passed.")
    else:
        raise SystemExit("\nSome checks failed; inspect the loss curves before proceeding.")


if __name__ == '__main__':
    main()
