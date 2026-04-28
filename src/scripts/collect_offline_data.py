import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import argparse
import retro
import torch as t
from torchrl.envs import TransformedEnv
from torchrl.envs.transforms import UnsqueezeTransform
from environment import _wrap_env, MARIO_ACTIONS
from utils import get_torch_compatible_actions

# Skip=3 to match training. Use rgb_array to avoid opening a display window.
_FRAME_SKIP = 3


def _make_env(level):
    raw = retro.make('SuperMarioWorld-Snes', state=level, render_mode='rgb_array')
    wrapped = _wrap_env(raw, skip=_FRAME_SKIP)
    return TransformedEnv(
        wrapped,
        UnsqueezeTransform(dim=0, allow_positive_dim=True,
                           in_keys=["observation", "reward", "done", "terminated"])
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--steps', type=int, default=50_000)
    parser.add_argument('--level', type=str, default='YoshiIsland2')
    parser.add_argument('--output', type=str, default='data/offline_transitions.pt')
    args = parser.parse_args()

    output_dir = os.path.dirname(args.output)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    env = _make_env(args.level)
    num_actions = len(MARIO_ACTIONS)

    obs_list, action_list, reward_list, done_list = [], [], [], []

    td = env.reset()
    current_obs = td["observation"].squeeze(0)   # (4, 84, 84) float32
    print_interval = 5000

    for i in range(args.steps):
        obs_uint8 = (current_obs * 255).to(t.uint8)

        action = t.randint(0, num_actions, (1,))
        td["action"] = get_torch_compatible_actions(action)
        td = env.step(td)

        next_obs = td["next"]["observation"].squeeze(0)
        reward = float(td["next"]["reward"].squeeze().item())
        done = bool(td["next"]["done"].squeeze().item())

        obs_list.append(obs_uint8.cpu())
        action_list.append(int(action.item()))
        reward_list.append(reward)
        done_list.append(done)

        if done:
            td = env.reset()
            current_obs = td["observation"].squeeze(0)
        else:
            current_obs = next_obs

        if (i + 1) % print_interval == 0:
            episodes = sum(done_list)
            print(f"Collected {i + 1}/{args.steps} transitions  ({episodes} episodes)")

    env.close()

    # Store obs only; next_obs[i] = obs[i+1] where done[i]=False (saves ~50% space).
    # Terminal transitions are masked out during training anyway.
    data = {
        'obs':     t.stack(obs_list),                              # (N, 4, 84, 84) uint8
        'actions': t.tensor(action_list, dtype=t.int64),           # (N,)
        'rewards': t.tensor(reward_list, dtype=t.float32),         # (N,)
        'dones':   t.tensor(done_list, dtype=t.bool),              # (N,)
        'level':   args.level,
        'frame_skip': _FRAME_SKIP,
    }

    t.save(data, args.output)
    obs_bytes = data['obs'].nbytes
    print(f"\nSaved {args.steps} transitions to {args.output}")
    print(f"Obs tensor: {obs_bytes / 1e9:.2f} GB  |  Episodes: {sum(done_list)}")


if __name__ == '__main__':
    main()
