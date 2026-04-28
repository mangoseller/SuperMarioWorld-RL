import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from dataclasses import replace
import torch as t
from muzero.config import MuZeroConfig
from muzero.replay_buffer import DualPriorityReplayBuffer, Trajectory
from utils import log_muzero_metrics


def _trajectory(config):
    length = config.unroll_steps
    obs = t.randint(0, 256, (length + 1, 4, 84, 84), dtype=t.uint8)
    return Trajectory(
        obs=obs,
        actions=t.zeros(length, dtype=t.long),
        rewards=t.ones(length),
        dones=t.zeros(length, dtype=t.bool),
        policy_targets=t.full((length, config.num_actions), 1.0 / config.num_actions),
        value_targets=t.zeros(length),
        level="L1",
        x_max=10,
        value_errors=t.ones(length),
    )


def main():
    config = replace(
        MuZeroConfig(),
        replay_capacity=20,
        frontier_capacity=10,
        unroll_steps=5,
    )
    replay = DualPriorityReplayBuffer(config)
    replay.add_trajectory(_trajectory(config))

    metrics = log_muzero_metrics(
        tracking={
            "episode_returns": [1.0, 3.0],
            "episode_lengths": [10, 20],
            "x_max": [8, 16],
            "episodes": 2,
            "env_steps": 30,
            "gradient_steps": 4,
        },
        diagnostics={
            "total_loss": 4.0,
            "dynamics_loss": 1.0,
            "reward_loss": 0.5,
            "value_loss": 1.5,
            "policy_loss": 1.0,
        },
        replay=replay,
        self_play={"queue_depth": 2, "mean_policy_entropy": 1.2},
        reanalyse={"updates": 3, "queue_depth": 1},
        search={"mean_depth": 2.5, "mean_root_value": 0.4},
        config=config,
        step=10,
    )

    required = [
        "train/mean_episode_return",
        "loss/dynamics",
        "loss/reward",
        "loss/value",
        "loss/policy",
        "replay/transitions",
        "replay/frontier_trajectories",
        "self_play/queue_depth",
        "reanalyse/updates",
        "search/mean_depth",
        "hyperparams/mcts_num_simulations",
    ]
    for key in required:
        assert key in metrics

    obsolete = [
        "loss/pixel_control",
        "hyperparams/entropy_coef",
        "hyperparams/value_coef",
    ]
    for key in obsolete:
        assert key not in metrics

    print("MuZero logging verification passed.")


if __name__ == '__main__':
    main()
