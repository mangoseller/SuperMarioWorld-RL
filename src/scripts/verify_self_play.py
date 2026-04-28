import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from dataclasses import replace
import torch as t
from muzero.config import MuZeroConfig
from muzero.mcts import SearchOutput
from muzero.replay_buffer import DualPriorityReplayBuffer
from muzero.self_play import (
    BehaviourProfile,
    SelfPlayProcessGroup,
    _td_info,
    assign_profiles,
    collect_episode,
    sample_action,
)


class FakeEnv:
    def __init__(self, length=6):
        self.length = length
        self.idx = 0

    def reset(self):
        self.idx = 0
        return self._obs(), {"xpos": 0}

    def step(self, action):
        self.idx += 1
        done = self.idx >= self.length
        reward = 1.0 if action == 1 else 0.0
        return self._obs(), reward, done, {"xpos": self.idx * 8}

    def close(self):
        pass

    def _obs(self):
        return t.full((4, 84, 84), self.idx / max(self.length, 1), dtype=t.float32)


class FakeSearch:
    def __init__(self, config):
        self.config = config

    def run(self, model, obs=None, state=None, invalid_actions=None,
            num_simulations=None, max_depth=None, generator=None,
            root_dirichlet_alpha=None, root_dirichlet_fraction=0.0):
        policy = t.tensor([0.1, 0.8, 0.1], dtype=t.float32)
        if root_dirichlet_fraction > 0:
            policy = (1.0 - root_dirichlet_fraction) * policy
            policy += root_dirichlet_fraction * t.full_like(policy, 1.0 / policy.shape[0])
        return SearchOutput(
            action=1,
            action_weights=policy / policy.sum(),
            visit_counts=t.tensor([1.0, 4.0, 1.0]),
            q_values=t.tensor([0.0, 1.0, 0.0]),
            root_value=0.0,
        )


class FakeNetwork:
    def load_weights(self, weights):
        pass


def fake_env_factory(level, config):
    return FakeEnv(length=config.self_play_max_episode_steps)


def fake_network_factory(config, device):
    return FakeNetwork()


def fake_search_factory(config):
    return FakeSearch(config)


def _make_config():
    return replace(
        MuZeroConfig(),
        num_actions=3,
        replay_capacity=100,
        frontier_capacity=50,
        unroll_steps=3,
        td_steps=2,
        self_play_workers=2,
        self_play_max_episode_steps=5,
        self_play_trajectory_queue_size=8,
        mcts_num_simulations=1,
        mcts_max_depth=1,
        self_play_search_batch_size=4,
    )


def _verify_profiles(config):
    expected = [
        "standard",
        "aggressive_dirichlet",
        "high_temperature",
        "epsilon_greedy",
        "standard",
        "aggressive_dirichlet",
        "high_temperature",
        "epsilon_greedy",
        "standard",
        "aggressive_dirichlet",
    ]
    got = [profile.name for profile in assign_profiles(10, config)]
    assert got == expected


def _verify_balanced_level_assignment():
    from muzero.self_play import _choose_level

    levels = ["a", "b", "c", "d"]
    assigned = [
        _choose_level(levels, [0.1, 0.2, 0.3, 0.4], "a", worker_id=i)
        for i in range(32)
    ]
    assert {level: assigned.count(level) for level in levels} == {
        "a": 8,
        "b": 8,
        "c": 8,
        "d": 8,
    }


def _verify_action_sampling():
    policy = t.tensor([0.05, 0.9, 0.05])
    assert sample_action(policy, temperature=0.0) == 1

    generator = t.Generator().manual_seed(1)
    counts = t.zeros(3)
    for _ in range(200):
        counts[sample_action(policy, temperature=2.0, generator=generator)] += 1
    assert counts[0] > 0 and counts[2] > 0

    generator = t.Generator().manual_seed(2)
    counts.zero_()
    for _ in range(200):
        counts[sample_action(policy, epsilon=1.0, generator=generator)] += 1
    assert (counts > 0).all()

    action = sample_action(t.zeros(3), generator=t.Generator().manual_seed(3))
    assert 0 <= action < 3


def _verify_collect_episode(config):
    env = FakeEnv(length=5)
    profile = BehaviourProfile("standard", 0.3, 0.25, 1.0)
    trajectory = collect_episode(
        env,
        FakeNetwork(),
        FakeSearch(config),
        config,
        profile,
        level="fake",
        generator=t.Generator().manual_seed(0),
    )

    assert trajectory.obs.shape == (6, 4, 84, 84)
    assert trajectory.actions.shape == (5,)
    assert trajectory.rewards.shape == (5,)
    assert trajectory.dones.shape == (5,)
    assert trajectory.policy_targets.shape == (5, config.num_actions)
    assert t.allclose(trajectory.policy_targets.sum(-1), t.ones(5))
    assert trajectory.x_max == 40.0
    assert trajectory.worker_type == "standard"
    assert _td_info({"xpos": t.tensor([7])})["xpos"].item() == 7


def _verify_process_group(config):
    group = SelfPlayProcessGroup(
        config,
        env_factory=fake_env_factory,
        levels=["fake"],
        device="cpu",
        network_factory=fake_network_factory,
        search_factory=fake_search_factory,
    )
    group.start()
    replay = DualPriorityReplayBuffer(config)
    try:
        received = []
        for _ in range(2):
            msg = group.get_trajectory(timeout=10)
            assert msg is not None
            assert msg["type"] == "trajectory"
            replay.add_trajectory(msg["trajectory"])
            received.append(msg["worker_id"])
        assert len(set(received)) >= 1
        assert replay.num_trajectories == 2
        batch = replay.sample_batch(2)
        assert batch.obs.shape[1] == config.unroll_steps + 1
    finally:
        group.close()
    assert all(not process.is_alive() for process in group.processes)


def _verify_batched_process_group(config):
    group = SelfPlayProcessGroup(
        config,
        env_factory=fake_env_factory,
        levels=["fake"],
        device="cpu",
    )
    group.start()
    try:
        group.broadcast_weights({}, 0, active_levels=["fake"], level_weights=[1.0])
        msg = group.get_trajectory(timeout=10)
        assert msg is not None
        assert msg["type"] == "trajectory"
        assert msg["trajectory"].policy_targets.shape[-1] == config.num_actions
    finally:
        group.close()
    assert all(not process.is_alive() for process in group.processes)
    assert group.search_process is None or not group.search_process.is_alive()


def main():
    config = _make_config()
    _verify_profiles(config)
    _verify_balanced_level_assignment()
    _verify_action_sampling()
    _verify_collect_episode(config)
    _verify_process_group(config)
    _verify_batched_process_group(config)
    print("Self-play verification passed.")


if __name__ == '__main__':
    main()
