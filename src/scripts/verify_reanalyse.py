import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from dataclasses import replace
import torch as t
from muzero.config import MuZeroConfig
from muzero.reanalyse import (
    CurrentNetworkTargetComputer,
    DeterministicTargetComputer,
    ReanalyseProcess,
    apply_reanalyse_update,
)
from muzero.replay_buffer import DualPriorityReplayBuffer, ReplayBatch, Trajectory


class _FakeEncoder:
    def __call__(self, obs):
        return t.zeros(obs.shape[0], 1)


class _FakePolicy:
    def __init__(self, num_actions):
        self.num_actions = num_actions

    def __call__(self, z):
        return t.zeros(z.shape[0], self.num_actions)


class _FakeValue:
    def __call__(self, z):
        return t.zeros(z.shape[0], 1)

    def predict(self, logits):
        return t.full((logits.shape[0],), 10.0)


class _FakeNetwork:
    def __init__(self, num_actions):
        self.encoder = _FakeEncoder()
        self.policy = _FakePolicy(num_actions)
        self.value = _FakeValue()


def _trajectory(config):
    length = config.unroll_steps + config.td_steps
    obs = t.randint(0, 256, (length + 1, 4, 84, 84), dtype=t.uint8)
    actions = t.arange(length, dtype=t.long) % config.num_actions
    rewards = t.arange(1, length + 1, dtype=t.float32)
    dones = t.zeros(length, dtype=t.bool)
    dones[1] = True
    policy = t.full((length, config.num_actions), 1.0 / config.num_actions)
    values = t.zeros(length, dtype=t.float32)
    errors = t.ones(length, dtype=t.float32)
    return Trajectory(
        obs=obs,
        actions=actions,
        rewards=rewards,
        dones=dones,
        policy_targets=policy,
        value_targets=values,
        level="L1",
        x_max=42,
        episode_return=float(rewards.sum().item()),
        worker_type="verify",
        value_errors=errors,
    )


def _make_config():
    return replace(
        MuZeroConfig(),
        replay_capacity=100,
        frontier_capacity=50,
        unroll_steps=5,
        td_steps=3,
        gamma=0.5,
    )


def _verify_current_network_targets(config):
    rewards = t.ones(1, config.unroll_steps + config.td_steps)
    dones = t.zeros_like(rewards, dtype=t.bool)
    dones[0, 1] = True

    batch = ReplayBatch(
        obs=t.zeros(1, config.unroll_steps + config.td_steps + 1, 4, 84, 84, dtype=t.uint8),
        actions=t.zeros(1, config.unroll_steps + config.td_steps, dtype=t.long),
        rewards=rewards,
        dones=dones,
        policy_targets=t.zeros(1, config.unroll_steps, config.num_actions),
        value_targets=t.zeros(1, config.unroll_steps),
        weights=t.ones(1),
        trajectory_ids=t.tensor([0]),
        start_indices=t.tensor([0]),
        sources=["main"],
        target_steps=config.unroll_steps,
    )

    out = CurrentNetworkTargetComputer().compute(
        batch,
        _FakeNetwork(config.num_actions),
        config,
        t.device("cpu"),
    )
    expected = t.tensor([1.5, 1.0, 3.0, 3.0, 3.0])
    assert t.allclose(out["value_targets"][0], expected)
    assert out["policy_targets"].shape == (1, config.unroll_steps, config.num_actions)


def main():
    config = _make_config()
    buffer = DualPriorityReplayBuffer(config)
    traj_id = buffer.add_trajectory(_trajectory(config))
    batch = buffer.sample_batch(1, frontier_fraction=0.0, extra_steps=config.td_steps)

    worker = ReanalyseProcess(
        config,
        target_computer=DeterministicTargetComputer(),
        device="cpu",
    )
    worker.start()

    try:
        worker.set_weights(None, step=5)
        msg = worker.get(timeout=10)
        assert msg["type"] == "weights_loaded"
        assert msg["step"] == 5

        worker.set_weights(None, step=3)
        msg = worker.get(timeout=10)
        assert msg["type"] == "weights_ignored"
        assert msg["latest_step"] == 5

        request_id = worker.submit_batch(batch)
        update = worker.get(timeout=10)
        assert update["type"] == "target_update"
        assert update["request_id"] == request_id
        assert update["trajectory_ids"].tolist() == [traj_id]
        assert update["start_indices"].tolist() == [0]
        assert update["policy_targets"].shape == (1, config.unroll_steps, config.num_actions)
        assert update["value_targets"].shape == (1, config.unroll_steps)
        assert update["value_errors"].shape == (1, config.unroll_steps)

        expected = t.tensor([
            1.0 + 0.5 * 2.0,
            2.0,
            3.0 + 0.5 * 4.0 + 0.25 * 5.0,
            4.0 + 0.5 * 5.0 + 0.25 * 6.0,
            5.0 + 0.5 * 6.0 + 0.25 * 7.0,
        ])
        assert t.allclose(update["value_targets"][0], expected)

        apply_reanalyse_update(buffer, update)
        traj = buffer._trajectories[traj_id]
        assert t.allclose(traj.value_targets[:config.unroll_steps], expected)
        assert t.allclose(traj.value_errors[:config.unroll_steps], expected.abs())
        assert t.allclose(traj.policy_targets[0], t.eye(config.num_actions)[0])

    finally:
        worker.close()

    assert worker.process is not None
    assert not worker.process.is_alive()
    _verify_current_network_targets(config)
    print("Reanalyse verification passed.")


if __name__ == '__main__':
    main()
