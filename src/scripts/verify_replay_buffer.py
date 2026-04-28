import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from dataclasses import replace
import torch as t
from muzero.config import MuZeroConfig
from muzero.replay_buffer import DualPriorityReplayBuffer, Trajectory


def _trajectory(length, level, x_max, num_actions):
    obs = t.randint(0, 256, (length + 1, 4, 84, 84), dtype=t.uint8)
    actions = t.arange(length, dtype=t.long) % num_actions
    rewards = t.linspace(0.0, 1.0, length)
    dones = t.zeros(length, dtype=t.bool)
    policy = t.full((length, num_actions), 1.0 / num_actions)
    values = t.zeros(length, dtype=t.float32)
    errors = t.ones(length, dtype=t.float32)
    return Trajectory(
        obs=obs,
        actions=actions,
        rewards=rewards,
        dones=dones,
        policy_targets=policy,
        value_targets=values,
        level=level,
        x_max=x_max,
        episode_return=float(rewards.sum().item()),
        worker_type="verify",
        value_errors=errors,
    )


def _make_config():
    return replace(
        MuZeroConfig(),
        replay_capacity=30,
        frontier_capacity=20,
        unroll_steps=5,
        td_steps=2,
        replay_alpha=0.6,
        replay_beta=0.4,
        frontier_sample_fraction=0.5,
        x_percentile_window=100,
    )


def _verify_priority_bias(config):
    buffer = DualPriorityReplayBuffer(config)
    low_id = buffer.add_trajectory(_trajectory(10, "P", 1, config.num_actions))
    high_id = buffer.add_trajectory(_trajectory(10, "P", 2, config.num_actions))
    errors = t.full((1, config.unroll_steps), 1000.0)
    buffer.update_targets([high_id], [0], value_errors=errors)

    hits = 0
    for _ in range(200):
        batch = buffer.sample_batch(8, frontier_fraction=0.0)
        pairs = set(zip(batch.trajectory_ids.tolist(), batch.start_indices.tolist()))
        hits += int((high_id, 0) in pairs)
    assert hits > 50
    assert low_id in buffer._trajectories


def _verify_frontier_eviction(config):
    config = replace(config, replay_capacity=80, frontier_capacity=10)
    buffer = DualPriorityReplayBuffer(config)
    first = buffer.add_trajectory(_trajectory(10, "F", 10, config.num_actions))
    second = buffer.add_trajectory(_trajectory(10, "F", 100, config.num_actions))

    assert first not in buffer._frontier_ids
    assert second in buffer._frontier_ids
    assert all(key[0] != first for key in buffer._frontier_train.key_to_idx)
    main = buffer.sample_batch(16, frontier_fraction=0.0)
    assert set(main.trajectory_ids.tolist()) <= set(buffer._trajectories.keys())


def main():
    t.manual_seed(0)
    config = _make_config()
    buffer = DualPriorityReplayBuffer(config)

    ids = []
    ids.append(buffer.add_trajectory(_trajectory(10, "L1", 10, config.num_actions)))
    ids.append(buffer.add_trajectory(_trajectory(10, "L1", 20, config.num_actions)))
    ids.append(buffer.add_trajectory(_trajectory(10, "L2", 5, config.num_actions)))

    assert len(buffer) == 30
    assert buffer.num_trajectories == 3
    assert buffer.num_frontier_trajectories > 0

    batch = buffer.sample_batch(16)
    assert batch.obs.shape == (16, config.unroll_steps + 1, 4, 84, 84)
    assert batch.actions.shape == (16, config.unroll_steps)
    assert batch.rewards.shape == (16, config.unroll_steps)
    assert batch.policy_targets.shape == (16, config.unroll_steps, config.num_actions)
    assert batch.value_targets.shape == (16, config.unroll_steps)
    assert batch.weights.shape == (16,)
    assert batch.target_steps == config.unroll_steps
    assert (batch.weights > 0).all()
    assert (batch.start_indices >= 0).all()
    assert (batch.start_indices <= 5).all()

    source_counts = {source: batch.sources.count(source) for source in set(batch.sources)}
    assert source_counts.get("frontier", 0) == 8
    assert source_counts.get("main", 0) == 8

    extended = buffer.sample_batch(8, extra_steps=2)
    assert extended.obs.shape == (8, config.unroll_steps + 3, 4, 84, 84)
    assert extended.actions.shape == (8, config.unroll_steps + 2)
    assert extended.policy_targets.shape == (8, config.unroll_steps, config.num_actions)
    assert (extended.start_indices <= 3).all()

    try:
        buffer.sample_batch(1, frontier_fraction=1.5)
        raise AssertionError("invalid frontier_fraction should fail")
    except ValueError:
        pass

    high_id = buffer.add_trajectory(_trajectory(10, "L1", 100, config.num_actions))
    assert high_id in buffer._frontier_ids
    assert ids[0] not in buffer._trajectories
    assert ids[0] not in buffer._frontier_ids
    assert len(buffer) == 30

    for _ in range(20):
        batch = buffer.sample_batch(32)
        assert ids[0] not in batch.trajectory_ids.tolist()

    new_policy = t.zeros(1, config.unroll_steps, config.num_actions)
    new_policy[:, :, 3] = 1.0
    new_values = t.arange(config.unroll_steps, dtype=t.float32).unsqueeze(0)
    new_errors = t.arange(1, config.unroll_steps + 1, dtype=t.float32).unsqueeze(0)
    buffer.update_targets(
        [high_id],
        [0],
        policy_targets=new_policy,
        value_targets=new_values,
        value_errors=new_errors,
    )
    traj = buffer._trajectories[high_id]
    assert t.allclose(traj.policy_targets[:config.unroll_steps], new_policy[0])
    assert t.allclose(traj.value_targets[:config.unroll_steps], new_values[0])
    assert t.allclose(traj.value_errors[:config.unroll_steps], new_errors[0])

    try:
        buffer.update_targets([high_id], [0], value_targets=t.zeros(1, config.unroll_steps + 1))
        raise AssertionError("invalid target shape should fail")
    except ValueError:
        pass

    stats = buffer.frontier_stats()
    assert "L1" in stats
    assert stats["L1"]["trajectories"] >= 1

    try:
        buffer.sample_batch(1, extra_steps=config.td_steps + 1)
        raise AssertionError("unsupported extra_steps should fail")
    except ValueError:
        pass

    _verify_priority_bias(config)
    _verify_frontier_eviction(config)

    print("Replay buffer verification passed.")


if __name__ == '__main__':
    main()
