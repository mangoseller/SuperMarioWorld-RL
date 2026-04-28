from collections import defaultdict, deque
from dataclasses import dataclass
import random
import torch as t


@dataclass
class Trajectory:
    obs: t.Tensor
    actions: t.Tensor
    rewards: t.Tensor
    dones: t.Tensor
    policy_targets: t.Tensor
    value_targets: t.Tensor
    level: str
    x_max: float
    episode_return: float = 0.0
    worker_type: str = ""
    value_errors: t.Tensor = None


@dataclass
class ReplayBatch:
    obs: t.Tensor
    actions: t.Tensor
    rewards: t.Tensor
    dones: t.Tensor
    policy_targets: t.Tensor
    value_targets: t.Tensor
    weights: t.Tensor
    trajectory_ids: t.Tensor
    start_indices: t.Tensor
    sources: list
    target_steps: int


class _SegmentTree:
    def __init__(self, capacity, op, neutral):
        tree_capacity = 1
        while tree_capacity < capacity:
            tree_capacity *= 2
        self.capacity = capacity
        self.tree_capacity = tree_capacity
        self.neutral = neutral
        self.op = op
        self.values = [neutral for _ in range(2 * tree_capacity)]

    def update(self, idx, value):
        if idx < 0 or idx >= self.capacity:
            raise IndexError("segment tree index out of range")
        tree_idx = idx + self.tree_capacity
        self.values[tree_idx] = value
        tree_idx //= 2
        while tree_idx >= 1:
            self.values[tree_idx] = self.op(
                self.values[2 * tree_idx],
                self.values[2 * tree_idx + 1],
            )
            tree_idx //= 2

    def reduce(self):
        return self.values[1]

    def find_prefixsum(self, prefixsum):
        idx = 1
        while idx < self.tree_capacity:
            left = 2 * idx
            if self.values[left] > prefixsum:
                idx = left
            else:
                prefixsum -= self.values[left]
                idx = left + 1
        return idx - self.tree_capacity


class _PriorityWindowIndex:
    def __init__(self, capacity, alpha, beta, eps):
        self.capacity = capacity
        self.alpha = alpha
        self.beta = beta
        self.eps = eps
        self.entries = [None] * capacity
        self.raw_priorities = [0.0] * capacity
        self.key_to_idx = {}
        self.free = list(range(capacity - 1, -1, -1))
        self.sum_tree = _SegmentTree(capacity, lambda a, b: a + b, 0.0)
        self.min_tree = _SegmentTree(capacity, min, float("inf"))

    def __len__(self):
        return len(self.key_to_idx)

    def add(self, key, priority):
        if key in self.key_to_idx:
            self.update(key, priority)
            return
        if not self.free:
            raise RuntimeError("replay index capacity exhausted")
        idx = self.free.pop()
        self.entries[idx] = key
        self.key_to_idx[key] = idx
        self.update(key, priority)

    def remove(self, key):
        idx = self.key_to_idx.pop(key, None)
        if idx is None:
            return
        self.entries[idx] = None
        self.raw_priorities[idx] = 0.0
        self.sum_tree.update(idx, 0.0)
        self.min_tree.update(idx, float("inf"))
        self.free.append(idx)

    def update(self, key, priority):
        idx = self.key_to_idx.get(key)
        if idx is None:
            return
        priority = max(float(priority), self.eps)
        scaled = priority ** self.alpha
        self.raw_priorities[idx] = priority
        self.sum_tree.update(idx, scaled)
        self.min_tree.update(idx, scaled)

    def sample(self, n, source):
        if len(self) == 0:
            return []
        total = self.sum_tree.reduce()
        if total <= 0:
            raise ValueError("cannot sample from zero-priority replay index")
        items = []
        max_weight = 0.0
        for _ in range(n):
            mass = random.random() * total
            idx = self.sum_tree.find_prefixsum(mass)
            key = self.entries[idx]
            if key is None:
                continue
            prob = self.sum_tree.values[idx + self.sum_tree.tree_capacity] / total
            weight = (len(self) * prob) ** (-self.beta)
            max_weight = max(max_weight, weight)
            traj_id, start = key
            items.append([traj_id, start, weight, source])

        if max_weight > 0:
            for item in items:
                item[2] /= max_weight
        return [tuple(item) for item in items]


class DualPriorityReplayBuffer:
    def __init__(self, config):
        self.config = config
        self.capacity = config.replay_capacity
        self.frontier_capacity = config.frontier_capacity
        self.unroll_steps = config.unroll_steps
        self.num_actions = config.num_actions
        self.alpha = config.replay_alpha
        self.beta = config.replay_beta
        self.eps = config.priority_eps
        self.frontier_fraction = config.frontier_sample_fraction
        self.frontier_margin = config.frontier_margin

        self._trajectories = {}
        self._order = deque()
        self._frontier_order = deque()
        self._frontier_ids = set()
        self._level_x = defaultdict(lambda: deque(maxlen=config.x_percentile_window))
        self._next_id = 0
        self._size = 0
        self._frontier_size = 0
        self._main_train = _PriorityWindowIndex(self.capacity, self.alpha, self.beta, self.eps)
        self._frontier_train = _PriorityWindowIndex(self.capacity, self.alpha, self.beta, self.eps)
        self._main_reanalyse = _PriorityWindowIndex(self.capacity, self.alpha, self.beta, self.eps)
        self._frontier_reanalyse = _PriorityWindowIndex(self.capacity, self.alpha, self.beta, self.eps)

    def __len__(self):
        return self._size

    @property
    def num_trajectories(self):
        return len(self._trajectories)

    @property
    def num_frontier_trajectories(self):
        self._compact_frontier()
        return len(self._frontier_ids)

    def add_trajectory(self, trajectory):
        trajectory = self._prepare_trajectory(trajectory)
        t_steps = trajectory.actions.shape[0]
        if t_steps < self.unroll_steps:
            raise ValueError("trajectory is shorter than unroll_steps")
        if t_steps > self.capacity:
            raise ValueError("single trajectory exceeds replay capacity")

        traj_id = self._next_id
        self._next_id += 1

        x90_before = self.level_x_percentile(trajectory.level, 0.90)
        frontier = x90_before is None or trajectory.x_max > x90_before

        self._trajectories[traj_id] = trajectory
        self._order.append(traj_id)
        self._size += t_steps
        self._level_x[trajectory.level].append(float(trajectory.x_max))
        self._index_trajectory(traj_id)

        if frontier:
            self._add_frontier(traj_id)

        self._evict_to_capacity()
        return traj_id

    def sample_batch(self, batch_size, frontier_fraction=None, extra_steps=0):
        if not self._trajectories:
            raise ValueError("cannot sample from an empty replay buffer")
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if extra_steps < 0:
            raise ValueError("extra_steps must be non-negative")
        if extra_steps not in (0, self.config.td_steps):
            raise ValueError("sample_batch only supports training or configured reanalyse windows")

        frontier_fraction = self.frontier_fraction if frontier_fraction is None else frontier_fraction
        if frontier_fraction < 0 or frontier_fraction > 1:
            raise ValueError("frontier_fraction must be between 0 and 1")
        n_frontier = int(round(batch_size * frontier_fraction))
        n_frontier = min(n_frontier, batch_size)

        items = []
        if n_frontier > 0 and self.num_frontier_trajectories > 0:
            items.extend(self._sample_items(n_frontier, source="frontier", extra_steps=extra_steps))

        remaining = batch_size - len(items)
        if remaining > 0:
            items.extend(self._sample_items(remaining, source="main", extra_steps=extra_steps))

        return self._build_batch(items, extra_steps)

    def update_targets(self, trajectory_ids, start_indices, policy_targets=None,
                       value_targets=None, value_errors=None):
        trajectory_ids = self._to_list(trajectory_ids)
        start_indices = self._to_list(start_indices)
        rows = len(trajectory_ids)
        if len(start_indices) != rows:
            raise ValueError("trajectory_ids and start_indices must have the same length")
        if policy_targets is not None and policy_targets.shape != (rows, self.unroll_steps, self.num_actions):
            raise ValueError("policy_targets must have shape (B, unroll_steps, num_actions)")
        if value_targets is not None and value_targets.shape != (rows, self.unroll_steps):
            raise ValueError("value_targets must have shape (B, unroll_steps)")
        if value_errors is not None and value_errors.shape != (rows, self.unroll_steps):
            raise ValueError("value_errors must have shape (B, unroll_steps)")

        for row, (traj_id, start) in enumerate(zip(trajectory_ids, start_indices)):
            traj_id = int(traj_id)
            start = int(start)
            if traj_id not in self._trajectories:
                continue

            traj = self._trajectories[traj_id]
            end = start + self.unroll_steps
            if end > traj.actions.shape[0]:
                raise ValueError("target update exceeds trajectory bounds")

            if policy_targets is not None:
                traj.policy_targets[start:end] = policy_targets[row].detach().cpu()
            if value_targets is not None:
                traj.value_targets[start:end] = value_targets[row].detach().cpu()
            if value_errors is not None:
                traj.value_errors[start:end] = value_errors[row].detach().cpu().abs().clamp_min(self.eps)
                self._refresh_index_priorities(traj_id, range(start, end))

    def level_x_percentile(self, level, q=0.90):
        values = self._level_x.get(str(level))
        if not values:
            return None
        tensor = t.tensor(list(values), dtype=t.float32)
        return float(t.quantile(tensor, q).item())

    def get_trajectory_level(self, traj_id):
        traj = self._trajectories.get(int(traj_id))
        return traj.level if traj is not None else None

    def frontier_stats(self):
        self._compact_frontier()
        stats = defaultdict(lambda: {"trajectories": 0, "transitions": 0})
        for traj_id in self._frontier_ids:
            traj = self._trajectories.get(traj_id)
            if traj is None:
                continue
            stats[traj.level]["trajectories"] += 1
            stats[traj.level]["transitions"] += traj.actions.shape[0]
        return dict(stats)

    def _prepare_trajectory(self, trajectory):
        t_steps = trajectory.actions.shape[0]
        if trajectory.obs.shape[0] != t_steps + 1:
            raise ValueError("obs must contain T + 1 frames")
        if trajectory.rewards.shape[0] != t_steps or trajectory.dones.shape[0] != t_steps:
            raise ValueError("actions, rewards, and dones must have length T")

        obs = trajectory.obs.detach().cpu()
        if obs.dtype != t.uint8:
            obs = (obs.float().clamp(0, 1) * 255).to(t.uint8)

        actions = trajectory.actions.detach().cpu().long()
        rewards = trajectory.rewards.detach().cpu().float()
        dones = trajectory.dones.detach().cpu().bool()

        if trajectory.policy_targets is None:
            policy_targets = t.full((t_steps, self.num_actions), 1.0 / self.num_actions)
        else:
            policy_targets = trajectory.policy_targets.detach().cpu().float()
        if policy_targets.shape != (t_steps, self.num_actions):
            raise ValueError("policy_targets must have shape (T, num_actions)")

        policy_sums = policy_targets.sum(-1, keepdim=True).clamp_min(1e-8)
        policy_targets = policy_targets / policy_sums

        if trajectory.value_targets is None:
            value_targets = t.zeros(t_steps, dtype=t.float32)
        else:
            value_targets = trajectory.value_targets.detach().cpu().float()
        if value_targets.shape != (t_steps,):
            raise ValueError("value_targets must have shape (T,)")

        if trajectory.value_errors is None:
            value_errors = t.ones(t_steps, dtype=t.float32)
        else:
            value_errors = trajectory.value_errors.detach().cpu().float().abs()
        if value_errors.shape != (t_steps,):
            raise ValueError("value_errors must have shape (T,)")
        value_errors = value_errors.clamp_min(self.eps)

        return Trajectory(
            obs=obs,
            actions=actions,
            rewards=rewards,
            dones=dones,
            policy_targets=policy_targets,
            value_targets=value_targets,
            level=str(trajectory.level),
            x_max=float(trajectory.x_max),
            episode_return=float(trajectory.episode_return),
            worker_type=str(trajectory.worker_type),
            value_errors=value_errors,
        )

    def _evict_to_capacity(self):
        while self._size > self.capacity and self._order:
            traj_id = self._order.popleft()
            traj = self._trajectories.pop(traj_id, None)
            if traj is None:
                continue
            self._size -= traj.actions.shape[0]
            if traj_id in self._frontier_ids:
                self._frontier_ids.remove(traj_id)
                self._frontier_size -= traj.actions.shape[0]
            self._remove_trajectory_windows(traj_id)
        self._compact_frontier()
        self._evict_frontier_to_capacity()

    def _add_frontier(self, traj_id):
        if traj_id in self._frontier_ids:
            return
        traj = self._trajectories[traj_id]
        self._frontier_order.append(traj_id)
        self._frontier_ids.add(traj_id)
        self._frontier_size += traj.actions.shape[0]
        self._index_frontier_trajectory(traj_id)
        self._evict_frontier_to_capacity()

    def _evict_frontier_to_capacity(self):
        while self._frontier_size > self.frontier_capacity and self._frontier_order:
            traj_id = self._frontier_order.popleft()
            if traj_id not in self._frontier_ids:
                continue
            traj = self._trajectories.get(traj_id)
            self._frontier_ids.remove(traj_id)
            if traj is not None:
                self._frontier_size -= traj.actions.shape[0]
            self._remove_frontier_windows(traj_id)

    def _compact_frontier(self):
        while self._frontier_order and self._frontier_order[0] not in self._trajectories:
            traj_id = self._frontier_order.popleft()
            self._frontier_ids.discard(traj_id)
        self._frontier_ids = {idx for idx in self._frontier_ids if idx in self._trajectories}
        self._frontier_size = sum(
            self._trajectories[idx].actions.shape[0]
            for idx in self._frontier_ids
        )

    def _sample_items(self, n, source, extra_steps):
        index = self._index_for(source, extra_steps)
        if len(index) == 0:
            if source == "frontier":
                return []
            raise ValueError("no valid replay windows available")
        return index.sample(n, source)

    def _candidate_items(self, source, extra_steps):
        if source == "frontier":
            self._compact_frontier()
            ids = list(self._frontier_ids)
        else:
            ids = list(self._trajectories.keys())

        candidates = []
        for traj_id in ids:
            traj = self._trajectories.get(traj_id)
            if traj is None:
                continue
            max_start = traj.actions.shape[0] - self.unroll_steps - extra_steps
            if max_start < 0:
                continue
            x90 = self.level_x_percentile(traj.level, 0.90)
            frontier_bonus = self._frontier_priority(traj, x90)
            value_priorities = traj.value_errors[:max_start + 1].clamp_min(self.eps)
            priorities = value_priorities * (1.0 + frontier_bonus)
            candidates.extend(
                (traj_id, start, float(priorities[start].item()))
                for start in range(max_start + 1)
            )
        return candidates

    def _frontier_priority(self, traj, x90):
        if x90 is None:
            return self.frontier_margin
        return max(0.0, float(traj.x_max) - x90 + self.frontier_margin)

    def _index_for(self, source, extra_steps):
        if source == "frontier":
            return self._frontier_reanalyse if extra_steps == self.config.td_steps else self._frontier_train
        return self._main_reanalyse if extra_steps == self.config.td_steps else self._main_train

    def _valid_starts(self, traj, extra_steps):
        max_start = traj.actions.shape[0] - self.unroll_steps - extra_steps
        if max_start < 0:
            return range(0)
        return range(max_start + 1)

    def _priority_for(self, traj, start):
        x90 = self.level_x_percentile(traj.level, 0.90)
        frontier_bonus = self._frontier_priority(traj, x90)
        return float(traj.value_errors[start].clamp_min(self.eps).item()) * (1.0 + frontier_bonus)

    def _index_trajectory(self, traj_id):
        traj = self._trajectories[traj_id]
        for start in self._valid_starts(traj, 0):
            self._main_train.add((traj_id, start), self._priority_for(traj, start))
        for start in self._valid_starts(traj, self.config.td_steps):
            self._main_reanalyse.add((traj_id, start), self._priority_for(traj, start))

    def _index_frontier_trajectory(self, traj_id):
        traj = self._trajectories.get(traj_id)
        if traj is None:
            return
        for start in self._valid_starts(traj, 0):
            self._frontier_train.add((traj_id, start), self._priority_for(traj, start))
        for start in self._valid_starts(traj, self.config.td_steps):
            self._frontier_reanalyse.add((traj_id, start), self._priority_for(traj, start))

    def _remove_trajectory_windows(self, traj_id):
        for index in (self._main_train, self._main_reanalyse, self._frontier_train, self._frontier_reanalyse):
            keys = [key for key in index.key_to_idx if key[0] == traj_id]
            for key in keys:
                index.remove(key)

    def _remove_frontier_windows(self, traj_id):
        for index in (self._frontier_train, self._frontier_reanalyse):
            keys = [key for key in index.key_to_idx if key[0] == traj_id]
            for key in keys:
                index.remove(key)

    def _refresh_index_priorities(self, traj_id, starts):
        traj = self._trajectories.get(traj_id)
        if traj is None:
            return
        for start in starts:
            start = int(start)
            priority = self._priority_for(traj, start)
            for index in (self._main_train, self._main_reanalyse):
                index.update((traj_id, start), priority)
            if traj_id in self._frontier_ids:
                for index in (self._frontier_train, self._frontier_reanalyse):
                    index.update((traj_id, start), priority)

    def _build_batch(self, items, extra_steps):
        obs, actions, rewards, dones = [], [], [], []
        policy_targets, value_targets, weights = [], [], []
        trajectory_ids, start_indices, sources = [], [], []

        for traj_id, start, weight, source in items:
            traj = self._trajectories[traj_id]
            target_end = start + self.unroll_steps
            data_end = target_end + extra_steps

            obs.append(traj.obs[start:data_end + 1])
            actions.append(traj.actions[start:data_end])
            rewards.append(traj.rewards[start:data_end])
            dones.append(traj.dones[start:data_end])
            policy_targets.append(traj.policy_targets[start:target_end])
            value_targets.append(traj.value_targets[start:target_end])
            weights.append(weight)
            trajectory_ids.append(traj_id)
            start_indices.append(start)
            sources.append(source)

        return ReplayBatch(
            obs=t.stack(obs),
            actions=t.stack(actions),
            rewards=t.stack(rewards),
            dones=t.stack(dones),
            policy_targets=t.stack(policy_targets),
            value_targets=t.stack(value_targets),
            weights=t.tensor(weights, dtype=t.float32),
            trajectory_ids=t.tensor(trajectory_ids, dtype=t.long),
            start_indices=t.tensor(start_indices, dtype=t.long),
            sources=sources,
            target_steps=self.unroll_steps,
        )

    def _to_list(self, value):
        if isinstance(value, t.Tensor):
            return value.detach().cpu().tolist()
        return list(value)
