from dataclasses import dataclass
import multiprocessing as mp
import numpy as np
import queue
import random
import time
import torch as t
from einops import rearrange

from muzero.mcts import GumbelMuZeroSearch, MuZeroModelAdapter
from muzero.network import deserialize_worker_weights
from muzero.reanalyse import ReanalyseNetwork, make_batch_search
from muzero.replay_buffer import Trajectory
from utils import get_torch_compatible_actions


@dataclass
class BehaviourProfile:
    name: str
    dirichlet_alpha: float
    dirichlet_weight: float
    temperature: float
    epsilon: float = 0.0


def make_profiles(config):
    return [
        BehaviourProfile(
            "standard",
            config.self_play_dirichlet_alpha,
            config.self_play_dirichlet_weight,
            1.0,
        ),
        BehaviourProfile(
            "aggressive_dirichlet",
            config.self_play_aggressive_dirichlet_alpha,
            config.self_play_aggressive_dirichlet_weight,
            1.0,
        ),
        BehaviourProfile(
            "high_temperature",
            config.self_play_dirichlet_alpha,
            config.self_play_dirichlet_weight,
            config.self_play_high_temperature,
        ),
        BehaviourProfile(
            "epsilon_greedy",
            config.self_play_dirichlet_alpha,
            config.self_play_dirichlet_weight,
            1.0,
            epsilon=config.self_play_epsilon,
        ),
    ]


def assign_profiles(num_workers, config):
    profiles = make_profiles(config)
    return [profiles[i % len(profiles)] for i in range(num_workers)]


def sample_action(policy, temperature=1.0, epsilon=0.0, generator=None):
    if epsilon > 0 and t.rand((), generator=generator).item() < epsilon:
        return int(t.randint(policy.shape[0], (), generator=generator).item())
    if temperature <= 0:
        return int(policy.argmax().item())
    probs = policy.float().clamp_min(0)
    if temperature != 1.0:
        probs = probs.pow(1.0 / temperature)
    total = probs.sum()
    if total <= 0 or not t.isfinite(total):
        probs = t.full_like(probs, 1.0 / probs.shape[0])
    else:
        probs = probs / total
    return int(t.multinomial(probs, 1, generator=generator).item())


def _obs_to_uint8(obs):
    obs = obs.detach().cpu()
    if obs.dtype == t.uint8:
        return obs
    return (obs.float().clamp(0, 1) * 255).to(t.uint8)


def _obs_to_float_batch(obs, device):
    device = t.device(device)
    if device.type == "cuda" and not t.cuda.is_available():
        device = t.device("cpu")
    obs = obs.to(device)
    if obs.dtype == t.uint8:
        obs = obs.float() / 255.0
    else:
        obs = obs.float()
        if obs.max() > 1.0:
            obs = obs / 255.0
    if obs.dim() == 3:
        obs = obs.unsqueeze(0)
    return obs


def _obs_payload(obs):
    obs = _obs_to_uint8(obs).contiguous()
    return {
        "shape": tuple(obs.shape),
        "data": obs.numpy().tobytes(),
    }


def _obs_from_payload(payload):
    array = np.frombuffer(payload["data"], dtype=np.uint8).copy()
    c, h, w = payload["shape"]
    obs = rearrange(t.from_numpy(array), '(c h w) -> c h w', c=c, h=h, w=w)
    return obs.float() / 255.0


def _tensor_payload(tensor):
    array = tensor.detach().cpu().contiguous().numpy()
    return {
        "shape": tuple(array.shape),
        "dtype": str(array.dtype),
        "data": array.tobytes(),
    }


def _tensor_from_payload(payload):
    array = np.frombuffer(
        payload["data"],
        dtype=np.dtype(payload["dtype"]),
    ).copy()
    return t.from_numpy(array.reshape(payload["shape"]))


def _trajectory_payload(trajectory):
    return {
        "obs": _tensor_payload(trajectory.obs),
        "actions": _tensor_payload(trajectory.actions),
        "rewards": _tensor_payload(trajectory.rewards),
        "dones": _tensor_payload(trajectory.dones),
        "policy_targets": _tensor_payload(trajectory.policy_targets),
        "value_targets": _tensor_payload(trajectory.value_targets),
        "value_errors": (
            _tensor_payload(trajectory.value_errors)
            if trajectory.value_errors is not None else None
        ),
        "level": trajectory.level,
        "x_max": float(trajectory.x_max),
        "episode_return": float(trajectory.episode_return),
        "worker_type": trajectory.worker_type,
    }


def _trajectory_from_payload(payload):
    value_errors = payload.get("value_errors")
    return Trajectory(
        obs=_tensor_from_payload(payload["obs"]),
        actions=_tensor_from_payload(payload["actions"]).long(),
        rewards=_tensor_from_payload(payload["rewards"]).float(),
        dones=_tensor_from_payload(payload["dones"]).bool(),
        policy_targets=_tensor_from_payload(payload["policy_targets"]).float(),
        value_targets=_tensor_from_payload(payload["value_targets"]).float(),
        level=payload["level"],
        x_max=float(payload["x_max"]),
        episode_return=float(payload.get("episode_return", 0.0)),
        worker_type=payload.get("worker_type", ""),
        value_errors=(
            _tensor_from_payload(value_errors).float()
            if value_errors is not None else None
        ),
    )


def _resolve_device(device):
    device = t.device(device)
    if device.type == "cuda" and not t.cuda.is_available():
        return t.device("cpu")
    return device


def _autocast_for(device, enabled=True):
    if t.device(device).type != "cuda" or not enabled:
        return t.autocast(device_type="cpu", enabled=False)
    return t.autocast(device_type="cuda", dtype=t.bfloat16)


def _extract_x(info):
    if not isinstance(info, dict):
        return 0.0
    for key in ("_max_x", "max_x", "_global_x", "global_x", "xpos", "x"):
        if key in info:
            value = info[key]
            if isinstance(value, t.Tensor):
                value = value.squeeze().item()
            return float(value)
    return 0.0


def _td_info(td):
    info = {}
    for key in ("_max_x", "max_x", "_global_x", "global_x", "xpos", "x"):
        try:
            value = td.get(key, None)
        except TypeError:
            value = None
        if value is not None:
            info[key] = value
    return info


class TorchRLEnvAdapter:
    def __init__(self, env, num_actions):
        self.env = env
        self.num_actions = num_actions
        self.td = None

    def reset(self):
        self.td = self.env.reset()
        return self.td["observation"].squeeze(0), _td_info(self.td)

    def step(self, action):
        self.td["action"] = get_torch_compatible_actions(t.tensor(action), self.num_actions)
        self.td = self.env.step(self.td)
        next_td = self.td["next"]
        obs = next_td["observation"].squeeze(0)
        reward = float(next_td["reward"].squeeze().item())
        done = bool(next_td["done"].squeeze().item())
        info = _td_info(next_td)
        return obs, reward, done, info

    def close(self):
        self.env.close()


def make_mario_env(level, config):
    import retro
    from torchrl.envs import TransformedEnv
    from torchrl.envs.transforms import UnsqueezeTransform
    from environment import _wrap_env

    raw = retro.make('SuperMarioWorld-Snes', state=level, render_mode='rgb_array')
    env = _wrap_env(raw, skip=3)
    env = TransformedEnv(
        env,
        UnsqueezeTransform(
            dim=0,
            allow_positive_dim=True,
            in_keys=["observation", "reward", "done", "terminated"],
        )
    )
    return TorchRLEnvAdapter(env, config.num_actions)


def make_reanalyse_network(config, device):
    return ReanalyseNetwork(config, _resolve_device(device))


def make_gumbel_search(config):
    return GumbelMuZeroSearch(config)


def _run_search(search, model, obs, generator, profile, num_simulations):
    if hasattr(search, "run_batch"):
        output = search.run_batch(
            model,
            obs=obs,
            generator=generator,
            root_dirichlet_alphas=profile.dirichlet_alpha,
            root_dirichlet_fractions=profile.dirichlet_weight,
            num_simulations=num_simulations,
        )
        return output.action_weights[0].cpu()
    output = search.run(
        model,
        obs=obs,
        generator=generator,
        root_dirichlet_alpha=profile.dirichlet_alpha,
        root_dirichlet_fraction=profile.dirichlet_weight,
        num_simulations=num_simulations,
    )
    return output.action_weights.cpu()


def collect_episode(env, model, search, config, profile, level="unknown",
                    device="cuda", generator=None, max_steps=None, num_simulations=None):
    max_steps = config.self_play_max_episode_steps if max_steps is None else max_steps
    obs, info = env.reset()
    obs_list, actions, rewards, dones, policies = [], [], [], [], []
    x_max = _extract_x(info)

    for _ in range(max_steps):
        obs_list.append(_obs_to_uint8(obs))
        obs_b = _obs_to_float_batch(obs, device)
        policy = _run_search(
            search, model, obs_b, generator, profile, num_simulations,
        )
        action = sample_action(policy, profile.temperature, profile.epsilon, generator)
        next_obs, reward, done, info = env.step(action)

        actions.append(action)
        rewards.append(float(reward))
        dones.append(bool(done))
        policies.append(policy)
        x_max = max(x_max, _extract_x(info))
        obs = next_obs
        if done:
            break

    obs_list.append(_obs_to_uint8(obs))
    length = len(actions)
    return Trajectory(
        obs=t.stack(obs_list),
        actions=t.tensor(actions, dtype=t.long),
        rewards=t.tensor(rewards, dtype=t.float32),
        dones=t.tensor(dones, dtype=t.bool),
        policy_targets=t.stack(policies),
        value_targets=t.zeros(length, dtype=t.float32),
        level=level,
        x_max=x_max,
        episode_return=float(sum(rewards)),
        worker_type=profile.name,
        value_errors=t.ones(length, dtype=t.float32),
    )


def _choose_level(active_levels, level_weights, current_level, worker_id=None):
    if not active_levels:
        return current_level
    if worker_id is not None:
        return active_levels[worker_id % len(active_levels)]
    total_w = sum(level_weights)
    if total_w > 0:
        probs = [w / total_w for w in level_weights]
        return random.choices(active_levels, weights=probs)[0]
    return active_levels[0]


def _put_trajectory(trajectory_queue, item, shutdown):
    while not shutdown.is_set():
        try:
            trajectory_queue.put(item, timeout=0.1)
            return
        except queue.Full:
            continue


def _search_service_loop(config, weight_queue, request_queue, response_queues,
                         shutdown, device):
    t.set_num_threads(1)
    device = _resolve_device(device)
    network = ReanalyseNetwork(config, device)
    adapter = MuZeroModelAdapter(network, config)
    search = make_batch_search(config)
    latest_step = -1

    while not shutdown.is_set():
        while True:
            try:
                msg = weight_queue.get_nowait()
            except queue.Empty:
                break
            if msg.get("type") == "weights" and int(msg["step"]) > latest_step:
                network.load_weights(deserialize_worker_weights(msg.get("weights")))
                latest_step = int(msg["step"])

        try:
            first = request_queue.get(timeout=0.05)
        except queue.Empty:
            continue
        if first.get("type") != "search":
            continue

        requests = [first]
        deadline = config.self_play_search_batch_wait_ms / 1000.0
        while len(requests) < config.self_play_search_batch_size:
            try:
                requests.append(request_queue.get(timeout=deadline))
            except queue.Empty:
                break

        by_sims = {}
        for req in requests:
            by_sims.setdefault(int(req["num_simulations"]), []).append(req)

        for num_sims, group in by_sims.items():
            obs = t.stack([_obs_from_payload(req["obs"]) for req in group]).to(device)
            alphas = [float(req["dirichlet_alpha"]) for req in group]
            fractions = [float(req["dirichlet_weight"]) for req in group]
            with t.inference_mode(), _autocast_for(device, config.use_amp):
                output = search.run_batch(
                    adapter,
                    obs=obs,
                    num_simulations=num_sims,
                    root_dirichlet_alphas=alphas,
                    root_dirichlet_fractions=fractions,
                )
            for row, req in enumerate(group):
                response_queues[int(req["worker_id"])].put({
                    "type": "search_result",
                    "request_id": req["request_id"],
                    "policy": output.action_weights[row].cpu().tolist(),
                    "weight_step": latest_step,
                })


def _request_policy(request_queue, response_queue, worker_id, request_id,
                    obs, profile, num_simulations, shutdown):
    request = {
        "type": "search",
        "worker_id": worker_id,
        "request_id": request_id,
        "obs": _obs_payload(obs),
        "dirichlet_alpha": profile.dirichlet_alpha,
        "dirichlet_weight": profile.dirichlet_weight,
        "num_simulations": int(num_simulations),
    }
    while not shutdown.is_set():
        try:
            request_queue.put(request, timeout=0.1)
            break
        except queue.Full:
            continue
    else:
        return None, -1

    while not shutdown.is_set():
        try:
            msg = response_queue.get(timeout=0.1)
        except queue.Empty:
            continue
        if msg.get("request_id") == request_id:
            return t.tensor(msg["policy"], dtype=t.float32), int(msg.get("weight_step", -1))
    return None, -1


def _worker_loop_batched(config, worker_id, profile, env_factory, level,
                         control_queue, search_queue, response_queue,
                         trajectory_queue, shutdown):
    t.set_num_threads(1)
    random.seed(worker_id)
    generator = t.Generator().manual_seed(worker_id)
    current_level = level
    env = env_factory(level, config)
    latest_step = -1
    request_id = 0

    active_levels = [level]
    level_weights = [1.0]
    level_sims = {level: config.mcts_num_simulations}
    max_steps = config.self_play_max_episode_steps

    try:
        while not shutdown.is_set():
            while True:
                try:
                    msg = control_queue.get_nowait()
                except queue.Empty:
                    break
                if msg.get("type") == "control":
                    latest_step = int(msg["step"])
                    active_levels = msg.get("active_levels", active_levels)
                    level_weights = msg.get("level_weights", level_weights)
                    level_sims = msg.get("level_sims", level_sims)
                    max_steps = msg.get("max_steps", max_steps)

            if latest_step < 0:
                time.sleep(0.01)
                continue

            new_level = _choose_level(active_levels, level_weights, current_level, worker_id)
            if new_level != current_level:
                if hasattr(env, "close"):
                    env.close()
                env = env_factory(new_level, config)
                current_level = new_level

            obs, info = env.reset()
            obs_list, actions, rewards, dones, policies = [], [], [], [], []
            x_max = _extract_x(info)
            num_sims = level_sims.get(current_level, config.mcts_num_simulations)
            episode_return = 0.0

            for _ in range(max_steps):
                obs_list.append(_obs_to_uint8(obs))
                policy, weight_step = _request_policy(
                    search_queue,
                    response_queue,
                    worker_id,
                    request_id,
                    obs,
                    profile,
                    num_sims,
                    shutdown,
                )
                if policy is None:
                    break
                request_id += 1
                latest_step = max(latest_step, weight_step)
                action = sample_action(policy, profile.temperature, profile.epsilon, generator)
                next_obs, reward, done, info = env.step(action)

                actions.append(action)
                rewards.append(float(reward))
                dones.append(bool(done))
                policies.append(policy)
                episode_return += float(reward)
                x_max = max(x_max, _extract_x(info))
                obs = next_obs
                if done:
                    break

            obs_list.append(_obs_to_uint8(obs))
            length = len(actions)
            if length < config.unroll_steps:
                continue
            trajectory = Trajectory(
                obs=t.stack(obs_list),
                actions=t.tensor(actions, dtype=t.long),
                rewards=t.tensor(rewards, dtype=t.float32),
                dones=t.tensor(dones, dtype=t.bool),
                policy_targets=t.stack(policies),
                value_targets=t.zeros(length, dtype=t.float32),
                level=current_level,
                x_max=x_max,
                episode_return=episode_return,
                worker_type=profile.name,
                value_errors=t.ones(length, dtype=t.float32),
            )
            _put_trajectory(
                trajectory_queue,
                {
                    "type": "trajectory",
                    "worker_id": worker_id,
                    "weight_step": latest_step,
                    "trajectory_payload": _trajectory_payload(trajectory),
                },
                shutdown,
            )
    finally:
        if hasattr(env, "close"):
            env.close()


def _worker_loop(config, worker_id, profile, env_factory, network_factory,
                 search_factory, level, weight_queue, trajectory_queue,
                 shutdown, device):
    t.set_num_threads(1)
    random.seed(worker_id)
    generator = t.Generator().manual_seed(worker_id)
    current_level = level
    env = env_factory(level, config)
    network = network_factory(config, device)
    adapter = MuZeroModelAdapter(network, config)
    search = search_factory(config)
    latest_step = -1

    active_levels = [level]
    level_weights = [1.0]
    level_sims = {level: config.mcts_num_simulations}
    max_steps = config.self_play_max_episode_steps

    try:
        while not shutdown.is_set():
            while True:
                try:
                    msg = weight_queue.get_nowait()
                except queue.Empty:
                    break
                if msg.get("type") == "weights" and int(msg["step"]) > latest_step:
                    network.load_weights(msg.get("weights"))
                    latest_step = int(msg["step"])
                    if "active_levels" in msg:
                        active_levels = msg["active_levels"]
                        level_weights = msg.get("level_weights", [1.0] * len(active_levels))
                    if "level_sims" in msg:
                        level_sims = msg["level_sims"]
                    if "max_steps" in msg:
                        max_steps = msg["max_steps"]

            new_level = _choose_level(active_levels, level_weights, current_level, worker_id)

            if new_level != current_level:
                if hasattr(env, "close"):
                    env.close()
                env = env_factory(new_level, config)
                current_level = new_level

            num_sims = level_sims.get(current_level, config.mcts_num_simulations)

            trajectory = collect_episode(
                env,
                adapter,
                search,
                config,
                profile,
                level=current_level,
                device=device,
                generator=generator,
                max_steps=max_steps,
                num_simulations=num_sims,
            )
            if trajectory.actions.shape[0] < config.unroll_steps:
                continue
            item = {
                "type": "trajectory",
                "worker_id": worker_id,
                "weight_step": latest_step,
                "trajectory_payload": _trajectory_payload(trajectory),
            }
            while not shutdown.is_set():
                try:
                    trajectory_queue.put(item, timeout=0.1)
                    break
                except queue.Full:
                    continue
    finally:
        if hasattr(env, "close"):
            env.close()


class SelfPlayProcessGroup:
    def __init__(self, config, env_factory=make_mario_env, levels=None, device="cuda",
                 network_factory=None, search_factory=make_gumbel_search):
        self.config = config
        self.env_factory = env_factory
        self.network_factory = network_factory
        self.search_factory = search_factory
        self.levels = levels or ["YoshiIsland2"]
        self.device = device
        self.shutdown = mp.Event()
        self.trajectory_queue = mp.Queue(maxsize=config.self_play_trajectory_queue_size)
        self.weight_queues = []
        self.search_weight_queue = None
        self.search_queue = None
        self.response_queues = []
        self.search_process = None
        self.processes = []

    def start(self):
        profiles = assign_profiles(self.config.self_play_workers, self.config)
        if self.network_factory is None:
            self.search_weight_queue = mp.Queue(maxsize=2)
            self.search_queue = mp.Queue(maxsize=self.config.self_play_search_queue_size)
            self.response_queues = [mp.Queue(maxsize=2) for _ in profiles]
            self.search_process = mp.Process(
                target=_search_service_loop,
                args=(
                    self.config,
                    self.search_weight_queue,
                    self.search_queue,
                    self.response_queues,
                    self.shutdown,
                    self.device,
                ),
            )
            self.search_process.start()

        for worker_id, profile in enumerate(profiles):
            weight_queue = mp.Queue(maxsize=2)
            level = self.levels[worker_id % len(self.levels)]
            if self.network_factory is None:
                process = mp.Process(
                    target=_worker_loop_batched,
                    args=(
                        self.config,
                        worker_id,
                        profile,
                        self.env_factory,
                        level,
                        weight_queue,
                        self.search_queue,
                        self.response_queues[worker_id],
                        self.trajectory_queue,
                        self.shutdown,
                    ),
                )
            else:
                process = mp.Process(
                    target=_worker_loop,
                    args=(
                        self.config,
                        worker_id,
                        profile,
                        self.env_factory,
                        self.network_factory,
                        self.search_factory,
                        level,
                        weight_queue,
                        self.trajectory_queue,
                        self.shutdown,
                        self.device,
                    ),
                )
            process.start()
            self.weight_queues.append(weight_queue)
            self.processes.append(process)

    def broadcast_weights(self, weights, step, **extras):
        if self.search_weight_queue is not None:
            while True:
                try:
                    self.search_weight_queue.get_nowait()
                except queue.Empty:
                    break
            self.search_weight_queue.put({"type": "weights", "weights": weights, "step": int(step)})

        msg_type = "control" if self.network_factory is None else "weights"
        msg = {"type": msg_type, "step": int(step)}
        if self.network_factory is not None:
            msg["weights"] = weights
        msg.update(extras)
        for weight_queue in self.weight_queues:
            while True:
                try:
                    weight_queue.get_nowait()
                except queue.Empty:
                    break
            weight_queue.put(msg)

    def get_trajectory(self, timeout=None):
        self.check_health()
        try:
            item = self.trajectory_queue.get(timeout=timeout)
        except queue.Empty:
            return None
        if item is not None and "trajectory_payload" in item:
            item["trajectory"] = _trajectory_from_payload(item.pop("trajectory_payload"))
        return item

    def check_health(self):
        failed = [
            (idx, process.exitcode)
            for idx, process in enumerate(self.processes)
            if process.exitcode not in (None, 0)
        ]
        if failed:
            details = ", ".join(f"{idx}:{code}" for idx, code in failed)
            raise RuntimeError(f"Self-play worker exited unexpectedly ({details})")
        if self.search_process is not None and self.search_process.exitcode not in (None, 0):
            raise RuntimeError(
                f"Self-play search process exited unexpectedly ({self.search_process.exitcode})"
            )

    def close(self, timeout=1.0):
        self.shutdown.set()
        for process in self.processes:
            process.join(timeout)
            if process.is_alive():
                process.terminate()
                process.join(timeout)
        if self.search_process is not None:
            self.search_process.join(timeout)
            if self.search_process.is_alive():
                self.search_process.terminate()
                self.search_process.join(timeout)
        for q in self.weight_queues + self.response_queues:
            q.close()
            q.cancel_join_thread()
        if self.search_weight_queue is not None:
            self.search_weight_queue.close()
            self.search_weight_queue.cancel_join_thread()
        if self.search_queue is not None:
            self.search_queue.close()
            self.search_queue.cancel_join_thread()
        self.trajectory_queue.close()
        self.trajectory_queue.cancel_join_thread()
