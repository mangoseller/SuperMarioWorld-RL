import multiprocessing as mp
import queue
import torch as t
import torch.nn.functional as F
from einops import rearrange

from muzero.dynamics import DynamicsModel
from muzero.encoder import MuZeroEncoder
from muzero.heads import PolicyHead, RewardHead, ValueHead
from muzero.mcts import (
    BatchedGumbelMuZeroSearch,
    GumbelMuZeroSearch,
    MuZeroModelAdapter,
    TensorGumbelMuZeroSearch,
)
from muzero.network import deserialize_worker_weights


def _as_float_obs(obs):
    if obs.dtype == t.uint8:
        return obs.float() / 255.0
    obs = obs.float()
    if obs.max() > 1.0:
        obs = obs / 255.0
    return obs


def _resolve_device(device):
    device = t.device(device)
    if device.type == "cuda" and not t.cuda.is_available():
        return t.device("cpu")
    return device


def _autocast_for(device, enabled=True):
    if t.device(device).type != "cuda" or not enabled:
        return t.autocast(device_type="cpu", enabled=False)
    return t.autocast(device_type="cuda", dtype=t.bfloat16)


class ReanalyseNetwork:
    def __init__(self, config, device):
        self.encoder = MuZeroEncoder(config).to(device)
        self.dynamics = DynamicsModel(config).to(device)
        self.policy = PolicyHead(config).to(device)
        self.value = ValueHead(config).to(device)
        self.reward = RewardHead(config).to(device)
        self.encoder.eval()
        self.dynamics.eval()
        self.policy.eval()
        self.value.eval()
        self.reward.eval()

    def load_weights(self, weights):
        if weights is None:
            return
        weights = deserialize_worker_weights(weights)
        if "encoder" in weights:
            self.encoder.load_state_dict(weights["encoder"])
        if "dynamics" in weights:
            self.dynamics.load_state_dict(weights["dynamics"])
        if "policy" in weights:
            self.policy.load_state_dict(weights["policy"])
        if "value" in weights:
            self.value.load_state_dict(weights["value"])
        if "reward" in weights:
            self.reward.load_state_dict(weights["reward"])


def make_batch_search(config):
    backend = getattr(config, "mcts_backend", "tensor")
    if backend == "tensor":
        return TensorGumbelMuZeroSearch(config)
    if backend == "python":
        return BatchedGumbelMuZeroSearch(config)
    raise ValueError(f"unknown mcts_backend: {backend!r}")


def _bootstrap_targets(batch, values, config):
    b = batch.actions.shape[0]
    target_steps = batch.target_steps
    reward_steps = batch.rewards.shape[1]
    rewards = batch.rewards.float()
    dones = batch.dones.bool()
    value_targets = t.zeros(b, target_steps, dtype=t.float32)

    for root in range(target_steps):
        alive = t.ones(b, dtype=t.bool)
        discount = t.ones(b, dtype=t.float32)
        target = t.zeros(b, dtype=t.float32)

        max_horizon = min(config.td_steps, reward_steps - root)
        for offset in range(max_horizon):
            step = root + offset
            target += alive.float() * discount * rewards[:, step]
            alive &= ~dones[:, step]
            discount *= config.gamma

        bootstrap_step = root + max_horizon
        if bootstrap_step < values.shape[1]:
            target += alive.float() * discount * values[:, bootstrap_step]

        value_targets[:, root] = target

    return value_targets


class CurrentNetworkTargetComputer:
    requires_network = True

    def compute(self, batch, network, config, device):
        obs = _as_float_obs(batch.obs.to(device))
        b, obs_steps = obs.shape[:2]
        target_steps = batch.target_steps

        flat_obs = rearrange(obs, 'b t c h w -> (b t) c h w')
        with t.no_grad(), _autocast_for(device, config.use_amp):
            z = network.encoder(flat_obs)
            policy_logits = network.policy(z)
            value_logits = network.value(z)
            policy = F.softmax(policy_logits, dim=-1)
            values = network.value.predict(value_logits)

        policy = rearrange(policy, '(b t) a -> b t a', b=b, t=obs_steps)[:, :target_steps].cpu()
        values = rearrange(values, '(b t) -> b t', b=b, t=obs_steps).cpu()

        value_targets = _bootstrap_targets(batch, values, config)
        value_errors = (values[:, :target_steps] - value_targets).abs()
        return {
            "policy_targets": policy,
            "value_targets": value_targets,
            "value_errors": value_errors,
        }


class MCTSTargetComputer:
    requires_network = True

    def __init__(self, num_simulations=None):
        self.num_simulations = num_simulations

    def compute(self, batch, network, config, device):
        obs = _as_float_obs(batch.obs.to(device))
        b, obs_steps = obs.shape[:2]
        target_steps = batch.target_steps

        flat_obs = rearrange(obs, 'b t c h w -> (b t) c h w')
        with t.no_grad(), _autocast_for(device, config.use_amp):
            z = network.encoder(flat_obs)
            value_logits = network.value(z)
            values = network.value.predict(value_logits)

        z = rearrange(z, '(b t) c h w -> b t c h w', b=b, t=obs_steps)
        values = rearrange(values, '(b t) -> b t', b=b, t=obs_steps).cpu()
        value_targets = _bootstrap_targets(batch, values, config)

        search = make_batch_search(config)
        adapter = MuZeroModelAdapter(network, config)
        policy_targets = t.zeros(b, target_steps, config.num_actions, dtype=t.float32)
        root_refs, root_states = [], []

        for i in range(b):
            for root in range(target_steps):
                if root > 0 and bool(batch.dones[i, root - 1]):
                    policy_targets[i, root] = F.one_hot(
                        batch.actions[i, root].long(),
                        num_classes=config.num_actions,
                    ).float()
                    continue
                root_refs.append((i, root))
                root_states.append(z[i, root])

        if root_states:
            root_states = t.stack(root_states, dim=0)
            with _autocast_for(device, config.use_amp):
                output = search.run_batch(
                    adapter,
                    state=root_states,
                    num_simulations=self.num_simulations,
                )
            for row, (i, root) in enumerate(root_refs):
                policy_targets[i, root] = output.action_weights[row].cpu()

        value_errors = (values[:, :target_steps] - value_targets).abs()
        return {
            "policy_targets": policy_targets,
            "value_targets": value_targets,
            "value_errors": value_errors,
        }


class DeterministicTargetComputer:
    requires_network = False

    def compute(self, batch, network, config, device):
        b = batch.actions.shape[0]
        target_steps = batch.target_steps
        reward_steps = batch.rewards.shape[1]
        policy = F.one_hot(batch.actions[:, :target_steps].long(), num_classes=config.num_actions).float()
        values = t.zeros(b, target_steps, dtype=t.float32)

        for root in range(target_steps):
            alive = t.ones(b, dtype=t.bool)
            discount = t.ones(b, dtype=t.float32)
            target = t.zeros(b, dtype=t.float32)
            for offset in range(min(config.td_steps, reward_steps - root)):
                step = root + offset
                target += alive.float() * discount * batch.rewards[:, step].float()
                alive &= ~batch.dones[:, step].bool()
                discount *= config.gamma
            values[:, root] = target

        return {
            "policy_targets": policy,
            "value_targets": values,
            "value_errors": values.abs().clamp_min(config.priority_eps),
        }


def apply_reanalyse_update(buffer, update):
    buffer.update_targets(
        update["trajectory_ids"],
        update["start_indices"],
        policy_targets=update["policy_targets"],
        value_targets=update["value_targets"],
        value_errors=update["value_errors"],
    )


def _worker_loop(config, in_queue, out_queue, target_computer, device):
    device = _resolve_device(device)
    network = None
    if getattr(target_computer, "requires_network", True):
        network = ReanalyseNetwork(config, device)
    latest_step = -1

    while True:
        message = in_queue.get()
        kind = message.get("type")

        if kind == "shutdown":
            out_queue.put({"type": "shutdown", "ok": True})
            return

        if kind == "weights":
            step = int(message.get("step", 0))
            if step > latest_step:
                try:
                    if network is not None:
                        network.load_weights(message.get("weights"))
                except Exception as exc:
                    out_queue.put({"type": "error", "step": step, "error": repr(exc)})
                    continue
                latest_step = step
                out_queue.put({"type": "weights_loaded", "step": step})
            else:
                out_queue.put({"type": "weights_ignored", "step": step, "latest_step": latest_step})
            continue

        if kind == "batch":
            batch = message["batch"]
            request_id = message.get("request_id")
            try:
                targets = target_computer.compute(batch, network, config, device)
            except Exception as exc:
                out_queue.put({"type": "error", "request_id": request_id, "error": repr(exc)})
                continue
            out_queue.put({
                "type": "target_update",
                "request_id": request_id,
                "weight_step": latest_step,
                "trajectory_ids": batch.trajectory_ids.cpu(),
                "start_indices": batch.start_indices.cpu(),
                "policy_targets": targets["policy_targets"].cpu(),
                "value_targets": targets["value_targets"].cpu(),
                "value_errors": targets["value_errors"].cpu(),
            })
            continue

        out_queue.put({"type": "error", "error": f"unknown message type: {kind}"})


class ReanalyseProcess:
    def __init__(self, config, target_computer=None, device="cuda"):
        self.config = config
        self.target_computer = target_computer or CurrentNetworkTargetComputer()
        self.device = device
        self.in_queue = mp.Queue(maxsize=4)
        self.out_queue = mp.Queue(maxsize=4)
        self.process = None
        self._request_id = 0

    def start(self):
        if self.process is not None and self.process.is_alive():
            return
        self.process = mp.Process(
            target=_worker_loop,
            args=(self.config, self.in_queue, self.out_queue, self.target_computer, self.device),
        )
        self.process.start()

    def set_weights(self, weights, step):
        self.in_queue.put({"type": "weights", "weights": weights, "step": int(step)})

    def submit_batch(self, batch):
        request_id = self._request_id
        self._request_id += 1
        self.in_queue.put({"type": "batch", "batch": batch, "request_id": request_id})
        return request_id

    def get(self, timeout=None):
        self.check_health()
        try:
            return self.out_queue.get(timeout=timeout)
        except queue.Empty:
            return None

    def check_health(self):
        if self.process is not None and self.process.exitcode not in (None, 0):
            raise RuntimeError(f"Reanalyse process exited with code {self.process.exitcode}")

    def close(self, timeout=5.0):
        if self.process is not None and self.process.is_alive():
            try:
                self.in_queue.put({"type": "shutdown"}, timeout=timeout)
            except queue.Full:
                pass
            self.process.join(timeout)
        if self.process is not None and self.process.is_alive():
            self.process.terminate()
            self.process.join(timeout)
        self.in_queue.close()
        self.in_queue.cancel_join_thread()
        self.out_queue.close()
        self.out_queue.cancel_join_thread()
