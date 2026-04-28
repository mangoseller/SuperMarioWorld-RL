import argparse
from contextlib import nullcontext
import glob
import multiprocessing as mp
import os
import shutil
import time
import warnings
warnings.filterwarnings("ignore")

from dataclasses import replace

import torch as t
import torch.nn.functional as F
from einops import rearrange
from tqdm import tqdm

from muzero.config import MUZERO_PRESETS, MuZeroConfig
from muzero.encoder import EMAEncoder
from muzero.eval import run_muzero_eval
from muzero.heads import scalar_to_support
from muzero.network import MuZeroNetwork, serialize_worker_weights
from muzero.plr import PLRSampler
from muzero.reanalyse import MCTSTargetComputer, ReanalyseProcess, apply_reanalyse_update
from muzero.replay_buffer import DualPriorityReplayBuffer
from muzero.rnd import RNDModule
from muzero.self_play import SelfPlayProcessGroup
from utils import log_muzero_metrics, readable_timestamp


def _as_float_obs(obs, device):
    obs = obs.to(device, non_blocking=True)
    if obs.dtype == t.uint8:
        return obs.float() / 255.0
    obs = obs.float()
    if obs.max() > 1.0:
        obs = obs / 255.0
    return obs


def _weighted_mean(loss, weights):
    while weights.dim() < loss.dim():
        weights = weights.unsqueeze(-1)
    return (loss * weights).mean()


def save_muzero_checkpoint(network, ema_encoder, optimizer, scheduler,
                           tracking, config, step, rnd=None, plr=None):
    checkpoint_dir = "model_checkpoints"
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint = {
        "network_state_dict": network.state_dict(),
        "ema_encoder_state_dict": ema_encoder.target.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
        "step": step,
        "tracking": tracking,
        "config_dict": config.to_wandb_config(),
    }
    if rnd is not None:
        checkpoint["rnd_state_dict"] = rnd.state_dict()
    if plr is not None:
        checkpoint["plr_state_dict"] = plr.state_dict()
    path = os.path.join(checkpoint_dir, f"{config.architecture}_ep{tracking['episodes']}.pt")
    # Drop any stale partial-writes from a prior crash before consuming more disk.
    for stale in glob.glob(os.path.join(checkpoint_dir, f"{config.architecture}_ep*.pt*.tmp")):
        try:
            os.remove(stale)
        except OSError:
            pass
    keep_last = max(int(getattr(config, "checkpoint_keep_last", 2)), 1)
    estimated_bytes = _estimate_checkpoint_bytes(checkpoint, checkpoint_dir, config.architecture)
    needed_bytes = int(estimated_bytes * 1.2) + 64 * 1024 * 1024
    tmp_path = f"{path}.{os.getpid()}.tmp"
    try:
        _ensure_disk_space(checkpoint_dir, config.architecture, needed_bytes, keep_last)
        t.save(checkpoint, tmp_path)
        os.replace(tmp_path, path)
    except Exception as exc:
        if os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except OSError:
                pass
        print(f"Skipping checkpoint save after error: {exc}")
        return None
    tracking["last_checkpoint_step"] = step
    _prune_old_checkpoints(checkpoint_dir, config.architecture, keep_last=keep_last)
    return path


def _list_arch_checkpoints(checkpoint_dir, architecture):
    pattern = os.path.join(checkpoint_dir, f"{architecture}_ep*.pt")
    paths = glob.glob(pattern)
    def _ep(path):
        name = os.path.basename(path)
        try:
            return int(name.rsplit("_ep", 1)[1].split(".pt", 1)[0])
        except (IndexError, ValueError):
            return -1
    paths.sort(key=_ep)
    return paths


def _estimate_checkpoint_bytes(checkpoint, checkpoint_dir, architecture):
    existing = _list_arch_checkpoints(checkpoint_dir, architecture)
    if existing:
        try:
            return max(os.path.getsize(p) for p in existing)
        except OSError:
            pass
    total = 0
    def _walk(obj):
        nonlocal total
        if isinstance(obj, t.Tensor):
            total += obj.numel() * obj.element_size()
        elif isinstance(obj, dict):
            for v in obj.values():
                _walk(v)
        elif isinstance(obj, (list, tuple)):
            for v in obj:
                _walk(v)
    _walk(checkpoint)
    return int(total * 1.05) + 1024 * 1024


def _ensure_disk_space(checkpoint_dir, architecture, needed_bytes, keep_last):
    free_bytes = shutil.disk_usage(checkpoint_dir).free
    if free_bytes >= needed_bytes:
        return
    existing = _list_arch_checkpoints(checkpoint_dir, architecture)
    # Drop the oldest first; never touch the most recent until forced.
    deletable = existing[:-1] if len(existing) > 0 else []
    while free_bytes < needed_bytes and deletable:
        old = deletable.pop(0)
        try:
            sz = os.path.getsize(old)
            os.remove(old)
            free_bytes += sz
            print(f"Removed old checkpoint to free space: {old}")
        except OSError as exc:
            print(f"Could not remove {old}: {exc}")
    if free_bytes < needed_bytes and existing:
        # Last resort: drop the most recent too. The new save will replace it.
        old = existing[-1]
        try:
            sz = os.path.getsize(old)
            os.remove(old)
            free_bytes += sz
            print(f"Removed most recent checkpoint to free space: {old}")
        except OSError as exc:
            print(f"Could not remove {old}: {exc}")
    if free_bytes < needed_bytes:
        raise RuntimeError(
            f"Insufficient disk space at {checkpoint_dir}: "
            f"{free_bytes / 1e9:.2f} GB free, need {needed_bytes / 1e9:.2f} GB."
        )


def _prune_old_checkpoints(checkpoint_dir, architecture, keep_last):
    if keep_last <= 0:
        return
    existing = _list_arch_checkpoints(checkpoint_dir, architecture)
    for old in existing[:-keep_last]:
        try:
            os.remove(old)
        except OSError as exc:
            print(f"Could not remove old checkpoint {old}: {exc}")


def load_muzero_checkpoint(path, network, ema_encoder, optimizer=None,
                           scheduler=None, resume=False, rnd=None, plr=None):
    device = next(network.parameters()).device
    checkpoint = t.load(path, map_location=device)
    network.load_state_dict(checkpoint["network_state_dict"])
    ema_encoder.target.load_state_dict(checkpoint["ema_encoder_state_dict"])
    if resume and optimizer is not None:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    if resume and scheduler is not None and checkpoint.get("scheduler_state_dict"):
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
    if resume and rnd is not None and checkpoint.get("rnd_state_dict"):
        rnd.load_state_dict(checkpoint["rnd_state_dict"])
    if resume and plr is not None and checkpoint.get("plr_state_dict"):
        plr.load_state_dict(checkpoint["plr_state_dict"])
    return checkpoint.get("step", 0), checkpoint.get("tracking") if resume else None


class _AsyncMuZeroTrainer:
    def __init__(self, config, device, levels=None, self_play_factory=SelfPlayProcessGroup,
                 reanalyse_factory=ReanalyseProcess):
        self.config = config
        self.device = t.device(device)
        self.plr = PLRSampler(config, level_pool=levels)
        self.levels = list(self.plr.active_levels)
        self.network = MuZeroNetwork(config).to(self.device)
        self.ema_encoder = EMAEncoder(self.network.encoder, config.ema_decay).to(self.device)
        self.optimizer = self._make_optimizer()
        self.scheduler = self._make_scheduler()
        self._setup_compiled_modules()
        self.replay = DualPriorityReplayBuffer(config)
        self.self_play = self_play_factory(
            config,
            levels=self.levels,
            device=config.self_play_device,
        )
        self.reanalyse = reanalyse_factory(
            config,
            target_computer=MCTSTargetComputer(),
            device=config.reanalyse_device,
        )
        self.rnd = RNDModule(config, device)
        self.pending_reanalyse = set()
        self.reanalyse_requests = 0
        self.reanalyse_updates = 0
        self.reanalyse_errors = 0
        self.self_play_errors = 0
        self.last_weight_step = -1
        self._priority_refresh_counter = 0

    def _setup_compiled_modules(self):
        compile_enabled = self.device.type == "cuda" and self.config.use_compile
        wrap = t.compile if compile_enabled else (lambda m: m)
        self._encoder_c = wrap(self.network.encoder)
        self._dynamics_c = wrap(self.network.dynamics)
        self._policy_c = wrap(self.network.policy)
        self._value_c = wrap(self.network.value)
        self._reward_c = wrap(self.network.reward)
        self._ema_target_c = wrap(self.ema_encoder.target)

    def _make_scheduler(self):
        if self.config.lr_schedule != "cosine":
            return None
        total_steps = max(1, self.config.num_training_steps)
        warmup = max(0, min(self.config.lr_warmup_steps, total_steps - 1))
        cosine = t.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=max(1, total_steps - warmup),
            eta_min=self.config.min_lr,
        )
        if warmup == 0:
            return cosine
        warmup_sched = t.optim.lr_scheduler.LinearLR(
            self.optimizer,
            start_factor=0.01,
            end_factor=1.0,
            total_iters=warmup,
        )
        return t.optim.lr_scheduler.SequentialLR(
            self.optimizer,
            schedulers=[warmup_sched, cosine],
            milestones=[warmup],
        )

    def _make_optimizer(self):
        kwargs = {
            "lr": self.config.learning_rate,
            "weight_decay": self.config.weight_decay,
        }
        if self.device.type == "cuda":
            kwargs["fused"] = True
        try:
            return t.optim.AdamW(self.network.parameters(), **kwargs)
        except TypeError:
            kwargs.pop("fused", None)
            return t.optim.AdamW(self.network.parameters(), **kwargs)

    def _autocast(self):
        if self.device.type != "cuda" or not self.config.use_amp:
            return nullcontext()
        return t.autocast(device_type="cuda", dtype=t.bfloat16)

    def _weights_for_workers(self):
        return serialize_worker_weights(self.network.weights_for_workers())

    def _broadcast_weights(self, step, force=False):
        if not force and step - self.last_weight_step < self.config.weight_refresh_interval:
            return
        weights = self._weights_for_workers()
        plr_info = self.plr.get_broadcast_info(step)
        self.self_play.broadcast_weights(weights, step, **plr_info)
        self.reanalyse.set_weights(weights, step)
        self.last_weight_step = step

    def _dynamics_loss(self, z_pred, z_target):
        pred = self.network.dynamics_head.predictor(
            self.network.dynamics_head.project(z_pred)
        )
        with t.no_grad():
            target = self.network.dynamics_head.project(z_target)
        pred = F.normalize(pred, dim=-1)
        target = F.normalize(target, dim=-1)
        return -(pred * target).sum(-1)

    def _value_loss(self, logits, targets):
        target = scalar_to_support(
            targets,
            self.config.value_support_min,
            self.config.value_support_max,
        )
        return -(target * F.log_softmax(logits, dim=-1)).sum(-1)

    def _reward_loss(self, logits, targets):
        target = scalar_to_support(
            targets,
            self.config.reward_support_min,
            self.config.reward_support_max,
        )
        return -(target * F.log_softmax(logits, dim=-1)).sum(-1)

    def _policy_loss(self, logits, targets):
        targets = targets / targets.sum(-1, keepdim=True).clamp_min(1e-8)
        return -(targets * F.log_softmax(logits, dim=-1)).sum(-1)

    def _check_batch(self, batch):
        expected_obs_steps = self.config.unroll_steps + 1
        if batch.obs.shape[1] != expected_obs_steps:
            raise ValueError("training batch has incorrect observation length")
        if not t.isfinite(batch.value_targets.float()).all():
            raise FloatingPointError("non-finite value target")
        if not t.isfinite(batch.rewards.float()).all():
            raise FloatingPointError("non-finite reward target")
        policy_sums = batch.policy_targets.float().sum(-1)
        if not t.allclose(policy_sums, t.ones_like(policy_sums), atol=1e-3):
            raise ValueError("policy targets must sum to one")

    def _compute_loss(self, batch):
        self._check_batch(batch)
        obs = _as_float_obs(batch.obs, self.device)
        actions = batch.actions.to(self.device, non_blocking=True)
        rewards = batch.rewards.to(self.device, non_blocking=True)
        policy_targets = batch.policy_targets.to(self.device, non_blocking=True)
        value_targets = batch.value_targets.to(self.device, non_blocking=True)
        weights = batch.weights.to(self.device, non_blocking=True)
        b = obs.shape[0]
        unroll = self.config.unroll_steps

        target_obs = rearrange(obs[:, 1:unroll + 1], 'b k c h w -> (b k) c h w')
        with t.no_grad():
            target_zs = self._ema_target_c(target_obs)
        target_zs = rearrange(target_zs, '(b k) c h w -> b k c h w', b=b, k=unroll)

        z = self._encoder_c(obs[:, 0])
        dynamics_losses, reward_losses = [], []
        value_losses, policy_losses = [], []

        for step in range(unroll):
            policy_logits = self._policy_c(z)
            value_logits = self._value_c(z)
            policy_losses.append(self._policy_loss(policy_logits, policy_targets[:, step]))
            value_losses.append(self._value_loss(value_logits, value_targets[:, step]))

            z_next = self._dynamics_c(z, actions[:, step])
            reward_logits = self._reward_c(z_next)
            reward_losses.append(self._reward_loss(reward_logits, rewards[:, step]))

            dynamics_losses.append(self._dynamics_loss(z_next, target_zs[:, step]))
            z = z_next

        dynamics_loss = _weighted_mean(t.stack(dynamics_losses, dim=1), weights)
        reward_loss = _weighted_mean(t.stack(reward_losses, dim=1), weights)
        value_loss = _weighted_mean(t.stack(value_losses, dim=1), weights)
        policy_loss = _weighted_mean(t.stack(policy_losses, dim=1), weights)
        total_loss = dynamics_loss + reward_loss + value_loss + policy_loss

        if not t.isfinite(total_loss):
            raise FloatingPointError("non-finite MuZero loss")

        return total_loss, {
            "total_loss": total_loss.detach(),
            "dynamics_loss": dynamics_loss.detach(),
            "reward_loss": reward_loss.detach(),
            "value_loss": value_loss.detach(),
            "policy_loss": policy_loss.detach(),
        }

    @t.no_grad()
    def _refresh_sample_priorities(self, batch):
        obs = _as_float_obs(batch.obs[:, :self.config.unroll_steps], self.device)
        b, k = obs.shape[:2]
        flat_obs = rearrange(obs, "b k c h w -> (b k) c h w")
        z = self._encoder_c(flat_obs)
        logits = self._value_c(z)
        values = self.network.value.predict(logits)
        values = rearrange(values, "(b k) -> b k", b=b, k=k).cpu()
        errors = (values - batch.value_targets.float()).abs().clamp_min(self.config.priority_eps)
        self.replay.update_targets(
            batch.trajectory_ids,
            batch.start_indices,
            value_errors=errors,
        )
        for i, traj_id in enumerate(batch.trajectory_ids.tolist()):
            level = self.replay.get_trajectory_level(int(traj_id))
            if level is not None:
                self.plr.record_value_error(level, float(errors[i].mean()))

    def _train_step(self):
        batch = self.replay.sample_batch(self.config.batch_size, extra_steps=0)
        self.optimizer.zero_grad(set_to_none=True)
        with self._autocast():
            total_loss, diagnostics = self._compute_loss(batch)
        total_loss.backward()
        t.nn.utils.clip_grad_norm_(
            self.network.parameters(),
            self.config.gradient_clip_norm,
        )
        self.optimizer.step()
        if self.scheduler is not None:
            self.scheduler.step()
        self.ema_encoder.update()
        self._priority_refresh_counter += 1
        if self._priority_refresh_counter % max(1, self.config.priority_refresh_interval) == 0:
            self._refresh_sample_priorities(batch)
        rnd_loss = self.rnd.train_step(batch.obs[:, 0])
        diagnostics["rnd_loss"] = rnd_loss
        return diagnostics

    def _submit_reanalyse(self):
        if len(self.pending_reanalyse) >= self.config.max_pending_reanalyse:
            return
        if self.reanalyse_requests % max(1, self.config.reanalyse_interval) != 0:
            self.reanalyse_requests += 1
            return
        batch = self.replay.sample_batch(
            self.config.reanalyse_batch_size,
            extra_steps=self.config.td_steps,
        )
        expected_obs_steps = self.config.unroll_steps + self.config.td_steps + 1
        if batch.obs.shape[1] != expected_obs_steps:
            raise ValueError("reanalyse batch has incorrect observation length")
        request_id = self.reanalyse.submit_batch(batch)
        self.pending_reanalyse.add(request_id)
        self.reanalyse_requests += 1

    def _drain_reanalyse(self):
        while True:
            msg = self.reanalyse.get(timeout=0.0)
            if msg is None:
                break
            kind = msg.get("type")
            if kind == "target_update":
                apply_reanalyse_update(self.replay, msg)
                self.pending_reanalyse.discard(msg.get("request_id"))
                self.reanalyse_updates += 1
            elif kind == "error":
                self.reanalyse_errors += 1
                raise RuntimeError(f"Reanalyse failed: {msg.get('error')}")

    def _check_process_health(self):
        if hasattr(self.self_play, "check_health"):
            self.self_play.check_health()
        if hasattr(self.reanalyse, "check_health"):
            self.reanalyse.check_health()

    def _drain_self_play(self, tracking, max_items=64):
        added = 0
        for _ in range(max_items):
            item = self.self_play.get_trajectory(timeout=0.0)
            if item is None:
                break
            if item.get("type") != "trajectory":
                self.self_play_errors += 1
                raise RuntimeError(f"Unexpected self-play message: {item}")
            trajectory = item["trajectory"]
            coef = self.rnd.current_coef(tracking["env_steps"])
            if coef > 0.0:
                intrinsic = self.rnd.compute_intrinsic(
                    trajectory.obs[:-1], tracking["env_steps"]
                )
                trajectory = replace(trajectory, rewards=trajectory.rewards + intrinsic)
            self.replay.add_trajectory(trajectory)
            length = int(trajectory.actions.shape[0])
            tracking["env_steps"] += length
            tracking["episodes"] += 1
            tracking["episode_returns"].append(float(trajectory.episode_return))
            tracking["episode_lengths"].append(length)
            tracking["x_max"].append(float(trajectory.x_max))
            tracking["worker_type_counts"][trajectory.worker_type] = (
                tracking["worker_type_counts"].get(trajectory.worker_type, 0) + 1
            )
            self.plr.record_episode(trajectory.level, float(trajectory.episode_return))
            added += 1
        return added

    def _tracking(self):
        return {
            "episode_returns": [],
            "episode_lengths": [],
            "x_max": [],
            "episodes": 0,
            "env_steps": 0,
            "gradient_steps": 0,
            "run_timestamp": readable_timestamp(),
            "last_checkpoint_step": 0,
            "last_eval_step": 0,
            "worker_type_counts": {},
        }

    def run(self, checkpoint_path=None, resume=False):
        run = self.config.setup_wandb()
        tracking = self._tracking()
        start_step = 0

        if checkpoint_path:
            start_step, saved_tracking = load_muzero_checkpoint(
                checkpoint_path,
                self.network,
                self.ema_encoder,
                self.optimizer,
                self.scheduler,
                resume=resume,
                rnd=self.rnd,
                plr=self.plr,
            )
            if resume and saved_tracking:
                tracking = saved_tracking
                tracking["run_timestamp"] = readable_timestamp()

        self.self_play.start()
        self.reanalyse.start()
        self._broadcast_weights(start_step, force=True)

        pbar = tqdm(
            total=self.config.num_training_steps,
            initial=min(tracking["env_steps"], self.config.num_training_steps),
            disable=not self.config.show_progress,
            desc="Async MuZero",
            unit="env_step",
        )
        last_log_step = tracking["env_steps"]
        last_progress_time = 0.0
        diagnostics = {}

        try:
            while tracking["env_steps"] < self.config.num_training_steps:
                new_episodes = self._drain_self_play(tracking)
                self._drain_reanalyse()
                self._check_process_health()

                if new_episodes > 0 and self.plr.maybe_add_levels():
                    self._broadcast_weights(tracking["env_steps"], force=True)

                replay_ready = (
                    len(self.replay) >= self.config.min_replay_transitions
                    and self.replay.num_trajectories > 0
                )
                if replay_ready:
                    train_steps = max(1, int(getattr(self.config, "train_steps_per_iter", 1)))
                    for _ in range(train_steps):
                        self._submit_reanalyse()
                        diagnostics = self._train_step()
                        tracking["gradient_steps"] += 1
                    self._broadcast_weights(tracking["env_steps"])

                    if (
                        tracking["env_steps"] - tracking.get("last_eval_step", 0)
                        >= self.config.eval_freq
                    ):
                        eval_stats = run_muzero_eval(
                            self.network,
                            self.config,
                            self.plr.active_levels,
                        )
                        log_muzero_metrics(
                            config=self.config,
                            step=tracking["env_steps"],
                            eval_stats=eval_stats,
                        )
                        tracking["last_eval_step"] = tracking["env_steps"]

                if tracking["env_steps"] - last_log_step >= self.config.log_freq:
                    log_muzero_metrics(
                        tracking=tracking,
                        diagnostics=diagnostics,
                        replay=self.replay,
                        self_play={
                            "episodes_added": tracking["episodes"],
                            "errors": self.self_play_errors,
                        },
                        reanalyse={
                            "pending": len(self.pending_reanalyse),
                            "updates": self.reanalyse_updates,
                            "errors": self.reanalyse_errors,
                        },
                        optimizer=self.optimizer,
                        config=self.config,
                        step=tracking["env_steps"],
                        plr_stats=self.plr.level_stats(),
                        rnd_coef=self.rnd.current_coef(tracking["env_steps"]),
                    )
                    tracking["episode_returns"].clear()
                    tracking["episode_lengths"].clear()
                    tracking["x_max"].clear()
                    last_log_step = tracking["env_steps"]

                if (
                    tracking["env_steps"] - tracking.get("last_checkpoint_step", 0)
                    >= self.config.checkpoint_freq
                ):
                    save_muzero_checkpoint(
                        self.network,
                        self.ema_encoder,
                        self.optimizer,
                        self.scheduler,
                        tracking,
                        self.config,
                        tracking["env_steps"],
                        rnd=self.rnd,
                        plr=self.plr,
                    )

                now = time.time()
                if (
                    self.config.show_progress
                    and now - last_progress_time >= self.config.progress_refresh_sec
                ):
                    pbar.n = min(tracking["env_steps"], self.config.num_training_steps)
                    pbar.set_postfix({
                        "ep": tracking["episodes"],
                        "replay": len(self.replay),
                        "grad": tracking["gradient_steps"],
                    })
                    pbar.refresh()
                    last_progress_time = now

                if not replay_ready:
                    time.sleep(0.05)

        except KeyboardInterrupt:
            print("\nInterrupted by user.")
        finally:
            try:
                self._drain_reanalyse()
            except Exception as exc:
                print(f"Skipping final reanalyse drain after error: {exc}")
            finally:
                self.self_play.close(timeout=1.0)
                self.reanalyse.close()
                pbar.close()
            path = save_muzero_checkpoint(
                self.network,
                self.ema_encoder,
                self.optimizer,
                self.scheduler,
                tracking,
                self.config,
                tracking["env_steps"],
                rnd=self.rnd,
                plr=self.plr,
            )
            if self.config.USE_WANDB:
                import wandb
                wandb.finish()
            if path is not None:
                print(f"Saved MuZero checkpoint: {path}")

        return self.network


def train_async(config=None, levels=None, checkpoint_path=None, resume=False,
                device=None):
    config = MuZeroConfig() if config is None else config
    device = device or config.training_device
    if str(device).startswith("cuda") and not t.cuda.is_available():
        device = "cpu"
    if str(device).startswith("cuda"):
        mp.set_start_method("spawn", force=True)
        t.backends.cuda.matmul.allow_tf32 = True
        t.backends.cudnn.allow_tf32 = True
        t._dynamo.config.cache_size_limit = 64
    t.multiprocessing.set_sharing_strategy("file_system")
    t.set_float32_matmul_precision("high")
    trainer = _AsyncMuZeroTrainer(config, device, levels=levels)
    return trainer.run(checkpoint_path=checkpoint_path, resume=resume)


def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--preset", type=str, default=None, choices=tuple(MUZERO_PRESETS.keys()))
    parser.add_argument("--total_steps", type=int, default=None)
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--mcts_sims", type=int, default=None)
    parser.add_argument("--mcts_depth", type=int, default=None)
    parser.add_argument("--mcts_backend", type=str, default=None, choices=("tensor", "python"))
    parser.add_argument("--max_episode_steps", type=int, default=None)
    parser.add_argument("--min_replay_transitions", type=int, default=None)
    parser.add_argument("--search_batch_size", type=int, default=None)
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--self_play_device", type=str, default=None)
    parser.add_argument("--reanalyse_device", type=str, default=None)
    parser.add_argument("--levels", type=str, default=None)
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--no_wandb", action="store_true")
    return parser.parse_args()


def main():
    args = _parse_args()
    preset_levels = None
    if args.preset is not None:
        config_factory, preset_levels = MUZERO_PRESETS[args.preset]
        config = config_factory()
    else:
        config = MuZeroConfig()
    updates = {}
    if args.total_steps is not None:
        updates["num_training_steps"] = args.total_steps
    if args.workers is not None:
        updates["self_play_workers"] = args.workers
    if args.batch_size is not None:
        updates["batch_size"] = args.batch_size
        updates["reanalyse_batch_size"] = args.batch_size
    if args.mcts_sims is not None:
        updates["mcts_num_simulations"] = args.mcts_sims
    if args.mcts_depth is not None:
        updates["mcts_max_depth"] = args.mcts_depth
    if args.mcts_backend is not None:
        updates["mcts_backend"] = args.mcts_backend
    if args.max_episode_steps is not None:
        updates["self_play_max_episode_steps"] = args.max_episode_steps
    if args.min_replay_transitions is not None:
        updates["min_replay_transitions"] = args.min_replay_transitions
    if args.search_batch_size is not None:
        updates["self_play_search_batch_size"] = args.search_batch_size
    if args.device is not None:
        updates["training_device"] = args.device
    if args.self_play_device is not None:
        updates["self_play_device"] = args.self_play_device
    if args.reanalyse_device is not None:
        updates["reanalyse_device"] = args.reanalyse_device
    if args.wandb:
        updates["USE_WANDB"] = True
    if args.no_wandb:
        updates["USE_WANDB"] = False
    if updates:
        config = replace(config, **updates)
    levels = list(preset_levels) if preset_levels is not None else None
    if args.levels:
        levels = [level.strip() for level in args.levels.split(",") if level.strip()]
    train_async(
        config=config,
        levels=levels,
        checkpoint_path=args.checkpoint,
        resume=args.resume,
        device=args.device,
    )


if __name__ == "__main__":
    main()
