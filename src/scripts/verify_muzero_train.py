import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from dataclasses import replace
import queue
import torch as t

from muzero.config import FIRST_REAL_RUN_LEVELS, MUZERO_PRESETS, MuZeroConfig
from muzero.replay_buffer import Trajectory
from muzero_train import _AsyncMuZeroTrainer, load_muzero_checkpoint, save_muzero_checkpoint


class FakeSelfPlayGroup:
    def __init__(self, config, levels=None, device="cpu"):
        self.config = config
        self.levels = levels or ["fake"]
        self.device = device
        self.started = False
        self.closed = False
        self.weights = None
        self.step = 0
        self.extras = {}
        self.queue = queue.Queue()

    def start(self):
        self.started = True
        for idx in range(6):
            self.queue.put({
                "type": "trajectory",
                "worker_id": idx % max(1, self.config.self_play_workers),
                "weight_step": 0,
                "trajectory": self._trajectory(idx),
            })

    def broadcast_weights(self, weights, step, **extras):
        self.weights = weights
        self.step = step
        self.extras = extras

    def get_trajectory(self, timeout=None):
        try:
            return self.queue.get(timeout=timeout)
        except queue.Empty:
            return None

    def close(self, timeout=None):
        self.closed = True

    def _trajectory(self, offset):
        length = self.config.unroll_steps + self.config.td_steps + 1
        obs = t.zeros(length + 1, 4, 84, 84, dtype=t.uint8)
        for i in range(length + 1):
            obs[i].fill_(min(255, (offset + i) * 8))
        actions = t.arange(length, dtype=t.long) % self.config.num_actions
        rewards = t.linspace(0.0, 1.0, length)
        dones = t.zeros(length, dtype=t.bool)
        policy = t.full((length, self.config.num_actions), 1.0 / self.config.num_actions)
        return Trajectory(
            obs=obs,
            actions=actions,
            rewards=rewards,
            dones=dones,
            policy_targets=policy,
            value_targets=t.zeros(length),
            level="fake",
            x_max=float(offset * 10),
            episode_return=float(rewards.sum().item()),
            worker_type="standard",
            value_errors=t.ones(length),
        )


class FakeReanalyseProcess:
    def __init__(self, config, target_computer=None, device="cpu"):
        self.config = config
        self.started = False
        self.closed = False
        self.weights = None
        self.request_id = 0
        self.out = queue.Queue()

    def start(self):
        self.started = True

    def set_weights(self, weights, step):
        self.weights = weights

    def submit_batch(self, batch):
        request_id = self.request_id
        self.request_id += 1
        self.out.put({
            "type": "target_update",
            "request_id": request_id,
            "trajectory_ids": batch.trajectory_ids,
            "start_indices": batch.start_indices,
            "policy_targets": batch.policy_targets,
            "value_targets": batch.rewards[:, :self.config.unroll_steps].float(),
            "value_errors": t.ones(
                batch.actions.shape[0],
                self.config.unroll_steps,
            ),
        })
        return request_id

    def get(self, timeout=None):
        try:
            return self.out.get(timeout=timeout)
        except queue.Empty:
            return None

    def close(self):
        self.closed = True


def _config():
    return replace(
        MuZeroConfig(),
        latent_channels=8,
        latent_spatial=2,
        encoder_widths=(8, 8, 8),
        encoder_res_blocks=1,
        dynamics_channels=8,
        dynamics_res_blocks=1,
        num_actions=3,
        proj_dim=16,
        head_hidden_dim=16,
        value_support_min=-5,
        value_support_max=10,
        reward_support_min=-2,
        reward_support_max=5,
        replay_capacity=200,
        frontier_capacity=50,
        unroll_steps=3,
        td_steps=2,
        batch_size=2,
        reanalyse_batch_size=2,
        min_replay_transitions=6,
        num_training_steps=12,
        self_play_workers=2,
        self_play_max_episode_steps=6,
        weight_refresh_interval=1,
        checkpoint_freq=10_000,
        log_freq=10_000,
        show_progress=False,
        USE_WANDB=False,
    )


def main():
    preset_factory, levels = MUZERO_PRESETS["first_real"]
    preset = preset_factory()
    assert tuple(levels) == FIRST_REAL_RUN_LEVELS
    assert preset.plr_initial_levels == 4
    assert preset.plr_levels_per_addition == 0
    assert preset.eval_episodes_per_level == 2
    assert preset.self_play_workers == 32
    assert preset.self_play_search_batch_size == 32
    assert preset.batch_size == 512
    assert preset.reanalyse_batch_size == 512

    config = _config()
    trainer = _AsyncMuZeroTrainer(
        config,
        "cpu",
        levels=["fake"],
        self_play_factory=FakeSelfPlayGroup,
        reanalyse_factory=FakeReanalyseProcess,
    )
    network = trainer.run()
    assert trainer.self_play.closed
    assert trainer.reanalyse.closed
    assert trainer.replay.num_trajectories > 0
    assert trainer.reanalyse_updates > 0

    tracking = trainer._tracking()
    tracking["episodes"] = 1
    path = save_muzero_checkpoint(
        trainer.network,
        trainer.ema_encoder,
        trainer.optimizer,
        trainer.scheduler,
        tracking,
        config,
        1,
        rnd=trainer.rnd,
        plr=trainer.plr,
    )
    step, loaded = load_muzero_checkpoint(
        path,
        trainer.network,
        trainer.ema_encoder,
        trainer.optimizer,
        trainer.scheduler,
        resume=True,
        rnd=trainer.rnd,
        plr=trainer.plr,
    )
    assert step == 1
    assert loaded["episodes"] == 1
    assert trainer.self_play.extras["active_levels"] == ["fake"]
    assert sum(p.numel() for p in network.parameters()) > 0
    print("MuZero training verification passed.")


if __name__ == '__main__':
    main()
