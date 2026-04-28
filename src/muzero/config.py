from dataclasses import dataclass, replace
import os
import wandb


FIRST_REAL_RUN_LEVELS = (
    "YoshiIsland2",
    "DonutPlains1",
    "VanillaDome4",
    "Bridges2",
)


@dataclass
class MuZeroConfig:
    architecture: str = "MuZero"
    latent_channels: int = 256
    latent_spatial: int = 8
    encoder_widths: tuple = (128, 256, 256)
    encoder_res_blocks: int = 8
    dynamics_channels: int = 512
    dynamics_res_blocks: int = 22
    num_actions: int = 14
    ema_decay: float = 0.99
    proj_dim: int = 512

    head_hidden_dim: int = 256

    value_support_min: int = -30
    value_support_max: int = 150
    reward_support_min: int = -15
    reward_support_max: int = 65

    gamma: float = 0.997

    replay_capacity: int = 1_000_000
    frontier_capacity: int = 100_000
    unroll_steps: int = 10
    td_steps: int = 5
    replay_alpha: float = 0.6
    replay_beta: float = 0.4
    priority_eps: float = 1e-3
    frontier_sample_fraction: float = 0.5
    frontier_margin: float = 16.0
    x_percentile_window: int = 10_000
    weight_refresh_interval: int = 2_000
    reanalyse_batch_size: int = 256
    priority_refresh_interval: int = 4

    mcts_num_simulations: int = 32
    mcts_max_depth: int = 10
    mcts_max_num_considered_actions: int = 16
    mcts_gumbel_scale: float = 1.0
    mcts_value_scale: float = 0.1
    mcts_maxvisit_init: float = 50.0
    mcts_backend: str = "tensor"

    self_play_workers: int = 16
    self_play_max_episode_steps: int = 12000
    self_play_trajectory_queue_size: int = 32
    self_play_dirichlet_alpha: float = 0.3
    self_play_dirichlet_weight: float = 0.25
    self_play_aggressive_dirichlet_alpha: float = 1.0
    self_play_aggressive_dirichlet_weight: float = 0.75
    self_play_high_temperature: float = 2.0
    self_play_epsilon: float = 0.3
    self_play_device: str = "cuda"
    reanalyse_device: str = "cuda"
    self_play_search_batch_size: int = 32
    self_play_search_queue_size: int = 128
    self_play_search_batch_wait_ms: float = 2.0

    training_device: str = "cuda"
    num_training_steps: int = 4_000_000
    batch_size: int = 256
    learning_rate: float = 1e-4
    min_lr: float = 1e-6
    lr_schedule: str = "cosine"
    lr_warmup_steps: int = 5_000
    weight_decay: float = 1e-4
    min_replay_transitions: int = 10_000
    gradient_clip_norm: float = 1.0
    checkpoint_freq: int = 50_000
    checkpoint_keep_last: int = 2
    log_freq: int = 1_000
    reanalyse_interval: int = 1
    max_pending_reanalyse: int = 1
    train_steps_per_iter: int = 4
    use_amp: bool = True
    use_compile: bool = True
    progress_refresh_sec: float = 0.5
    USE_WANDB: bool = False
    wandb_project: str = "marioRL"
    show_progress: bool = True

    rnd_coef_start: float = 0.5
    rnd_coef_end: float = 0.0
    # All `*_steps` schedule thresholds below are in env_steps (not gradient_steps).
    rnd_anneal_steps: int = 1_500_000
    rnd_clip: float = 2.0
    rnd_channels: int = 64
    rnd_embedding_dim: int = 256

    plr_beta: float = 0.3
    plr_rho: float = 0.1
    plr_score_window: int = 10
    plr_solved_return: float = 5000.0
    plr_initial_levels: int = 6
    plr_levels_per_addition: int = 3

    # mcts_ramp / max_steps thresholds are in env_steps.
    mcts_ramp_start: int = 100_000
    mcts_ramp_end: int = 1_500_000
    mcts_sims_frontier: int = 64
    mcts_sims_mid: int = 48
    mcts_sims_easy: int = 32

    max_steps_early_multiplier: float = 2.0
    max_steps_early_end_step: int = 500_000

    eval_freq: int = 200_000
    eval_episodes_per_level: int = 3
    eval_mcts_sims: int = 64

    def to_wandb_config(self):
        return self.__dict__.copy()

    def setup_wandb(self):
        if not self.USE_WANDB:
            return
        api_key = os.environ.get("WANDB_API_KEY")
        if api_key is None:
            raise RuntimeError("WANDB_API_KEY not set in environment")
        wandb.login(key=api_key)
        return wandb.init(
            project=self.wandb_project,
            config=self.to_wandb_config(),
        )


def first_real_run_config():
    return replace(
        MuZeroConfig(),
        num_training_steps=4_000_000,
        self_play_workers=32,
        self_play_search_batch_size=32,
        self_play_search_queue_size=256,
        self_play_trajectory_queue_size=64,
        batch_size=512,
        reanalyse_batch_size=512,
        min_replay_transitions=20_000,
        plr_initial_levels=4,
        plr_levels_per_addition=0,
        eval_episodes_per_level=2,
        eval_freq=50_000,
        checkpoint_freq=50_000,
        log_freq=1_000,
        mcts_backend="tensor",
        training_device="cuda",
        self_play_device="cuda",
        reanalyse_device="cuda",
        USE_WANDB=True,
    )


MUZERO_PRESETS = {
    "first_real": (first_real_run_config, FIRST_REAL_RUN_LEVELS),
}
