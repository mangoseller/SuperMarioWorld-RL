from dataclasses import dataclass
import os
import wandb

@dataclass
class TrainingConfig: 
    num_envs: int
    num_training_steps: int 
    eval_freq: int
    checkpoint_freq: int
    show_progress: bool
    architecture: str

    steps_per_env: int = 4096
    learning_rate: float = 1e-4
    gamma: float = 0.998
    lambda_gae: float = 0.95
    epsilon_advantage: float = 1e-8
    clip_eps: float = 0.2
    c1: float = 0.5
    c2: float = 0.01
    epochs: int = 8
    lr_schedule: str = 'constant'
    min_lr: float = 1e-6
    USE_WANDB: bool = False
    wandb_project: str = 'marioRL'
    use_curriculum: bool = False
    
    @property
    def buffer_size(self):
        return self.steps_per_env * self.num_envs
    
    @property
    def minibatch_size(self):
        return max(32, self.buffer_size // 32)
    
    def to_wandb_config(self):
        return {
            "learning_rate": self.learning_rate,
            "gamma": self.gamma,
            "lambda": self.lambda_gae,
            "epsilon": self.epsilon_advantage,
            "clip_eps": self.clip_eps,
            "c1": self.c1,
            "c2": self.c2,
            "architecture": self.architecture,
            "epochs": self.epochs,
            "buffer_size": self.buffer_size, 
            "minibatch_size": self.minibatch_size,
            "lr_schedule": self.lr_schedule,
            "min_lr": self.min_lr,
            "steps_per_env": self.steps_per_env
        }
    
    def setup_wandb(self):
        if not self.USE_WANDB:
            return
        api_key = os.environ.get("WANDB_API_KEY")
        if api_key is None:
            raise RuntimeError("WANDB_API_KEY not set in environment")
        wandb.login(key=api_key)
        return wandb.init(
            project=self.wandb_project,
            config=self.to_wandb_config()
        )
    
    @classmethod
    def from_wandb(cls, base_config):
        config_dict = base_config.__dict__.copy()
        for k, v, in wandb.config.items():
            if hasattr(base_config, k):
                config_dict[k] = v
        return cls(**config_dict)


TRANSPALA_TRAIN_CONFIG = TrainingConfig(
    architecture='TransPala',
    lr_schedule='linear',
    learning_rate=1.5e-4,
    min_lr=1e-6,
    epochs=3,
    clip_eps=0.2,
    c1=0.5,
    c2=0.02,
    gamma=0.995,
    lambda_gae=0.95,
    num_envs=28,
    steps_per_env=512,  # 28 × 512 = 14,336 batch size
    num_training_steps=200,
    checkpoint_freq=200_000,
    eval_freq=150_000,
    show_progress=True,
    USE_WANDB=False
)

TRANSPALA_TUNE_CONFIG = TrainingConfig(
    architecture='TransPala',
    lr_schedule='linear',
    learning_rate=2e-5,
    min_lr=1e-6,
    epochs=2,
    clip_eps=0.1,
    c1=0.5,
    c2=0.005,
    gamma=0.995,
    lambda_gae=0.95,
    num_envs=28,
    steps_per_env=512,
    num_training_steps=400_000,
    checkpoint_freq=100_000,
    eval_freq=50_000,
    show_progress=True,
    USE_WANDB=True
)

TRANSPALA_TEST_CONFIG = TrainingConfig(
    architecture='TransPala',
    lr_schedule='linear',
    learning_rate=1.5e-4,
    min_lr=1e-6,
    epochs=3,
    clip_eps=0.2,
    c1=0.5,
    c2=0.02,
    gamma=0.995,
    lambda_gae=0.95,
    num_envs=1,
    steps_per_env=4096,
    num_training_steps=500_000,
    checkpoint_freq=100_000,
    eval_freq=100_000,
    show_progress=True,
    USE_WANDB=False
)
IMPALA_TRAIN_CONFIG = TrainingConfig(
    architecture='ImpalaLike',
    lr_schedule='linear',
    learning_rate=2.5e-4,
    min_lr=1e-6,
    epochs=4,
    clip_eps=0.2,
    c1=0.5,
    c2=0.02,
    num_envs=16,
    steps_per_env=512,
    num_training_steps=2000,
    checkpoint_freq=100_000,
    eval_freq=100_000,
    show_progress=True,
    USE_WANDB=True
)


IMPALA_TEST_CONFIG = TrainingConfig(
    architecture='ImpalaLike',
    lr_schedule='linear',
    learning_rate=2.5e-4,
    min_lr=1e-6,
    epochs=4,
    clip_eps=0.2,
    c1=0.5,
    c2=0.02,
    num_envs=1,
    steps_per_env=4096,
    num_training_steps=4_000_000,
    checkpoint_freq=2,
    eval_freq=200_000,
    show_progress=True,
    USE_WANDB=True
)

IMPALA_TUNE_CONFIG = TrainingConfig(
    architecture='ImpalaLike',
    lr_schedule='linear',
    learning_rate=1e-6,
    min_lr=1e-6,
    epochs=4,
    clip_eps=0.2,
    c1=0.5,
    c2=0.005,
    num_envs=16,
    steps_per_env=512,
    num_training_steps=500_000,
    checkpoint_freq=50_000,
    eval_freq=50_000,
    show_progress=True,
    USE_WANDB=True
)

CONV_TRAIN_CONFIG = TrainingConfig(
    architecture="ConvolutionalSmall",
    num_envs=16,
    num_training_steps=2000,
    steps_per_env=25, 
    eval_freq=250_000,
    checkpoint_freq=200_0000,
    USE_WANDB=False,
    show_progress=True,
    learning_rate=2e-5,
    lr_schedule='linear'
)

CONV_TEST_CONFIG = TrainingConfig(
    architecture="ConvolutionalSmall",
    num_envs=1,
    num_training_steps=1_250_000,
    steps_per_env=4096,
    eval_freq=200_000,
    checkpoint_freq=125_000,
    USE_WANDB=False,
    show_progress=True,
    c1=0.8,
    c2=0.01,
    learning_rate=2e-5,
    lr_schedule='cosine'
)


CONV_TUNE_CONFIG = TrainingConfig(
    architecture="ConvolutionalSmall",
    num_envs=12,
    num_training_steps=500_000,
    steps_per_env=4096, 
    eval_freq=1,
    checkpoint_freq=1000000,
    USE_WANDB=False,
    show_progress=True,
    learning_rate=1e-5,
    c1=0.8,
    c2=0.00005,
    lr_schedule='linear'
)
