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
    
    steps_per_env: int = 4096
    learning_rate: float = 1e-4
    gamma: float = 0.9995
    lambda_gae: float = 0.95
    epsilon_advantage: float = 1e-8
    clip_eps: float = 0.2
    c1: float = 0.5
    c2: float = 0.01
    epochs: int = 8
    lr_schedule: str = 'constant'
    min_lr: float = 1e-6
    architecture = 'ImpalaSmall'
    USE_WANDB: bool = False
    wandb_project: str = 'marioRL'
    
    
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

TRAINING_CONFIG = TrainingConfig(
    num_envs=1,
    num_training_steps=int(1.5e6),
    steps_per_env=4096, 
    eval_freq=250_000,
    checkpoint_freq=200_000,
    USE_WANDB=True,
    show_progress=True,
    learning_rate=2e-5,
    lr_schedule='linear'
)

TESTING_CONFIG = TrainingConfig(
    num_envs=12,
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

SWEEPRUN_CONFIG = TrainingConfig(
    num_envs=1,
    num_training_steps=800_000, 
    steps_per_env=4096,         
    eval_freq=80_000,
    checkpoint_freq=100_000_000,
    USE_WANDB=True,
    show_progress=True,
    learning_rate=2e-5, 
    c1=0.5,
    c2=0.01,               
    gamma=0.99,
    lr_schedule='cosine'
)

FINETUNE_CONFIG = TrainingConfig(
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
