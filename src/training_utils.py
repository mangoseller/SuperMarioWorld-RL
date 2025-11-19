from dataclasses import dataclass
import os
import wandb
import torch as t

@dataclass
class TrainingConfig: 
    num_envs: int
    num_training_steps: int 
    buffer_size: int
    eval_freq: int
    checkpoint_freq: int
    show_progress: bool

    # PPO Parameters
    learning_rate: float = 1e-4
    gamma: float = 0.99
    lambda_gae: float = 0.95
    epsilon_advantage: float = 1e-8
    clip_eps: float = 0.2
    c1: float = 0.5 # Value loss coefficient 
    c2: float = 0.01 # Entropy coefficient
    epochs: int = 4 # Epochs for PPO update
   
    # Model Params
    architecture = 'ImpalaSmall'
    minibatch_size: int = 64

    USE_WANDB: bool = False
    wandb_project: str = 'marioRL'

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
            "minibatch_size": self.minibatch_size
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

def get_torch_compatible_actions(actions, num_envs, num_actions=13):
# Convert integer actions into one-hot format for torchrl
    onehot_actions = t.nn.functional.one_hot(actions, num_classes=num_actions).float()
    if num_envs == 1:
        return onehot_actions.squeeze(0)
    return onehot_actions

# Sweep config
SWEEPRUN_CONFIG = TrainingConfig(
    num_envs = 6,
    num_training_steps=5_000_000,
    buffer_size=2048,
    eval_freq=500_000,
    checkpoint_freq=int(1e40),
    USE_WANDB=True,
    show_progress=False
)

TRAINING_CONFIG = TrainingConfig(
    num_envs=8,
    num_training_steps=int(1e6),
    buffer_size=4096,
    eval_freq=250_000,
    checkpoint_freq=200_000,
    USE_WANDB=False,
    show_progress=True
)

TESTING_CONFIG = TrainingConfig(
    num_envs=1,
    num_training_steps=100000000,
        buffer_size=4096,
    eval_freq= 2000000, # 20000
    checkpoint_freq=5000000000,
    USE_WANDB=False,
    show_progress=True
)
