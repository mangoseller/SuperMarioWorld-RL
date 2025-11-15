from dataclasses import dataclass
import os
import wandb

@dataclass
class TrainingConfig:
    
    num_envs: int
    num_training_steps: int 
    buffer_size: int
    eval_freq: int
    checkpoint_freq: int

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
    # Initialize wandb
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

TRAINING_CONFIG = TrainingConfig(
    num_envs=8,
    num_training_steps=int(1e6),
    buffer_size=4096,
    eval_freq=250_000,
    checkpoint_freq=200_000,
    USE_WANDB=True
)

TESTING_CONFIG = TrainingConfig(
    num_envs=8,
    num_training_steps=100000,
    buffer_size=50,
    eval_freq=200,
    checkpoint_freq=500,
    USE_WANDB=True
)
