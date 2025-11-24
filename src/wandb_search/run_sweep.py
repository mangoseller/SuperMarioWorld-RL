import wandb
import sys
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.dirname(current_dir)
if src_dir not in sys.path:
    sys.path.append(src_dir)

from config import SWEEPRUN_CONFIG, TrainingConfig
from model_small import ImpalaSmall
from train import training_loop
from wandb_search.sweep_config import sweep_config


def train_sweep(config=None):
    with wandb.init(config=config):

        updated_config = TrainingConfig.from_wandb(SWEEPRUN_CONFIG)
        agent = ImpalaSmall()
        training_loop(agent, updated_config)


if __name__ == "__main__":
    sweep_id = wandb.sweep(
        sweep_config,
        project="marioRL"
    )
    print(f"Sweep initialized with ID: {sweep_id}")
    
    # Start the agent to run the search
    wandb.agent(
        sweep_id,
        function=train_sweep,
        count=10
    )
