import sys
import os

c_dir = os.path.dirname(os.path.realpath(__file__))
p_dir = os.path.dirname(c_dir)
sys.path.append(p_dir)

import wandb
from train import train_sweep
from sweep_config import sweep_config
#
# sweep_id = wandb.sweep(
#     sweep_config,
#     project="marioRL"
# )
if __name__ == "__main__":
    sweep_id = wandb.sweep(
        sweep_config,
        project="marioRL"
)
    wandb.agent(
        sweep_id,
        function=train_sweep,
        count=20
    )
