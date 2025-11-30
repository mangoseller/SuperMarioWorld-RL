import torch as t
import numpy as np
import os
import wandb
from ppo import PPO
from buffer import RolloutBuffer
from environment import make_env
from datetime import datetime


def readable_timestamp():
    return datetime.now().strftime("%d-%m_%H-%M")


def log_training_metrics(tracking, diagnostics, policy, config, step):
    if len(tracking['completed_rewards']) > 0:
        mean_reward = np.mean(tracking['completed_rewards'])
        mean_length = np.mean(tracking['completed_lengths'])
    else:
        mean_reward = 0
        mean_length = 0
    
    metrics = {
        'train/mean_reward': mean_reward,
        'train/mean_episode_length': mean_length,
        'train/episodes': tracking['episode_num'],
        'train/total_env_steps': tracking['total_env_steps'],
        
        'loss/total': diagnostics['total_loss'],
        'loss/policy': diagnostics['policy_loss'],
        'loss/value': diagnostics['value_loss'],
        'loss/pixel_control': diagnostics['pixel_control_loss'],

        
        'hyperparams/entropy_coef': policy.c2,
        'hyperparams/learning_rate': policy.get_current_lr(),
        'hyperparams/value_coef': policy.c1,
    }
    
    if config.USE_WANDB:
        wandb.log(metrics, step=step)


def save_checkpoint(agent, policy, tracking, config, run, step, curriculum_option=None):
    checkpoint_dir = "model_checkpoints"
    os.makedirs(checkpoint_dir, exist_ok=True)   
    checkpoint = {
        'model_state_dict': agent.state_dict(),
        'optimizer_state_dict': policy.optimizer.state_dict(),
        'scheduler_state_dict': policy.scheduler.state_dict() if policy.scheduler else None,
        'step': step,
        'tracking': tracking,
        'config_dict': {
            'architecture': config.architecture,
            'num_training_steps': config.num_training_steps,
            'learning_rate': config.learning_rate,
            'min_lr': config.min_lr,
            'lr_schedule': config.lr_schedule,
            'c2': config.c2,
            'use_curriculum': config.use_curriculum,
            'curriculum_option': curriculum_option,
        }
    }
    model_path = os.path.join(checkpoint_dir, f"{config.architecture}_ep{tracking['episode_num']}.pt")
    t.save(checkpoint, model_path)
    if config.USE_WANDB:
        artifact = wandb.Artifact(f"marioRLep{tracking['episode_num']}", type="model")
        artifact.add_file(model_path)
        run.log_artifact(artifact)
    tracking['last_checkpoint'] = step
    return model_path


def load_checkpoint(checkpoint_path, agent, policy, device):
    checkpoint = t.load(checkpoint_path, map_location=device)
    

    policy.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    if checkpoint['scheduler_state_dict'] and policy.scheduler:
        policy.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    
    start_step = checkpoint['step']
    tracking = checkpoint['tracking']
    
    config_dict = checkpoint.get('config_dict', {})
    max_entropy = config_dict.get('c2', 0.02)
    total_steps = config_dict.get('num_training_steps', 1_000_000)
    policy.c2 = get_entropy(start_step, total_steps, max_entropy=max_entropy)
    
    return start_step, tracking



def handle_env_resets(env, environment, next_state, terminated, num_envs):
    """
    Handle environment resets for terminated episodes.
    
    For parallel envs, only resets the environments that are done.
    """
    if num_envs == 1:
        if terminated.item():
            environment = env.reset()
            state = environment["pixels"]
            if state.dim() == 3:
                state = state.unsqueeze(0)
        else:
            state = next_state
    else:
        done_mask = terminated.squeeze(-1) if terminated.dim() > 1 else terminated
        
        if done_mask.any():
            env_device = next_state.device
            reset_td = environment.clone()
            done_mask = done_mask.to(env_device)
            reset_td["_reset"] = done_mask.unsqueeze(-1)
            reset_td = reset_td.to('cpu')
            reset_output = env.reset(reset_td)
            reset_output = reset_output.to(env_device)
            reset_pixels = reset_output["pixels"]
            done_mask = done_mask.to(env_device)
            reset_pixels = reset_pixels.to(env_device)
            next_state = next_state.to(env_device)
            
            state = t.where(
                done_mask.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1),
                reset_pixels,
                next_state
            )
            environment = reset_output
        else:
            state = next_state
    
    return state, environment


def get_entropy(step, total_steps, max_entropy=0.02, min_entropy=0.005):
    """Linearly decay entropy coefficient over training."""
    progress = step / total_steps
    current_entropy = max_entropy - (max_entropy - min_entropy) * progress
    return current_entropy


def compute_pixel_change_targets(observations, cell_size=12, device='cuda'):
    """
    Compute spatial pixel changes from consecutive frames for auxiliary task.
    
    Args:
        observations: (T, C, H, W) tensor of observations
        cell_size: Size of spatial cells (84/12 = 7x7 grid)
    
    Returns:
        targets: (T-1, grid_h, grid_w) tensor of pixel change magnitudes
    """
    observations = observations.to(device)
    
    current = observations[:-1]
    next_obs = observations[1:]
    
    # Absolute difference averaged over channels
    diff = t.abs(next_obs - current).mean(dim=1)  # (T-1, H, W)
    diff = diff.unsqueeze(1)  # (T-1, 1, H, W)
    
    h, w = diff.shape[2], diff.shape[3]
    grid_h = h // cell_size
    grid_w = w // cell_size
    
    # Crop to exact multiple of cell_size
    diff = diff[:, :, :grid_h*cell_size, :grid_w*cell_size]
    
    # Average pool to grid
    targets = t.nn.functional.avg_pool2d(diff, kernel_size=cell_size, stride=cell_size)
    
    return targets.squeeze(1)  # (T-1, grid_h, grid_w)
