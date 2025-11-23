import torch as t
import numpy as np
import os
import wandb
from ppo import PPO
from buffer import RolloutBuffer
from environment import make_training_env
from datetime import datetime


def readable_timestamp():
    return datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

def init_training(agent, config, device):
    # Initialize PPO policy, buffer, environment, and get initial state
    policy = PPO(agent, config, device)   
    buffer = RolloutBuffer(config.buffer_size // config.num_envs, config.num_envs, device)
    env = make_training_env(num_envs=config.num_envs)
    environment = env.reset()
    state = environment['pixels']
    if config.num_envs == 1 and state.dim() == 3:
        state = state.unsqueeze(0)
    
    return policy, buffer, env, environment, state


def init_tracking(config):
    # Initialize tracking dictionary for training metrics
    return {
        'current_episode_rewards': [0.0] * config.num_envs,
        'current_episode_lengths': [0] * config.num_envs,
        'completed_rewards': [],
        'completed_lengths': [],
        'num_updates': 0,
        'episode_num': 0,
        'last_checkpoint': 0,
        'total_env_steps': 0,
        'last_eval_steps': 0,
        'run_timestamp': readable_timestamp()
    }


def update_episode_tracking(tracking, config, rewards, dones):
    # Update per-environment episode tracking
    for i in range(config.num_envs):
        tracking['current_episode_rewards'][i] += rewards[i].item()
        tracking['current_episode_lengths'][i] += 1

        if dones[i].item():
            tracking['completed_rewards'].append(tracking['current_episode_rewards'][i])
            tracking['completed_lengths'].append(tracking['current_episode_lengths'][i])
            tracking['current_episode_rewards'][i] = 0.0
            tracking['current_episode_lengths'][i] = 0
            tracking['episode_num'] += 1


def log_training_metrics(tracking, diagnostics, policy, config, step):
    # Log training metrics to wandb or console
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
        
        'diagnostics/entropy': diagnostics['entropy'],
        'diagnostics/clip_fraction': diagnostics['clip_fraction'],
        'diagnostics/approx_kl': diagnostics['approx_kl'],
        'diagnostics/explained_variance': diagnostics['explained_variance'],
        
        'hyperparams/entropy_coef': policy.c2,
        'hyperparams/learning_rate': policy.get_current_lr(),
        'hyperparams/value_coef': policy.c1,
    }
    
    if config.USE_WANDB:
        wandb.log(metrics, step=step)

def save_checkpoint(agent, tracking, config, run, step):
    checkpoint_dir = "model_checkpoints"
    os.makedirs(checkpoint_dir, exist_ok=True)
    model_path = os.path.join(checkpoint_dir, f"ImpalaSmall{tracking['episode_num']}.pt")
    t.save(agent.state_dict(), model_path)

    if config.USE_WANDB:
        artifact = wandb.Artifact(f"marioRLep{tracking['episode_num']}", type="model")
        artifact.add_file(model_path)
        run.log_artifact(artifact)
    
    tracking['last_checkpoint'] = step

def handle_env_resets(env, environment, next_state, terminated, num_envs): # TODO: Fix this up

    if num_envs == 1:
        # Single environment case
        if terminated.item():
            environment = env.reset()
            state = environment["pixels"]
            # Ensure correct shape [1, C, H, W]
            if state.dim() == 3:
                state = state.unsqueeze(0)
        else:
            state = next_state
    else:
        # Parallel environments case
        # Extract done mask: [num_envs]
        done_mask = terminated.squeeze(-1) if terminated.dim() > 1 else terminated
        
        if done_mask.any():
            # Get the device of the environment
            env_device = next_state.device
            
            # Create a TensorDict with _reset key for done environments
            reset_td = environment.clone()
            
            # Ensure done_mask is on the same device as environment
            done_mask = done_mask.to(env_device)
            
            # Set _reset with shape [num_envs, 1] as expected by TorchRL
            reset_td["_reset"] = done_mask.unsqueeze(-1)
            
            # Move reset_td to CPU for the reset operation (ParallelEnv expects CPU)
            reset_td = reset_td.to('cpu')
            
            # Reset only the done environments
            reset_output = env.reset(reset_td)
            
            # Move reset output back to original device
            reset_output = reset_output.to(env_device)
            
            # Get the pixels from the reset output
            reset_pixels = reset_output["pixels"]
            
            # Ensure all tensors are on the same device for the where operation
            done_mask = done_mask.to(env_device)
            reset_pixels = reset_pixels.to(env_device)
            next_state = next_state.to(env_device)
            
            # Update state: use reset pixels for done envs, next_state for others
            # Broadcast done_mask to match pixel dimensions [num_envs, C, H, W]
            state = t.where(
                done_mask.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1),
                reset_pixels,
                next_state
            )
            
            # Update environment reference for consistency
            environment = reset_output
        else:
            state = next_state
    
    return state, environment


def get_torch_compatible_actions(actions, num_actions=14): 
    # Convert integer actions into one-hot format for torchrl
    onehot_actions = t.nn.functional.one_hot(actions, num_classes=num_actions).float()
    return onehot_actions


def get_entropy(step, total_steps, max_entropy=0.1, min_entropy=0.001):
       progress = step / total_steps
       current_entropy = max_entropy - (max_entropy - min_entropy) * progress
       return current_entropy
