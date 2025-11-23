import torch as t
import numpy as np
import os
import wandb
from ppo import PPO
from buffer import RolloutBuffer
from environment import make_training_env
from training_utils import readable_timestamp


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


def log_training_metrics(tracking, diagnostics, policy, config, step, temp):
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
        
        'hyperparams/temperature': temp,
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
