import torch as t
import numpy as np
import os
import wandb
from datetime import datetime

def readable_timestamp():
    return datetime.now().strftime("%d-%m_%H-%M")

def get_checkpoint_info(checkpoint_path):
    checkpoint = t.load(checkpoint_path, map_location='cpu')
    config_dict = checkpoint.get('config_dict', {})
    tracking = checkpoint.get('tracking', {})
    
    return {
        'architecture': config_dict.get('architecture'),
        'step': checkpoint.get('step', 0),
        'episode_num': tracking.get('episode_num', 0),
        'total_steps': config_dict.get('num_training_steps'),
        'use_curriculum': config_dict.get('use_curriculum', False),
        'curriculum_option': config_dict.get('curriculum_option'),
    }

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
        
        'loss/total': diagnostics.get('total_loss', 0),
        'loss/policy': diagnostics.get('policy_loss', 0),
        'loss/value': diagnostics.get('value_loss', 0),
        'loss/pixel_control': diagnostics.get('pixel_control_loss', 0),

        
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

def load_checkpoint(checkpoint_path, agent, policy, resume=False):
    device = next(agent.parameters()).device
    checkpoint = t.load(checkpoint_path, map_location=device)
    
    agent.load_state_dict(checkpoint['model_state_dict'])
    
    if resume:
        policy.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        if checkpoint.get('scheduler_state_dict') and policy.scheduler:
            policy.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    
    start_step = checkpoint.get('step', 0)
    tracking = checkpoint.get('tracking') if resume else None
    
    if resume:
        config_dict = checkpoint.get('config_dict', {})
        max_entropy = config_dict.get('c2', 0.02)
        total_steps = config_dict.get('num_training_steps', 1_000_000)
        policy.c2 = get_entropy(start_step, total_steps, max_entropy=max_entropy)
    
    return start_step, tracking


def get_torch_compatible_actions(actions, num_actions=14): 
    # Convert integer actions into one-hot format for torchrl
    onehot_actions = t.nn.functional.one_hot(actions, num_classes=num_actions).float()
    return onehot_actions


def get_entropy(step, total_steps, max_entropy=0.02, min_entropy=0.005):
  # Linearly decay entropy coefficient over training
    progress = step / total_steps
    current_entropy = max_entropy - (max_entropy - min_entropy) * progress
    return current_entropy



