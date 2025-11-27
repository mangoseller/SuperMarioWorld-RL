import torch as t
import numpy as np
import os
import wandb
from ppo import PPO
from buffer import RolloutBuffer
from environment import make_env
from datetime import datetime


def readable_timestamp():
    """Timestamp for run directories: DD-MM_HH-MM"""
    return datetime.now().strftime("%d-%m_%H-%M")


def init_training(agent, config, device):
    """
    Initialize PPO policy, buffer, environment, and get initial state.
    
    Note: For curriculum learning, use init_training_components() from train.py instead.
    """
    policy = PPO(agent, config, device)   
    buffer = RolloutBuffer(config.buffer_size // config.num_envs, config.num_envs, device)
    
    env = make_env(
        num_envs=config.num_envs,
        render_human=(config.num_envs == 1),
    )
    
    environment = env.reset()
    state = environment['pixels']
    if config.num_envs == 1 and state.dim() == 3:
        state = state.unsqueeze(0)
    
    return policy, buffer, env, environment, state


def init_tracking(config):
    """Initialize tracking dictionary for training metrics."""
    return {
        'current_episode_rewards': [0.0] * config.num_envs,
        'current_episode_lengths': [0] * config.num_envs,
        'completed_rewards': [],
        'completed_lengths': [],
        'num_updates': 0,
        'episode_num': 0,
        'last_checkpoint': 0,
        'total_env_steps': 0,
        'last_eval_step': 0,
        'run_timestamp': readable_timestamp()
    }


def update_episode_tracking(tracking, config, rewards, dones):
    """Update per-environment episode tracking."""
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
    """Log training metrics to wandb or console."""
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


def save_checkpoint(agent, policy, tracking, config, run, step, curriculum_option=None):
    """Save model checkpoint and optionally log to wandb."""
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
    """Load model and optimizer state from checkpoint."""
    checkpoint = t.load(checkpoint_path, map_location=device)
    agent.load_state_dict(checkpoint['model_state_dict'])
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


def get_checkpoint_info(checkpoint_path):
    """
    Extract info from a checkpoint without loading model weights.
    
    Returns:
        dict with keys: architecture, step, episode_num, total_steps, 
                       use_curriculum, curriculum_option
    """
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
        'learning_rate': config_dict.get('learning_rate'),
        'lr_schedule': config_dict.get('lr_schedule'),
    }


def fix_compiled_checkpoint(checkpoint_path):
    """
    Fix checkpoint saved with torch.compile() by stripping '_orig_mod.' prefix.
    Modifies the file in place if needed.
    
    Returns:
        True if fix was applied, False if not needed
    """
    checkpoint = t.load(checkpoint_path, map_location='cpu')
    model_state = checkpoint.get('model_state_dict', {})
    
    if any(k.startswith('_orig_mod.') for k in model_state.keys()):
        print(f"Detected torch.compile() checkpoint, stripping '_orig_mod.' prefix...")
        fixed_state = {k.replace('_orig_mod.', ''): v for k, v in model_state.items()}
        checkpoint['model_state_dict'] = fixed_state
        t.save(checkpoint, checkpoint_path)
        print(f"Fixed checkpoint saved to {checkpoint_path}")
        return True
    return False


def setup_from_checkpoint(checkpoint_path, agent, policy, config, device, resume=False):
    """
    Setup training state from checkpoint.
    
    Args:
        checkpoint_path: Path to checkpoint file
        agent: Model to load weights into
        policy: PPO policy (optimizer/scheduler will be loaded if resuming)
        config: Training config
        device: Torch device
        resume: If True, load full state. If False, only load model weights.
    
    Returns:
        Tuple of (start_step, tracking_dict)
    """
    if checkpoint_path is None:
        return 0, init_tracking(config)
    
    # Fix compiled checkpoint if needed
    fix_compiled_checkpoint(checkpoint_path)
    
    if resume:
        print(f"Resuming training from {checkpoint_path}")
        start_step, tracking = load_checkpoint(checkpoint_path, agent, policy, device)
        tracking['run_timestamp'] = readable_timestamp()
        print(f"Resumed at step {start_step}, episode {tracking['episode_num']}")
        return start_step, tracking
    else:
        print(f"Loading weights from {checkpoint_path}")
        weights = t.load(checkpoint_path, map_location=device)
        # Handle both old (raw state_dict) and new (full checkpoint) formats
        if 'model_state_dict' in weights:
            agent.load_state_dict(weights['model_state_dict'])
        else:
            agent.load_state_dict(weights)
        return 0, init_tracking(config)


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


def get_torch_compatible_actions(actions, num_actions=14): 
    """Convert integer actions into one-hot format for torchrl."""
    onehot_actions = t.nn.functional.one_hot(actions, num_classes=num_actions).float()
    return onehot_actions


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
