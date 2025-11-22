import warnings
# Supress NVML warning
warnings.filterwarnings(
        'ignore', 
        message="Can't initialize NVML",
        category=UserWarning,
        module="torch.cuda"
)
# Suppress numpy scalar conversion warning from torchrl
warnings.filterwarnings(
        'ignore',
        message="Conversion of an array with ndim > 0 to a scalar is deprecated",
        category=DeprecationWarning,
        module="torchrl.envs.libs.gym"
)
# Suppress video overwriting warning from gymnasium
warnings.filterwarnings(
        'ignore',
        message=".*Overwriting existing videos.*",
        category=UserWarning,
        module="gymnasium.wrappers.rendering"
)
import torch as t 
import numpy as np
import wandb
from model_small import ImpalaSmall
from ppo import PPO
from buffer import RolloutBuffer
from environment import eval_parallel_safe, make_training_env
import argparse
from training_utils import TRAINING_CONFIG, TESTING_CONFIG, get_torch_compatible_actions, SWEEPRUN_CONFIG, readable_timestamp, get_entropy, get_temp
import time
from tqdm import tqdm 
import os

parser = argparse.ArgumentParser()
parser.add_argument('--config', choices=['train', 'test'], default='test')
args=parser.parse_args()
config = TRAINING_CONFIG if args.config == 'train' else TESTING_CONFIG
max_updates = config.num_training_steps // config.buffer_size

def init_training(agent, config, device):
    policy = PPO(
        model=agent,
        lr=config.learning_rate,  
        epsilon=config.clip_eps,
        optimizer=t.optim.Adam,
        device=device,
        c1=config.c1,
        c2=config.c2,
        max_steps=config.num_training_steps,
        use_lr_scheduling=True
    )
 # Create buffer, initialize environment and get first state 
    buffer = RolloutBuffer(config.buffer_size // config.num_envs, config.num_envs, device)
    env = make_training_env(num_envs=config.num_envs)
    environment = env.reset()
    state = environment['pixels']
    if config.num_envs == 1 and state.dim() == 3:
        state = state.unsqueeze(0) # Handle 1 env only, (4, 84, 84) -> (1, 4, 84, 84)

    return agent, policy, buffer, env, environment, state

def init_tracking(config):
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
    # Update per env tracking
    for i in range(config.num_envs):
        tracking['current_episode_rewards'][i] += rewards[i].item()
        tracking['current_episode_lengths'][i] += 1

        if dones[i].item():
            tracking['completed_rewards'].append(tracking['current_episode_rewards'][i])
            tracking['completed_lengths'].append(tracking['current_episode_lengths'][i])
            tracking['current_episode_rewards'][i] = 0.0
            tracking['current_episode_lengths'][i] = 0
            tracking['episode_num'] += 1

def run_evaluation(model, policy, tracking, config, run, episodes, temp):
    eval_timestamp = readable_timestamp()
    run_dir = f'evals/run_{tracking["run_timestamp"]}'
    eval_dir = f'{run_dir}/eval_step_{tracking["total_env_steps"]}_time_{eval_timestamp}'
    os.makedirs(eval_dir, exist_ok=True)
    eval_metrics = eval_parallel_safe(model, policy, config, num_episodes=episodes, record_dir=eval_dir, eval_temp=temp)
    
    if config.USE_WANDB:
        wandb.log(eval_metrics)
        vids = wandb.Artifact(
            f"{readable_timestamp()}_step_{tracking['total_env_steps']}",
            type="eval_videos"
        )
        vids.add_dir(eval_dir)
        run.log_artifact(vids)

    tracking['last_eval_steps'] = tracking['total_env_steps']

def log_training_metrics(tracking, diagnostics, policy, config, step, temp):
   
    if len(tracking['completed_rewards']) > 0:
        mean_reward = np.mean(tracking['completed_rewards'])
        mean_length = np.mean(tracking['completed_lengths'])
    else:
        mean_reward = 0
        mean_length = 0
    
    metrics = {
        # Training performance
        'train/mean_reward': mean_reward,
        'train/mean_episode_length': mean_length,
        'train/episodes': tracking['episode_num'],
        'train/total_env_steps': tracking['total_env_steps'],
        
        # Loss components
        'loss/total': diagnostics['total_loss'],
        'loss/policy': diagnostics['policy_loss'],
        'loss/value': diagnostics['value_loss'],
        
        # Diagnostics
        'diagnostics/entropy': diagnostics['entropy'],
        'diagnostics/clip_fraction': diagnostics['clip_fraction'],
        'diagnostics/approx_kl': diagnostics['approx_kl'],
        'diagnostics/explained_variance': diagnostics['explained_variance'],
        
        # Hyperparameters
        'hyperparams/temperature': temp,
        'hyperparams/entropy_coef': policy.c2,
        'hyperparams/learning_rate': policy.get_current_lr(),
        'hyperparams/value_coef': policy.c1,
    }
    
    if config.USE_WANDB:
        wandb.log(metrics, step=step)
    else:
        # Print key metrics
        if step % 10_000 == 0:
            print(f"\n[Step {step}] Metrics:")
            print(f"  Reward: {mean_reward:.2f}")
            print(f"  Entropy: {diagnostics['entropy']:.3f}")
            print(f"  Value Loss: {diagnostics['value_loss']:.3f}")
            print(f"  Explained Var: {diagnostics['explained_variance']:.3f}")
            print(f"  Clip Frac: {diagnostics['clip_fraction']:.3f}")
            print(f"  KL: {diagnostics['approx_kl']:.4f}")

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

def train(model, num_eval_episodes=5):
    run = config.setup_wandb()
    device = "cuda" if t.cuda.is_available() else "cpu"
    agent = model().to(device)

    agent, policy, buffer, env, environment, state = init_training(agent, config, device)
    if device == "cuda":
        assert next(agent.parameters()).is_cuda, "Model is not on GPU!"
        print("Training on GPU")
    else:
        print("Training on CPU")
    tracking = init_tracking(config)
    pbar = tqdm(range(config.num_training_steps), disable=not config.show_progress)
    
    for step in pbar:
        policy.c2 = get_entropy(step, total_steps=config.num_training_steps) 
        temp = get_temp(step, total_steps=config.num_training_steps) 
        actions, log_probs, values = policy.action_selection(state, temp)


        environment["action"] = get_torch_compatible_actions(actions)
        environment = env.step(environment)
        next_state = environment["next"]["pixels"]

        if config.num_envs == 1 and next_state.dim() == 3:
            next_state = next_state.unsqueeze(0) # Handle single env

        rewards = environment["next"]["reward"]
        dones = environment["next"]["done"]
        trunc = environment["next"].get("truncated", t.zeros_like(dones))
        terminated = dones | trunc
        if config.num_envs == 1: # Single env shape correction
            if rewards.dim() == 0:
                rewards = rewards.unsqueeze(0)
            if dones.dim() == 0:
                dones = dones.unsqueeze(0)

        # Store experience
        buffer.store(
            state.cpu().numpy(),
            rewards.squeeze().cpu().numpy(),
            actions.cpu().numpy(),
            log_probs.cpu().numpy(),
            values.cpu().numpy(),
            dones.squeeze().cpu().numpy(),
        )
        
        tracking['total_env_steps'] += config.num_envs

        # Update episode tracking
        update_episode_tracking(tracking, config, rewards, terminated)

        # Update progress bar:
        if config.show_progress and len(tracking['completed_rewards']) > 0:
            pbar.set_postfix({
                'episodes': tracking['episode_num'],
                'mean_reward': f"{np.mean(tracking['completed_rewards']):.2f}",
                'updates': tracking['num_updates'],
                'temp': f"{temp:.3f}",
                'lr': f"{policy.get_current_lr():.2e}",
                'c2': f"{policy.c2}",
            })
    
        # Evaluation
        if tracking['total_env_steps'] - tracking['last_eval_steps'] >= config.eval_freq:
            run_evaluation(model, policy, tracking, config, run, num_eval_episodes, temp=0.1)
            print(f"C2: {policy.c2}, Temp: {temp}")
        if config.num_envs == 1:
            if dones.item(): # TODO: Separate training with 1 env and multienvs into different functions
                environment = env.reset() # Handle single env
                state = environment["pixels"]
        state = next_state
        
        # PPO update when buffer is full
        if buffer.idx == buffer.capacity:
            mean_loss = policy.update(buffer, next_state=state)
            tracking['num_updates'] += 1
            
            # Log metrics
            log_training_metrics(tracking, mean_loss, policy, config, step, temp)
            tracking['completed_rewards'].clear()
            tracking['completed_lengths'].clear()
            
            buffer.clear()
            
            # Save model checkpoints
            if step - tracking['last_checkpoint'] >= config.checkpoint_freq:
                save_checkpoint(agent, tracking, config, run, step)
                print(f"Model checkpoint saved at step {step}")
    
    if config.USE_WANDB:
        wandb.finish()
    else:
        print("Test completed without incident.")

if __name__ == "__main__":
    train(ImpalaSmall)

