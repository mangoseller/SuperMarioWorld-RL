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
from training_utils import TRAINING_CONFIG, TESTING_CONFIG
import time
from tqdm import tqdm 
import os
parser = argparse.ArgumentParser()
parser.add_argument('--config', choices=['train', 'test'], default='test')
args=parser.parse_args()
config = TRAINING_CONFIG if args.config == 'train' else TESTING_CONFIG


def init_training(agent, config, device):
    policy = PPO(
        model=agent,
        lr=config.learning_rate, # TODO: Implement LR Scheduling
        epsilon=config.clip_eps,
        optimizer=t.optim.Adam,
        device=device,
        c1=config.c1,
        c2=config.c2
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
        'last_eval_steps': 0
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

def run_evaluation(model, policy, tracking, config, run, episodes):
    timestamp = int(time.time())
    eval_dir = f'evals/eval_step_{tracking["total_env_steps"]}_time_{timestamp}'
    eval_metrics = eval_parallel_safe(model, policy, config, num_episodes=episodes, record_dir=eval_dir)
    
    if config.USE_WANDB:
        wandb.log(eval_metrics)
        vids = wandb.Artifact(
            f"eval_videos_step_{tracking['total_env_steps']}",
            type="eval_videos"
        )
        vids.add_dir(eval_dir)
        run.log_artifact(vids)

    tracking['last_eval_steps'] = tracking['total_env_steps']

def log_training_metrics(tracking, mean_loss, config, step):
    if len(tracking['completed_rewards']) > 0 and config.USE_WANDB:
        wandb.log({
            "train/loss": mean_loss,
            "train/mean_reward": np.mean(tracking['completed_rewards']),
            "train/mean_length": np.mean(tracking['completed_lengths']),
            "train/num_episodes": len(tracking['completed_rewards']),
            "global_step": step,
            "num_updates": tracking['num_updates']
        })

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

def train(model, num_eval_episodes=2):

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




        actions, log_probs, values = policy.action_selection(state)
        action_onehot = t.nn.functional.one_hot(actions, num_classes=13).float()
        environment["action"] = action_onehot.squeeze(0)

        #print(f"Generated actions: {actions}, shape: {actions.shape}, dtype: {actions.dtype}")
        #  environment["action"] = actions

        #print(f"Action in environment dict: {environment['action']}")
        environment = env.step(environment)
        next_state = environment["next"]["pixels"]

        if config.num_envs == 1 and next_state.dim() == 3:
            next_state = next_state.unsqueeze(0) # Handle single env

        rewards = environment["next"]["reward"]
        dones = environment["next"]["done"]

        if config.num_envs == 1:
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
        update_episode_tracking(tracking, config, rewards, dones)

        # Update progress bar:
        if config.show_progress and len(tracking['completed_rewards']) > 0:
            pbar.set_postfix({
                'episodes': tracking['episode_num'],
                'mean_reward': f"{np.mean(tracking['completed_rewards']):.2f}",
                'updates': tracking['num_updates']
            })

        
        # Evaluation
        if tracking['total_env_steps'] - tracking['last_eval_steps'] >= config.eval_freq:
            run_evaluation(model, policy, tracking, config, run, num_eval_episodes)
        
       # if dones.item(): SingleENV logic
        #     environment = env.reset()
        #     state = environment["pixels"].unsqueeze(0)

        state = next_state
        
        # PPO update when buffer is full
        if buffer.idx == buffer.capacity:
            mean_loss = policy.update(buffer, next_state=state)
            tracking['num_updates'] += 1
            
            # Log metrics
            log_training_metrics(tracking, mean_loss, config, step)
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
