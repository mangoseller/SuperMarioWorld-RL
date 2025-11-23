import warnings
warnings.filterwarnings('ignore', message="Can't initialize NVML", category=UserWarning, module="torch.cuda")
warnings.filterwarnings('ignore', message="Conversion of an array with ndim > 0 to a scalar is deprecated", 
                       category=DeprecationWarning, module="torchrl.envs.libs.gym")
warnings.filterwarnings('ignore', message=".*Overwriting existing videos.*", 
                       category=UserWarning, module="gymnasium.wrappers.rendering")

import torch as t 
import numpy as np
import argparse
from tqdm import tqdm
from model_small import ImpalaSmall
from config import TRAINING_CONFIG, TESTING_CONFIG, FINETUNE_CONFIG
from training_helpers import (
    init_training, init_tracking, update_episode_tracking,
    log_training_metrics, save_checkpoint
)
from evals import run_evaluation
from training_utils import get_torch_compatible_actions, get_temp, get_entropy


def training_loop(agent, config, num_eval_episodes=5, checkpoint_path=None):
    run = config.setup_wandb()
    device = "cuda" if t.cuda.is_available() else "cpu"
    agent = agent.to(device)
    
    if checkpoint_path:
        print(f"Loading checkpoint from {checkpoint_path}")
        weights = t.load(checkpoint_path, map_location=device)
        agent.load_state_dict(weights)
    
    policy, buffer, env, environment, state = init_training(agent, config, device)
    
    if device == "cuda":
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
            next_state = next_state.unsqueeze(0)

        rewards = environment["next"]["reward"]
        dones = environment["next"]["done"] # Isn't Dones Terminated and Truncated? 
        trunc = environment["next"].get("truncated", t.zeros_like(dones))
        terminated = dones | trunc
        
        if config.num_envs == 1:
            if rewards.dim() == 0:
                rewards = rewards.unsqueeze(0)
            if dones.dim() == 0:
                dones = dones.unsqueeze(0)

        buffer.store(
            state.cpu().numpy(),
            rewards.squeeze().cpu().numpy(),
            actions.cpu().numpy(),
            log_probs.cpu().numpy(),
            values.cpu().numpy(),
            dones.squeeze().cpu().numpy(),
        )
        
        tracking['total_env_steps'] += config.num_envs
        update_episode_tracking(tracking, config, rewards, terminated)

        if config.show_progress and len(tracking['completed_rewards']) > 0:
            pbar.set_postfix({
                'episodes': tracking['episode_num'],
                'mean_reward': f"{np.mean(tracking['completed_rewards']):.2f}",
                'updates': tracking['num_updates'],
                'temp': f"{temp:.3f}",
                'lr': f"{policy.get_current_lr():.2e}",
                'c2': f"{policy.c2:.4f}",
            })
    
        if tracking['total_env_steps'] - tracking['last_eval_steps'] >= config.eval_freq:
            run_evaluation(ImpalaSmall, policy, tracking, config, run, num_eval_episodes, temp=0.1)
        
        if config.num_envs == 1:
            if dones.item():
                environment = env.reset()
                state = environment["pixels"]
        
        state = next_state
        
        if buffer.idx == buffer.capacity:
            mean_loss = policy.update(buffer, next_state=state, temp=temp)
            tracking['num_updates'] += 1
            
            log_training_metrics(tracking, mean_loss, policy, config, step, temp)
            tracking['completed_rewards'].clear()
            tracking['completed_lengths'].clear()
            
            buffer.clear()
            
            if step - tracking['last_checkpoint'] >= config.checkpoint_freq:
                save_checkpoint(agent, tracking, config, run, step)
                print(f"Model checkpoint saved at step {step}")

    # Perform final evaluation and store last weights
    run_evaluation(ImpalaSmall, policy, tracking, config, run, num_eval_episodes, temp=0.1)
    save_checkpoint(agent, tracking, config, run, step)
    
    if config.USE_WANDB:
        import wandb
        wandb.finish()
    else:
        print("Training completed successfully")
    
    return agent


def train(model, config, num_eval_episodes=5):
    agent = model()
    return training_loop(agent, config, num_eval_episodes)


def finetune(model, checkpoint_path, config, num_eval_episodes=5):
    agent = model()
    return training_loop(agent, config, num_eval_episodes, checkpoint_path=checkpoint_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['train', 'test', 'finetune'], default='test')
    parser.add_argument('--checkpoint', type=str, default='finetune.pt', 
                       help='Checkpoint path for fine-tuning')
    args = parser.parse_args()
    
    if args.mode == 'train':
        train(ImpalaSmall, TRAINING_CONFIG)
    elif args.mode == 'test':
        train(ImpalaSmall, TESTING_CONFIG)
    elif args.mode == 'finetune':
        finetune(ImpalaSmall, args.checkpoint, FINETUNE_CONFIG)


if __name__ == "__main__":
    main()
