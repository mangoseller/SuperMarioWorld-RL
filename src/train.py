import warnings
warnings.filterwarnings('ignore')

import torch as t
import numpy as np
from einops import rearrange
from tqdm import tqdm
from runner import run_training
from ppo import PPO
from buffer import RolloutBuffer
from environment import make_env
from curriculum import Curriculum, assign_levels
from evals import run_evaluation
from utils import (
    readable_timestamp,
    get_entropy,
    save_checkpoint,
    load_checkpoint,
    log_training_metrics,
    get_torch_compatible_actions
)

def make_env_for_curriculum(curriculum, config):
    level_dist = assign_levels(config.num_envs, curriculum.weights)
    env = make_env(
        num_envs=config.num_envs,
        level_distribution=level_dist,
    )
    return env, level_dist


def train(model_class, config, curriculum_option=None, 
          checkpoint_path=None, resume=False):

    run = config.setup_wandb()
    device = "cuda" if t.cuda.is_available() else "cpu"
    
    original_agent = model_class().to(device)
    policy = PPO(original_agent, config, device)
    
    tracking = {
        'current_episode_rewards': [0.0] * config.num_envs,
        'current_episode_lengths': [0] * config.num_envs,
        'completed_rewards': [],
        'completed_lengths': [],
        'episode_num': 0,
        'total_env_steps': 0,
        'last_eval_step': 0,
        'run_timestamp': readable_timestamp(),
    }
    start_step = 0
    
    if checkpoint_path:
        start_step, saved_tracking = load_checkpoint(checkpoint_path, original_agent, policy, resume)
        if resume and saved_tracking:
            tracking = saved_tracking
            tracking['run_timestamp'] = readable_timestamp()
        print(f"{'Resumed from' if resume else 'Loaded weights from'} {checkpoint_path} at step {start_step}")
    
    curriculum = None
    if config.use_curriculum and curriculum_option:
        curriculum = Curriculum.create(curriculum_option)
        if start_step > 0:
            while curriculum.update(start_step, config.num_training_steps):
                pass
    
    if curriculum:
        env, level_dist = make_env_for_curriculum(curriculum, config)
        print(f"Curriculum stage {curriculum.stage}: {level_dist[:5]}...")
    else:
        env = make_env(num_envs=config.num_envs)
    
    buffer = RolloutBuffer(config.steps_per_env, config.num_envs, device)
    td = env.reset()
    state = td['pixels']    
    print(f"Training {config.architecture} | {sum(p.numel() for p in original_agent.parameters()):,} params | {device}")
    
    pbar = tqdm(range(start_step, config.num_training_steps), disable=not config.show_progress)
    print("Compiling model for training...")

    agent = t.compile(original_agent)
    policy.model = agent # Compiled model and original model share same underlying memory 
    for step in pbar:

        # Curriculum stage transition
        if curriculum and curriculum.update(step, config.num_training_steps):
            env.close()
            env, level_dist = make_env_for_curriculum(curriculum, config)
            td = env.reset()
            state = td['pixels']

            buffer.clear()
            tracking['current_episode_rewards'] = [0.0] * config.num_envs
            tracking['current_episode_lengths'] = [0] * config.num_envs
            print(f"\nCurriculum -> Stage {curriculum.stage}: {curriculum.describe(curriculum.stage)}")
        
        # Entropy decay with boost for low-variance rewards
        entropy = get_entropy(step, config.num_training_steps, max_entropy=config.c2)
        recent = tracking['completed_rewards'][-40:]
        if len(recent) >= 40 and np.std(recent) < 1.0 and np.mean(recent) < 150:
            entropy *= 3
        policy.c2 = entropy
        
        # Step environment
        actions, log_probs, values = policy.action_selection(state)
        td["action"] = get_torch_compatible_actions(actions)
        td = env.step(td) 
        next_state = td["next"]["pixels"]
        rewards = td["next"]["reward"]
        dones = td["next"]["done"] | td["next"].get("truncated", t.zeros_like(td["next"]["done"]))
        

        buffer.store(
            state,
            rewards.squeeze(),
            actions,
            log_probs,
            values,
            dones.squeeze(),
        )
        
        # Update episode tracking
        tracking['total_env_steps'] += config.num_envs
        for i in range(config.num_envs):
            tracking['current_episode_rewards'][i] += rewards[i].item()
            tracking['current_episode_lengths'][i] += 1
            if dones[i].item():
                tracking['completed_rewards'].append(tracking['current_episode_rewards'][i])
                tracking['completed_lengths'].append(tracking['current_episode_lengths'][i])
                tracking['current_episode_rewards'][i] = 0.0
                tracking['current_episode_lengths'][i] = 0
                tracking['episode_num'] += 1
        
        if config.show_progress and tracking['completed_rewards']:
            pbar.set_postfix({
                'ep': tracking['episode_num'],
                'reward': f"{np.mean(tracking['completed_rewards']):.1f}",
                'lr': f"{policy.get_current_lr():.1e}",
                'c2': f"{policy.c2:.1e}",
            })
        

        if dones.any():
            """Handle episode ends across multiple parallel envs, for environments marked as true in dones,
            reset the env. For finished environments, state sets the frame to the starting frames of the env.
            Without this logic, parallel envs will never reset upon completion, destroying training"""
            reset_td = td.clone()
            # Prepare reset signal
            reset_td["_reset"] = dones.clone()
            # Reset finished environments
            reset_out = env.reset(reset_td.to('cpu')).to(state.device)
            # Reshape dones (batch, 1) -> (batch, 1, 1, 1) to match the shape of the envs and allow broadcasting
            mask = rearrange(dones, 'b c -> b c 1 1')
            # Apply the mask, reset completed environments
            state = t.where(mask, reset_out["pixels"], next_state)
            td = reset_out
        else:
            state = next_state
        
        # Evaluation and checkpoint
        if step - tracking['last_eval_step'] >= config.eval_freq:
            save_checkpoint(original_agent, policy, tracking, config, run, step, curriculum_option)
            policy.model = original_agent # Use non-compiled model for evals
            run_evaluation(policy, tracking, config, run, step, curriculum)
            policy.model = agent
            print(f"Eval + checkpoint at step {step}")
            tracking['last_eval_step'] = step
        
        # PPO update
        if buffer.idx == buffer.capacity:
            diagnostics = policy.update(buffer, config, next_state=state)
            log_training_metrics(tracking, diagnostics, policy, config, step)
            tracking['completed_rewards'].clear()
            tracking['completed_lengths'].clear()
            buffer.clear()
    
    env.close()
    policy.model = original_agent
    run_evaluation(policy, tracking, config, run, step, curriculum)
    save_checkpoint(original_agent, policy, tracking, config, run, step, curriculum_option)
    
    if config.USE_WANDB:
        import wandb
        wandb.finish() 
    print("Training complete.")
    return agent

if __name__ == "__main__":
    run_training()
