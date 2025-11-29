import warnings

warnings.filterwarnings('ignore', message=".*Overwriting existing videos.*", category=UserWarning)
warnings.filterwarnings('ignore', message="Can't initialize NVML", category=UserWarning, module="torch.cuda")
warnings.filterwarnings('ignore', message="Conversion of an array with ndim > 0 to a scalar is deprecated", 
                       category=DeprecationWarning, module="torchrl.envs.libs.gym")
warnings.filterwarnings('ignore', message=".*Overwriting existing videos.*", 
                    category=UserWarning, module="gymnasium.wrappers.rendering")

import torch as t 
import numpy as np
from tqdm import tqdm
from evals import run_evaluation
from runner import run_training
from utils import (
    init_tracking, 
    update_episode_tracking,
    log_training_metrics, 
    save_checkpoint, 
    handle_env_resets,
    get_torch_compatible_actions,
    get_entropy,
    readable_timestamp,
    setup_from_checkpoint
)
from environment import make_env
from ppo import PPO
from buffer import RolloutBuffer
from curriculum import (
    CurriculumState,
    compute_level_distribution,
    DEFAULT_CURRICULUM,
)


def create_env_from_curriculum(curriculum_state, config, render_human=False):
    """
    Create environment with level distribution from current curriculum phase.
    """
    weights = curriculum_state.get_phase_weights()
    level_dist = compute_level_distribution(config.num_envs, weights)
    
    env = make_env(
        num_envs=config.num_envs,
        level_distribution=level_dist,
        render_human=render_human and config.num_envs == 1,
    )
    
    return env, level_dist


def transition_curriculum_phase(env, curriculum_state, config, device):
    """
    Handle transition to a new curriculum phase.
    """
    env.close()
    
    new_env, level_dist = create_env_from_curriculum(curriculum_state, config)
    
    print(f"\n{'='*60}")
    print(f"CURRICULUM TRANSITION: {curriculum_state.get_description()}")
    # print(f"Level distribution: {level_dist}")
    print(f"{'='*60}\n")
    
    environment = new_env.reset()
    state = environment['pixels']
    if config.num_envs == 1 and state.dim() == 3:
        state = state.unsqueeze(0)
    
    return new_env, environment, state


def init_training_components(agent, config, device, curriculum_state=None):
    """
    Initialize all training components: policy, buffer, environment.
    """
    policy = PPO(agent, config, device)
    buffer = RolloutBuffer(config.buffer_size // config.num_envs, config.num_envs, device)
    
    if curriculum_state is not None:
        env, level_dist = create_env_from_curriculum(curriculum_state, config, render_human=(config.num_envs == 1))
        print(f"\n{'='*60}")
        print(f"CURRICULUM LEARNING ENABLED")
        print(f"Starting with: {curriculum_state.get_description()}")
        print(f"Level distribution: {level_dist}")
        print(f"{'='*60}\n")
    else:
        env = make_env(
            num_envs=config.num_envs,
            render_human=(config.num_envs == 1),
        )
    
    environment = env.reset()
    state = environment['pixels']
    if config.num_envs == 1 and state.dim() == 3:
        state = state.unsqueeze(0)
    
    return policy, buffer, env, environment, state


def training_loop(agent, config, num_eval_episodes=5, checkpoint_path=None, resume=False, curriculum_option=None):
    """
    Main training loop supporting both standard and curriculum learning.
    """
    run = config.setup_wandb()
    device = "cuda" if t.cuda.is_available() else "cpu"
    agent = agent.to(device)
    
    # Initialize curriculum if enabled
    curriculum_state = None
    if config.use_curriculum:
        if curriculum_option is not None:
            curriculum_state = CurriculumState.from_option(curriculum_option)
        else:
            curriculum_state = CurriculumState.from_schedule(DEFAULT_CURRICULUM)
    
    # Initialize training components
    policy, buffer, env, environment, state = init_training_components(
        agent, config, device, curriculum_state
    )
    
    # Handle checkpoint loading
    if checkpoint_path is not None:
        start_step, tracking = setup_from_checkpoint(
            checkpoint_path, agent, policy, config, device, resume
        )
        # Sync curriculum phase if resuming
        if curriculum_state is not None and start_step > 0:
            # Fast-forward curriculum to correct phase
            progress = start_step / config.num_training_steps
            while curriculum_state.check_phase_transition(start_step, config.num_training_steps):
                pass  # Updates internal state
            
            if curriculum_state.current_phase != 0:
                env, environment, state = transition_curriculum_phase(
                    env, curriculum_state, config, device
                )
    else:
        tracking = init_tracking(config)
        start_step = 0
    
    # Log model info
    total_params = sum(p.numel() for p in agent.parameters())
    print(f"Total Parameters: {total_params:,}")
    print(f"Training {config.architecture} on {device.upper()}")
    
    pbar = tqdm(range(start_step, config.num_training_steps), disable=not config.show_progress)
    agent_class = agent.__class__
    agent = t.compile(agent)
    for step in pbar:
        # Check for curriculum phase transition
        if curriculum_state is not None:
            if curriculum_state.check_phase_transition(step, config.num_training_steps):
                env, environment, state = transition_curriculum_phase(
                    env, curriculum_state, config, device
                )
                buffer.clear()
                # Reset current stats on transition
                tracking['current_episode_rewards'] = [0.0] * config.num_envs
                tracking['current_episode_lengths'] = [0] * config.num_envs
        
        # Entropy decay
        policy.c2 = get_entropy(step, total_steps=config.num_training_steps, max_entropy=config.c2)
        
        # Action selection and environment step
        actions, log_probs, values = policy.action_selection(state)
        environment["action"] = get_torch_compatible_actions(actions)
        environment = env.step(environment)
        next_state = environment["next"]["pixels"]

        if config.num_envs == 1 and next_state.dim() == 3:
            next_state = next_state.unsqueeze(0)

        rewards = environment["next"]["reward"]
        dones = environment["next"]["done"]
        trunc = environment["next"].get("truncated", t.zeros_like(dones))
        terminated = dones | trunc
        
        if config.num_envs == 1:
            if rewards.dim() == 0:
                rewards = rewards.unsqueeze(0)
            if dones.dim() == 0:
                dones = dones.unsqueeze(0)

        # Store transition
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
        recent_rewards = tracking['completed_rewards'][-20:]
        curr_entropy = get_entropy(step, total_steps=config.num_training_steps, max_entropy=config.c2)
        if len(recent_rewards) >= 20:
            std = np.std(recent_rewards)
            recent_rewards = np.mean(recent_rewards) 
            if std < 1.0 and recent_rewards < 150:
                curr_entropy *= 3
                if step % 1000 == 0:
                    print(f"[{step}] Entropy Boost Triggered: StdDev {std:.2f}")
        policy.c2 = curr_entropy
        # Update progress bar
        if config.show_progress and len(tracking['completed_rewards']) > 0:
            postfix = {
                'ep': tracking['episode_num'],
                'reward': f"{np.mean(tracking['completed_rewards']):.1f}",
                'updates': tracking['num_updates'],
                'lr': f"{policy.get_current_lr():.1e}",
            }
            if curriculum_state is not None:
                postfix['phase'] = curriculum_state.current_phase
            pbar.set_postfix(postfix)
            
        # Checkpointing
        if step - tracking['last_checkpoint'] >= config.checkpoint_freq:
            save_checkpoint(agent, policy, tracking, config, run, step, curriculum_option=curriculum_option)
            print(f"Model checkpoint saved at step {step}")
    
        # Evaluation
        if step - tracking['last_eval_step'] >= config.eval_freq:
            run_evaluation(agent_class, policy, tracking, config, run, step, num_eval_episodes, curriculum_state)

        # Handle environment resets
        state, environment = handle_env_resets(env, environment, next_state, terminated, config.num_envs)
          
        # PPO update when buffer is full
        if buffer.idx == buffer.capacity:
            mean_loss = policy.update(buffer, config, next_state=state)
            tracking['num_updates'] += 1
            
            log_training_metrics(tracking, mean_loss, policy, config, step)
            tracking['completed_rewards'].clear()
            tracking['completed_lengths'].clear()
            
            buffer.clear()
            

    # Cleanup
    env.close()
    
    # Final evaluation and checkpoint
    run_evaluation(agent.__class__, policy, tracking, config, run, step, num_eval_episodes, curriculum_state)
    save_checkpoint(agent, policy, tracking, config, run, step, curriculum_option=curriculum_option)

    if config.USE_WANDB:
        import wandb
        wandb.finish()
    else:
        print("Training completed successfully")
    
    return agent


def train(model, config, num_eval_episodes=9, curriculum_option=None):
    """Start fresh training run."""
    agent = model()
    return training_loop(agent, config, num_eval_episodes, curriculum_option=curriculum_option)


def finetune(model, checkpoint_path, config, num_eval_episodes=9, curriculum_option=None):
    """Load weights from checkpoint but start training fresh (step 0)."""
    agent = model()
    return training_loop(agent, config, num_eval_episodes, checkpoint_path=checkpoint_path, curriculum_option=curriculum_option)


def resume(model, checkpoint_path, config, num_eval_episodes=9, curriculum_option=None):
    """Resume training from checkpoint step."""
    agent = model()
    return training_loop(agent, config, num_eval_episodes, checkpoint_path=checkpoint_path, resume=True, curriculum_option=curriculum_option)


if __name__ == "__main__":
    run_training()
