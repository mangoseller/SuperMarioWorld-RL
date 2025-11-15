import warnings
# Supress NVML warning
warnings.filterwarnings(
        'ignore', 
        message="Can't initialize NVML",
        category=UserWarning,
        module="torch.cuda"
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
import gc
import time

parser = argparse.ArgumentParser()
parser.add_argument('--config', choices=['train', 'test'], default='test')
args=parser.parse_args()
config = TRAINING_CONFIG if args.config == 'train' else TESTING_CONFIG
run = config.setup_wandb()

if __name__ == "__main__":
    # assert t.cuda.is_available(), "GPU is not available!"
    # device = 'cuda'
    device = 'cpu'
    agent = ImpalaSmall().to(device)
    # assert next(agent.parameters()).is_cuda, "Model is not on GPU!"
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

    # Initialize tracking variables for each env
    episode_rewards = [0.0] * config.num_envs
    episode_lengths = [0] * config.num_envs
    completed_lengths = []
    completed_rewards = []
    num_updates = 0 # Number of PPO updates performed
    episode_reward = 0
    episode_length = 0
    episode_num = 0
    last_checkpoint = 0
    total_env_steps = 0
    last_eval_steps = 0

    for step in range(config.num_training_steps):
        actions, log_probs, values = policy.action_selection(state)
        # if step % 50 == 0:
        #    print(f"Step {step}: Action={action.item()}, Value={value:.3f}")

        # Take a step
        environment["action"] = actions.unsqueeze(-1)
        environment = env.step(environment)
        next_state = environment["next"]["pixels"]
        rewards = environment["next"]["reward"]
        # if reward != 0:
        #   print(f"Reward at step {step} is: {reward}")
        dones = environment["next"]["done"]

        # Store batched data in buffer
        buffer.store(
            state.cpu().numpy(), 
            rewards.squeeze().cpu().numpy(),
            actions.cpu().numpy(),
            log_probs.cpu().numpy(),
            values.cpu().numpy(),
            dones.squeeze().cpu().numpy(),
        )

        total_env_steps += config.num_envs

        # Update per-env tracking
        for i in range(config.num_envs):
            episode_rewards[i] += rewards[i].item()
            episode_lengths[i] += 1

            if dones[i].item(): 
                completed_rewards.append(episode_rewards[i])
                completed_lengths.append(episode_lengths[i])
                # Reset this envs tracking

                episode_rewards[i] = 0.0 
                episode_lengths[i] = 0
                episode_num += 1

        # Evaluate with agent taking optimal actions as per current policy
        if total_env_steps - last_eval_steps >= config.eval_freq:
            timestamp = int(time.time())
            eval_dir = f'evals/eval_step_{total_env_steps}_time_{timestamp}'

            eval_metrics = eval_parallel_safe(policy, num_episodes=1, record_dir='evals')
            eval_metrics["total_env_steps"] = total_env_steps
            if config.USE_WANDB:
                wandb.log(eval_metrics)
                vids = wandb.Artifact(
                    f"eval_videos_step_{total_env_steps}", 
                        type="eval_videos"
                )
                vids.add_dir(eval_dir)  
                run.log_artifact(vids)
                
            last_eval_steps = total_env_steps

        state = next_state


        # Update PPO when buffer is full
        if buffer.idx == buffer.capacity:
            mean_loss = policy.update(buffer, next_state=state)
            num_updates += 1
            # Log metrics
            if len(episode_rewards) > 0:
                if config.USE_WANDB:
                    wandb.log({
                        "train/loss": mean_loss,
                        "train/mean_reward": np.mean(episode_rewards),
                        "train/mean_length": np.mean(episode_lengths),
                        "train/num_episodes": len(episode_rewards),
                        "global_step": step,
                        "num_updates": num_updates,
                    })
                completed_rewards.clear()
                completed_lengths.clear()
            buffer.clear()

            # Save model at Checkpoints
            if step - last_checkpoint >= config.checkpoint_freq:

                model_path = f"ImpalaSmall{episode_num}.pt"
                t.save(agent.state_dict(), model_path)
                if config.USE_WANDB:
                    artifact = wandb.Artifact(f"marioRLep{episode_num}", type="model")
                    artifact.add_file(model_path)
                    run.log_artifact(artifact)
                last_checkpoint = step
                print("Check WANDB for test progress.")
                exit(0)

    if config.USE_WANDB:
        wandb.finish()
    else:
        print("Test completed without incident.")











