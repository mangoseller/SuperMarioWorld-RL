import torch as t 
import numpy as np
import wandb
from model_small import ImpalaSmall
from ppo import PPO
from buffer import RolloutBuffer
from environment import evaluate, make_training_env
import argparse
from training_utils import TRAINING_CONFIG, TESTING_CONFIG


parser = argparse.ArgumentParser()
parser.add_argument('--config', choices=['train', 'testing'], default='test')
args=parser.parse_args()
config = TRAINING_CONFIG if args.config == 'train' else TESTING_CONFIG
run = config.setup_wandb()


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
buffer = RolloutBuffer(config.buffer_size, device)
env = make_training_env()
environment = env.reset()
state = environment['pixels']

# Initialize tracking variables 
episode_rewards = []
episode_lengths = []
num_updates = 0 # Number of PPO updates performed
episode_reward = 0
episode_length = 0
episode_num = 0
last_checkpoint = 0
last_eval = 0

for step in range(config.num_training_steps):
    action, log_prob, value = policy.action_selection(state)
    # if step % 50 == 0:
    #    print(f"Step {step}: Action={action.item()}, Value={value:.3f}")

    # Take a step
    environment["action"] = action.unsqueeze(0)
    environment = env.step(environment)
    next_state = environment["next"]["pixels"]
    reward = environment["next"]["reward"].item()
    if reward != 0:
        print(f"Reward at step {step} is: {reward}")
    done = environment["next"]["done"].item()

    # Store step data in buffer
    buffer.store(state, reward, action.item(), log_prob, value, done)
    episode_reward += reward
    episode_length += 1

    if done:
        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)
        episode_reward = 0
        episode_length = 0 
        episode_num += 1

    # Evaluate with agent taking optimal actions
    if step - last_eval >= config.eval_freq:
        # Close training env
        env.close()
        eval_metrics = evaluate(policy, num_episodes=5, record_dir='evals')
        eval_metrics["Step"] = step 
        if config.USE_WANDB:
            wandb.log(eval_metrics)

        env = make_training_env()
        environment = env.reset()
        state = environment["pixels"]
        last_eval = step


    # Update PPO when buffer is full
    if buffer.idx == buffer.capacity:
        policy.update(buffer, next_state=state)
        num_updates += 1
        # Log metrics
        if len(episode_rewards) > 0:
            if config.USE_WANDB:
                wandb.log({
                    "train/mean_reward": np.mean(episode_rewards),
                    "train/mean_length": np.mean(episode_lengths),
                    "train/num_episodes": len(episode_rewards),
                    "global_step": step,
                    "num_updates": num_updates,
                })
            episode_rewards.clear()
            episode_lengths.clear()
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

if config.USE_WANDB:
    wandb.finish()











