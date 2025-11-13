import torch as t 
import numpy as np
import os 
import wandb
from model_small import ImpalaSmall
from ppo import PPO
from buffer import RolloutBuffer
from environment import env, eval_env, evaluate

USE_WANDB = False
if USE_WANDB:
    api_key = os.environ.get("WANDB_API_KEY")
    if api_key is None:
        raise RuntimeError("WANDB_API_KEY not set in environment")

    wandb.login(key=api_key)
    run = wandb.init(
    project="marioRL",
    config={
       "learning_rate": 1e-4, # TODO: add learning rate scheduling
       "gamma": 0.99,
       "lambda": 0.95, 
       "epsilon": 1e-8, # Advantage Normalization
       "clip_eps": 0.2, # PPO clipping
       "c1": 0.5,
       "c2": 0.01,
       "architecture": "IMAPLASmall",
       "epochs": 4,
       "buffer_size": 4096,
       "minibatch_size": 64,
    },
    )

# assert t.cuda.is_available(), "GPU is not available!"
# device = 'cuda'
device = 'cpu'
agent = ImpalaSmall().to(device)
# assert next(agent.parameters()).is_cuda, "Model is not on GPU!"
policy = PPO(
    model=agent,
    lr=1e-4, # TODO: Implement LR Scheduling
    epsilon=0.2,
    optimizer=t.optim.Adam,
    device=device,
    c1=0.5,
    c2=0.01
)

# Create buffer, initialize environment and get first state 
# buffer = RolloutBuffer(4096, device)
# environment = env.reset()
# state = environment['pixels']
# num_training_steps = int(1e6) # Change to 20-50M steps when training is set up correctly
# eval_freq = 250_000
# last_eval = 0
# checkpoint_freq = 200_000
# last_checkpoint = 0


# Testing Params:
buffer = RolloutBuffer(512, device)
environment = env.reset()
state = environment['pixels']
num_training_steps = 50000
eval_freq = int(1e14)
checkpoint_freq = 10000
last_checkpoint = 0
last_eval=0

# Initialize tracking variables 
episode_rewards = []
episode_lengths = []
num_updates = 0 # Number of PPO updates performed
episode_reward = 0
episode_length = 0
episode_num = 0

for step in range(num_training_steps):
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

    # Evaluate with agent taking optimal actions
    if step - last_eval >= eval_freq:
        eval_metrics = evaluate(policy, eval_env, num_episodes=5)
        eval_metrics["Step"] = step 
        wandb.log(eval_metrics)
        last_eval = step

    if done:
        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)
        episode_reward = 0
        episode_length = 0 
        episode_num += 1

        environment = env.reset()
        state = environment["pixels"]
    else:
        state = next_state

    # Update PPO when buffer is full
    if buffer.idx == buffer.capacity:
        policy.update(buffer, next_state=state)
        num_updates += 1
        # Log metrics
        if len(episode_rewards) > 0:
            if USE_WANDB:
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
        if step - last_checkpoint >= checkpoint_freq:
            model_path = f"ImpalaSmall{episode_num}.pt"
            t.save(agent.state_dict(), model_path)
            if USE_WANDB:
                artifact = wandb.Artifact(f"marioRLep{episode_num}", type="model")
                artifact.add_file(model_path)
                run.log_artifact(artifact)
            last_checkpoint = step

if USE_WANDB:
    wandb.finish()











