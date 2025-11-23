
import numpy as np
import retro
import os
import torch as t
import wandb
from multiprocessing import Process, Queue
from environment import prepare_env
from training_utils import get_torch_compatible_actions, readable_timestamp
from ppo import PPO

def evaluate(agent, num_episodes=5, record_dir='/evals', temp=0.1):
    eval_rewards, eval_lengths = [], []
    os.makedirs(record_dir, exist_ok=True)
    eval_env = retro.make('SuperMarioWorld-Snes',
                          render_mode='rgb_array',
                          state='YoshiIsland2',
                          )
    eval_env = prepare_env(eval_env, record=True, record_dir=record_dir)
    for _ in range(num_episodes):
        eval_environment = eval_env.reset()
        state = eval_environment["pixels"]
        episode_reward = 0
        episode_length = 0
        done = False

        while not done:
            action = agent.eval_action_selection(state, temp)
            eval_environment["action"] = get_torch_compatible_actions(t.tensor(action))
            eval_environment = eval_env.step(eval_environment)
            state = eval_environment["next"]["pixels"]
            reward = eval_environment["next"]["reward"].item()
            done = eval_environment["next"]["done"].item()
            episode_reward += reward
            episode_length += 1

        eval_rewards.append(episode_reward)
        eval_lengths.append(episode_length)

    eval_env.close()

    return {
        "eval/mean_reward": np.mean(eval_rewards),
        "eval/std_reward": np.std(eval_rewards),
        "eval/mean_length": np.mean(eval_lengths),
        "eval/max_reward": np.max(eval_rewards),
        "eval/min_reward": np.min(eval_rewards)
    }


def _run_eval_(model, model_state_dict, config, num_episodes, record_dir, result_queue, temp):   
# Internal function called to setup and run eval
    agent = model().to('cpu')
    agent.load_state_dict(model_state_dict)
    eval_policy = PPO(agent, config, device="cpu")
    metrics = evaluate(eval_policy, num_episodes, record_dir, temp)
    result_queue.put(metrics)


def eval_parallel_safe(model, policy, config, record_dir, eval_temp=0.1, num_episodes=3):
# Run evaluate in a sub-process 
    result_queue = Queue()
    # Move model weights to cpu
    cpu_state_dicts = {k: v.cpu() for k, v in policy.model.state_dict().items()}
    process = Process(target=_run_eval_, args=(
        model, cpu_state_dicts, config, num_episodes, record_dir, result_queue, eval_temp
    ))
    process.start()
    process.join()
    return result_queue.get()


def run_evaluation(model, policy, tracking, config, run, episodes, temp):
    # Run evaluation and log results to wandb - used in main training calls
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
