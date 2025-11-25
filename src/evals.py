
import numpy as np
import retro
import os
import torch as t
import wandb
from multiprocessing import Process, Queue
from environment import prepare_env
from utils import get_torch_compatible_actions, readable_timestamp
from ppo import PPO

EVAL_LEVELS = ['YoshiIsland2', 'YoshiIsland1', 'DonutPlains1']

def evaluate(agent, num_episodes=9, record_dir='/evals'):

    assert num_episodes >= 3 and num_episodes % 3 == 0, f"Number of episodes must be a multiple of 3, got {num_episodes}"
    episodes_per_level = num_episodes // 3
    all_rewards = []
    all_lengths = []
    level_metrics = {}
    os.makedirs(record_dir, exist_ok=True)

    for level in EVAL_LEVELS:
        level_rewards = []
        level_lengths = []

        level_record_dir = os.path.join(record_dir, level)
        os.makedirs(level_record_dir, exist_ok=True)
        eval_env = retro.make(
            'SuperMarioWorld-Snes',
            render_mode='rgb_array',
            state=level,
            )
        eval_env = prepare_env(eval_env, record = True, record_dir=level_record_dir)

        for episode in range(episodes_per_level):
            eval_environment = eval_env.reset()
            state = eval_environment["pixels"]
            episode_reward = 0
            episode_length = 0
            done = False

            while not done:
                action, _, _ = agent.action_selection(state)
                eval_environment["action"] = get_torch_compatible_actions(t.tensor(action))
                eval_environment = eval_env.step(eval_environment)
                state = eval_environment["next"]["pixels"]
                reward = eval_environment["next"]["reward"].item()
                done = eval_environment["next"]["done"].item()
                episode_reward += reward
                episode_length += 1
            
            level_rewards.append(episode_reward)
            level_lengths.append(episode_length)
            all_rewards.append(episode_reward)
            all_lengths.append(episode_length)

        eval_env.close()
        
        # Track per-level metrics
        level_metrics[f"eval/{level}/mean_reward"] = np.mean(level_rewards)
        level_metrics[f"eval/{level}/std_reward"] = np.std(level_rewards)
        level_metrics[f"eval/{level}/mean_length"] = np.mean(level_lengths)
        level_metrics[f"eval/{level}/max_reward"] = np.max(level_rewards)
        level_metrics[f"eval/{level}/min_reward"] = np.min(level_rewards)

    # Aggregate metrics across all levels
    aggregate_metrics = {
        "eval/mean_reward": np.mean(all_rewards),
        "eval/std_reward": np.std(all_rewards),
        "eval/mean_length": np.mean(all_lengths),
        "eval/max_reward": np.max(all_rewards),
        "eval/min_reward": np.min(all_rewards)
    }
    
    aggregate_metrics.update(level_metrics)

    return aggregate_metrics 


def _run_eval_(model, model_state_dict, config, num_episodes, record_dir, result_queue):   
# Internal function called to setup and run eval
    agent = model().to('cpu')
    agent.load_state_dict(model_state_dict)
    eval_policy = PPO(agent, config, device="cpu")
    metrics = evaluate(eval_policy, num_episodes, record_dir)
    result_queue.put(metrics)


def eval_parallel_safe(model, policy, config, record_dir, num_episodes=5):
# Run evaluate in a sub-process 
    result_queue = Queue()
    # Move model weights to cpu
    cpu_state_dicts = {k: v.cpu() for k, v in policy.model.state_dict().items()}
    process = Process(target=_run_eval_, args=(
        model, cpu_state_dicts, config, num_episodes, record_dir, result_queue 
    ))
    process.start()
    process.join()
    return result_queue.get()


def run_evaluation(model, policy, tracking, config, run, step, episodes):
    # Run evaluation and log results to wandb - used in main training calls
    eval_timestamp = readable_timestamp()
    run_dir = f'evals/run_{tracking["run_timestamp"]}'
    eval_dir = f'{run_dir}/eval_step_{tracking["total_env_steps"] // config.num_envs}_time_{eval_timestamp}'
    os.makedirs(eval_dir, exist_ok=True)
    eval_metrics = eval_parallel_safe(model, policy, config, num_episodes=episodes, record_dir=eval_dir)
    
    if config.USE_WANDB:
        wandb.log(eval_metrics)
        vids = wandb.Artifact(
            f"{readable_timestamp()}_step_{tracking['total_env_steps'] // config.num_envs}",
            type="eval_videos"
        )
        vids.add_dir(eval_dir)
        run.log_artifact(vids)

    tracking['last_eval_step'] = step
