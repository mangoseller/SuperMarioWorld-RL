import numpy as np
import os
import torch as t
import wandb
import random
from datetime import datetime
from multiprocessing import Process, Queue
from environment import make_eval_env
from utils import get_torch_compatible_actions
from ppo import PPO
from curriculum import ALL_LEVELS


# Default evaluation levels when not using curriculum
DEFAULT_EVAL_LEVELS = ['YoshiIsland2', 'YoshiIsland3', 'DonutPlains1', 'DonutPlains4', 'ChocolateIsland1']


def eval_timestamp():
    """Short timestamp for eval directories: HH-MM"""
    return datetime.now().strftime("%H-%M")


def run_timestamp():
    """Timestamp for run directories: DD-MM_HH-MM"""
    return datetime.now().strftime("%d-%m_%H-%M")


def get_eval_levels_for_training(curriculum_state=None, trained_levels=None):
    """
    Determine which levels to evaluate on.
    
    If curriculum_state is provided, uses all levels from the curriculum plus one random holdout.
    If trained_levels is provided as a set/list, uses those plus one random holdout.
    Otherwise, uses DEFAULT_EVAL_LEVELS.
    
    Args:
        curriculum_state: CurriculumState object (optional)
        trained_levels: Set or list of level names that were trained on (optional)
    
    Returns:
        List of level names to evaluate on
    """
    if curriculum_state is not None:
        return curriculum_state.get_eval_levels()
    
    if trained_levels is not None:
        trained_set = set(trained_levels)
        holdout_candidates = [level for level in ALL_LEVELS if level not in trained_set]
        
        eval_levels = list(trained_set)
        if holdout_candidates:
            eval_levels.append(random.choice(holdout_candidates))
        return sorted(eval_levels)
    
    return DEFAULT_EVAL_LEVELS


def evaluate(agent, num_episodes=9, record_dir='./evals', eval_levels=None):
    """
    Evaluate agent across multiple levels.
    
    Args:
        agent: PPO policy wrapper with action_selection method
        num_episodes: Total episodes to run (distributed across levels)
        record_dir: Base directory for saving evaluation videos
        eval_levels: List of level names to evaluate on
    
    Returns:
        Dictionary of evaluation metrics
    """
    if eval_levels is None:
        eval_levels = DEFAULT_EVAL_LEVELS

    episodes_per_level = max(3, num_episodes // len(eval_levels))
    
    all_rewards = []
    all_lengths = []
    level_metrics = {}
    os.makedirs(record_dir, exist_ok=True)

    for level in eval_levels:
        level_rewards = []
        level_lengths = []

        level_record_dir = os.path.join(record_dir, level)
        os.makedirs(level_record_dir, exist_ok=True)
        
        eval_env = make_eval_env(level, record_dir=level_record_dir)

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
        "eval/min_reward": np.min(all_rewards),
        "eval/num_episodes": len(all_rewards),
        "eval/num_levels": len(eval_levels),
    }
    
    aggregate_metrics.update(level_metrics)
    return aggregate_metrics 


def _run_eval_(model, model_state_dict, config, num_episodes, record_dir, eval_levels, result_queue):
    """Internal function for subprocess evaluation."""
    agent = model().to('cpu')
    agent.load_state_dict(model_state_dict)
    eval_policy = PPO(agent, config, device="cpu")
    metrics = evaluate(eval_policy, num_episodes, record_dir, eval_levels=eval_levels)
    result_queue.put(metrics)


def eval_parallel_safe(model, policy, config, record_dir, num_episodes=5, eval_levels=None):
    """Run evaluation in a subprocess to avoid environment conflicts."""
    result_queue = Queue()
    cpu_state_dicts = {k: v.cpu() for k, v in policy.model.state_dict().items()}
    process = Process(target=_run_eval_, args=(
        model, cpu_state_dicts, config, num_episodes, record_dir, eval_levels, result_queue 
    ))
    process.start()
    process.join()
    return result_queue.get()


def run_evaluation(model, policy, tracking, config, run, step, episodes, curriculum_state=None):
    """Run evaluation and log results to wandb."""
    episode_num = tracking['episode_num']
    time_str = eval_timestamp()
    
    # Format: evals/TransPala_27-11/eval_ep1234_14-30/
    run_dir = f'evals/{config.architecture}_{tracking["run_timestamp"]}'
    eval_dir = f'{run_dir}/eval_ep{episode_num}_{time_str}'
    os.makedirs(eval_dir, exist_ok=True)
    
    # Determine eval levels based on curriculum or config
    eval_levels = get_eval_levels_for_training(curriculum_state)
    
    eval_metrics = eval_parallel_safe(
        model, policy, config, 
        num_episodes=episodes, 
        record_dir=eval_dir,
        eval_levels=eval_levels
    )
    
    if config.USE_WANDB:
        wandb.log(eval_metrics)
        artifact_name = f"{config.architecture}_ep{episode_num}_{time_str}"
        vids = wandb.Artifact(artifact_name, type="eval_videos")
        vids.add_dir(eval_dir)
        run.log_artifact(vids)

    tracking['last_eval_step'] = step
