import numpy as np
import os
import torch as t
import wandb
from datetime import datetime
from multiprocessing import Process, Queue
from environment import make_eval_env
from utils import get_torch_compatible_actions, get_base_model
from ppo import PPO

def _run_episode(agent, env):
    # Run 1 eval episode, return (total_reward, length)
    td = env.reset()
    state = td["pixels"]
    reward_sum, length = 0.0, 0
    
    while True:
        action, _, _ = agent.action_selection(state)
        td["action"] = get_torch_compatible_actions(t.tensor(action))
        td = env.step(td)
        
        reward_sum += td["next"]["reward"].item()
        length += 1
        
        if td["next"]["done"].item():
            break
        state = td["next"]["pixels"]
    
    return reward_sum, length


def _compute_stats(values, prefix):
    return {
        f"{prefix}/mean": np.mean(values),
        f"{prefix}/std": np.std(values),
        f"{prefix}/max": np.max(values),
        f"{prefix}/min": np.min(values),
    }


def evaluate(agent, record_dir='./evals', eval_levels=None):

    levels = eval_levels or ['YoshiIsland2', 'YoshiIsland3', 'DonutPlains1', 'DonutPlains4', 'ChocolateIsland1']     
    all_rewards, all_lengths = [], []
    metrics = {}
    
    for level in levels:
        level_dir = os.path.join(record_dir, level)
        os.makedirs(level_dir, exist_ok=True)
        env = make_eval_env(level, record_dir=level_dir)
        rewards, lengths = zip(*[
            _run_episode(agent, env) 
            for _ in range(3)
        ])

        env.close()
        all_rewards.extend(rewards)
        all_lengths.extend(lengths)
        metrics.update(_compute_stats(rewards, f"eval/{level}/reward"))
        metrics[f"eval/{level}/mean_length"] = np.mean(lengths)

    metrics.update(_compute_stats(all_rewards, "eval/reward"))
    metrics["eval/mean_length"] = np.mean(all_lengths)
    metrics["eval/num_episodes"] = len(all_rewards)
    metrics["eval/num_levels"] = len(levels)

    return metrics


def _eval_worker(model_class, state_dict, config, record_dir, eval_levels, queue):
    # Subprocess worker that runs evaluation
    agent = model_class().to('cpu')
    agent.load_state_dict(state_dict)
    policy = PPO(agent, config, device="cpu")
    queue.put(evaluate(policy, record_dir, eval_levels))



def evaluate_in_subprocess(model_class, config, record_dir, eval_levels):
    # Run evaluation in subprocess to avoid environment conflicts with training
    queue = Queue()
    state_dict = {k: v.cpu() for k, v in model_class.state_dict().items()}
    proc = Process(target=_eval_worker, args=(
        model_class, state_dict, config, record_dir, eval_levels, queue
    ))
    proc.start()
    proc.join()

    return queue.get()


def run_evaluation(model_class, policy, tracking, config, run, step, curriculum=None):

    timestamp = datetime.now().strftime("%H-%M")
    episode_num = tracking['episode_num']
    eval_dir = f'evals/{config.architecture}_{tracking["run_timestamp"]}/eval_ep{episode_num}_{timestamp}'
    os.makedirs(eval_dir, exist_ok=True)
    
    eval_levels = curriculum.eval_levels if curriculum else ['YoshiIsland2', 'YoshiIsland3', 'DonutPlains1', 'DonutPlains4', 'ChocolateIsland1']      
    metrics = evaluate_in_subprocess(
        model_class, policy, config,
        record_dir=eval_dir,
        eval_levels=eval_levels,
    )
    
    if config.USE_WANDB:
        wandb.log(metrics)
        artifact = wandb.Artifact(f"{config.architecture}_ep{episode_num}_{timestamp}", type="eval_videos")
        artifact.add_dir(eval_dir)
        run.log_artifact(artifact)

    tracking['last_eval_step'] = step


