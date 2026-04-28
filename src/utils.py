import torch as t
import numpy as np
import os
import wandb
from datetime import datetime

def readable_timestamp():
    return datetime.now().strftime("%d-%m_%H-%M")

def get_checkpoint_info(checkpoint_path):
    checkpoint = t.load(checkpoint_path, map_location='cpu')
    config_dict = checkpoint.get('config_dict', {})
    tracking = checkpoint.get('tracking', {})
    
    return {
        'architecture': config_dict.get('architecture'),
        'step': checkpoint.get('step', 0),
        'episode_num': tracking.get('episode_num', 0),
        'total_steps': config_dict.get('num_training_steps'),
        'use_curriculum': config_dict.get('use_curriculum', False),
        'curriculum_option': config_dict.get('curriculum_option'),
    }

def log_training_metrics(tracking, diagnostics, policy, config, step):

    if len(tracking['completed_rewards']) > 0:
        mean_reward = np.mean(tracking['completed_rewards'])
        mean_length = np.mean(tracking['completed_lengths'])
    else:
        mean_reward = 0
        mean_length = 0
    
    metrics = {
        'train/mean_reward': mean_reward,
        'train/mean_episode_length': mean_length,
        'train/episodes': tracking['episode_num'],
        'train/total_env_steps': tracking['total_env_steps'],
        
        'loss/total': diagnostics.get('total_loss', 0),
        'loss/policy': diagnostics.get('policy_loss', 0),
        'loss/value': diagnostics.get('value_loss', 0),
        'loss/pixel_control': diagnostics.get('pixel_control_loss', 0),

        
        'hyperparams/entropy_coef': policy.c2,
        'hyperparams/learning_rate': policy.get_current_lr(),
        'hyperparams/value_coef': policy.c1,
    }
    
    if config.USE_WANDB:
        wandb.log(metrics, step=step)

    return metrics


def _mean(values):
    if values is None or len(values) == 0:
        return 0
    return float(np.mean(values))


def _distribution_stats(values, prefix):
    if values is None or len(values) == 0:
        return {}
    arr = np.asarray(values, dtype=np.float64)
    return {
        f"{prefix}/min": float(arr.min()),
        f"{prefix}/p10": float(np.percentile(arr, 10)),
        f"{prefix}/p50": float(np.percentile(arr, 50)),
        f"{prefix}/p90": float(np.percentile(arr, 90)),
        f"{prefix}/max": float(arr.max()),
    }


# Thresholds in (frame-skipped) env steps and game-x units.
# instant_steps=60 ≈ 3 s at frame_skip=3, 60 Hz. Spawn x is ~16, so x≤32 ≈ "barely moved",
# x≤100 ≈ "tiny progress in a several-thousand-unit-long level".
def _episode_outcomes(episode_lengths, x_max_values,
                      instant_steps=60, no_movement_x=32, wandering_x=100):
    if not episode_lengths or not x_max_values:
        return {}
    n = min(len(episode_lengths), len(x_max_values))
    if n == 0:
        return {}
    counts = {"instant_death": 0, "no_movement": 0, "wandering": 0, "progressing": 0}
    for L, xm in zip(episode_lengths[:n], x_max_values[:n]):
        if L <= instant_steps:
            counts["instant_death"] += 1
        elif xm <= no_movement_x:
            counts["no_movement"] += 1
        elif xm <= wandering_x:
            counts["wandering"] += 1
        else:
            counts["progressing"] += 1
    metrics = {f"train/outcome/{k}_frac": v / n for k, v in counts.items()}
    metrics["train/outcome/degenerate_frac"] = (
        counts["instant_death"] + counts["no_movement"] + counts["wandering"]
    ) / n
    return metrics


def _metric_value(value):
    if isinstance(value, t.Tensor):
        if value.numel() != 1:
            return None
        return float(value.detach().cpu().item())
    if isinstance(value, np.generic):
        return float(value)
    if isinstance(value, (int, float, bool)):
        return value
    return None


def _add_prefixed(metrics, prefix, values):
    if not values:
        return
    for key, value in values.items():
        metric = _metric_value(value)
        if metric is not None:
            metrics[f"{prefix}/{key}"] = metric


def log_muzero_metrics(tracking=None, diagnostics=None, replay=None,
                       self_play=None, reanalyse=None, search=None,
                       optimizer=None, config=None, step=None,
                       eval_stats=None, plr_stats=None, rnd_coef=None):
    tracking = tracking or {}
    diagnostics = diagnostics or {}
    episode_returns = tracking.get("episode_returns", [])
    episode_lengths = tracking.get("episode_lengths", [])
    total_episode_steps = sum(episode_lengths)
    mean_reward = (
        float(sum(episode_returns) / total_episode_steps)
        if total_episode_steps > 0 else 0.0
    )
    metrics = {
        "train/mean_episode_return": _mean(episode_returns),
        "train/mean_episode_length": _mean(episode_lengths),
        "train/mean_reward": mean_reward,
        "train/mean_x_max": _mean(tracking.get("x_max", [])),
        "train/episodes": tracking.get("episodes", 0),
        "train/env_steps": tracking.get("env_steps", 0),
        "train/gradient_steps": tracking.get("gradient_steps", 0),
    }
    metrics.update(_distribution_stats(episode_lengths, "train/episode_length"))
    metrics.update(_distribution_stats(tracking.get("x_max", []), "train/x_max"))
    metrics.update(_distribution_stats(episode_returns, "train/episode_return"))
    metrics.update(_episode_outcomes(episode_lengths, tracking.get("x_max", [])))
    for worker_type, count in tracking.get("worker_type_counts", {}).items():
        metrics[f"self_play/worker_type/{worker_type}/episodes"] = count

    loss_keys = ("total", "dynamics", "reward", "value", "policy", "rnd")
    for key in loss_keys:
        metric = _metric_value(diagnostics.get(f"{key}_loss", diagnostics.get(key)))
        if metric is not None:
            metrics[f"loss/{key}"] = metric

    _add_prefixed(metrics, "self_play", self_play)
    _add_prefixed(metrics, "reanalyse", reanalyse)
    _add_prefixed(metrics, "search", search)

    if replay is not None:
        metrics.update({
            "replay/transitions": len(replay),
            "replay/trajectories": replay.num_trajectories,
            "replay/frontier_trajectories": replay.num_frontier_trajectories,
        })
        if replay.num_trajectories > 0:
            metrics["replay/frontier_fraction"] = (
                replay.num_frontier_trajectories / replay.num_trajectories
            )
        for level, stats in replay.frontier_stats().items():
            metrics[f"replay/frontier/{level}/trajectories"] = stats["trajectories"]
            metrics[f"replay/frontier/{level}/transitions"] = stats["transitions"]

    if optimizer is not None and optimizer.param_groups:
        metrics["hyperparams/learning_rate"] = optimizer.param_groups[0]["lr"]

    if config is not None:
        metrics.update({
            "hyperparams/gamma": config.gamma,
            "hyperparams/td_steps": config.td_steps,
            "hyperparams/unroll_steps": config.unroll_steps,
            "hyperparams/mcts_num_simulations": config.mcts_num_simulations,
            "hyperparams/replay_beta": config.replay_beta,
        })

    if eval_stats:
        for level, stats in eval_stats.items():
            for k, v in stats.items():
                metrics[f"eval/{level}/{k}"] = v

    if plr_stats:
        for level, stats in plr_stats.items():
            for k, v in stats.items():
                metrics[f"plr/{level}/{k}"] = v

    if rnd_coef is not None:
        metrics["hyperparams/rnd_coef"] = rnd_coef

    if config is not None and getattr(config, "USE_WANDB", False):
        wandb.log(metrics, step=step)

    return metrics

def save_checkpoint(agent, policy, tracking, config, run, step, curriculum_option=None):

    checkpoint_dir = "model_checkpoints"
    os.makedirs(checkpoint_dir, exist_ok=True)   
    checkpoint = {
        'model_state_dict': agent.state_dict(),
        'optimizer_state_dict': policy.optimizer.state_dict(),
        'scheduler_state_dict': policy.scheduler.state_dict() if policy.scheduler else None,
        'step': step,
        'tracking': tracking,
        'config_dict': {
            'architecture': config.architecture,
            'num_training_steps': config.num_training_steps,
            'learning_rate': config.learning_rate,
            'min_lr': config.min_lr,
            'lr_schedule': config.lr_schedule,
            'c2': config.c2,
            'use_curriculum': config.use_curriculum,
            'curriculum_option': curriculum_option,
        }
    }
    model_path = os.path.join(checkpoint_dir, f"{config.architecture}_ep{tracking['episode_num']}.pt")
    t.save(checkpoint, model_path)
    if config.USE_WANDB:
        artifact = wandb.Artifact(f"marioRLep{tracking['episode_num']}", type="model")
        artifact.add_file(model_path)
        run.log_artifact(artifact)
    tracking['last_checkpoint'] = step

    return model_path

def load_checkpoint(checkpoint_path, agent, policy, resume=False):
    device = next(agent.parameters()).device
    checkpoint = t.load(checkpoint_path, map_location=device)
    
    agent.load_state_dict(checkpoint['model_state_dict'])
    
    if resume:
        policy.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        if checkpoint.get('scheduler_state_dict') and policy.scheduler:
            policy.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    
    start_step = checkpoint.get('step', 0)
    tracking = checkpoint.get('tracking') if resume else None
    
    if resume:
        config_dict = checkpoint.get('config_dict', {})
        max_entropy = config_dict.get('c2', 0.02)
        total_steps = config_dict.get('num_training_steps', 1_000_000)
        policy.c2 = get_entropy(start_step, total_steps, max_entropy=max_entropy)
    
    return start_step, tracking


def get_torch_compatible_actions(actions, num_actions=14): 
    onehot_actions = t.nn.functional.one_hot(actions, num_classes=num_actions).float()
    return onehot_actions


def get_entropy(step, total_steps, max_entropy=0.02, min_entropy=0.005):
    progress = step / total_steps
    current_entropy = max_entropy - (max_entropy - min_entropy) * progress
    return current_entropy
