import multiprocessing as mp
import numpy as np
import torch as t

from muzero.mcts import MuZeroModelAdapter
from muzero.network import serialize_worker_weights
from muzero.reanalyse import ReanalyseNetwork, make_batch_search
from muzero.self_play import BehaviourProfile, collect_episode, make_mario_env


_EVAL_PROFILE = BehaviourProfile(
    name="eval",
    dirichlet_alpha=0.0,
    dirichlet_weight=0.0,
    temperature=0.0,
    epsilon=0.0,
)


def _resolve_device(device):
    device = t.device(device)
    if device.type == "cuda" and not t.cuda.is_available():
        return t.device("cpu")
    return device


def _eval_worker(weights, config, levels, episodes_per_level, result_queue):
    device = _resolve_device(config.self_play_device)
    network = ReanalyseNetwork(config, device)
    network.load_weights(weights)
    adapter = MuZeroModelAdapter(network, config)
    search = make_batch_search(config)
    stats = {}

    for level in levels:
        env = make_mario_env(level, config)
        returns, x_maxes = [], []
        for _ in range(episodes_per_level):
            traj = collect_episode(
                env, adapter, search, config,
                profile=_EVAL_PROFILE,
                level=level,
                device=str(device),
                num_simulations=config.eval_mcts_sims,
            )
            returns.append(float(traj.episode_return))
            x_maxes.append(float(traj.x_max))
        env.close()
        stats[level] = {
            "mean_return": float(np.mean(returns)),
            "std_return": float(np.std(returns)) if len(returns) > 1 else 0.0,
            "mean_x_max": float(np.mean(x_maxes)),
            "max_x_max": float(max(x_maxes)),
        }

    result_queue.put({"type": "eval_done", "stats": stats})


def run_muzero_eval(network, config, levels, timeout=600.0):
    weights = serialize_worker_weights(network.weights_for_workers())
    result_queue = mp.Queue()
    p = mp.Process(
        target=_eval_worker,
        args=(weights, config, list(levels), config.eval_episodes_per_level, result_queue),
    )
    p.start()
    try:
        result = result_queue.get(timeout=timeout)
        return result.get("stats", {})
    except Exception:
        return {}
    finally:
        p.join(timeout=10.0)
        if p.is_alive():
            p.terminate()
            p.join()
