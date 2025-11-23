import torch as t
from datetime import datetime
import math


def get_torch_compatible_actions(actions, num_actions=14): 
    # Convert integer actions into one-hot format for torchrl
    onehot_actions = t.nn.functional.one_hot(actions, num_classes=num_actions).float()
    return onehot_actions


def readable_timestamp():
    return datetime.now().strftime("%Y-%m-%d_%H-%M-%S")


def get_temp(step, total_steps):
# Temperature schedule for action sampling during training
    start_threshold = int(0.1 * total_steps)
    end_threshold = int(0.5 * total_steps)
    if step < start_threshold:
        return 1.0
    elif step < end_threshold:
        progress = (step - start_threshold) / (end_threshold - start_threshold)
        return 1.0 - (0.5 * progress)
    else:
        return 0.5


def get_entropy(step, total_steps, num_cycles=3, max_entropy=0.1, min_entropy=0.001, decay_power=4.0):

    """
    Cyclical entropy schedule, intended to mimic LR warm restarts.
    Spikes entropy high to break local optima, then decays aggressively
    """

    cycle_length = total_steps / num_cycles
    cycle_progress = (step % cycle_length) / cycle_length
    
    decay_factor = math.pow((1 - cycle_progress), decay_power)
    
    current_entropy = min_entropy + (max_entropy - min_entropy) * decay_factor
    
    return current_entropy
