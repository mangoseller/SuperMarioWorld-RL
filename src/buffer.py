import torch as t
import numpy as np
from einops import rearrange

class RolloutBuffer:
    # Store rollout data as numpy arrays on CPU to save GPU memory 
    # Could store all of these as uint8 if optimization is needed
    def __init__(self, capacity, num_envs, device):
        self.states = np.zeros((capacity, num_envs, 4, 84, 84), dtype=np.float32) # Environment at time t 
        self.rewards = np.zeros((capacity, num_envs), dtype=np.float32)
        self.actions = np.zeros((capacity, num_envs), dtype=np.int64)
        self.log_probs = np.zeros((capacity, num_envs), dtype=np.float32) # Probabilities of actions that were taken
        self.values = np.zeros((capacity, num_envs), dtype=np.float32) # Model estimations of the value of observed states
        self.dones = np.zeros((capacity, num_envs), dtype=np.float32)
        self.idx = 0
        self.capacity = capacity
        self.device = device 

    def store(self, state, reward, action, log_prob, value, done):
        if self.idx >= self.capacity:
            raise ValueError("Buffer is full!")
        self.states[self.idx] = state.cpu().numpy() if isinstance(state, t.Tensor) else state
        self.rewards[self.idx] = reward 
        self.actions[self.idx] = action
        self.log_probs[self.idx] = log_prob
        self.values[self.idx] = value
        self.dones[self.idx] = done
        self.idx += 1

    def get(self):
        # If the buffer is empty, raise an Exception
        if self.idx == 0:
            raise ValueError("Buffer is empty!")
        
        # Flatten buffers into 1d tensor for use on the GPU
        prep_tensor = lambda buf: t.from_numpy(rearrange(buf[:self.idx], 'n ... -> (n ...)')).to(self.device) 

        reward_tensor = prep_tensor(self.rewards)
        action_tensor = prep_tensor(self.actions)
        log_prob_tensor = prep_tensor(self.log_probs) 
        value_tensor = prep_tensor(self.values) 
        done_tensor = prep_tensor(self.dones)

        """State tensor needs a different shape - batch time and num_envs together - 
        Mixing states from different environments together and permuting the order in ppo.py gives us a 
        a more accurate estimate of the policy gradient. Temporally ordered data would not be iid and so would give a
        higher variance/less accurate estimate of this gradient. """
        state_tensor = t.from_numpy(rearrange(self.states[:self.idx], 't n c h w -> (t n) c h w'))
        return state_tensor, reward_tensor, action_tensor, log_prob_tensor, value_tensor, done_tensor

    def clear(self):
        self.idx = 0 # Will overwrite previous data
 
    def __len__(self): # How much of the buffer is filled? 
        return self.idx

    

