import torch as t
import numpy as np
from einops import rearrange

class RolloutBuffer:
    def __init__(self, capacity, num_envs, device):
        self.device = device
        self.capacity = capacity
        self.idx = 0

        self.states = t.zeros((capacity, num_envs, 4, 84, 84), dtype=t.uint8, device=device)
        self.actions = t.zeros((capacity, num_envs), dtype=t.int64, device=device)
        self.rewards = t.zeros((capacity, num_envs), dtype=t.float32, device=device)
        self.log_probs = t.zeros((capacity, num_envs), dtype=t.float32, device=device)
        self.values = t.zeros((capacity, num_envs), dtype=t.float32, device=device)
        self.dones = t.zeros((capacity, num_envs), dtype=t.float32, device=device)

    def store(self, state, reward, action, log_prob, value, done):
        if self.idx >= self.capacity:
            raise ValueError("Buffer is full!")

        if state.dtype == t.float32: # Handle state compression (float 0-1 -> uint8 0-255)
            state = (state * 255).to(t.uint8)
            
        self.states[self.idx] = state.to(self.device, non_blocking=True)
        self.actions[self.idx] = t.as_tensor(action, device=self.device)
        self.rewards[self.idx] = t.as_tensor(reward, device=self.device)
        self.log_probs[self.idx] = t.as_tensor(log_prob, device=self.device)
        self.values[self.idx] = t.as_tensor(value, device=self.device)
        self.dones[self.idx] = t.as_tensor(done, device=self.device)

    def get(self):
            if self.idx == 0:
                raise ValueError("Buffer is empty!")
                
            # Flatten: (Time, Env, ...) -> (Time * Env, ...) 
            # Decompress state: uint8 0-255 -> float 0-1
            state_tensor = rearrange(self.states[:self.idx], 't n c h w -> (t n) c h w').float() / 255.0
            flatten = lambda collection: rearrange(collection, 't n -> (t n)')
            others = tuple(map(flatten, [
                    self.rewards[:self.idx],
                    self.actions[:self.idx],
                    self.log_probs[:self.idx],
                    self.values[:self.idx],
                    self.dones[:self.idx]
                ]))

            return (state_tensor,) + others


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

    

