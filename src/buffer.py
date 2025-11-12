import torch as t
import numpy as np

class RolloutBuffer:
    def __init__(self, capacity, device):
        self.states = np.zeros((capacity, 4, 84, 84), dtype=np.float32)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.actions = np.zeros(capacity, dtype=np.int64)
        self.log_probs = np.zeros(capacity, dtype=np.float32)
        self.values = np.zeros(capacity, dtype=np.float32)
        self.dones = np.zeros(capacity, dtype=np.float32)
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
        # If the tensors are empty, raise an Exception
        if self.idx == 0:
            raise ValueError("Buffer is empty!")
        state_tensor = t.from_numpy(self.states[:self.idx]).to(self.device)
        reward_tensor = t.from_numpy(self.rewards[:self.idx]).to(self.device)
        action_tensor = t.from_numpy(self.actions[:self.idx]).to(self.device)
        log_prob_tensor = t.from_numpy(self.log_probs[:self.idx]).to(self.device)
        value_tensor = t.from_numpy(self.values[:self.idx]).to(self.device) 
        done_tensor = t.from_numpy(self.dones[:self.idx]).to(self.device) 
        return state_tensor, reward_tensor, action_tensor, log_prob_tensor, value_tensor, done_tensor

    def clear(self):
        self.idx = 0 # Will overwrite previous data
 
    def __len__(self):
        return self.idx

    

