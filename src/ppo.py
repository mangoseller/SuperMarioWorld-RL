import torch as t 
from torch.distributions import Categorical
import numpy as np

class PPO: # TODO: Implement lots of rollout at once, multiple envs
    def __init__(self, model, lr, epsilon, optimizer, device, c1=0.5, c2 =0.01):
        self.model = model
        self.optimizer = optimizer(model.parameters(), lr=lr)
        self.eps = epsilon
        self.device = device
        # Scaling terms for loss computation
        self.c1 = c1 
        self.c2 = c2
    def action_selection(self, state):

        with t.inference_mode():
            state = state.to(self.device)
            state_logits, value = self.model(state.unsqueeze(0)) # add batch dim, shape (1, 4, 84, 84)
        distributions = Categorical(state_logits)
        action = distributions.sample()
        action_prob = distributions.log_prob(action)
        return action.item(), action_prob.item(), value.squeeze().item() # Return scalars

    def compute_loss(self, states, actions, old_log_probs, advantages, returns):
        
        # Move params to correct device
        states = states.to(self.device)
        actions = actions.to(self.device)
        old_log_probs = old_log_probs.to(self.device)
        advantages = advantages.to(self.device)
        returns = returns.to(self.device)

        logits, values = self.model(states)
        # logits - (batch_size, num_actions), values (batch_size, 1)

        distributions = Categorical(logits=logits)
        new_log_probs = distributions.log_prob(actions) # shape (batch_size,)

        old_log_probs = old_log_probs.detach() # Detach gradients
        # Now compute the probability ratio
        ratio = t.exp(new_log_probs - old_log_probs)
        unclipped_ratio = ratio * advantages # Unclipped surrogate objective

        # ratio > 1 -> old actions more likely under new policy
        # ratio < 1 -> new actions less likely
        # ratio = 1 -> same likelihood

        # Clip probability ratio to desired range
        clipped_ratio = t.clamp(ratio, min=1-self.eps, max=1+self.eps)
        clipped_ratio = clipped_ratio * advantages # Clipped surrogate objective

        # Now take the minimum of the two ratios to get pessimistic estimate of the objective

        min_ratio = t.minimum(clipped_ratio, unclipped_ratio)
        policy_loss = -t.mean(min_ratio) 
        
        # MSE loss on value-head predictions
        value_loss = t.mean((values.squeeze() - returns)**2)
        return policy_loss + (self.c1 * value_loss) + (self.c2 * -distributions.entropy().mean()) # Total loss



        
     




