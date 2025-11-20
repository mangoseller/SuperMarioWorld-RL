import torch as t 
from torch.distributions import Categorical
import numpy as np

class PPO:
    def __init__(self, model, lr, epsilon, optimizer, device, c1=0.5, c2=0.01, use_lr_scheduling=True, max_updates=None):
        self.model = model
        self.optimizer = optimizer(model.parameters(), lr=lr)
        self.eps = epsilon
        self.device = device
        # Scaling terms for loss computation
        self.c1 = c1 
        self.c2 = c2

        self.use_lr_scheduling = use_lr_scheduling
        self.initial_lr = lr
        if use_lr_scheduling and max_updates:
            # Linear decay to 0 over training
            self.scheduler = t.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=max_updates,
                eta_min=0.
            )
        else:
            self.scheduler = None

    def action_selection(self, states):
        with t.inference_mode():
            states = states.to(self.device)
            # states has shape (num_envs, 4, 84, 84) or (4, 84, 84) if num_envs == 1
            if states.dim() == 3: # If num_envs == 1, add a batch dim
                states = states.unsqueeze(0)

            state_logits, value = self.model(states) # add batch dim, shape (1, 4, 84, 84)
        
        distributions = Categorical(logits=state_logits)
        actions = distributions.sample()
        action_probs = distributions.log_prob(actions)
        return actions, action_probs, value.squeeze(-1)

    def eval_action_selection(self, state):
        with t.inference_mode():
            state = state.to(self.device)
            logits, _ = self.model(state.unsqueeze(0))

        action = t.argmax(logits, dim=-1)
        return action.item()

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

    def compute_advantages(self, buffer, gamma=0.99, lambda_=0.95, next_state=None):
        _, rewards, _, _, values, dones =  buffer.get()
        assert all(i.shape == (rewards.shape[0],) for i in [rewards, values, dones]), "Tensors are of unexpected shape!"
        values = values.detach()       

        # Reshape from flattened(steps * num_envs,) to (steps, num_envs)
        reshape = lambda tensor: tensor.view(len(buffer), buffer.rewards.shape[1])
        rewards = reshape(rewards)
        values = reshape(values)
        dones = reshape(dones)

        # At timestep t we need V(s_{t+1})
        advantages = t.zeros_like(rewards)
       

        # Get the value of the next state to find the value of the most recent, (first) state
        if next_state is not None:
            with t.no_grad():
                _, last_values = self.model(next_state.to(self.device))
                last_values = last_values.squeeze(-1)
        else:
            last_values = t.zeros(buffer.rewards.shape[1], device=self.device)

        gaes = t.zeros(buffer.rewards.shape[1], device=self.device)

        for i in range(len(buffer)-1, -1, -1):
            if i == len(buffer) - 1:
                next_value = last_values * (1-dones[i])
            else:
                next_value = values[i+1]

            # Compute TD Error
            delta = rewards[i] + gamma * next_value * (1-dones[i]) - values[i]
            
            # Accumulate gae
            gaes = delta + gamma * lambda_ * (1 - dones[i]) * gaes
            advantages[i] = gaes

        # Flatten for the update
        advantages = advantages.view(-1)
        values = values.view(-1)
        returns = advantages + values
        return advantages, returns


    def update(self, buffer, num_epochs=5, minibatch_size=64, eps=1e-8, next_state=None):
        self.model.train()
        advantages, returns = self.compute_advantages(buffer, next_state=next_state)
        normalized_advantages = (advantages - advantages.mean()) / (advantages.std() + eps)
        states, _, actions, log_probs, _, _ = buffer.get()
        total_losses = []
        for _ in range(1, num_epochs+1):
            permuted_indices = t.randperm(states.shape[0]) 
            states = states[permuted_indices]
            actions = actions[permuted_indices]
            log_probs = log_probs[permuted_indices]
            advantages = normalized_advantages[permuted_indices]
            returns = returns[permuted_indices]

            for start_idx in range(0, len(buffer), minibatch_size):
                end_idx = min(start_idx + minibatch_size, len(buffer))
                mb_states = states[start_idx:end_idx]
                mb_actions = actions[start_idx:end_idx]
                mb_log_probs = log_probs[start_idx:end_idx]
                mb_advantages = advantages[start_idx:end_idx]
                mb_returns = returns[start_idx:end_idx]

            # Compute loss over this batch
                loss = self.compute_loss(mb_states, mb_actions, mb_log_probs, 
                                         mb_advantages, mb_returns)
                total_losses.append(loss)

                self.optimizer.zero_grad()
                loss.backward()
                # Clip grad norms to prevent exploding gradients
                t.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()
        if self.scheduler is not None:
            self.scheduler.step()
        
        return np.mean([loss.item() for loss in total_losses])

    def get_current_lr(self):
        return self.optimizer.param_groups[0]['lr']


        
        
     




