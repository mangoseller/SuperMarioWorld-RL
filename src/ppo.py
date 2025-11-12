import torch as t 
from torch.distributions import Categorical

class PPO: # TODO: Implement lots of rollout at once, multiple envs
    def __init__(self, model, lr, epsilon, optimizer, device, c1=0.5, c2=0.01):
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

    def compute_advantages(self, buffer, gamma=0.99, lambda_=0.95, next_state=None):
        _, rewards, _, _, values, dones =  buffer.get()
        assert all(i.shape == (len(buffer), ) for i in [rewards, values, dones]), "Tensors are of unexpected shape!"
        values = values.detach()        

        # At timestep t we need V(s_{t+1})
        advantages = t.zeros_like(rewards)
        gae = 0

        # Get the value of the next state to find the value of the most recent, (first) state
        if next_state is not None:
            with t.no_grad():
                _, last_value = self.model(next_state.unsqueeze(0).to(self.device))
                last_value = last_value.squeeze().item()
        else:
            last_value = 0

        for i in range(len(buffer)-1, -1, -1):
            if i == len(buffer) - 1:
                next_value = last_value * (1-dones[i])
            else:
                next_value = values[i+1]

            # Compute TD Error
            delta = rewards[i] + gamma * next_value * (1-dones[i]) - values[i]
            
            # Accumulate gae
            gae = delta + gamma * lambda_ * (1 - dones[i]) * gae
            advantages[i] = gae

        returns = advantages + values
        return advantages, returns


    def update(self, buffer, num_epochs=5, minibatch_size=64, eps=1e-8, next_state=None):
        self.model.train()
        advantages, returns = self.compute_advantages(buffer, next_state=next_state)
        normalized_advantages = (advantages - advantages.mean()) / (advantages.std() + eps)
        states, _, actions, log_probs, _, _ = buffer.get()
        for _ in range(1, num_epochs+1):
            permuted_indices = t.randperm(len(buffer))
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

                self.optimizer.zero_grad()
                loss.backward()
                # Clip grad norms to prevent exploding gradients
                t.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()

        return


        
        
     




