import torch as t 
from torch.distributions import Categorical
import numpy as np


class PPO:
    def __init__(self, model, config, device):
        self.model = model
        self.config = config
        self.device = device
        self.optimizer = t.optim.Adam(model.parameters(), lr=config.learning_rate)
        self.eps = config.clip_eps
        self.c1 = config.c1
        self.c2 = config.c2    
        self.initial_lr = config.learning_rate
        lr_schedule = getattr(config, 'lr_schedule', 'cosine').lower() 

        if lr_schedule == 'cosine':
            self.scheduler = t.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=config.num_training_steps,
                eta_min=config.min_lr,
            )
        elif lr_schedule == 'linear':
            self.scheduler = t.optim.lr_scheduler.LinearLR(
                self.optimizer,
                start_factor=1.0,
                end_factor=config.min_lr / config.learning_rate,
                total_iters=config.num_training_steps,
            )
        elif lr_schedule == 'constant':
            self.scheduler = None
    
    def action_selection(self, states, temp):
       # Select actions during training (stochastic)
        with t.inference_mode():
            states = states.to(self.device)
            if states.dim() == 3:
                states = states.unsqueeze(0)

            state_logits, value = self.model(states)
        
        scaled_logits = state_logits / temp
        distributions = Categorical(logits=scaled_logits)
        actions = distributions.sample()
        log_probs = distributions.log_prob(actions)
       
        return actions, log_probs, value.squeeze(-1)

    def eval_action_selection(self, state, temp):
      # Select actions during evals - should ideally be entirely deterministic 
        with t.inference_mode():
            state = state.to(self.device)
            logits, _ = self.model(state.unsqueeze(0))

        if temp == 0:
            return t.argmax(logits, dim=-1).item()

        scaled_logits = logits / temp
        distribution = Categorical(logits=scaled_logits)
        action = distribution.sample()

        return action.item()
    
    def _compute_diagnostics(self, ratio, values, returns):
        # Compute diagnostic metrics for logging
        with t.no_grad():
            clip_fraction = ((ratio < 1 - self.eps) | (ratio > 1 + self.eps)).float().mean()
            approx_kl = ((ratio - 1) - t.log(ratio)).mean()
            
            y_pred = values.squeeze()
            y_true = returns
            var_y = t.var(y_true)
            explained_var = 1 - t.var(y_true - y_pred) / (var_y + 1e-8)
            
        return {
            'clip_fraction': clip_fraction.item(),
            'approx_kl': approx_kl.item(),
            'explained_variance': explained_var.item(),
        }
    
    def compute_loss(self, states, actions, old_log_probs, advantages, returns, temp):
        # Compute PPO loss
        states = states.to(self.device)
        actions = actions.to(self.device)
        old_log_probs = old_log_probs.to(self.device)
        advantages = advantages.to(self.device)
        returns = returns.to(self.device)

        logits, values = self.model(states)
        scaled_logits = logits / temp

        distributions = Categorical(logits=scaled_logits)
        new_log_probs = distributions.log_prob(actions)

        old_log_probs = old_log_probs.detach()
        ratio = t.exp(new_log_probs - old_log_probs)
        unclipped_ratio = ratio * advantages

        clipped_ratio = t.clamp(ratio, min=1-self.eps, max=1+self.eps)
        clipped_ratio = clipped_ratio * advantages

        min_ratio = t.minimum(clipped_ratio, unclipped_ratio)
        policy_loss = -t.mean(min_ratio) 
        entropy_loss = distributions.entropy().mean()
        value_loss = t.mean((values.squeeze() - returns)**2)
        total_loss = policy_loss + (self.c1 * value_loss) + (self.c2 * -entropy_loss)

        diagnostics = self._compute_diagnostics(ratio, values, returns)
        diagnostics.update({
            'policy_loss': policy_loss.item(),
            'value_loss': value_loss.item(),
            'entropy': entropy_loss.item(),
        })
        
        return total_loss, diagnostics

    def compute_advantages(self, buffer, gamma=0.99, lambda_=0.95, next_state=None): #TODO: Comment this a lot better
        # Compute GAE advantages and returns

        _, rewards, _, _, values, dones = buffer.get()
        assert all(i.shape == (rewards.shape[0],) for i in [rewards, values, dones]), "Tensors are of unexpected shape!"
        values = values.detach()       

        reshape = lambda tensor: tensor.view(len(buffer), buffer.rewards.shape[1])
        rewards = reshape(rewards)
        values = reshape(values)
        dones = reshape(dones)

        advantages = t.zeros_like(rewards)

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

            delta = rewards[i] + gamma * next_value * (1-dones[i]) - values[i]
            gaes = delta + gamma * lambda_ * (1 - dones[i]) * gaes
            advantages[i] = gaes

        advantages = advantages.view(-1)
        values = values.view(-1)
        returns = advantages + values
        return advantages, returns

    def update(self, buffer, temp, num_epochs=5, minibatch_size=64, eps=1e-8, next_state=None):
        # Perform a PPO update 
        self.model.train()
        advantages, returns = self.compute_advantages(buffer, next_state=next_state)
        normalized_advantages = (advantages - advantages.mean()) / (advantages.std() + eps)
        states, _, actions, log_probs, _, _ = buffer.get()
        
        total_losses = []
        all_diagnostics = {
            'policy_loss': [],
            'value_loss': [],
            'entropy': [],
            'clip_fraction': [],
            'approx_kl': [],
            'explained_variance': [],
        }
        
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

                loss, diagnostics = self.compute_loss(
                    mb_states, mb_actions, mb_log_probs, mb_advantages, mb_returns, temp=temp
                )
                total_losses.append(loss.item())
                
                for key, value in diagnostics.items():
                    all_diagnostics[key].append(value)

                self.optimizer.zero_grad()
                loss.backward()
                t.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()
                
        if self.scheduler is not None:
            self.scheduler.step()
        
        averaged_diagnostics = {
            key: np.mean(values) for key, values in all_diagnostics.items()
        }
        averaged_diagnostics['total_loss'] = np.mean(total_losses)
        
        return averaged_diagnostics

    def get_current_lr(self):
        return self.optimizer.param_groups[0]['lr']
