import torch as t 
from torch.distributions import Categorical
import numpy as np
from utils import compute_pixel_change_targets

class PPO:
    def __init__(self, model, config, device):
        self.model = model
        self.config = config
        self.device = device
        self.optimizer = t.optim.Adam(model.parameters(), lr=config.learning_rate, eps=1e-5)
        self.eps = config.clip_eps
        self.c1 = config.c1
        self.c2 = config.c2    
        self.initial_lr = config.learning_rate
        lr_schedule = getattr(config, 'lr_schedule', 'linear').lower() 

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
    @property
    def _has_pixel_control(self):
        return hasattr(self.model, 'pixel_control_head')

    @t.inference_mode()
    def action_selection(self, states):

        states = states.to(self.device)
        if states.dim() == 3:
            states = states.unsqueeze(0)
        state_logits, value = self.model(states)
        distributions = Categorical(logits=state_logits)
        actions = distributions.sample()
        log_probs = distributions.log_prob(actions)

        return actions, log_probs, value.squeeze(-1)
    
    def compute_loss(self, states, actions, old_log_probs, advantages, returns, pixel_targets=None, pixel_loss_weight=0.1):
        self.model.train()

        states = states.to(self.device)
        actions = actions.to(self.device)
        old_log_probs = old_log_probs.to(self.device)
        advantages = advantages.to(self.device)
        returns = returns.to(self.device)
        
        if pixel_targets is not None:
            logits, values, pixel_pred = self.model(states, return_pixel_control=True)
            pixel_targets = pixel_targets.to(self.device)
        else:
            logits, values = self.model(states)

        distributions = Categorical(logits=logits)
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

        # Pixel control loss (Auxiliary Task)
        if pixel_targets is not None:
            # MSE Loss: Targets are positive (abs diff), PixelHead uses Softplus (positive)
            pixel_control_loss = t.nn.functional.mse_loss(pixel_pred, pixel_targets)
        else:
            pixel_control_loss = t.tensor(0.0, device=self.device)
        
        total_loss = (policy_loss + 
                    (self.c1 * value_loss) + 
                    (self.c2 * -entropy_loss) +
                    (pixel_loss_weight * pixel_control_loss))

        diagnostics = {
            'policy_loss': policy_loss.item(),
            'value_loss': value_loss.item(),
            'entropy': entropy_loss.item(),
            'pixel_control_loss': pixel_control_loss.item()
        }
        
        return total_loss, diagnostics

    def compute_advantages(self, buffer, next_state=None):
        # Compute GAE advantages and returns
        
        gamma = self.config.gamma
        lambda_ = self.config.lambda_gae
        _, rewards, _, _, values, dones = buffer.get()
        
        values = values.detach()       
        # Reshape to (Batch, Num_Envs) for GAE calculation
        reshape = lambda tensor: tensor.view(len(buffer), buffer.rewards.shape[1])
        
        rewards = reshape(rewards)
        values = reshape(values)
        dones = reshape(dones)
        advantages = t.zeros_like(rewards)

        if next_state is not None:
                out = self.model(next_state.to(self.device))
                last_values = out[1] if isinstance(out, tuple) else out
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

        # Flatten back to match buffer storage
        advantages = advantages.view(-1)
        values = values.view(-1)
        returns = advantages + values

        return advantages, returns

    def update(self, buffer, config, eps=1e-8, next_state=None):
 
            self.model.train()
            minibatch_size = config.minibatch_size
            num_epochs = config.epochs
            
            # Get flattened buffer data
            states, _, actions, log_probs, _, _ = buffer.get()
            
            advantages, returns = self.compute_advantages(buffer, next_state=next_state)
            normalized_advantages = (advantages - advantages.mean()) / (advantages.std() + eps)
            

            
            pixel_targets = None
            
            if self._has_pixel_control: # TODO: Refactor

                # Reconstruct (Time, Env) structure to compute temporal differences
                buffer_len = buffer.capacity
                num_envs = buffer.rewards.shape[1]
                
                # (Time * Env, ...) -> (Time, Env, ...)
                states_reshaped = states.view(buffer_len, num_envs, *states.shape[1:])
                
                t_dim, e_dim, c, h, w = states_reshaped.shape
                
                # (Time, Env, C, H, W)
                current = states_reshaped[:-1]
                next_obs = states_reshaped[1:]
            
                # Calculate diffs for all envs and timesteps in one go
                # diff: (Time-1, Env, H, W)
                diff = t.abs(next_obs - current).mean(dim=2) 
                
                # Reshape for pooling: ((Time-1)*Env, 1, H, W)
                diff_flat = diff.reshape(-1, 1, h, w)
                
                # Pool: ((Time-1)*Env, 1, 7, 7)
                targets_flat = t.nn.functional.avg_pool2d(
                    diff_flat, kernel_size=12, stride=12
                )
                
                # Reshape back: (Time-1, Env, 7, 7)
                pixel_targets = targets_flat.reshape(t_dim-1, e_dim, 7, 7)
                
                # Flatten to match buffer structure (t0e0, t0e1, ... t1e0...)
                pixel_targets = pixel_targets.reshape(-1, 7, 7)
                    
                # We must trim the training data to match pixel targets length (Time - 1)
                # Since buffer is flattened Time-Major (t0e0, t0e1... t1e0, t1e1...), 
                # removing the last `num_envs` items removes the last timestep T.

                states = states[:-num_envs]
                actions = actions[:-num_envs]
                log_probs = log_probs[:-num_envs]
                normalized_advantages = normalized_advantages[:-num_envs]
                returns = returns[:-num_envs]
                            
            # Mini-batch Updates
            for _ in range(1, num_epochs+1):

                indices = t.randperm(states.shape[0])    
                states_shuffled = states[indices]
                actions_shuffled = actions[indices]
                log_probs_shuffled = log_probs[indices]
                advantages_shuffled = normalized_advantages[indices]
                returns_shuffled = returns[indices]
                
                pixel_targets_shuffled = None
                if self._has_pixel_control:
                    pixel_targets_shuffled = pixel_targets[indices]

                for start_idx in range(0, states.shape[0], minibatch_size):
                    end_idx = min(start_idx + minibatch_size, states.shape[0])
                    
                    mb_states = states_shuffled[start_idx:end_idx]
                    mb_actions = actions_shuffled[start_idx:end_idx]
                    mb_log_probs = log_probs_shuffled[start_idx:end_idx]
                    mb_advantages = advantages_shuffled[start_idx:end_idx]
                    mb_returns = returns_shuffled[start_idx:end_idx]
                    
                    mb_pixel_targets = None
                    if self._has_pixel_control:
                        mb_pixel_targets = pixel_targets_shuffled[start_idx:end_idx]

 
                    with t.amp.autocast(device_type='cuda', dtype=t.bfloat16):
                        loss, diagnostics = self.compute_loss(
                        mb_states, mb_actions, mb_log_probs, mb_advantages, mb_returns,
                        pixel_targets=mb_pixel_targets
                    )
                    total_losses = []
                    total_losses.append(loss.item())

                    self.optimizer.zero_grad()
                    loss.backward()
                    t.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=0.5) # Is this using the correct param? 
                    self.optimizer.step()
                    
            if self.scheduler is not None:
                self.scheduler.step()
            
            averaged_diagnostics = {np.mean(total_losses)}     
            return averaged_diagnostics
    
    def get_current_lr(self):
        return self.optimizer.param_groups[0]['lr']
