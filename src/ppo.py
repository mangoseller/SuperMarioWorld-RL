from einops import rearrange
import torch as t 
from torch.distributions import Categorical
import numpy as np

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
        
        if pixel_targets is not None: # Pixel prediction auxiliary task
            logits, values, pixel_pred = self.model(states, return_pixel_control=True)
            pixel_targets = pixel_targets.to(self.device)
        else:
            logits, values = self.model(states)
     
        distributions = Categorical(logits=logits)
        new_log_probs = distributions.log_prob(actions) # How likely is the action we took before under the new policy?
        old_log_probs = old_log_probs.detach()
        # The point of this ratio is that we can learn from stale data, after the first update gradient update
        ratio = t.exp(new_log_probs - old_log_probs)
        unclipped_ratio = ratio * advantages

        # ratio > 1 -> old actions more likely under new policy
        # ratio < 1 -> new actions less likely
        # ratio = 1 -> same likelihood
        
        # Limit the size of any particular gradient update
        clipped_ratio = t.clamp(ratio, min=1-self.eps, max=1+self.eps)
        clipped_ratio = clipped_ratio * advantages 
        
        # Take the minimum of the clipped and unclipped to get a pessimistic estimate of the objective function
        min_ratio = t.minimum(clipped_ratio, unclipped_ratio)
        policy_loss = -t.mean(min_ratio)  # Clipped surrogate objective, we minimise negative mean returns (i.e. maximise returns)
        entropy_loss = distributions.entropy().mean() # Used to punish deterministic actions during training
        value_loss = t.mean((values.squeeze() - returns)**2) # MSE

        # Pixel control loss (Auxiliary Task)
        if pixel_targets is not None:
            # MSE Loss: Targets are positive (abs diff), PixelHead uses Softplus (positive)
            pixel_control_loss = t.nn.functional.mse_loss(pixel_pred, pixel_targets)
        else:
            pixel_control_loss = t.tensor(0.0, device=self.device)

        # C1 controls how much the shared weights 'care' about the value head, vs the action head, c2 controls how deterministic the model is
        total_loss = (
                     policy_loss + 
                     (self.c1 * value_loss) + 
                     (self.c2 * -entropy_loss) +
                     (pixel_loss_weight * pixel_control_loss)
            )

        diagnostics = {
            'policy_loss': policy_loss.item(),
            'value_loss': value_loss.item(),
            'entropy': entropy_loss.item(),
            'pixel_control_loss': pixel_control_loss.item(),
            'total_loss': total_loss.item(),
        }
        
        return total_loss, diagnostics

    def compute_advantages(self, buffer, next_state=None):
        # Compute Generalized Advantage Returns
        
        gamma = self.config.gamma # Future reward discounting factor

        """Lambda = GAE horizon. λ≈0 trust value head (high bias, low variance) 
        λ≈1 trust true observed rewards (low bias, high variance)
        λ=0.95 uses real rewards for ~20 steps then uses the value function"""

        lambda_ = self.config.lambda_gae 

        _, rewards, _, _, values, dones = buffer.get()
        
        values = values.detach()      

        # Reshape to (Batch, Num_Envs) for GAE calculations
        reshape = lambda tensor: tensor.view(len(buffer), buffer.rewards.shape[1])
        
        rewards = reshape(rewards)
        values = reshape(values)
        dones = reshape(dones)
        advantages = t.zeros_like(rewards)

        if next_state is not None: 
            # Handle the last state in our buffer - GAE calculation requires state t+1 to compute advantage t
            # If the episode didn't end upon the last state in the buffer, pass the following state. Otherwise, set the next state to be 0 reward
                out = self.model(next_state.to(self.device))
                last_values = out[1] if isinstance(out, tuple) else out
                last_values = last_values.squeeze(-1).detach()
        else:
            last_values = t.zeros(buffer.rewards.shape[1], device=self.device)

        gaes = t.zeros(buffer.rewards.shape[1], device=self.device)
        """Computing the GAE at t requires knowledge of the GAE at t+1 - hence similar to backprop, we iterate through the buffer backwards in time, using the advantage
        at t+1 to compute the advantage at t, and so on. If an episode terminated at step i, dones[i] = 1. Hence, next_value = last_values * 1-(dones[i]) = next_value if the episode
        didn't terminate (has a next value), or 0 if it did terminate, as we require. This is also why we have next_value * (1-dones[i]) in the delta computation, if the episode ended, 
        the value of the next state needs to be set to 0 for PPO to work correctly."""

        for i in range(len(buffer)-1, -1, -1):
            if i == len(buffer) - 1:
                next_value = last_values * (1-dones[i])
            else:
                next_value = values[i+1]

            delta = rewards[i] + gamma * next_value * (1-dones[i]) - values[i] # How good is this state compared to the value functions estimate?
            gaes = delta + gamma * lambda_ * (1 - dones[i]) * gaes # The GAE the exponential moving average of deltas (TD Errors)
            advantages[i] = gaes

        # Flatten back to match buffer storage, break temporal ordering again
        advantages = advantages.view(-1)
        values = values.view(-1)

        """Real returns of the states, given the actual trajectory experienced.
        i.e. for states k.....t, returns[k] holds the sum of the predicted value of the state and the true advantage observed for that state - which together form an 
        estimate for the true return for state k. This is a 'bootstrapped return' our best estimate of the true return
        of a state given our predictions and true observations"""
        returns = advantages + values 

        return advantages, returns

    def compute_pixel_change_targets(self, states, buffer, dones, next_state=None): 
        """Compute ground_truth for pixel changes, for each state s_t, we predict
        |s_{t+1} - s_t|, i.e. changes in pixels, for a downsampled 7x7 grid. Aims to help
        the model learn beneficial spacial representations, in other words, what to pay attention to"""
        
        buffer_len = len(buffer)
        num_envs = buffer.rewards.shape[1]

        # Prepare dones to mask out pixel transitions between episodes
        dones = rearrange(
            dones, 
            '(time env) -> time env',
            time=buffer_len,
            env=num_envs
        )
        expanded_dones = dones[:, :, None, None] # (time, env, 1, 1)

        # Unpack data into seperate environments again        
        time_added_states = rearrange(
            states, 
            '(time env) c h w -> time env c h w',
            time=buffer_len,
            env=num_envs
        )
        time_dim, env_dim, channels, height, width = time_added_states.shape

        # Get consecutive frame pairs
        if next_state is not None: 
            # add a dummy time dimension to the single next_state
            next_state = rearrange(
            next_state.to(self.device),
            'env c h w -> 1 env c h w'
            )

            # Concat along time dim
            full_seq = t.cat([time_added_states, next_state], dim=0)
            current = full_seq[:-1] # Times [0, 1, ..., T-1]
            next_obs = full_seq[1:] # Times [1, 2, ...., T]
       
        else:
            # If we don't have a next state, we need to lose the final state in our buffer
            current = time_added_states[:-1]
            next_obs = time_added_states[1:]
            dones_mask = dones[:-1]  
            
    

        # Compute pixel-wise absolute differences, averaged over the 4 frame stacks 
        diff = t.abs(next_obs - current).mean(dim=2)
        diff = diff * (1 - expanded_dones.float()) # Difference is zero at episode boundaries
        """Merge time and env into a single batch dimension, and add a channel dimension
         We want to treat each (time, env) pair as a seperate sample in the batch
         (time, env, height, width) -> (time*env, 1, height, width), channels = 1, 
         a greyscale difference map"""
        diff_batched = rearrange(diff, 'time env h w -> (time env) 1 h w')

        # Now down sample, since the pixel control head only ouputs predictions for a 7x7 grid
        # 84*84 input, means with kernel_size 12 and stride 12, we end up reducing to a 7x7 grid as required
        targets_pooled = t.nn.functional.avg_pool2d(
            diff_batched, 
            kernel_size=12,
            stride=12
        )

        # Now remove the channel dimension to return shape (time*env, 7, 7) 
        return rearrange(
            targets_pooled,
            '(time env) 1 h w -> (time env) h w',
            time=current.shape[0],
            env=env_dim
        )

    def update(self, buffer, config, next_state=None):
 
            self.model.train()
            minibatch_size = config.minibatch_size
            num_epochs = config.epochs
            
            # Get flattened buffer data
            states, _, actions, log_probs, _, dones = buffer.get()      
            advantages, returns = self.compute_advantages(buffer, next_state=next_state)

            """0 mean computes relative advantages, so that a positive reward action is able to be negatively reinforced by the optimizer,
            e.g. if an action was positive reward but lower in reward than the average for the batch, we want this action to occur less.
            Without this, we may naively reinforce all positive reward actions, however suboptimal.
            Unit variance (dividing by std) helps scale the gradients correctly to avoid extensive lr tuning"""

            normalized_advantages = (advantages - advantages.mean()) / (advantages.std() + self.eps) 
            pixel_targets = None

            if self._has_pixel_control:
                pixel_targets = self.compute_pixel_change_targets(states, buffer, dones, next_state)

            # If next_state wasn't provided, pixel_targets has fewer samples, need to trim the training data to match
            if next_state is None:
                num_envs = buffer.rewards.shape[1]
                states = states[:-num_envs]
                actions = actions[:-num_envs]
                log_probs = log_probs[:-num_envs]
                normalized_advantages = normalized_advantages[:-num_envs]
                returns = returns[:-num_envs]

            all_diagnostics = []       

            # Mini-batch Updates
            for _ in range(1, num_epochs+1):
            # Shuffle the data get i.i.d datapoints

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

                # Handle left over data, e.g. if our buffer size isn't divisible by our minibatch size
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

                    all_diagnostics.append(diagnostics)
                    self.optimizer.zero_grad()
                    loss.backward()
                    t.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=0.5)
                    self.optimizer.step()
                    
            if self.scheduler is not None:
                self.scheduler.step()
            
            # Average diagnostics across all minibatches
            averaged_diagnostics = {
                key: np.mean([d[key] for d in all_diagnostics])
                for key in all_diagnostics[0].keys()
            }
            self.model.eval() # Disable data augmentation for rollout collection again
            return averaged_diagnostics
    
    def get_current_lr(self):
        return self.optimizer.param_groups[0]['lr']
