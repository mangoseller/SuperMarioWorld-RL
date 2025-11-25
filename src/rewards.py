import gymnasium as gym


REWARD_CONFIG = {
    'movement_reward': 0.01,
    'step_penalty': 0.001,
    'damage_penalty': 1,
    'life_loss_penalty': 5.0,
    'max_steps_penalty': 15.0,
    'coin_reward': 0.3,
    'score_reward': 0.001,
    'midway_reward': 20.0,
    'level_complete_reward': 50.0,
}


class MovementReward:
    # Reward forward progress through the level
    def __init__(self):
        self.scale = REWARD_CONFIG['movement_reward']
        self._max_x = 0
        self._global_x = 0
        self._last_x = 0
    
    def reset(self, _info):
        self._max_x = 0
        self._global_x = 0
        self._last_x = 0

    def calculate(self, info):
        """Mario's x-position is a 1byte uint that ranges from 0-255, and resets multiple times over the course of a single level.
        We can work out when Mario's change in x-position as follows:
        1. If the change in Mario's position is less then |128|, we haven't passed a boundary point where Mario's x-position resets, so we simply reward positive change in x-position.
        2. If the change in Mario's position is < -128, we reached a boundary point from progressing through the level, and Mario's x-position reset from 255 back to 0.
        To recover the actual positive change in x-position Mario attained, add 256 to the change in x-position to offset this restart. 
        e.g. If Mario's x-position goes from 255 to 4, we compute: 4 - 255 = -251, then add 256 to get the actual change in x-position: -251 + 256 = 5 - so Mario moved 5 units in the positive x-direction.
        3. Similarly, if the change is > 128, Mario went far enough backward to trigger a reset so we subtract 256 from Mario's change in x-position to recover the true backward movement."""

        raw_x = info.get("xpos", 16) # The starting x-position in YoshisIsland2 is 16
        delta = raw_x - self._last_x

        if delta < -128:
            delta += 256
        elif delta > 128:
            delta -= 256
        
        self._global_x += delta
        self._last_x = raw_x
        
        if self._global_x > self._max_x:

            gain = self._global_x - self._max_x
            self._max_x = self._global_x
            return gain * self.scale

        return 0.0


class DamageReward:
 # Penalty for taking damage, small bonus for getting powerups

    def __init__(self):
        self.penalty = REWARD_CONFIG['damage_penalty']
        self.last_state = 0
    
    def reset(self, info):
        self.last_state = info.get('powerup', 0) # 0 represents Mario's default state
    
    def calculate(self, info, terminated):
        current_state = info.get('powerup', 0)
        reward = 0.0
        
        if self.last_state > 0 and current_state == 0 and not terminated: # Don't double punish death
            reward -= self.penalty

        elif self.last_state == 0 and current_state > 0: # A powerup is worth 1/2 the penalty for taking a hit in reward
            reward += self.penalty / 2
        
        self.last_state = current_state

        return reward


class CoinReward:
    def __init__(self):
        self.reward_per_coin = REWARD_CONFIG['coin_reward']
        self.last_coins = 0

    def reset(self, info):
        self.last_coins = info.get('coins', 0)
    
    def calculate(self, info):
        current_coins = info.get('coins', 0)
        coins_gained = current_coins - self.last_coins
        self.last_coins = current_coins
        return coins_gained * self.reward_per_coin


class ScoreReward:
    def __init__(self):
        self.scale = REWARD_CONFIG['score_reward']
        self.last_score = 0

    def reset(self, info):
        self.last_score = info.get('score', 0)
    
    def calculate(self, info):
        current_score = info.get('score', 0)
        score_gained = current_score - self.last_score
        self.last_score = current_score
        return score_gained * self.scale


class MidwayFlagReward:
 # Reward for reaching the midway checkpoint
    def __init__(self):
        self.reward = REWARD_CONFIG['midway_reward']
        self.last_flag_state = 0
    
    def reset(self, info):
        self.last_flag_state = info.get('midway_flag', 0) # 0 = Flag not reached, 1 = Flag has been reached
    
    def calculate(self, info):
        current_flag_state = info.get('midway_flag', 0)
        if current_flag_state == 1 and self.last_flag_state == 0:
            self.last_flag_state = current_flag_state
            return self.reward
        self.last_flag_state = current_flag_state
        return 0.0


class LifeLossPenalty:
    def __init__(self):
        self.penalty = REWARD_CONFIG['life_loss_penalty']
        self.last_lives = None
    
    def reset(self, info):
        self.last_lives = info.get('lives', None)
    
    def calculate(self, info):
        current_lives = info.get('lives', None)
        if self.last_lives is not None and current_lives is not None:
            if current_lives < self.last_lives:
                self.last_lives = current_lives
                return self.penalty
        self.last_lives = current_lives

        return 0.0


class LevelCompleteReward:
    """Large reward for completing the level. The main level we train on, YoshisIsland2 corresponds to 80 in the WRAM address 7E13D6
     This value changes on level complete, when the overworld is loaded. We can therefore use it as a proxy for a true-level complete flag,
     but this solution is not very robust."""

    def __init__(self):
        self.reward = REWARD_CONFIG['level_complete_reward']
    
    def reset(self, _info):
        pass
    
    def calculate(self, info):
        level_complete = info.get('level_complete', 80)
        if level_complete != 80:
            return self.reward
        return 0.0


class ComposedRewardWrapper(gym.Wrapper):
    """Handle reward components and apply the step penalty to punish inaction/long episodes."""
    
    def __init__(self, env):
        super().__init__(env)
        self.step_penalty = REWARD_CONFIG['step_penalty']     
        self.components = {
            'movement': MovementReward(),
            'damage': DamageReward(),
            'life_loss': LifeLossPenalty(),
            'coins': CoinReward(),
            'score': ScoreReward(),
            'midway': MidwayFlagReward(),
            'level_complete': LevelCompleteReward(),
        }
    
    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        for name, component in self.components.items():
            component.reset(info)
        
        return obs, info

    def step(self, action):
    # Calculate reward for taking a step in the environment
        obs, reward, terminated, truncated, info = self.env.step(action)
        
        reward -= self.step_penalty
        reward -= self.components['damage'].calculate(info, terminated=terminated)
        reward -= self.components['life_loss'].calculate(info)
        reward += self.components['movement'].calculate(info)
        reward += self.components['coins'].calculate(info)
        reward += self.components['score'].calculate(info)
        reward += self.components['midway'].calculate(info)
        reward += self.components['level_complete'].calculate(info)
        
        return obs, reward, terminated, truncated, info
