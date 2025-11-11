from torchrl.envs import TransformedEnv, GymWrapper
from torchrl.envs.transforms import (
    ToTensorImage,
    Resize,
    GrayScale,
    CatFrames,
    StepCounter,
    RewardSum, 
    Compose,
    FrameSkipTransform,
)
import numpy as np
import retro
import gymnasium as gym
training=False

class Discretizer(gym.ActionWrapper):
# Wrap an env to use COMBOS as its discrete action space

    def __init__(self, env, combos):
        super().__init__(env)
        buttons = env.unwrapped.buttons 
        self._decode_discrete_action = []
        for c in combos:
            arr = np.array([False] * env.action_space.n) # All positions except for the button we want to press should be False
            for button in c:
                arr[buttons.index(button)] = True # Set the button position in the array to True
            self._decode_discrete_action.append(arr)
        self.action_space = gym.spaces.Discrete(len(self._decode_discrete_action)) 
    
    def action(self, action):
        return self._decode_discrete_action[action].copy() # Convert integer into expected boolean arr

MARIO_ACTIONS = [
    [],                   # Do nothing
    ['RIGHT'],            # Walk right
    ['RIGHT', 'B'],       # Run right  
    ['RIGHT', 'A'],       # Jump right
    ['RIGHT', 'B', 'A'],  # Run + jump right
    ['LEFT'],             # Walk left
    ['LEFT', 'B'],        # Run left
    ['LEFT', 'A'],        # Jump left
    ['LEFT', 'B', 'A'],   # Run + jump left
    ['A'],                # Jump in place
    ['Y'],                # Attack
    ['DOWN'],             # Duck/enter pipe
    ['UP'],               # Look up/climb
]
base_env = retro.make(
        'SuperMarioWorld-Snes', 
        render_mode='rgb_array' if training else 'human'
    )      

base_env = GymWrapper(Discretizer(base_env, MARIO_ACTIONS))

env = TransformedEnv(base_env, Compose(*[
    FrameSkipTransform(frame_skip = 4),
    ToTensorImage(), # Convert stable-retro return values to PyTorch Tensors
    Resize(84, 84), # Can also do 96x96, 128x128
    GrayScale(),
    CatFrames(N=4, dim=-3), # dim -3 stacks frames over the channel dimension (does this make sense with gray frames?)
    StepCounter(),
    RewardSum(),
  ]))
print(env.action_space) 

