from torchrl.envs import TransformedEnv, GymWrapper, ParallelEnv 
from torchrl.envs.transforms import (
    ToTensorImage,
    Resize,
    GrayScale,
    CatFrames,
    StepCounter,
    RewardSum, 
    Compose,
)
import retro
import gymnasium as gym
import numpy as np
import multiprocessing
from datetime import datetime
from gymnasium.wrappers import RecordVideo
from wrappers import Discretizer, FrameSkipAndTermination, MaxStepWrapper, DilatedFrameStack
from rewards import ComposedRewardWrapper
from torchvision.transforms import InterpolationMode
from curriculum import assign_levels
from utils import readable_timestamp

MARIO_ACTIONS = [
    [],                   # Do nothing
    ['RIGHT'],            # Walk right
    ['RIGHT', 'Y'],       # Run right  
    ['RIGHT', 'B'],       # Jump right
    ['RIGHT', 'Y', 'B'],  # Run + jump right
    ['LEFT'],             # Walk left
    ['LEFT', 'Y'],        # Run left
    ['LEFT', 'B'],        # Jump left
    ['LEFT', 'Y', 'B'],   # Run + jump left
    ['B'],                # Jump in place
    ['X'],                # Attack
    ['DOWN'],             # Duck/enter pipe
    ['UP'],               # Look up/climb
    ['A'],                # Spin Jump
]


class MockRetro(gym.Env):
    """Provides environment specs for ParallelEnv's main-process initialization
    without launching the actual SNES emulator (which has singleton constraints)."""

    observation_space = gym.spaces.Box(0, 255, (224, 256, 3), np.uint8)
    action_space = gym.spaces.MultiBinary(12)
    buttons = ['B', 'Y', 'SELECT', 'START', 'UP', 'DOWN', 'LEFT', 'RIGHT', 'A', 'X', 'L', 'R']
    
    def reset(self, **_): return self.observation_space.sample(), {}
    def step(self, _): return self.observation_space.sample(), 0.0, False, False, {}
    def render(self): pass

def _wrap_env(env, skip=2, record=False, record_dir=None):
    # Apply all wrappers to a raw retro environment 

    wrapped_env = Discretizer(env, MARIO_ACTIONS) # Actions
    wrapped_env = ComposedRewardWrapper(wrapped_env) # Reward
    wrapped_env = FrameSkipAndTermination(wrapped_env, skip=skip) # Dones
    wrapped_env = MaxStepWrapper(wrapped_env, max_steps=8000)
    
    if record:
        wrapped_env = RecordVideo(
            wrapped_env,
            video_folder=record_dir,
            episode_trigger=lambda x: True, # Record all episodes
            name_prefix=f"vid_{readable_timestamp()}"
        )
    
    wrapped_env = GymWrapper(wrapped_env)
    return TransformedEnv(wrapped_env, Compose([
        ToTensorImage(),
        Resize(84, 84, interpolation=InterpolationMode.NEAREST),
        GrayScale(),
        CatFrames(N=4, dim=-3),
        StepCounter(),
        RewardSum(),
    ]))


def make_env(
    num_envs = 1,
    level_weights = None,
    level_distribution = None,
    frame_skip = 2,
    record = False,
    record_dir = None,
):
    dist = get_level_distribution(level_distribution, level_weights, num_envs) 
    render_mode = 'human' if num_envs == 1 else 'rgb_array'

    if num_envs == 1:
        raw_env = retro.make(
            'SuperMarioWorld-Snes',
            state=dist[0],
            render_mode=render_mode,
        )
        return _wrap_env(raw_env, skip=frame_skip, record=record, record_dir=record_dir)
    else:

        def create_env(level):

            # If we are in the MainProcess, ParallelEnv is running a dummy check.
            # We return a MockRetro to satisfy the check without triggering Retro's 1 env per process error

            if multiprocessing.current_process().name == 'MainProcess':
                raw_env = MockRetro()
            else:
                raw_env = retro.make(
                    'SuperMarioWorld-Snes',
                    state=level,
                    render_mode='rgb_array',
                )
            return _wrap_env(raw_env, skip=frame_skip)  

        return ParallelEnv(
            num_workers=num_envs,
            create_env_fn=create_env,
            create_env_kwargs=[{'level': level} for level in dist],
        )


def make_eval_env(level, record_dir = None):
    # Create a single environment for evaluation.
    return _wrap_env(
        retro.make(
        'SuperMarioWorld-Snes',
        state=level,
        render_mode='rgb_array',
    ),
        record=record_dir is not None,
        record_dir=record_dir,
    )

def get_level_distribution(level_distribution, level_weights, num_envs):
    if level_distribution is not None:
        return level_distribution
    elif level_weights is not None:
        return assign_levels(num_envs, level_weights)
    else:
        return assign_levels(num_envs, {
    'YoshiIsland2': 0.5,
    'YoshiIsland3': 0.5,
})

    

