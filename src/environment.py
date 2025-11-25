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
from gymnasium.wrappers import RecordVideo
from wrappers import Discretizer, FrameSkipAndTermination, MaxStepWrapper
from rewards import ComposedRewardWrapper
from torchvision.transforms import InterpolationMode

DEFAULT_LEVELS = {
    'level1': 'YoshiIsland2',
    'level2': 'DonutPlains1',
}

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

def compute_level_distribution(num_envs, **level_kwargs):
    """Distribute environments uniformly across specified levels.
      Usage: compute_level_distribution(16, level1='YoshiIsland2', level2='YoshiIsland1', level3='DonutPlains1')
    """
    levels = level_kwargs if level_kwargs else DEFAULT_LEVELS
    level_list = list(levels.values())
    num_levels = len(level_list)
    
    # Distribute as uniformly as possible
    base_count = num_envs // num_levels
    remainder = num_envs % num_levels
    
    distribution = []
    for i, level in enumerate(level_list):
        # Give one extra env to the first 'remainder' levels
        count = base_count + (1 if i < remainder else 0)
        distribution.extend([level] * count)
    
    return distribution

def prepare_env(env, skip=2, record=False, record_dir=None):
    wrapped_env = Discretizer(env, MARIO_ACTIONS)
    wrapped_env = ComposedRewardWrapper(wrapped_env)
    wrapped_env = FrameSkipAndTermination(wrapped_env, skip=skip)
    wrapped_env = MaxStepWrapper(wrapped_env, max_steps=8000)
    if record:
        from utils import readable_timestamp # Avoid circular imports
        wrapped_env = RecordVideo(
            wrapped_env,
            video_folder=record_dir,
            episode_trigger=lambda x: True, # Record every episode, used for eval runs
            name_prefix=f"eval_{readable_timestamp()}"
        )
    wrapped_env = GymWrapper(wrapped_env)

    return TransformedEnv(wrapped_env, Compose([
    ToTensorImage(), # Convert stable-retro return values to PyTorch Tensors
    Resize(84, 84, interpolation=InterpolationMode.NEAREST), # Can also do 96x96, 128x128
    GrayScale(),
    CatFrames(N=4, dim=-3), # dim -3 stacks frames over the channel dimension
    StepCounter(),
    RewardSum(),
  ]))
 
def make_training_env(num_envs=1, **level_kwargs):
    if num_envs == 1:
        return prepare_env(
            retro.make(
            'SuperMarioWorld-Snes',
            state='DonutPlains5', # YoshiIsland2
            render_mode='human', # Change to 'rgb_array' when debugging finished,
        ))
    else:
        if num_envs == 28:
            level_dist = compute_level_distribution(num_envs, level1='YoshiIsland2', level2='DonutPlains1', level3='DonutPlains4')
        else:
            level_dist = compute_level_distribution(num_envs)
        create_env = lambda level: prepare_env(
            retro.make(
                'SuperMarioWorld-Snes',
                state=level,
                render_mode='rgb_array'
            )
        )
        return ParallelEnv(
            num_workers=num_envs,
            create_env_fn=create_env,
            create_env_kwargs=[{'level': state} for state in level_dist]
        )
