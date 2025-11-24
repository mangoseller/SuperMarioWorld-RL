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

def _get_level_distribution(num_envs):
    # 2/3 of the envs should train on the easier YoshiIsland2
    assert num_envs >= 3 and num_envs % 3 == 0, "Number of environments must be a multiple of 3"
    third = num_envs // 3
    two_thirds = third * 2
    states = ['YoshiIsland2' for _ in range(two_thirds)]
    states.extend(['YoshiIsland1' for _ in range(third)])
    
    return states
    
def _get_level_distribution(num_envs):
    # 2/3 of the envs should train on the easier YoshiIsland2
    assert num_envs >= 3 and num_envs % 3 == 0, "Number of environments must be a multiple of 3"
    third = num_envs // 3
    two_thirds = third * 2
    states = ['YoshiIsland2' for _ in range(two_thirds)]
    states.extend(['YoshiIsland1' for _ in range(third)])
    
    return states
    
def make_training_env(num_envs=1):
    if num_envs == 1:
        return prepare_env(
            retro.make(
            'SuperMarioWorld-Snes',
            state='DonutPlains1', # YoshiIsland2
            render_mode='human', # Change to 'rgb_array' when debugging finished,
        ))
    else:
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
            create_env_kwargs=[{'level': state} for state in _get_level_distribution(num_envs)]
        )

