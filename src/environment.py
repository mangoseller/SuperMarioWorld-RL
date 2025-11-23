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
from training_utils import readable_timestamp


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
    wrapped_env = MaxStepWrapper(wrapped_env, max_steps=5000)
    if record:
        wrapped_env = RecordVideo(
            wrapped_env,
            video_folder=record_dir,
            episode_trigger=lambda x: True, # Record every episode, used for eval runs
            name_prefix=f"eval_{readable_timestamp()}"
        )
    wrapped_env = GymWrapper(wrapped_env)

    return TransformedEnv(wrapped_env, Compose([
    ToTensorImage(), # Convert stable-retro return values to PyTorch Tensors
    Resize(84, 84), # Can also do 96x96, 128x128
    GrayScale(),
    CatFrames(N=4, dim=-3), # dim -3 stacks frames over the channel dimension
    StepCounter(),
    RewardSum(),
  ]))


def make_training_env(num_envs=1):
    if num_envs == 1:
        return prepare_env(
            retro.make(
            'SuperMarioWorld-Snes',
            state='YoshiIsland2', # YoshiIsland2
            render_mode='human', # Change to 'rgb_array' when debugging finished,
        ))
    else:
        return ParallelEnv(
            num_workers=num_envs,
            create_env_fn=lambda: prepare_env(
        retro.make(
        'SuperMarioWorld-Snes',
        state='YoshiIsland2',
        render_mode='rgb_array' # human doesn't work for parallel envs
    ))
)
