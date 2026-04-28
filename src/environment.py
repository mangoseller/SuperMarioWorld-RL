from torchrl.envs import TransformedEnv, GymWrapper, ParallelEnv
from torchrl.envs.gym_like import default_info_dict_reader
from torchrl.envs.transforms import (
    ToTensorImage,
    Resize,
    GrayScale,
    CatFrames,
    StepCounter,
    RewardSum,
    Compose,
    UnsqueezeTransform,
    RenameTransform
)
import retro
import gymnasium as gym
import numpy as np
import multiprocessing
from datetime import datetime
from gymnasium.wrappers import RecordVideo
from wrappers import Discretizer, FrameSkipAndTermination, MaxStepWrapper
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

    """torchrl's ParallelEnv needs to instantiate the environment in the main process to validate
    environment specs before spawning workers to create the different envs. The underlying emulator, gym-retro,
    is not thread-safe however: attempting to spawn workers/additional environments after calling retro.make in the main
    environment throws a fatal 1 env per process error. To work around this, in the main process we call this mock which matches
    the actual api of our environment, but does not actually call retro.make. In the subprocesses where the ParallelEnvs are actually,
    spawned we are then safe to call retro.make and create our environments as needed."""

    observation_space = gym.spaces.Box(0, 1, (4, 84, 84), np.float32)
    observation_space = gym.spaces.Box(0, 255, (224, 256, 3), np.uint8)
    action_space = gym.spaces.Discrete(14)
    buttons = ['B', 'Y', 'SELECT', 'START', 'UP', 'DOWN', 'LEFT', 'RIGHT', 'A', 'X', 'L', 'R']

    def reset(self, **_): return self.observation_space.sample(), {}
    def step(self, _): return self.observation_space.sample(), 0.0, False, False, {}
    def render(self): pass

def _wrap_env(env, skip=4, record=False, record_dir=None):
    # Apply all wrappers to a raw retro environment 

    wrapped_env = Discretizer(env, MARIO_ACTIONS) # Actions
    wrapped_env = ComposedRewardWrapper(wrapped_env) # Reward
    wrapped_env = FrameSkipAndTermination(wrapped_env, skip=skip) # Dones
    wrapped_env = MaxStepWrapper(wrapped_env, max_steps=12000)
    
    if record:
        wrapped_env = RecordVideo(
            wrapped_env,
            video_folder=record_dir,
            episode_trigger=lambda x: True, # Record all episodes
            name_prefix=f"vid_{readable_timestamp()}"
        )
    
    wrapped_env = GymWrapper(wrapped_env)
    wrapped_env.set_info_dict_reader(
        default_info_dict_reader(["_global_x", "_max_x"], ignore_private=False)
    )
    return TransformedEnv(wrapped_env, Compose([
    ToTensorImage(),
    Resize(84, 84, interpolation=InterpolationMode.NEAREST),
    GrayScale(),
    CatFrames(N=4, dim=-3),
    RenameTransform(in_keys=["pixels"], out_keys=["observation"]),
]))



def make_env(
    num_envs = 1,
    level_weights = None,
    level_distribution = None,
    frame_skip = 3,
    record = False,
    record_dir = None,
):

    dist = get_level_distribution(level_distribution, level_weights, num_envs) 
    if num_envs == 1:
        env = retro.make(
            'SuperMarioWorld-Snes',
            state=dist[0],
            render_mode='human', # Enables window if running locally
        )
        env = _wrap_env(env, skip=frame_skip, record=record, record_dir=record_dir)
        # Manually unsqueeze to match ParallelEnv's [1, ...] output shape
        return TransformedEnv(
            env, 
            UnsqueezeTransform(
                dim=0, 
                allow_positive_dim=True,
                in_keys=["observation", "reward", "done", "terminated"]
            )
        )

    else:

        def _create_parallel_worker(level):
            if multiprocessing.current_process().name == 'MainProcess':
    # The initial python process when train.py is called is 'MainProccess' - we return a mock in this case
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
            create_env_fn=_create_parallel_worker,
            create_env_kwargs=[{'level': level} for level in dist],
        )

def make_eval_env(level, record_dir = None):

    env = _wrap_env(
        retro.make(
            'SuperMarioWorld-Snes',
            state=level,
            render_mode='rgb_array',
        ),
        record=record_dir is not None, # We should record if passed a record_dir
        record_dir=record_dir,
    )

    return TransformedEnv(
        env, 
        UnsqueezeTransform(
            dim=0,
            allow_positive_dim=True, 
            in_keys=["observation", "reward", "done", "terminated"]
        )
    )

def get_level_distribution(level_distribution, level_weights, num_envs):
    if level_distribution is not None:
        return level_distribution
    elif level_weights is not None:
        return assign_levels(num_envs, level_weights)
    else: # Reasonable defaults for testing
        return assign_levels(num_envs, {
    'YoshiIsland2': 0.5,
    'YoshiIsland3': 0.5,
})

    
