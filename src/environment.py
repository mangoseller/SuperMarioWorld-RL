from torchrl.envs import TransformedEnv, GymWrapper, ParallelEnv
from torchrl.envs.transforms import (
    ToTensorImage,
    Resize,
    GrayScale,
    CatFrames,
    StepCounter,
    RewardSum, 
    Compose,
    Transform
)
import numpy as np
import retro
import gymnasium as gym
import os
from gymnasium.wrappers import RecordVideo
from multiprocessing import Process, Queue
from model_small import ImpalaSmall
from ppo import PPO
import torch as t
import time

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
        return self._decode_discrete_action[action].copy() # Convert integer action into expected boolean arr of button presses

class HandleMarioLifeLoss(gym.Wrapper):
    # Frame skip that stops on life loss to allow for episode termination on death
    def __init__(self, env, skip=4):
        super().__init__(env)
        self.skip = skip
        self.prev_lives = None

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        # print(self.env.env.unwrapped.data['7E0DBE'])
        # print(self.env.env.get_ram()[00007'])
        self.prev_lives = info.get('lives', None)
        return obs, info

    def step(self, action):
        total_reward = 0.0
        terminated, truncated = False, False

        for _ in range(self.skip):
            obs, reward, term, trunc, info = self.env.step(action)

            total_reward += reward
            current_lives = info.get('lives', None)

            # Check for life loss
            if self.prev_lives is not None and current_lives is not None \
            and current_lives < self.prev_lives:
                self.prev_lives = current_lives

                terminated = True
                break # Stop frame skipping

            self.prev_lives = current_lives
            terminated = term
            truncated = trunc
            if term or trunc:
                break # Stop on other done conditions

        return obs, total_reward, terminated, truncated, info

def prepare_env(env, skip=4, record=False, record_dir=None):
    wrapped_env = Discretizer(env, MARIO_ACTIONS)
    wrapped_env = HandleMarioLifeLoss(wrapped_env, skip=skip) # Frame skip
    if record:
        wrapped_env = RecordVideo(
            wrapped_env,
            video_folder=record_dir,
            episode_trigger=lambda x: True, # Record every episode, used for eval runs
            name_prefix=f"eval_{int(time.time())}"
        )
    wrapped_env = GymWrapper(wrapped_env)

    return TransformedEnv(wrapped_env, Compose([
    ToTensorImage(), # Convert stable-retro return values to PyTorch Tensors
    Resize(84, 84), # Can also do 96x96, 128x128
    GrayScale(),
    CatFrames(N=4, dim=-3), # dim -3 stacks frames over the channel dimension (does this make sense with gray frames?)
    StepCounter(),
    RewardSum(),
  ]))
 
def evaluate(agent, num_episodes=5, record_dir='/evals'):
    eval_rewards, eval_lengths = [], []
    os.makedirs(record_dir, exist_ok=True)
    eval_env = retro.make('SuperMarioWorld-Snes',
                          render_mode='rgb_array',
                          state='YoshiIsland2',
                          )
    eval_env = prepare_env(eval_env, record=True, record_dir=record_dir)
    for _ in range(num_episodes):
        eval_environment = eval_env.reset()
        state = eval_environment["pixels"]
        episode_reward = 0
        episode_length = 0
        done = False

        while not done:
            action = agent.eval_action_selection(state)
            eval_environment["action"] = action
            eval_environment = eval_env.step(eval_environment)
            state = eval_environment["next"]["pixels"]
            reward = eval_environment["next"]["reward"].item()
            done = eval_environment["next"]["done"].item()
            episode_reward += reward
            episode_length += 1
        
        eval_rewards.append(episode_reward)
        eval_lengths.append(episode_length)

    eval_env.close()

    return {
        "eval/mean_reward": np.mean(eval_rewards),
        "eval/std_reward": np.std(eval_rewards),
        "eval/mean_length": np.mean(eval_lengths),
        "eval/max_reward": np.max(eval_rewards),
        "eval/min_reward": np.min(eval_rewards)
    }


def _run_eval_(model, model_state_dict, config, num_episodes, record_dir, result_queue):    
    agent = model().to('cpu')
    agent.load_state_dict(model_state_dict)
    eval_policy = PPO(agent, config.learning_rate, epsilon=config.clip_eps, 
                      optimizer=t.optim.Adam, device='cpu', 
                      c1=config.c1, c2=config.c2)
    metrics = evaluate(eval_policy, num_episodes, record_dir)
    result_queue.put(metrics)

def eval_parallel_safe(model, policy, config, record_dir, num_episodes=3):
# Run evaluate in a sub-process 
    result_queue = Queue()
    # Move model weights to cpu
    cpu_state_dicts = {k: v.cpu() for k, v in policy.model.state_dict().items()}
    process = Process(target=_run_eval_, args=(
        model, cpu_state_dicts, config, num_episodes, record_dir, result_queue
    ))
    process.start()
    process.join()
    return result_queue.get()


def make_training_env(num_envs=1):
    if num_envs == 1:
        return prepare_env(
            retro.make(
            'SuperMarioWorld-Snes',
            state='YoshiIsland2',
            render_mode='human' # Change to 'rgb_array' when debugging finished
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

# print(env.action_space) 

