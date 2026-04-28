"""
Async PPO training: environment collection and model training overlap.

Architecture:
    Collection thread (GPU inference)  Training thread (GPU training)
    +---------------------------+     +---------------------------+
    | GPU copy of model         |     | GPU model + PPO           |
    | (Inference, no grad)      |-->  | Trains on completed buffer|
    | Weight sync (CPU staging) |<--  | Runs PPO epochs on GPU    |
    +---------------------------+     +---------------------------+
                    |                             |
                    +--- rollout_queue (size=1) ---+
"""

import warnings
warnings.filterwarnings('ignore')

import threading
import time
import traceback
import math
import copy
from queue import Queue, Empty
from contextlib import contextmanager

import torch as t
import numpy as np
from tqdm import tqdm
from einops import rearrange

from ppo import PPO
from buffer import RolloutBuffer
from environment import make_env
from curriculum import Curriculum, assign_levels
from evals import run_evaluation
from utils import (
    readable_timestamp,
    save_checkpoint,
    load_checkpoint,
    log_training_metrics,
    get_torch_compatible_actions,
    get_entropy,
)


def _make_env_for_curriculum(curriculum, config):
    level_dist = assign_levels(config.num_envs, curriculum.weights)
    return make_env(num_envs=config.num_envs, level_distribution=level_dist)


def _make_env_default(config):
    return make_env(num_envs=config.num_envs)


class _AsyncPPOTrainer:
    """Manages two threads: collection (daemon) and training (main)."""

    def __init__(self, model_class, config, device):
        self.config = config
        self.device = device

        # Create model and PPO wrapper
        original_agent = model_class().to(device)
        self.policy = PPO(original_agent, config, device)
        self.original_agent = original_agent

        # GPU copy for collection inference
        self.collect_model = copy.deepcopy(original_agent)
        self.collect_model.eval()
        self.collect_policy = PPO(self.collect_model, config, device)

        # Synchronisation primitives
        self._stop = threading.Event()
        self._pause_requested = threading.Event()
        self._collection_paused = threading.Event()
        self._weight_lock = threading.Lock()
        self._episode_queue = Queue()
        self._rollout_queue = Queue(maxsize=1)
        self._collection_error = None

        # Staged weights: main thread writes, collector reads (between rollouts)
        self._staged_weights = None
        self._weights_ready = False

        # Collection-thread state
        self._env = None
        self._env_needs_reset = False

    # ------------------------------------------------------------------
    #  Weight synchronisation (GPU -> CPU staging -> GPU collector)
    # ------------------------------------------------------------------

    def _sync_weights(self):
        """Copy trained weights to CPU staging area."""
        sd = {k: v.cpu() for k, v in self.original_agent.state_dict().items()}
        with self._weight_lock:
            self._staged_weights = sd
            self._weights_ready = True

    def _maybe_apply_staged_weights(self):
        """Collector calls this between rollouts to pick up new weights."""
        with self._weight_lock:
            if self._weights_ready:
                self.collect_model.load_state_dict(self._staged_weights)
                self._weights_ready = False

    # ------------------------------------------------------------------
    #  Pause / resume collection (for eval + checkpoint)
    # ------------------------------------------------------------------

    @contextmanager
    def _paused_collection(self):
        self._pause_requested.set()
        ack = self._collection_paused.wait(timeout=30)
        if not ack:
            raise RuntimeError(
                "Collection thread did not acknowledge pause within 30 s. "
                "It may have crashed — check self._collection_error."
            )
        try:
            yield
        finally:
            self._collection_paused.clear()
            self._pause_requested.clear()

    # ------------------------------------------------------------------
    #  Collection thread
    # ------------------------------------------------------------------

    def _collection_loop(self, tracking):
        """Collect rollouts continuously. Each completed rollout goes onto
        self._rollout_queue for the main thread to train on."""
        try:
            td = self._env.reset()
            state = td['observation']

            while not self._stop.is_set():
                # Handle pause requests
                if self._pause_requested.is_set():
                    self._collection_paused.set()
                    while self._pause_requested.is_set() and not self._stop.is_set():
                        time.sleep(0.01)
                    if self._stop.is_set():
                        break
                    if self._env_needs_reset:
                        td = self._env.reset()
                        state = td['observation']
                        tracking['current_episode_rewards'] = [0.0] * self.config.num_envs
                        tracking['current_episode_lengths'] = [0] * self.config.num_envs
                        self._env_needs_reset = False
                    continue

                # Apply staged weights (between rollouts)
                self._maybe_apply_staged_weights()

                # Collect one full rollout
                buffer = RolloutBuffer(
                    self.config.steps_per_env, self.config.num_envs, 'cpu',
                )
                aborted = False

                for step_idx in range(self.config.steps_per_env):
                    if self._stop.is_set():
                        return
                    if self._pause_requested.is_set():
                        aborted = True
                        break

                    state_gpu = state.to(self.device, non_blocking=True)
                    actions, log_probs, values = (
                        self.collect_policy.action_selection(state_gpu)
                    )
                    actions_cpu = actions.cpu()

                    td['action'] = get_torch_compatible_actions(actions_cpu)
                    td = self._env.step(td)

                    next_state = td['next']['observation']
                    rewards = td['next']['reward']
                    dones = (
                        td['next']['done']
                        | td['next'].get('truncated', t.zeros_like(td['next']['done']))
                    )

                    buffer.store(
                        state, rewards.squeeze(), actions,
                        log_probs, values, dones.squeeze(),
                    )

                    # Episode tracking
                    tracking['total_env_steps'] += self.config.num_envs
                    for i in range(self.config.num_envs):
                        tracking['current_episode_rewards'][i] += rewards[i].item()
                        tracking['current_episode_lengths'][i] += 1
                        if dones.squeeze(-1)[i].item():
                            self._episode_queue.put({
                                'reward': tracking['current_episode_rewards'][i],
                                'length': tracking['current_episode_lengths'][i],
                            })
                            tracking['current_episode_rewards'][i] = 0.0
                            tracking['current_episode_lengths'][i] = 0
                            tracking['episode_num'] += 1

                    # Handle env resets
                    if dones.any():
                        reset_td = td.clone()
                        reset_td['_reset'] = dones.clone()
                        reset_out = self._env.reset(reset_td.to('cpu')).to(state.device)
                        mask = rearrange(dones, 'b c -> b c 1 1')
                        state = t.where(mask, reset_out['observation'], next_state)
                        td = reset_out
                    else:
                        state = next_state

                if aborted:
                    continue  # Discard partial rollout, handle pause on next iter

                # Submit completed rollout
                self._rollout_queue.put({
                    'buffer': buffer,
                    'next_state': state.clone(),
                })

        except Exception as e:
            self._collection_error = e
            traceback.print_exc()

    # ------------------------------------------------------------------
    #  Drain completed episode metrics
    # ------------------------------------------------------------------

    def _drain_episodes(self):
        rewards, lengths = [], []
        while True:
            try:
                ep = self._episode_queue.get_nowait()
                rewards.append(ep['reward'])
                lengths.append(ep['length'])
            except Empty:
                break
        return rewards, lengths

    # ------------------------------------------------------------------
    #  Main entry point
    # ------------------------------------------------------------------

    def run(self, curriculum_option=None, checkpoint_path=None, resume=False):
        run = self.config.setup_wandb()

        tracking = {
            'current_episode_rewards': [0.0] * self.config.num_envs,
            'current_episode_lengths': [0] * self.config.num_envs,
            'completed_rewards': [],
            'completed_lengths': [],
            'episode_num': 0,
            'total_env_steps': 0,
            'last_eval_step': 0,
            'run_timestamp': readable_timestamp(),
        }
        start_step = 0

        if checkpoint_path:
            start_step, saved_tracking = load_checkpoint(
                checkpoint_path, self.original_agent, self.policy, resume,
            )
            if resume and saved_tracking:
                tracking = saved_tracking
                tracking['run_timestamp'] = readable_timestamp()
            print(f"{'Resumed from' if resume else 'Loaded weights from'} "
                  f"{checkpoint_path} at step {start_step}")
            # Re-sync collection model after loading
            self.collect_model = copy.deepcopy(self.original_agent)
            self.collect_model.eval()
            self.collect_policy.model = self.collect_model

            # start_step from checkpoint is per-env; total_env_steps is across all envs
            total_from_checkpoint = start_step * self.config.num_envs
            tracking['total_env_steps'] = max(tracking['total_env_steps'], total_from_checkpoint)
            if not resume:
                tracking['last_eval_step'] = tracking['total_env_steps']

        curriculum = None
        if self.config.use_curriculum and curriculum_option:
            curriculum = Curriculum.create(curriculum_option)
            if start_step > 0:
                while curriculum.update(start_step, self.config.num_training_steps):
                    pass

        if curriculum:
            self._env = _make_env_for_curriculum(curriculum, self.config)
            print(f'Curriculum: {curriculum.describe(curriculum.stage)}')
        else:
            self._env = _make_env_default(self.config)

        param_count = sum(p.numel() for p in self.original_agent.parameters())
        print(f'Training {self.config.architecture} (async) | {param_count:,} params | {self.device}')
        print('Compiling GPU model...')

        compiled_model = t.compile(self.original_agent)
        self.policy.model = compiled_model

        def _swap_to_original():
            self.policy.model = self.original_agent

        def _swap_to_compiled():
            self.policy.model = compiled_model

        # Launch collector thread
        collector = threading.Thread(
            target=self._collection_loop,
            args=(tracking,),
            daemon=True,
            name='ppo-collector',
        )
        collector.start()
        print('Collection thread started.')

        # Training bookkeeping
        # num_training_steps is per-env steps; each rollout = steps_per_env per-env steps
        total_updates = math.ceil(self.config.num_training_steps / self.config.steps_per_env)
        train_updates = start_step // self.config.steps_per_env
        last_log_step = tracking['total_env_steps']
        last_wait_log_env_steps = tracking['total_env_steps']

        # Entropy scheduling
        entropy_boost = 1.0
        BOOST_MAGNITUDE = 3.0
        BOOST_DECAY = 0.0005
        last_mean_reward = 'N/A'

        # pbar tracks per-env steps (num_training_steps is per-env)
        per_env_steps = tracking['total_env_steps'] // self.config.num_envs
        pbar = tqdm(
            total=self.config.num_training_steps,
            initial=min(self.config.num_training_steps, per_env_steps),
            disable=not self.config.show_progress,
            desc='Async PPO',
            unit='env_step',
        )

        self.policy.model.eval()

        try:
            while train_updates < total_updates:
                self._check_collection_alive(collector)

                # Wait for a completed rollout
                try:
                    rollout = self._rollout_queue.get(timeout=1.0)
                except Empty:
                    if self.config.show_progress:
                        per_env_steps = tracking['total_env_steps'] // self.config.num_envs
                        pbar.n = min(self.config.num_training_steps, per_env_steps)
                        pbar.set_postfix({
                            'ep': tracking['episode_num'],
                            'rew': last_mean_reward,
                            'phase': 'collect',
                        })
                        pbar.refresh()

                    # Heartbeat logging for WandB during long collection phases
                    if (
                        self.config.USE_WANDB
                        and tracking['total_env_steps'] - last_wait_log_env_steps >= 5000
                    ):
                        import wandb
                        wandb.log({
                            'train/total_env_steps': tracking['total_env_steps'],
                            'train/episodes': tracking['episode_num'],
                            'system/phase': 0,  # 0 = collection
                        }, step=tracking['total_env_steps'])
                        last_wait_log_env_steps = tracking['total_env_steps']
                    continue

                buffer = rollout['buffer'].to(self.device)
                next_state = rollout['next_state'].to(self.device)

                # per-env step for schedules (entropy, curriculum) that expect per-env units
                per_env_step = tracking['total_env_steps'] // self.config.num_envs

                # Entropy scheduling (per-env steps, matching sync train.py)
                base_entropy = get_entropy(
                    per_env_step, self.config.num_training_steps,
                    max_entropy=self.config.c2,
                )
                self.policy.c2 = base_entropy * entropy_boost
                entropy_boost = max(1.0, entropy_boost - BOOST_DECAY)

                # Curriculum transition (per-env steps)
                if curriculum and curriculum.update(
                    per_env_step, self.config.num_training_steps,
                ):
                    with self._paused_collection():
                        self._env.close()
                        self._env = _make_env_for_curriculum(curriculum, self.config)
                        self._env_needs_reset = True
                        print(f'\nCurriculum -> Stage {curriculum.stage}: '
                              f'{curriculum.describe(curriculum.stage)}')
                        entropy_boost = BOOST_MAGNITUDE

                # PPO update
                diagnostics = self.policy.update(buffer, self.config, next_state=next_state)
                train_updates += 1

                # Sync weights to collector
                self._sync_weights()

                # Drain episode metrics
                ep_rewards, ep_lengths = self._drain_episodes()
                tracking['completed_rewards'].extend(ep_rewards)
                tracking['completed_lengths'].extend(ep_lengths)

                # Progress bar
                if tracking['completed_rewards']:
                    last_mean_reward = f"{np.mean(tracking['completed_rewards']):.1f}"

                if self.config.show_progress:
                    pbar.set_postfix({
                        'ep': tracking['episode_num'],
                        'rew': last_mean_reward,
                        'lr': f"{self.policy.get_current_lr():.1e}",
                        'c2': f"{self.policy.c2:.1e}",
                        'phase': 'train'
                    })
                    pbar.n = min(self.config.num_training_steps, per_env_step)
                    pbar.refresh()

                # Logging to wandb
                env_steps_since_log = tracking['total_env_steps'] - last_log_step
                if env_steps_since_log >= 1000 or ep_rewards:
                    log_training_metrics(
                        tracking, diagnostics, self.policy, self.config,
                        per_env_step,
                    )
                    tracking['completed_rewards'].clear()
                    tracking['completed_lengths'].clear()
                    last_log_step = tracking['total_env_steps']

                # Eval + checkpoint (eval_freq is in total env steps across all envs)
                if tracking['total_env_steps'] - tracking['last_eval_step'] >= self.config.eval_freq:
                    with self._paused_collection():
                        _swap_to_original()
                        save_checkpoint(
                            self.original_agent, self.policy, tracking, self.config, run,
                            per_env_step, curriculum_option,
                        )
                        run_evaluation(
                            self.policy, tracking, self.config, run,
                            per_env_step, curriculum,
                        )
                        _swap_to_compiled()
                        self._sync_weights()
                    print(f'Eval + checkpoint at step ~{per_env_step}')
                    tracking['last_eval_step'] = tracking['total_env_steps']

        except KeyboardInterrupt:
            print('\nInterrupted by user.')
        finally:
            print('Shutting down...')
            self._stop.set()
            collector.join(timeout=10)
            pbar.close()

            _swap_to_original()

            final_per_env_step = tracking['total_env_steps'] // self.config.num_envs
            run_evaluation(
                self.policy, tracking, self.config, run,
                final_per_env_step, curriculum,
            )
            save_checkpoint(
                self.original_agent, self.policy, tracking, self.config, run,
                final_per_env_step, curriculum_option,
            )

            self._env.close()
            if self.config.USE_WANDB:
                import wandb
                wandb.finish()

        print(f'Async PPO training complete. '
              f'{train_updates} gradient updates, '
              f'{tracking["total_env_steps"]:,} env steps.')
        return self.original_agent

    def _check_collection_alive(self, thread):
        if self._collection_error is not None:
            raise RuntimeError(
                f"Collection thread crashed: {self._collection_error}"
            ) from self._collection_error
        if not thread.is_alive():
            raise RuntimeError("Collection thread exited unexpectedly.")


def train_async(model_class, config, curriculum_option=None,
                checkpoint_path=None, resume=False):
    """Drop-in async replacement for synchronous PPO training."""

    device = 'cuda' if t.cuda.is_available() else 'cpu'
    print(f'[async] Initialising PPO on {device}')

    t.set_float32_matmul_precision('high')

    trainer = _AsyncPPOTrainer(model_class, config, device)
    return trainer.run(
        curriculum_option=curriculum_option,
        checkpoint_path=checkpoint_path,
        resume=resume,
    )
