# CLAUDE.md

## Project Overview

Super Mario World (SNES) reinforcement learning using PPO. The agent plays via `gym-retro` (emulator), `torchrl` (environment management), and PyTorch. Three model architectures of increasing size; the largest (`ImpalaWide`/`TransPala`) uses pixel-control SSL as an auxiliary task.

Entry point: `python train.py` (via `runner.py` which handles CLI args and routes to `train.py` or `async_train.py`).

---

## File Map

| File | Purpose |
|---|---|
| `runner.py` | CLI, model registry, `run_training()` dispatcher |
| `train.py` | Synchronous training loop |
| `async_train.py` | Async training: collection thread + training thread |
| `async_demo.py` | Prototype sketch for LSTM-PPO (imports don't exist yet) |
| `config.py` | `TrainingConfig` dataclass + named preset configs |
| `models.py` | `ConvolutionalSmall`, `ImpalaLike`, `ImpalaWide` |
| `model_components.py` | `FourierCoordConv`, `ResidualBlock`, `SpatialSoftmax`, `PixelControlHead`, `RandomShifts` |
| `ppo.py` | `PPO`: optimizer, action selection, GAE, loss, update |
| `buffer.py` | `RolloutBuffer`: uint8-compressed rollout storage |
| `environment.py` | `make_env`, `make_eval_env`, `MockRetro`, action table |
| `wrappers.py` | `Discretizer`, `FrameSkipAndTermination`, `MaxStepWrapper` |
| `rewards.py` | `ComposedRewardWrapper` + per-component reward classes |
| `curriculum.py` | `Curriculum`, `assign_levels`, level schedules |
| `evals.py` | `evaluate`, `evaluate_in_subprocess`, `run_evaluation` |
| `utils.py` | Checkpoint I/O, WandB logging, `get_entropy`, helpers |

---

## Coding Style

### Imports
- `import torch as t` — always, never `import torch`
- `import numpy as np`
- `from einops import rearrange, repeat, einsum` — used throughout for all tensor reshaping

### Naming
- `snake_case` for functions and variables
- `ALL_CAPS` for module-level constants (`REWARD_CONFIG`, `MARIO_ACTIONS`, `SCHEDULES`)
- `_leading_underscore` for private/internal names (`_wrap_env`, `_run_episode`, `_AsyncPPOTrainer`)
- Class names: `PascalCase`

### Comments
Comments appear only when the *why* is non-obvious. Algorithmic complexity (GAE backward pass, xpos wraparound logic, MockRetro thread-safety workaround) gets a block comment or docstring explaining the invariant or constraint. Routine code has no comments. Never comment what the code does; only comment why it must be this way.

### Lambdas
Used freely for small inline transforms: `flatten = lambda collection: rearrange(...)`, `reshape = lambda tensor: tensor.view(...)`.

### No type hints
The existing code has no function-level type annotations except in `curriculum.py:assign_levels`. Don't add them.

### Tensor operations
Prefer `einops.rearrange` over `.view()`, `.reshape()`, `.permute()`. Use `einops.repeat` for broadcasting. Use `einops.einsum` for batched contractions.

---

## Architecture

### Models (`models.py` / `model_components.py`)

All three models share the same `forward(x, return_pixel_control=False)` signature returning `(policy_logits, value)` or `(policy_logits, value, pixel_pred)`.

**`ConvolutionalSmall`** (~600k params): plain conv → flatten → FC, no augmentation, no coord encoding.

**`ImpalaLike`** (~1.1M params): `RandomShifts` → `FourierCoordConv(scales=[2.0])` → 3 residual layers with dilation → `SpatialSoftmax` → trunk. No pixel control head; returns `None` if `return_pixel_control=True`.

**`ImpalaWide`** (`TransPala`, ~3.8M params): same as ImpalaLike but wider channels, dual Fourier scales `[1.0, 2.0]`, and a real `PixelControlHead`.

**Key components:**
- `FourierCoordConv`: appends sinusoidal coordinate channels to the input tensor. Each scale produces 4 channels (sin/cos × x/y). Input grows from 4 → 8 or 12 channels.
- `ResidualBlock`: pre-activation (GroupNorm→SiLU→Conv→GroupNorm→SiLU→Conv + skip). `GroupNorm(8, channels)` — chosen over BatchNorm for stability with small batches.
- `SpatialSoftmax`: flattens spatial dims, softmax, computes expected (x,y) per channel. Converts `(B,C,H,W)` to `(B, 2C)` while preserving spatial information.
- `PixelControlHead`: linear deconv → reshape → two spatial convs → Softplus. Predicts absolute pixel change on a 7×7 grid.
- `RandomShifts`: disabled in `model.eval()` mode. Replicates padding to avoid black borders.

**Weight init**: orthogonal init, `gain=sqrt(2)` for conv/linear; `gain=0.01` for policy head (near-uniform initial distribution); `gain=1.0` for value head.

### PPO (`ppo.py`)

`PPO` wraps a model with an Adam optimizer and optional LR scheduler.

- `action_selection`: runs under `@t.inference_mode()`, returns `(actions, log_probs, values)`
- `compute_advantages`: backward GAE pass over buffer. Properly zeroes next-value at episode boundaries via `(1 - dones[i])` masking.
- `compute_pixel_change_targets`: computes `|s_{t+1} - s_t|` averaged over frame stack, pooled to 7×7. Masks out transitions at episode boundaries.
- `compute_loss`: clipped surrogate objective + value MSE + entropy regularisation + pixel control MSE. Uses `bfloat16` autocast (applied in `update()`).
- `update`: shuffles buffer, iterates minibatches across `epochs`. Clips gradients at 0.5. Calls `model.eval()` after update.
- `_has_pixel_control`: duck-typed check (`hasattr(self.model, 'pixel_control_head')`).

Loss: `policy_loss + c1 * value_loss - c2 * entropy + pixel_weight * pixel_loss`  
(note entropy enters negated — we *subtract* entropy loss because higher entropy is desirable)

### Buffer (`buffer.py`)

`RolloutBuffer` stores `(capacity, num_envs, ...)` shaped tensors on `device`. States stored as `uint8` (×255 on store, ÷255 on `get()`). `get()` flattens `(Time, Envs, ...)` → `(Time×Envs, ...)`. `clear()` resets `idx=0` without reallocating. Full when `idx == capacity`.

### Environment Stack

```
retro.make('SuperMarioWorld-Snes', state=level)
  → Discretizer(MARIO_ACTIONS)           # 14 discrete button combos
  → ComposedRewardWrapper                # custom reward shaping
  → FrameSkipAndTermination(skip=3)      # accumulate reward, terminate on death/complete
  → MaxStepWrapper(max_steps=12000)      # episode length cap
  [→ RecordVideo]                        # eval only
  → GymWrapper                           # TorchRL adapter
  → TransformedEnv(ToTensorImage, Resize(84,84,NEAREST), GrayScale,
                   CatFrames(N=4,dim=-3), RenameTransform(pixels→observation))
  [→ UnsqueezeTransform(dim=0)]          # single-env batch dim
```

**`MockRetro`**: `ParallelEnv` validates specs in the main process before spawning workers. `retro.make` is not thread-safe — calling it in the main process then forking causes a fatal error. `MockRetro` matches the API (spaces, `buttons`, `reset`, `step`) without calling retro, allowing safe spec validation.

**14 actions** defined in `MARIO_ACTIONS` as button combo lists.

### Rewards (`rewards.py`)

Each component has `reset(info)` and `calculate(info)` (plus `terminated` for damage). `ComposedRewardWrapper.step()` calls all of them and sums. Scales in `REWARD_CONFIG` dict at top of file.

**`MovementReward`**: only rewards new forward progress (`_global_x > _max_x`). Handles Mario's 1-byte x-position wraparound (±128 threshold). Stuck-detection gives small exploration bonus after 300 steps with no progress.

### Curriculum (`curriculum.py`)

`Curriculum` is a `@dataclass` with a `schedule`: list of `(end_fraction, weights_dict)` pairs. `update(step, total)` advances `self.stage` and returns `True` if it changed. `assign_levels` uses the largest-remainder method for whole-number allocation.

Three named schedules: `PROGRESSIVE_SCHEDULE`, `GRADUAL_SCHEDULE`, `SEQUENTIAL_SCHEDULE`, registered in `SCHEDULES = {1: ..., 2: ..., 3: ...}`.

### Config (`config.py`)

`TrainingConfig` is a `@dataclass`. `buffer_size` and `minibatch_size` are `@property`. Use `dataclasses.replace(config, field=val)` to override fields. Named presets follow the pattern `{ARCH}_{MODE}_CONFIG` where MODE ∈ {TRAIN, TEST, TUNE}.

### Training Loop (`train.py`)

Entropy schedule: linearly decayed `c2` from `get_entropy()`, with a curriculum-change spike (`entropy_boost = 3.0`) that decays back at `BOOST_DECAY = 0.0005` per step.

Compile pattern: `agent = t.compile(original_agent)`, then swap `policy.model = agent` for training and `policy.model = original_agent` for eval/checkpoint.

Done handling: when any env finishes, a masked reset merges the reset observation back in:
```python
mask = rearrange(dones, 'b c -> b c 1 1')
state = t.where(mask, reset_out["observation"], next_state)
```

Eval runs in a subprocess (`evaluate_in_subprocess`) to avoid retro thread-safety issues with the main process.

### Async Training (`async_train.py`)

Two threads on one GPU:
- **Collector thread** (daemon): runs `_collection_loop`, fills `RolloutBuffer` on CPU, pushes to `_rollout_queue(maxsize=1)`. Applies staged weights between rollouts.
- **Main thread**: waits on queue, runs PPO update, calls `_sync_weights()` (GPU→CPU staging area).

Synchronisation: `threading.Event` for stop/pause, `threading.Lock` for weight staging, `_paused_collection()` context manager for eval/checkpoint.

`bfloat16` autocast and `t.compile` used on the training side; collection model stays in default dtype.

---

## Checkpoints

Saved under `model_checkpoints/` as `{architecture}_ep{episode_num}.pt`:
```python
{
    'model_state_dict': ...,
    'optimizer_state_dict': ...,
    'scheduler_state_dict': ...,   # None if no scheduler
    'step': int,
    'tracking': {...},
    'config_dict': {architecture, num_training_steps, learning_rate, ...}
}
```
`load_checkpoint(path, agent, policy, resume)` — `resume=True` restores optimizer/scheduler state and tracking; `resume=False` loads weights only.

---

## WandB Integration

Controlled by `config.USE_WANDB`. Metrics namespaced: `train/*`, `loss/*`, `hyperparams/*`, `eval/{level}/reward/*`. Artifacts: model checkpoints and eval video directories. `config.setup_wandb()` reads `WANDB_API_KEY` from environment.

---

## Running

```bash
python train.py --mode train --model ImpalaWide
python train.py --mode test  --model ImpalaWide
python train.py --mode finetune --model ImpalaWide --checkpoint model_checkpoints/X.pt
python train.py --mode resume  --model ImpalaWide --checkpoint model_checkpoints/X.pt
python train.py --mode train --model ImpalaWide --curriculum --curriculum_option 1
python train.py --mode train --model ImpalaWide --async
```

`--model` accepts `1/2/3` or the full name. Omitting `--model` launches interactive selection. `--total_steps N` overrides `num_training_steps`.

---

## async_demo.py

This file is a **prototype sketch** for an LSTM-PPO architecture (`LSTMPolicy`, `LSTMPPOAgent`). The imports it references (`ppo.lstm_policy`, `ppo.lstm_agent`) do not yet exist. It describes the design intent for the upcoming `mu_zero` refactor — treat it as a design document, not runnable code.
