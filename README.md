# Super Mario World — Reinforcement Learning

An implementation of **Proximal Policy Optimization (PPO)** in PyTorch, trained to play *Super Mario World* (SNES) directly from pixels.

The agent successfully completed two previously unseen levels, including the challenging Vanilla Dome 5. While successful completions were rare and generalization remained inconsistent, these results demonstrate that the learned policy could transfer beyond its training environments.

### Vanilla Dome 5 — Zero-shot completion

https://github.com/user-attachments/assets/0818a76c-0c20-4c9b-898f-9a591c901d13

### Yoshi's Island 1 — Zero-shot completion

https://github.com/user-attachments/assets/0dbeaa94-010e-44ab-a91e-5b492f1a5f8d

## Implementation

I implemented PPO from scratch in PyTorch, including the clipped surrogate objective, Generalized Advantage Estimation, and the rollout and optimization pipeline. The agent learns from stacked grayscale frames, selecting actions from a discrete set of SNES controller inputs. Emulator RAM is used to construct rewards, but the policy itself receives only visual observations.

I experimented with three convolutional actor-critic architectures, progressing from a small CNN to wider [IMPALA](https://arxiv.org/abs/1802.01561)-inspired residual networks incorporating Fourier positional embeddings, spatial softmax, and an auxiliary pixel-change prediction objective. The latter was intended to introduce an inductive bias toward dynamic regions of the environment, such as moving enemies, rather than largely static background elements.

Training is parallelized across emulator instances and uses experiment tracking through Weights & Biases.

## Reward Hacking

A few notable instances of **reward hacking** were encountered during my experiments:

### Farming movement reward

An early variant of the reward function incentivized horizontal movement naively, assigning positive reward to any positive change in the x-axis. 

```json
{
  "reward": {
    "variables": {
      "xpos": {
        "measurement": "delta",
        "reward": 0.015,
        "penalty": 0.0
      }
    }
  }
}
```
The intention was to provide a dense learning signal for progress through the level, rather than relying exclusively on the extremely sparse reward associated with level completion.

https://github.com/user-attachments/assets/924fa10f-fb9d-4ca5-858b-e6164de54b88

Instead, agents learned to repeatedly traverse the same ground, receiving positive reward for rightward movement without an equivalent penalty for moving left, as shown above.

I addressed this by tracking the maximum horizontal progress achieved during an episode and rewarding newly reached positions rather than movement alone.

### Exploiting the game clock

A surprising exploit involved an inactivity penalty tied to the in-game timer.

Agents learned to exploit message blocks that temporarily pause the game clock, avoiding time penalties while remaining stationary to avoid facing a difficult jump in the Yoshi's Island 2 level.

```json
{
  "reward": {
    "variables": {
      "time_remaining": {
        "measurement": "delta",
        "penalty": 0.01
      }
    }
  }
}
```
Because the penalty depended on changes in the countdown timer, no penalty accumulated while the clock was paused. In one instance, this produced an episode lasting over 20 minutes, ultimately crashing the training process.


https://github.com/user-attachments/assets/9bbe6a29-f34e-42cd-9aa4-0a5b8eb974e0


To fix this, the penalty was changed to depend on environment steps rather than the in-game timer.

## Code

The implementation is available in [`src/`](src/), using PyTorch, TorchRL, Gymnasium, and Stable-Retro.
