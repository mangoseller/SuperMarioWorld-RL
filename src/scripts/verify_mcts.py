import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from dataclasses import replace
import torch as t
from muzero.config import MuZeroConfig
from muzero.mcts import (
    BatchedGumbelMuZeroSearch,
    GumbelMuZeroSearch,
    InferenceOutput,
    TensorGumbelMuZeroSearch,
    apply_dirichlet_noise,
)


class ToyModel:
    def __init__(self, config):
        self.config = config

    def initial_inference(self, obs=None, state=None):
        return InferenceOutput(
            prior_logits=t.tensor([2.0, 0.0, 0.0]),
            value=t.tensor(0.0),
            state=0,
        )

    def initial_inference_batch(self, obs=None, state=None):
        batch = obs.shape[0] if obs is not None else state.shape[0]
        return InferenceOutput(
            prior_logits=t.tensor([[2.0, 0.0, 0.0]]).repeat(batch, 1),
            value=t.zeros(batch),
            state=t.zeros(batch, 1),
        )

    def recurrent_inference(self, state, action):
        reward = [0.0, 3.0, 1.0][action] if state == 0 else 0.0
        return InferenceOutput(
            prior_logits=t.zeros(self.config.num_actions),
            value=t.tensor(0.0),
            state=state + 1,
            reward=t.tensor(reward),
            discount=t.tensor(0.0),
        )

    def recurrent_inference_batch(self, states, actions):
        rewards = t.tensor([0.0, 3.0, 1.0])[actions.cpu()]
        rewards = t.where(states[:, 0].cpu() == 0, rewards, t.zeros_like(rewards))
        return InferenceOutput(
            prior_logits=t.zeros(actions.shape[0], self.config.num_actions),
            value=t.zeros(actions.shape[0]),
            state=states + 1,
            reward=rewards,
            discount=t.zeros(actions.shape[0]),
        )


def main():
    config = replace(
        MuZeroConfig(),
        num_actions=3,
        mcts_num_simulations=32,
        mcts_max_depth=2,
        mcts_max_num_considered_actions=3,
        mcts_gumbel_scale=0.0,
    )
    search = GumbelMuZeroSearch(config)
    output = search.run(ToyModel(config))

    assert output.action == 1
    assert output.visit_counts[1] > output.visit_counts[0]
    assert output.visit_counts[1] > output.visit_counts[2]
    assert output.action_weights.shape == (config.num_actions,)
    assert t.allclose(output.action_weights.sum(), t.tensor(1.0))
    assert output.action_weights.argmax().item() == 1

    invalid = t.tensor([False, True, False])
    masked = search.run(ToyModel(config), invalid_actions=invalid)
    assert masked.action != 1
    assert masked.action_weights[1].item() == 0.0

    stochastic = search.run(
        ToyModel(config),
        generator=t.Generator().manual_seed(0),
    )
    assert stochastic.visit_counts.sum().item() == config.mcts_num_simulations

    batch_search = BatchedGumbelMuZeroSearch(config)
    batch = batch_search.run_batch(ToyModel(config), obs=t.zeros(4, 1))
    assert batch.action_weights.shape == (4, config.num_actions)
    assert (batch.action_weights.argmax(-1) == 1).all()
    assert t.allclose(batch.action_weights.sum(-1), t.ones(4))

    logits = t.tensor([2.0, 0.0, -1.0])
    invalid = t.tensor([False, False, True])
    noisy = apply_dirichlet_noise(
        logits,
        invalid,
        alpha=0.3,
        fraction=0.5,
        noise=t.tensor([0.0, 1.0]),
    )
    probs = t.softmax(noisy, dim=-1)
    assert t.allclose(probs.sum(), t.tensor(1.0))
    assert probs[2].item() < 1e-6
    assert probs[1] > t.softmax(logits.masked_fill(invalid, -1e9), dim=-1)[1]

    _verify_tensor_search(config)
    _verify_tensor_python_agreement(config)
    _verify_depth_chain()
    _verify_considered_mask_per_row()

    print("MCTS verification passed.")


class DepthChainModel:
    """Toy MDP where reward only appears at depth 2 along action 1.

    state[:, 0] holds the current depth. Initial state is depth 0; the
    recurrent step increments by 1 and pays out reward 5 only when the
    transition was made from depth 1 with action 1.
    """

    def __init__(self, config):
        self.config = config

    def initial_inference_batch(self, obs=None, state=None):
        batch = obs.shape[0] if obs is not None else state.shape[0]
        return InferenceOutput(
            prior_logits=t.zeros(batch, self.config.num_actions),
            value=t.zeros(batch),
            state=t.zeros(batch, 1),
        )

    def recurrent_inference_batch(self, states, actions):
        depth = states[:, 0]
        actions_cpu = actions.cpu()
        reward = t.where(
            (depth.cpu() == 1) & (actions_cpu == 1),
            t.full_like(actions_cpu, 5.0, dtype=t.float32),
            t.zeros_like(actions_cpu, dtype=t.float32),
        )
        next_state = states.clone()
        next_state[:, 0] = depth + 1.0
        return InferenceOutput(
            prior_logits=t.zeros(actions.shape[0], self.config.num_actions),
            value=t.zeros(actions.shape[0]),
            state=next_state,
            reward=reward,
            discount=t.full((actions.shape[0],), 1.0),
        )


def _verify_tensor_search(config):
    search = TensorGumbelMuZeroSearch(config)
    output = search.run_batch(ToyModel(config), obs=t.zeros(4, 1))

    assert output.action_weights.shape == (4, config.num_actions)
    assert t.allclose(output.action_weights.sum(-1), t.ones(4), atol=1e-5)
    assert (output.action_weights.argmax(-1) == 1).all()
    visit_sums = output.visit_counts.sum(-1)
    assert (visit_sums == config.mcts_num_simulations).all()
    assert output.action_weights.device.type == "cpu"

    invalid = t.tensor([False, True, False])
    masked = search.run_batch(
        ToyModel(config),
        obs=t.zeros(2, 1),
        invalid_actions=invalid,
    )
    assert (masked.action_weights[:, 1] == 0).all()
    assert (masked.action_weights.argmax(-1) != 1).all()


def _verify_tensor_python_agreement(config):
    deterministic = replace(config, mcts_gumbel_scale=0.0)
    tensor_search = TensorGumbelMuZeroSearch(deterministic)
    python_search = BatchedGumbelMuZeroSearch(deterministic)

    obs = t.zeros(3, 1)
    tensor_out = tensor_search.run_batch(ToyModel(deterministic), obs=obs)
    python_out = python_search.run_batch(ToyModel(deterministic), obs=obs)

    assert (tensor_out.action_weights.argmax(-1) == python_out.action_weights.argmax(-1)).all()
    assert t.allclose(
        tensor_out.visit_counts.sum(-1),
        python_out.visit_counts.sum(-1),
    )


def _verify_considered_mask_per_row():
    """Regression: row-varying invalid masks must not clobber kept actions.

    Caught a scatter bug where ``masked_fill(~keep, 0)`` collapsed dropped
    top-k slots to index 0, overwriting legitimate True writes when action 0
    was already in the considered set.
    """
    config = replace(
        MuZeroConfig(),
        num_actions=3,
        mcts_num_simulations=8,
        mcts_max_depth=2,
        mcts_max_num_considered_actions=3,
        mcts_gumbel_scale=0.0,
    )
    search = TensorGumbelMuZeroSearch(config)

    # Row 0: all valid → reward action 1 should win.
    # Row 1: action 1 invalid → only valid actions are 0 and 2; action 2
    # gives reward 1 vs action 0's 0, so action 2 should be the argmax even
    # though action 0 is the lowest topk index of the considered set.
    invalid = t.tensor([[False, False, False],
                        [False, True, False]])
    out = search.run_batch(ToyModel(config), obs=t.zeros(2, 1), invalid_actions=invalid)
    assert out.action_weights[0].argmax().item() == 1
    assert out.action_weights[1, 1].item() == 0.0
    assert out.action_weights[1].argmax().item() == 2
    assert out.visit_counts[1, 0] > 0
    assert out.visit_counts[1, 2] > 0


def _verify_depth_chain():
    base = replace(
        MuZeroConfig(),
        num_actions=3,
        mcts_num_simulations=12,
        mcts_max_num_considered_actions=3,
        mcts_gumbel_scale=0.0,
        gamma=1.0,
    )
    deep = replace(base, mcts_max_depth=2)
    shallow = replace(base, mcts_max_depth=1)

    deep_out = TensorGumbelMuZeroSearch(deep).run_batch(
        DepthChainModel(deep), obs=t.zeros(1, 1),
    )
    shallow_out = TensorGumbelMuZeroSearch(shallow).run_batch(
        DepthChainModel(shallow), obs=t.zeros(1, 1),
    )

    deep_q = deep_out.q_values[0]
    shallow_q = shallow_out.q_values[0]
    assert deep_q.max() > 1.0
    assert shallow_q.max() < 1e-6


if __name__ == '__main__':
    main()
