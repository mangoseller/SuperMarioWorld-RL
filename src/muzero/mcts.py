from dataclasses import dataclass
import functools
import math
import torch as t
import torch.nn.functional as F


@dataclass
class InferenceOutput:
    prior_logits: t.Tensor
    value: t.Tensor
    state: object
    reward: t.Tensor = None
    discount: t.Tensor = None


@dataclass
class SearchOutput:
    action: int
    action_weights: t.Tensor
    visit_counts: t.Tensor
    q_values: t.Tensor
    root_value: float


@dataclass
class BatchSearchOutput:
    action_weights: t.Tensor
    visit_counts: t.Tensor
    q_values: t.Tensor
    root_values: t.Tensor


class _Node:
    def __init__(self, state, prior_logits, value):
        self.state = state
        self.prior_logits = prior_logits.detach().float().cpu()
        self.value = float(value.detach().float().cpu().item())
        self.children = {}
        self.visit_counts = t.zeros_like(self.prior_logits, dtype=t.float32)
        self.value_sums = t.zeros_like(self.prior_logits, dtype=t.float32)

    def q_values(self):
        return t.where(
            self.visit_counts > 0,
            self.value_sums / self.visit_counts.clamp_min(1.0),
            t.zeros_like(self.value_sums),
        )


class MuZeroModelAdapter:
    def __init__(self, network, config):
        self.network = network
        self.config = config

    @t.no_grad()
    def initial_inference(self, obs=None, state=None):
        if state is None:
            state = self.network.encoder(obs)
        policy_logits = self.network.policy(state)
        value_logits = self.network.value(state)
        return InferenceOutput(
            prior_logits=policy_logits.squeeze(0),
            value=self.network.value.predict(value_logits).squeeze(0),
            state=state,
        )

    @t.no_grad()
    def initial_inference_batch(self, obs=None, state=None):
        if state is None:
            state = self.network.encoder(obs)
        policy_logits = self.network.policy(state)
        value_logits = self.network.value(state)
        return InferenceOutput(
            prior_logits=policy_logits,
            value=self.network.value.predict(value_logits),
            state=state,
        )

    @t.no_grad()
    def recurrent_inference(self, state, action):
        actions = t.tensor([action], dtype=t.long, device=state.device)
        next_state = self.network.dynamics(state, actions)
        reward_logits = self.network.reward(next_state)
        policy_logits = self.network.policy(next_state)
        value_logits = self.network.value(next_state)
        return InferenceOutput(
            prior_logits=policy_logits.squeeze(0),
            value=self.network.value.predict(value_logits).squeeze(0),
            state=next_state,
            reward=self.network.reward.predict(reward_logits).squeeze(0),
            discount=t.tensor(self.config.gamma, dtype=t.float32),
        )

    @t.no_grad()
    def recurrent_inference_batch(self, states, actions):
        next_states = self.network.dynamics(states, actions.to(states.device))
        reward_logits = self.network.reward(next_states)
        policy_logits = self.network.policy(next_states)
        value_logits = self.network.value(next_states)
        return InferenceOutput(
            prior_logits=policy_logits,
            value=self.network.value.predict(value_logits),
            state=next_states,
            reward=self.network.reward.predict(reward_logits),
            discount=t.full(
                (actions.shape[0],),
                self.config.gamma,
                dtype=t.float32,
                device=next_states.device,
            ),
        )


class GumbelMuZeroSearch:
    def __init__(self, config):
        self.config = config

    def run(self, model, obs=None, state=None, invalid_actions=None,
            num_simulations=None, max_depth=None, generator=None,
            root_dirichlet_alpha=None, root_dirichlet_fraction=0.0):
        num_simulations = self.config.mcts_num_simulations if num_simulations is None else num_simulations
        max_depth = self.config.mcts_max_depth if max_depth is None else max_depth
        if num_simulations <= 0:
            raise ValueError("num_simulations must be positive")

        root_out = model.initial_inference(obs=obs, state=state)
        root = _Node(root_out.state, root_out.prior_logits, root_out.value)
        invalid_actions = self._invalid_actions(invalid_actions, root.prior_logits.shape[0])
        root.prior_logits = root.prior_logits.masked_fill(invalid_actions, -1e9)
        if root_dirichlet_alpha is not None and root_dirichlet_fraction > 0:
            root.prior_logits = self._add_dirichlet_noise(
                root.prior_logits,
                invalid_actions,
                root_dirichlet_alpha,
                root_dirichlet_fraction,
            )

        gumbel = self._gumbel(root.prior_logits.shape, generator) * self.config.mcts_gumbel_scale
        gumbel = gumbel.masked_fill(invalid_actions, -1e9)
        num_valid = int((~invalid_actions).sum().item())
        num_considered = min(self.config.mcts_max_num_considered_actions, num_valid)
        considered_mask = self._considered_actions(root.prior_logits, gumbel, invalid_actions, num_considered)
        considered_visits = _sequence_of_considered_visits(num_considered, num_simulations)

        for simulation in range(num_simulations):
            value = self._simulate(
                model,
                root,
                gumbel,
                invalid_actions,
                considered_mask,
                considered_visits[simulation],
                max_depth,
            )
            if not math.isfinite(value):
                raise FloatingPointError("non-finite MCTS backup value")

        completed_q = self._completed_q(root)
        action_weights = F.softmax(root.prior_logits + completed_q, dim=-1)
        action_weights = action_weights.masked_fill(invalid_actions, 0.0)
        action_weights = action_weights / action_weights.sum().clamp_min(1e-8)

        considered_visit = root.visit_counts.max()
        final_score = self._root_score(root, gumbel, invalid_actions, considered_mask, considered_visit)
        action = int(final_score.argmax().item())
        return SearchOutput(
            action=action,
            action_weights=action_weights,
            visit_counts=root.visit_counts.clone(),
            q_values=root.q_values(),
            root_value=root.value,
        )

    def _simulate(self, model, root, gumbel, root_invalid_actions, considered_mask, considered_visit, max_depth):
        node = root
        path = []
        depth = 0

        while True:
            if depth == 0:
                action = self._select_root(node, gumbel, root_invalid_actions, considered_mask, considered_visit)
            else:
                action = self._select_interior(node)

            if action not in node.children:
                out = model.recurrent_inference(node.state, action)
                child = _Node(out.state, out.prior_logits, out.value)
                reward = float(out.reward.detach().float().cpu().item())
                discount = float(out.discount.detach().float().cpu().item())
                node.children[action] = (child, reward, discount)
                path.append((node, action, reward, discount))
                value = child.value
                break

            child, reward, discount = node.children[action]
            path.append((node, action, reward, discount))
            node = child
            depth += 1
            if max_depth is not None and depth >= max_depth:
                value = node.value
                break

        self._backup(path, value)
        return value

    def _backup(self, path, value):
        for node, action, reward, discount in reversed(path):
            value = reward + discount * value
            node.visit_counts[action] += 1.0
            node.value_sums[action] += value

    def _select_root(self, node, gumbel, invalid_actions, considered_mask, considered_visit):
        score = self._root_score(node, gumbel, invalid_actions, considered_mask, considered_visit)
        return int(score.argmax().item())

    def _root_score(self, node, gumbel, invalid_actions, considered_mask, considered_visit):
        logits = node.prior_logits - node.prior_logits.max()
        completed_q = self._completed_q(node)
        eligible = node.visit_counts == float(considered_visit)
        score = gumbel + logits + completed_q
        score = score.masked_fill(invalid_actions | ~considered_mask | ~eligible, -float("inf"))
        if t.isneginf(score).all():
            score = gumbel + logits + completed_q
            score = score.masked_fill(invalid_actions | ~considered_mask, -float("inf"))
        return score

    def _select_interior(self, node):
        completed_q = self._completed_q(node)
        probs = F.softmax(node.prior_logits + completed_q, dim=-1)
        score = probs - node.visit_counts / (1.0 + node.visit_counts.sum())
        return int(score.argmax().item())

    def _completed_q(self, node):
        q_values = node.q_values()
        visits = node.visit_counts
        prior = F.softmax(node.prior_logits, dim=-1)
        visited = visits > 0
        if visited.any():
            prob_mass = prior[visited].sum().clamp_min(t.finfo(prior.dtype).tiny)
            weighted_q = (prior[visited] * q_values[visited] / prob_mass).sum()
            mixed_value = (node.value + visits.sum() * weighted_q) / (visits.sum() + 1.0)
        else:
            mixed_value = t.tensor(node.value, dtype=t.float32)
        completed = t.where(visited, q_values, mixed_value)
        min_value = completed.min()
        max_value = completed.max()
        normalized = (completed - min_value) / (max_value - min_value).clamp_min(1e-8)
        visit_scale = self.config.mcts_maxvisit_init + visits.max()
        return visit_scale * self.config.mcts_value_scale * normalized

    def _invalid_actions(self, invalid_actions, num_actions):
        if invalid_actions is None:
            return t.zeros(num_actions, dtype=t.bool)
        invalid_actions = invalid_actions.detach().cpu().bool()
        if invalid_actions.shape != (num_actions,):
            raise ValueError("invalid_actions must have shape (num_actions,)")
        if invalid_actions.all():
            raise ValueError("at least one action must be valid")
        return invalid_actions

    def _considered_actions(self, prior_logits, gumbel, invalid_actions, num_considered):
        score = prior_logits + gumbel
        score = score.masked_fill(invalid_actions, -float("inf"))
        idx = t.topk(score, k=num_considered).indices
        mask = t.zeros_like(invalid_actions)
        mask[idx] = True
        return mask

    def _gumbel(self, shape, generator):
        uniform = t.rand(shape, generator=generator).clamp(1e-8, 1.0 - 1e-8)
        return -t.log(-t.log(uniform))

    def _add_dirichlet_noise(self, prior_logits, invalid_actions, alpha, fraction):
        valid = ~invalid_actions
        concentration = t.full((int(valid.sum().item()),), float(alpha), dtype=t.float32)
        noise = t.distributions.Dirichlet(concentration).sample()
        return apply_dirichlet_noise(prior_logits, invalid_actions, alpha, fraction, noise)


class BatchedGumbelMuZeroSearch(GumbelMuZeroSearch):
    def run_batch(self, model, obs=None, state=None, invalid_actions=None,
                  num_simulations=None, max_depth=None, generator=None,
                  root_dirichlet_alphas=None, root_dirichlet_fractions=None):
        num_simulations = self.config.mcts_num_simulations if num_simulations is None else num_simulations
        max_depth = self.config.mcts_max_depth if max_depth is None else max_depth
        if num_simulations <= 0:
            raise ValueError("num_simulations must be positive")

        root_out = model.initial_inference_batch(obs=obs, state=state)
        batch_size = root_out.state.shape[0]
        invalid_actions = self._batch_invalid_actions(
            invalid_actions,
            batch_size,
            root_out.prior_logits.shape[-1],
        )

        roots, gumbels, considered_masks, visit_sequences = [], [], [], []
        for i in range(batch_size):
            root = _Node(
                root_out.state[i:i + 1],
                root_out.prior_logits[i],
                root_out.value[i],
            )
            root.prior_logits = root.prior_logits.masked_fill(invalid_actions[i], -1e9)
            alpha = self._row_value(root_dirichlet_alphas, i)
            fraction = self._row_value(root_dirichlet_fractions, i, default=0.0)
            if alpha is not None and fraction > 0:
                root.prior_logits = self._add_dirichlet_noise(
                    root.prior_logits,
                    invalid_actions[i],
                    alpha,
                    fraction,
                )

            gumbel = self._gumbel(root.prior_logits.shape, generator) * self.config.mcts_gumbel_scale
            gumbel = gumbel.masked_fill(invalid_actions[i], -1e9)
            num_valid = int((~invalid_actions[i]).sum().item())
            num_considered = min(self.config.mcts_max_num_considered_actions, num_valid)
            roots.append(root)
            gumbels.append(gumbel)
            considered_masks.append(
                self._considered_actions(root.prior_logits, gumbel, invalid_actions[i], num_considered)
            )
            visit_sequences.append(_sequence_of_considered_visits(num_considered, num_simulations))

        for simulation in range(num_simulations):
            leaf_jobs = []
            for i, root in enumerate(roots):
                job = self._select_leaf(
                    root,
                    gumbels[i],
                    invalid_actions[i],
                    considered_masks[i],
                    visit_sequences[i][simulation],
                    max_depth,
                )
                if job["leaf"]:
                    leaf_jobs.append((i, job))
                else:
                    self._backup(job["path"], job["value"])

            if leaf_jobs:
                states = t.cat([job["node"].state for _, job in leaf_jobs], dim=0)
                actions = t.tensor(
                    [job["action"] for _, job in leaf_jobs],
                    dtype=t.long,
                    device=states.device,
                )
                out = model.recurrent_inference_batch(states, actions)
                for row, (_root_idx, job) in enumerate(leaf_jobs):
                    child = _Node(
                        out.state[row:row + 1],
                        out.prior_logits[row],
                        out.value[row],
                    )
                    reward = float(out.reward[row].detach().float().cpu().item())
                    discount = float(out.discount[row].detach().float().cpu().item())
                    job["node"].children[job["action"]] = (child, reward, discount)
                    path = job["path"] + [(job["node"], job["action"], reward, discount)]
                    value = child.value
                    if not math.isfinite(value):
                        raise FloatingPointError("non-finite MCTS backup value")
                    self._backup(path, value)

        action_weights, visit_counts, q_values, root_values = [], [], [], []
        for i, root in enumerate(roots):
            completed_q = self._completed_q(root)
            weights = F.softmax(root.prior_logits + completed_q, dim=-1)
            weights = weights.masked_fill(invalid_actions[i], 0.0)
            weights = weights / weights.sum().clamp_min(1e-8)
            action_weights.append(weights)
            visit_counts.append(root.visit_counts.clone())
            q_values.append(root.q_values())
            root_values.append(root.value)

        return BatchSearchOutput(
            action_weights=t.stack(action_weights),
            visit_counts=t.stack(visit_counts),
            q_values=t.stack(q_values),
            root_values=t.tensor(root_values, dtype=t.float32),
        )

    def _select_leaf(self, root, gumbel, root_invalid_actions, considered_mask,
                     considered_visit, max_depth):
        node = root
        path = []
        depth = 0

        while True:
            if depth == 0:
                action = self._select_root(
                    node,
                    gumbel,
                    root_invalid_actions,
                    considered_mask,
                    considered_visit,
                )
            else:
                action = self._select_interior(node)

            if action not in node.children:
                return {
                    "leaf": True,
                    "node": node,
                    "action": action,
                    "path": path,
                }

            child, reward, discount = node.children[action]
            path.append((node, action, reward, discount))
            node = child
            depth += 1
            if max_depth is not None and depth >= max_depth:
                return {"leaf": False, "path": path, "value": node.value}

    def _batch_invalid_actions(self, invalid_actions, batch_size, num_actions):
        if invalid_actions is None:
            return t.zeros(batch_size, num_actions, dtype=t.bool)
        invalid_actions = invalid_actions.detach().cpu().bool()
        if invalid_actions.shape == (num_actions,):
            invalid_actions = invalid_actions.unsqueeze(0).expand(batch_size, num_actions)
        if invalid_actions.shape != (batch_size, num_actions):
            raise ValueError("invalid_actions must have shape (B, num_actions)")
        if invalid_actions.all(dim=-1).any():
            raise ValueError("each root must have at least one valid action")
        return invalid_actions

    def _row_value(self, values, row, default=None):
        if values is None:
            return default
        if isinstance(values, (int, float)):
            return float(values)
        return float(values[row])


def apply_dirichlet_noise(prior_logits, invalid_actions, alpha, fraction, noise):
    valid = ~invalid_actions
    probs = F.softmax(prior_logits, dim=-1)
    noise = noise.float()
    if noise.shape != (int(valid.sum().item()),):
        raise ValueError("noise must have one entry per valid action")
    noise = noise / noise.sum().clamp_min(1e-8)
    mixed = probs.clone()
    mixed[valid] = (1.0 - fraction) * probs[valid] + fraction * noise
    mixed = mixed / mixed.sum().clamp_min(1e-8)
    return mixed.clamp_min(1e-8).log().masked_fill(invalid_actions, -1e9)


@functools.lru_cache(maxsize=128)
def _sequence_of_considered_visits(max_num_considered_actions, num_simulations):
    if max_num_considered_actions <= 1:
        return tuple(range(num_simulations))

    log2max = int(math.ceil(math.log2(max_num_considered_actions)))
    sequence = []
    visits = [0] * max_num_considered_actions
    num_considered = max_num_considered_actions
    while len(sequence) < num_simulations:
        num_extra_visits = max(1, int(num_simulations / (log2max * num_considered)))
        for _ in range(num_extra_visits):
            sequence.extend(visits[:num_considered])
            for i in range(num_considered):
                visits[i] += 1
        num_considered = max(2, num_considered // 2)
    return tuple(sequence[:num_simulations])


def _sequence_of_considered_visits_tensor(num_considered_per_row, num_simulations,
                                          device):
    rows = [
        _sequence_of_considered_visits(int(nc), num_simulations)
        for nc in num_considered_per_row
    ]
    return t.tensor(rows, dtype=t.float32, device=device)


class TensorGumbelMuZeroSearch:
    def __init__(self, config):
        self.config = config

    def run_batch(self, model, obs=None, state=None, invalid_actions=None,
                  num_simulations=None, max_depth=None, generator=None,
                  root_dirichlet_alphas=None, root_dirichlet_fractions=None):
        num_simulations = self.config.mcts_num_simulations if num_simulations is None else num_simulations
        max_depth = self.config.mcts_max_depth if max_depth is None else max_depth
        if num_simulations <= 0:
            raise ValueError("num_simulations must be positive")

        root_out = model.initial_inference_batch(obs=obs, state=state)
        root_state = root_out.state
        device = root_state.device
        batch_size = root_state.shape[0]
        num_actions = root_out.prior_logits.shape[-1]
        max_nodes = 1 + num_simulations

        invalid_actions = self._batch_invalid_actions(
            invalid_actions, batch_size, num_actions, device,
        )

        states = t.zeros(
            (batch_size, max_nodes) + tuple(root_state.shape[1:]),
            dtype=root_state.dtype,
            device=device,
        )
        priors = t.zeros(batch_size, max_nodes, num_actions, dtype=t.float32, device=device)
        values = t.zeros(batch_size, max_nodes, dtype=t.float32, device=device)
        rewards = t.zeros(batch_size, max_nodes, dtype=t.float32, device=device)
        discounts = t.zeros(batch_size, max_nodes, dtype=t.float32, device=device)
        visit_counts = t.zeros(batch_size, max_nodes, num_actions, dtype=t.float32, device=device)
        value_sums = t.zeros(batch_size, max_nodes, num_actions, dtype=t.float32, device=device)
        children = t.full((batch_size, max_nodes, num_actions), -1, dtype=t.long, device=device)
        node_count = t.ones(batch_size, dtype=t.long, device=device)

        states[:, 0] = root_state
        root_logits = root_out.prior_logits.float().masked_fill(invalid_actions, -1e9)
        root_logits = self._maybe_apply_dirichlet(
            root_logits, invalid_actions,
            root_dirichlet_alphas, root_dirichlet_fractions,
        )
        priors[:, 0] = root_logits
        values[:, 0] = root_out.value.float()
        discounts[:, 0] = 1.0

        gumbel = self._gumbel((batch_size, num_actions), generator, device) * self.config.mcts_gumbel_scale
        gumbel = gumbel.masked_fill(invalid_actions, -1e9)

        num_valid = (~invalid_actions).sum(dim=-1)
        max_considered = self.config.mcts_max_num_considered_actions
        num_considered = t.clamp(num_valid, max=max_considered)
        considered_mask = self._considered_actions_tensor(
            root_logits, gumbel, invalid_actions, num_considered,
        )
        visit_seqs = _sequence_of_considered_visits_tensor(
            num_considered.tolist(), num_simulations, device=device,
        )

        batch_idx = t.arange(batch_size, device=device)

        for sim in range(num_simulations):
            considered_visit = visit_seqs[:, sim]
            self._run_simulation(
                model,
                batch_idx,
                states,
                priors,
                values,
                rewards,
                discounts,
                visit_counts,
                value_sums,
                children,
                node_count,
                invalid_actions,
                gumbel,
                considered_mask,
                considered_visit,
                max_depth,
                num_actions,
            )

        completed_q = self._completed_q_tensor(
            priors[:, 0], values[:, 0], visit_counts[:, 0], value_sums[:, 0],
        )
        weights = F.softmax(priors[:, 0] + completed_q, dim=-1)
        weights = weights.masked_fill(invalid_actions, 0.0)
        weights = weights / weights.sum(dim=-1, keepdim=True).clamp_min(1e-8)

        root_visits = visit_counts[:, 0]
        root_q = t.where(
            root_visits > 0,
            value_sums[:, 0] / root_visits.clamp_min(1.0),
            t.zeros_like(value_sums[:, 0]),
        )

        return BatchSearchOutput(
            action_weights=weights.detach().cpu(),
            visit_counts=root_visits.detach().cpu(),
            q_values=root_q.detach().cpu(),
            root_values=values[:, 0].detach().cpu(),
        )

    def _run_simulation(self, model, batch_idx, states, priors, values, rewards,
                        discounts, visit_counts, value_sums, children, node_count,
                        invalid_actions, gumbel, considered_mask, considered_visit,
                        max_depth, num_actions):
        batch_size = batch_idx.shape[0]
        device = batch_idx.device

        node = t.zeros(batch_size, dtype=t.long, device=device)
        active = t.ones(batch_size, dtype=t.bool, device=device)
        path_parents = t.zeros(batch_size, max_depth, dtype=t.long, device=device)
        path_actions = t.zeros(batch_size, max_depth, dtype=t.long, device=device)
        path_children = t.zeros(batch_size, max_depth, dtype=t.long, device=device)
        path_len = t.zeros(batch_size, dtype=t.long, device=device)
        needs_expansion = t.zeros(batch_size, dtype=t.bool, device=device)
        expand_parent = t.zeros(batch_size, dtype=t.long, device=device)
        expand_action = t.zeros(batch_size, dtype=t.long, device=device)

        for d in range(max_depth):
            if d == 0:
                action = self._select_root_tensor(
                    priors[:, 0], values[:, 0], visit_counts[:, 0], value_sums[:, 0],
                    gumbel, considered_mask, considered_visit, invalid_actions,
                )
            else:
                node_priors = priors[batch_idx, node]
                node_values = values[batch_idx, node]
                node_visits = visit_counts[batch_idx, node]
                node_value_sums = value_sums[batch_idx, node]
                action = self._select_interior_tensor(
                    node_priors, node_values, node_visits, node_value_sums,
                )

            child_slice = children[batch_idx, node]
            child = child_slice.gather(1, action.unsqueeze(1)).squeeze(1)
            is_leaf = active & (child < 0)
            advance = active & ~is_leaf

            path_parents[:, d] = t.where(active, node, path_parents[:, d])
            path_actions[:, d] = t.where(active, action, path_actions[:, d])
            path_children[:, d] = t.where(advance, child, path_children[:, d])
            path_len = t.where(active, path_len + 1, path_len)

            needs_expansion = needs_expansion | is_leaf
            expand_parent = t.where(is_leaf, node, expand_parent)
            expand_action = t.where(is_leaf, action, expand_action)

            node = t.where(advance, child, node)
            active = advance

        leaf_value = t.zeros(batch_size, dtype=t.float32, device=device)
        if bool(needs_expansion.any()):
            expand_idx = needs_expansion.nonzero(as_tuple=True)[0]
            parent_states = states[expand_idx, expand_parent[expand_idx]]
            actions_to_take = expand_action[expand_idx]

            out = model.recurrent_inference_batch(parent_states, actions_to_take)

            new_node_id = node_count[expand_idx].clone()
            node_count[expand_idx] = new_node_id + 1

            states[expand_idx, new_node_id] = out.state
            priors[expand_idx, new_node_id] = out.prior_logits.float()
            values[expand_idx, new_node_id] = out.value.float()
            rewards[expand_idx, new_node_id] = out.reward.float()
            discounts[expand_idx, new_node_id] = out.discount.float()
            children[expand_idx, expand_parent[expand_idx], expand_action[expand_idx]] = new_node_id

            last_d = path_len[expand_idx] - 1
            path_children[expand_idx, last_d] = new_node_id

            leaf_value[expand_idx] = out.value.float()

        traversed_to_max = ~needs_expansion & (path_len > 0)
        if bool(traversed_to_max.any()):
            trav_idx = traversed_to_max.nonzero(as_tuple=True)[0]
            leaf_value[trav_idx] = values[trav_idx, node[trav_idx]]

        if not t.isfinite(leaf_value).all():
            raise FloatingPointError("non-finite MCTS backup value")

        v = leaf_value
        for d in reversed(range(max_depth)):
            edge_active = (d < path_len)
            parent = path_parents[:, d]
            action = path_actions[:, d]
            child = path_children[:, d]

            r = rewards[batch_idx, child]
            g = discounts[batch_idx, child]
            backed = r + g * v
            v = t.where(edge_active, backed, v)

            scale = edge_active.float()
            visit_counts[batch_idx, parent, action] += scale
            value_sums[batch_idx, parent, action] += scale * v

    def _select_root_tensor(self, root_priors, root_values, root_visits,
                            root_value_sums, gumbel, considered_mask,
                            considered_visit, invalid_actions):
        logits = root_priors - root_priors.max(dim=-1, keepdim=True).values
        completed_q = self._completed_q_tensor(
            root_priors, root_values, root_visits, root_value_sums,
        )
        eligible = root_visits == considered_visit.unsqueeze(-1)
        raw = gumbel + logits + completed_q
        strict_mask = invalid_actions | ~considered_mask | ~eligible
        score = raw.masked_fill(strict_mask, -float("inf"))

        all_neg = score.isneginf().all(dim=-1, keepdim=True)
        relaxed_mask = invalid_actions | ~considered_mask
        relaxed = raw.masked_fill(relaxed_mask, -float("inf"))
        score = t.where(all_neg, relaxed, score)
        return score.argmax(dim=-1)

    def _select_interior_tensor(self, priors, values, visits, value_sums):
        completed_q = self._completed_q_tensor(priors, values, visits, value_sums)
        probs = F.softmax(priors + completed_q, dim=-1)
        score = probs - visits / (1.0 + visits.sum(dim=-1, keepdim=True))
        return score.argmax(dim=-1)

    def _completed_q_tensor(self, priors, values, visits, value_sums):
        prior = F.softmax(priors, dim=-1)
        visited = visits > 0
        q_values = t.where(
            visited,
            value_sums / visits.clamp_min(1.0),
            t.zeros_like(value_sums),
        )
        prob_mass = (prior * visited.float()).sum(dim=-1).clamp_min(t.finfo(prior.dtype).tiny)
        weighted_q = (prior * q_values * visited.float()).sum(dim=-1) / prob_mass
        visit_total = visits.sum(dim=-1)
        any_visited = visited.any(dim=-1)
        mixed = t.where(
            any_visited,
            (values + visit_total * weighted_q) / (visit_total + 1.0),
            values,
        )
        completed = t.where(visited, q_values, mixed.unsqueeze(-1))
        min_v = completed.min(dim=-1, keepdim=True).values
        max_v = completed.max(dim=-1, keepdim=True).values
        normalized = (completed - min_v) / (max_v - min_v).clamp_min(1e-8)
        visit_scale = self.config.mcts_maxvisit_init + visits.max(dim=-1).values
        return (visit_scale * self.config.mcts_value_scale).unsqueeze(-1) * normalized

    def _considered_actions_tensor(self, prior_logits, gumbel, invalid_actions,
                                   num_considered):
        score = (prior_logits + gumbel).masked_fill(invalid_actions, -float("inf"))
        max_k = int(num_considered.max().item()) if num_considered.numel() else 0
        if max_k <= 0:
            return t.zeros_like(invalid_actions)
        batch_size, num_actions = invalid_actions.shape
        topk = score.topk(k=max_k, dim=-1).indices
        rank_values = t.arange(max_k, device=score.device).unsqueeze(0).expand(batch_size, -1)
        ranks = t.full((batch_size, num_actions), num_actions, dtype=t.long, device=score.device)
        ranks.scatter_(1, topk, rank_values)
        return ranks < num_considered.unsqueeze(-1)

    def _gumbel(self, shape, generator, device):
        if generator is not None and generator.device != device:
            uniform = t.rand(shape, generator=generator).clamp(1e-8, 1.0 - 1e-8).to(device)
        else:
            uniform = t.rand(shape, generator=generator, device=device).clamp(1e-8, 1.0 - 1e-8)
        return -t.log(-t.log(uniform))

    def _batch_invalid_actions(self, invalid_actions, batch_size, num_actions, device):
        if invalid_actions is None:
            return t.zeros(batch_size, num_actions, dtype=t.bool, device=device)
        invalid_actions = invalid_actions.detach().bool().to(device)
        if invalid_actions.shape == (num_actions,):
            invalid_actions = invalid_actions.unsqueeze(0).expand(batch_size, num_actions).contiguous()
        if invalid_actions.shape != (batch_size, num_actions):
            raise ValueError("invalid_actions must have shape (B, num_actions)")
        if invalid_actions.all(dim=-1).any():
            raise ValueError("each root must have at least one valid action")
        return invalid_actions

    def _maybe_apply_dirichlet(self, root_logits, invalid_actions, alphas, fractions):
        if alphas is None:
            return root_logits
        out = root_logits.clone()
        batch_size = root_logits.shape[0]
        for i in range(batch_size):
            alpha = alphas if isinstance(alphas, (int, float)) else alphas[i]
            fraction = (
                fractions if fractions is None or isinstance(fractions, (int, float))
                else fractions[i]
            )
            if alpha is None or fraction is None or float(fraction) <= 0:
                continue
            valid_count = int((~invalid_actions[i]).sum().item())
            concentration = t.full(
                (valid_count,), float(alpha), dtype=t.float32, device=root_logits.device,
            )
            noise = t.distributions.Dirichlet(concentration).sample()
            out[i] = apply_dirichlet_noise(
                out[i].cpu(),
                invalid_actions[i].cpu(),
                float(alpha),
                float(fraction),
                noise.cpu(),
            ).to(root_logits.device)
        return out
