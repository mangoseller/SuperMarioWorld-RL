from collections import deque
from dataclasses import dataclass, field
import numpy as np

from curriculum import ALL_LEVELS


@dataclass
class _LevelState:
    staleness: int = 0
    episodes: int = 0
    value_errors: deque = field(default_factory=deque)
    solved: bool = False
    triggered_expansion: bool = False


class PLRSampler:
    def __init__(self, config, level_pool=None):
        pool = list(level_pool) if level_pool is not None else list(ALL_LEVELS)
        if not pool:
            raise ValueError("PLRSampler requires at least one level")
        n = min(config.plr_initial_levels, len(pool))
        self.active_levels = pool[:n]
        self.inactive_levels = pool[n:]
        self._state = {
            level: _LevelState(value_errors=deque(maxlen=config.plr_score_window))
            for level in pool
        }
        self.config = config

    def record_episode(self, level, episode_return):
        state = self._state.get(level)
        if state is None:
            return
        state.episodes += 1
        state.staleness = 0
        if episode_return >= self.config.plr_solved_return:
            state.solved = True
        for l in self.active_levels:
            if l != level and l in self._state:
                self._state[l].staleness += 1

    def record_value_error(self, level, mean_value_error):
        state = self._state.get(level)
        if state is None:
            return
        state.value_errors.append(float(mean_value_error))

    def state_dict(self):
        return {
            "active_levels": list(self.active_levels),
            "inactive_levels": list(self.inactive_levels),
            "state": {
                level: {
                    "staleness": state.staleness,
                    "episodes": state.episodes,
                    "value_errors": list(state.value_errors),
                    "solved": state.solved,
                    "triggered_expansion": state.triggered_expansion,
                }
                for level, state in self._state.items()
            },
        }

    def load_state_dict(self, state_dict):
        self.active_levels = list(state_dict.get("active_levels", self.active_levels))
        self.inactive_levels = list(state_dict.get("inactive_levels", self.inactive_levels))
        state = {}
        for level, payload in state_dict.get("state", {}).items():
            state[level] = _LevelState(
                staleness=int(payload.get("staleness", 0)),
                episodes=int(payload.get("episodes", 0)),
                value_errors=deque(
                    payload.get("value_errors", []),
                    maxlen=self.config.plr_score_window,
                ),
                solved=bool(payload.get("solved", False)),
                triggered_expansion=bool(payload.get("triggered_expansion", False)),
            )
        if state:
            self._state = state

    def maybe_add_levels(self):
        if not self.inactive_levels or self.config.plr_levels_per_addition <= 0:
            return False
        triggers = [
            l for l in self.active_levels
            if self._state[l].solved and not self._state[l].triggered_expansion
        ]
        if not triggers:
            return False
        n = self.config.plr_levels_per_addition
        added = self.inactive_levels[:n]
        self.inactive_levels = self.inactive_levels[n:]
        self.active_levels = self.active_levels + added
        for level in added:
            if level not in self._state:
                self._state[level] = _LevelState(
                    value_errors=deque(maxlen=self.config.plr_score_window)
                )
        for l in triggers:
            self._state[l].triggered_expansion = True
        return True

    def sampling_weights(self):
        scores = self._scores()
        raw = np.array([
            scores[l] ** (1.0 / self.config.plr_beta)
            * (self._state[l].staleness + 1) ** self.config.plr_rho
            for l in self.active_levels
        ], dtype=np.float64)
        total = raw.sum()
        if total <= 0:
            return [1.0 / len(self.active_levels)] * len(self.active_levels)
        return (raw / total).tolist()

    def tiered_sims(self, grad_step):
        config = self.config
        if grad_step < config.mcts_ramp_start:
            return {level: config.mcts_num_simulations for level in self.active_levels}

        scores = self._scores()
        score_vals = [scores[l] for l in self.active_levels]

        if grad_step >= config.mcts_ramp_end:
            return self._tiered_by_quartile(score_vals)

        frac = (grad_step - config.mcts_ramp_start) / max(
            config.mcts_ramp_end - config.mcts_ramp_start, 1
        )
        ramp_sims = max(
            config.mcts_num_simulations,
            round(config.mcts_num_simulations + frac * (config.mcts_sims_frontier - config.mcts_num_simulations)),
        )
        return {level: int(ramp_sims) for level in self.active_levels}

    def current_max_steps(self, grad_step):
        if grad_step < self.config.max_steps_early_end_step:
            return int(
                self.config.self_play_max_episode_steps * self.config.max_steps_early_multiplier
            )
        return self.config.self_play_max_episode_steps

    def get_broadcast_info(self, grad_step):
        return {
            "active_levels": list(self.active_levels),
            "level_weights": self.sampling_weights(),
            "level_sims": self.tiered_sims(grad_step),
            "max_steps": self.current_max_steps(grad_step),
        }

    def level_stats(self):
        result = {}
        scores = self._scores()
        for level in self.active_levels:
            s = self._state[level]
            result[level] = {
                "score": scores[level],
                "staleness": s.staleness,
                "episodes": s.episodes,
                "solved": s.solved,
            }
        return result

    def _scores(self):
        result = {}
        for level in self.active_levels:
            errors = self._state[level].value_errors
            result[level] = float(np.mean(errors)) if errors else 1.0
        return result

    def _tiered_by_quartile(self, score_vals):
        config = self.config
        if len(score_vals) < 4:
            return {level: config.mcts_sims_frontier for level in self.active_levels}
        arr = np.array(score_vals, dtype=np.float64)
        q75 = float(np.percentile(arr, 75))
        q25 = float(np.percentile(arr, 25))
        result = {}
        for level, sv in zip(self.active_levels, score_vals):
            if sv >= q75:
                result[level] = config.mcts_sims_frontier
            elif sv >= q25:
                result[level] = config.mcts_sims_mid
            else:
                result[level] = config.mcts_sims_easy
        return result
