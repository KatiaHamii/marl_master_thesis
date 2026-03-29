"""
agents/qlearning.py — Q-Learning agent
All hyperparameters come from config.yaml via utils.load_config()
"""

import numpy as np
from utils import load_config, get, set_seed, EpsilonSchedule


class QLearningAgent:
    """
    Tabular Q-Learning.

    CartPole:   continuous state → discretised into bins → table index
    FrozenLake: state is already a single integer tile index → used directly
    """

    def __init__(self, cfg: dict, state_dim: int, action_dim: int):
        self.cfg = cfg
        self.action_dim = action_dim
        self.lr = get(cfg, "qlearning", "learning_rate")
        self.discount = get(cfg, "training", "discount")
        self.bins = get(cfg, "qlearning", "bins")
        self.epsilon = EpsilonSchedule(cfg)
        self.env_name = get(cfg, "environment", "name")

        if self.env_name == "FrozenLake-v1":
            # State is already a discrete tile index — Q-table is (n_tiles, n_actions)
            # state_dim passed in as 1, but we read n_states from frozenlake config
            size = get(cfg, "frozenlake", "size", default=4)
            n_states = size * size
            self.q_table = np.zeros([n_states, action_dim])
            self._mode = "discrete"
        else:
            # CartPole: continuous → discretise into bins
            self.bins = get(cfg, "qlearning", "bins")
            self._low = [-2.4, -3.0, -0.25, -3.5]
            self._high = [2.4, 3.0, 0.25, 3.5]
            self.q_table = np.zeros([self.bins] * 4 + [action_dim])
            self._mode = "continuous"

    # ── state preprocessing ──────────────────────────────────
    def _to_index(self, obs):
        if self._mode == "discrete":
            return int(obs)  # FrozenLake tile index used directly
        # CartPole discretisation
        indices = []
        for i, val in enumerate(obs):
            lo, hi = self._low[i], self._high[i]
            bucket = int((np.clip(val, lo, hi) - lo) / (hi - lo) * self.bins)
            indices.append(min(bucket, self.bins - 1))
        return tuple(indices)

    # ── action selection ─────────────────────────────────────
    def select_action(self, obs) -> int:
        if self.epsilon.explore():
            return np.random.randint(self.action_dim)
        return int(np.argmax(self.q_table[self._to_index(obs)]))

    # ── learning update ──────────────────────────────────────
    def update(self, obs, action, reward, next_obs, done):
        s = self._to_index(obs)
        s_next = self._to_index(next_obs)
        future = 0.0 if done else np.max(self.q_table[s_next])
        target = reward + self.discount * future
        self.q_table[s][action] += self.lr * (target - self.q_table[s][action])

    def end_episode(self):
        self.epsilon.step()
