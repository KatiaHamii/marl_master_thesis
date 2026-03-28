"""
agents/qlearning.py — Q-Learning agent
All hyperparameters come from config.yaml via utils.load_config()
"""

import numpy as np
from utils import load_config, get, set_seed, EpsilonSchedule


class QLearningAgent:
    """
    Tabular Q-Learning with discretised state space.
    Reads all params from the shared config.
    """

    def __init__(self, cfg: dict, state_dim: int, action_dim: int):
        self.cfg = cfg
        self.action_dim = action_dim
        self.lr = get(cfg, "qlearning", "learning_rate")
        self.discount = get(cfg, "training", "discount")
        self.bins = get(cfg, "qlearning", "bins")
        self.epsilon = EpsilonSchedule(cfg)

        # Observation bounds for CartPole
        self._low = [-2.4, -3.0, -0.25, -3.5]
        self._high = [2.4, 3.0, 0.25, 3.5]

        # Q-table: (bins^state_dim) × action_dim
        self.q_table = np.zeros([self.bins] * state_dim + [action_dim])

    # ── state preprocessing ──────────────────────────────────
    def _discretise(self, obs: np.ndarray) -> tuple:
        indices = []
        for i, val in enumerate(obs):
            lo, hi = self._low[i], self._high[i]
            bucket = int((np.clip(val, lo, hi) - lo) / (hi - lo) * self.bins)
            indices.append(min(bucket, self.bins - 1))
        return tuple(indices)

    # ── action selection ─────────────────────────────────────
    def select_action(self, obs: np.ndarray) -> int:
        if self.epsilon.explore():
            return np.random.randint(self.action_dim)
        return int(np.argmax(self.q_table[self._discretise(obs)]))

    # ── learning update ──────────────────────────────────────
    def update(self, obs, action, reward, next_obs, done):
        s = self._discretise(obs)
        s_next = self._discretise(next_obs)
        future = 0.0 if done else np.max(self.q_table[s_next])
        target = reward + self.discount * future
        self.q_table[s][action] += self.lr * (target - self.q_table[s][action])

    def end_episode(self):
        self.epsilon.step()
