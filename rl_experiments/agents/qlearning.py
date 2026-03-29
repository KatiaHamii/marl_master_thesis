"""
agents/qlearning.py — Q-Learning agent
All hyperparameters come from config.yaml via utils.load_config()
"""

import numpy as np
from utils import load_config, get, EpsilonSchedule


class QLearningAgent:
    """
    Tabular Q-Learning.

    CartPole:   continuous state → discretised into bins → table index
    FrozenLake: state is already a single integer tile index → used directly

    Curriculum note: when the map grows (4x4 → 6x6 → 8x8) the Q-table
    must grow too. Call resize_table(new_size) whenever the env changes.
    Existing knowledge for the smaller grid is preserved in the top-left
    corner; new tiles are initialised to zero.
    """

    def __init__(self, cfg: dict, state_dim: int, action_dim: int):
        self.cfg = cfg
        self.action_dim = action_dim
        self.lr = get(cfg, "qlearning", "learning_rate")
        self.discount = get(cfg, "training", "discount")
        self.epsilon = EpsilonSchedule(cfg)
        self.env_name = get(cfg, "environment", "name")

        if self.env_name == "FrozenLake-v1":
            size = get(cfg, "frozenlake", "size", default=4)
            self._size = size
            n_states = size * size
            self.q_table = np.zeros([n_states, action_dim])
            self._mode = "discrete"
        else:
            self.bins = get(cfg, "qlearning", "bins")
            self._low = [-2.4, -3.0, -0.25, -3.5]
            self._high = [2.4, 3.0, 0.25, 3.5]
            self.q_table = np.zeros([self.bins] * 4 + [action_dim])
            self._mode = "continuous"
            self._size = None

    # ── curriculum support ───────────────────────────────────
    def resize_table(self, new_size: int):
        """
        Grow the Q-table when the map size increases.
        Called automatically by the training loop when curriculum triggers.
        Existing values are preserved — new rows initialised to zero.
        """
        if self._mode != "discrete" or new_size == self._size:
            return
        new_n = new_size * new_size
        old_n = len(self.q_table)  # actual length, not recalculated
        if new_n <= old_n:
            return  # already big enough
        new_tbl = np.zeros([new_n, self.action_dim])
        new_tbl[:old_n] = self.q_table  # copy old knowledge
        self.q_table = new_tbl
        self._size = new_size
        print(f"  Q-table resized: {old_n} → {new_n} states")

    # ── state preprocessing ──────────────────────────────────
    def _to_index(self, obs):
        if self._mode == "discrete":
            idx = int(obs)
            # safety clamp — should not happen after resize, but guards against
            # edge cases during the transition episode
            return min(idx, len(self.q_table) - 1)
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
