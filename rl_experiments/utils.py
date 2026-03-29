"""
utils.py — shared utilities for all RL experiments
===================================================
Import anything you need:

    from utils import load_config, set_seed, ReplayBuffer, Plotter, evaluate_agent
"""

import os
import random
import torch
from pathlib import Path
from typing import Callable

import numpy as np
import matplotlib.pyplot as plt
import yaml


# ─────────────────────────────────────────────
#  CONFIG
# ─────────────────────────────────────────────


def load_config(path: str = "config.yaml") -> dict:
    """
    Load the shared YAML config.
    Returns a plain dict — access with cfg["training"]["episodes"] etc.
    """
    with open(path, "r") as f:
        cfg = yaml.safe_load(f)
    return cfg


def get(cfg: dict, *keys, default=None):
    """
    Safe nested key access.
    Example: get(cfg, "dqn", "learning_rate")
    """
    val = cfg
    for k in keys:
        if not isinstance(val, dict) or k not in val:
            return default
        val = val[k]
    return val


def get_device(cfg: dict) -> torch.device:
    preference = get(cfg, "system", "device", default="auto")
    if preference == "auto":
        if torch.backends.mps.is_available():
            return torch.device("mps")
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")
    return torch.device(preference)


# ─────────────────────────────────────────────
#  REPRODUCIBILITY
# ─────────────────────────────────────────────


def set_seed(seed: int):
    """Seed Python, NumPy, and PyTorch for reproducible runs."""
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass  # torch not needed for Q-learning runs


# ─────────────────────────────────────────────
#  EPSILON-GREEDY SCHEDULE
# ─────────────────────────────────────────────


class EpsilonSchedule:
    """
    Tracks and decays epsilon.  Keeps all agents on the same schedule.

    Usage:
        eps = EpsilonSchedule(cfg)
        action = random_action if eps.explore() else best_action
        eps.step()
    """

    def __init__(self, cfg: dict):
        self.value = get(cfg, "training", "epsilon_start")
        self.minimum = get(cfg, "training", "epsilon_end")
        self.decay = get(cfg, "training", "epsilon_decay")

    def explore(self) -> bool:
        return random.random() < self.value

    def step(self):
        self.value = max(self.minimum, self.value * self.decay)

    def __float__(self):
        return float(self.value)

    def __repr__(self):
        return f"EpsilonSchedule(ε={self.value:.4f})"


# ─────────────────────────────────────────────
#  REPLAY BUFFER  (used by DQN and future MARL agents)
# ─────────────────────────────────────────────

from collections import deque


class ReplayBuffer:
    """
    Circular buffer of (state, action, reward, next_state, done) transitions.

    Why random sampling?
        Consecutive steps are correlated — training on them in order
        destabilises the network.  Random batches break that correlation.
    """

    def __init__(self, capacity: int):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size: int):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        try:
            import torch

            return (
                torch.FloatTensor(np.array(states)),
                torch.LongTensor(actions),
                torch.FloatTensor(rewards),
                torch.FloatTensor(np.array(next_states)),
                torch.FloatTensor(dones),
            )
        except ImportError:
            return (
                np.array(states),
                np.array(actions),
                np.array(rewards),
                np.array(next_states),
                np.array(dones),
            )

    def __len__(self):
        return len(self.buffer)

    def ready(self, batch_size: int) -> bool:
        return len(self) >= batch_size


# ─────────────────────────────────────────────
#  ENVIRONMENT FACTORY
#  Returns a callable that creates fresh envs.
#  Handles domain randomisation for FrozenLake.
# ─────────────────────────────────────────────


def make_env_factory(cfg: dict, size: int = None):
    """
    Returns a callable: () -> (gym.Env, map_desc | None)

    For CartPole:    returns (env, None)
    For FrozenLake:  returns (env, map_desc) so RewardShaper can use the
                     actual map layout for distance-based reward shaping.
    """
    import gymnasium as gym
    from gymnasium.envs.toy_text.frozen_lake import generate_random_map

    env_name = get(cfg, "environment", "name")

    if env_name == "FrozenLake-v1":
        grid_size = size or get(cfg, "frozenlake", "size", default=4)
        slippery = get(cfg, "frozenlake", "is_slippery", default=False)
        randomise = get(cfg, "frozenlake", "domain_randomisation", default=True)

        def factory():
            desc = generate_random_map(size=grid_size) if randomise else None
            env = gym.make("FrozenLake-v1", desc=desc, is_slippery=slippery)
            map_desc = env.unwrapped.desc  # actual map used (even if desc=None)
            return env, map_desc

        return factory

    # default — fixed env, no map desc needed
    return lambda: (gym.make(env_name), None)


def onehot(state: int, n_states: int) -> np.ndarray:
    """
    FrozenLake returns a single integer tile index as observation.
    Neural networks need a vector — one-hot encoding converts it.
    Q-Learning uses the integer directly as a table index.

    Example: state=5, n_states=16  →  [0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0]
    """
    v = np.zeros(n_states, dtype=np.float32)
    v[state] = 1.0
    return v


# ─────────────────────────────────────────────
#  EVALUATION
# ─────────────────────────────────────────────


def evaluate_agent(
    env_factory: Callable,
    select_action: Callable,
    cfg: dict,
    epsilon: float = 0.0,
) -> dict:
    """
    Run the agent greedily for eval_episodes episodes.
    Returns a dict with mean, min, max, std, and per-episode rewards.

    Args:
        env_factory:   callable that returns a fresh gym env
        select_action: fn(state) -> int  (should use epsilon=0 internally)
        cfg:           the shared config dict
        epsilon:       exploration rate during eval (default 0 = pure greedy)
    """
    n = get(cfg, "environment", "eval_episodes", default=20)
    rewards = []
    env = env_factory()

    for _ in range(n):
        obs, _ = env.reset()
        total = 0
        while True:
            action = select_action(obs)
            obs, r, term, trunc, _ = env.step(action)
            total += r
            if term or trunc:
                break
        rewards.append(total)

    env.close()
    return {
        "mean": float(np.mean(rewards)),
        "std": float(np.std(rewards)),
        "min": float(np.min(rewards)),
        "max": float(np.max(rewards)),
        "rewards": rewards,
    }


# ─────────────────────────────────────────────
#  ENVIRONMENT FACTORY
#  Returns a callable that creates fresh envs.
#  Handles domain randomisation for FrozenLake.
# ─────────────────────────────────────────────

# ─────────────────────────────────────────────
#  REWARD SHAPING  (FrozenLake)
#
#  FrozenLake's default reward is +1 at goal, 0 everywhere else.
#  With domain randomisation the agent almost never finds the goal
#  by random exploration, so it receives zero signal for thousands
#  of episodes and never learns.
#
#  Shaping adds intermediate signals:
#    - penalty for falling in a hole  → discourages recklessness
#    - small step penalty             → encourages efficiency
#    - distance bonus                 → nudges agent toward the goal
#
#  The shaped reward is ONLY used during training.
#  Evaluation always uses the original reward so results are comparable.
# ─────────────────────────────────────────────


class RewardShaper:
    """
    Wraps FrozenLake's sparse reward with dense shaping signals.

    Usage:
        shaper = RewardShaper(cfg, map_desc)
        shaped_r = shaper.shape(obs, next_obs, reward, done)
    """

    def __init__(self, cfg: dict, map_desc):
        self.enabled = get(cfg, "frozenlake", "reward_shaping", default=False)
        self.r_goal = get(cfg, "frozenlake", "reward_goal", default=1.0)
        self.r_hole = get(cfg, "frozenlake", "reward_hole", default=-0.5)
        self.r_step = get(cfg, "frozenlake", "reward_step", default=-0.01)
        self.r_closer = get(cfg, "frozenlake", "reward_closer", default=0.05)
        self.map_desc = map_desc  # list of strings e.g. ["SFFF","FHFH",...]
        self.size = len(map_desc)
        self._goal_pos = self._find_goal()

    def _find_goal(self):
        for r, row in enumerate(self.map_desc):
            for c, cell in enumerate(row):
                if cell == b"G" or cell == "G":
                    return (r, c)
        return (self.size - 1, self.size - 1)  # fallback

    def _tile(self, state: int) -> str:
        row, col = divmod(state, self.size)
        cell = self.map_desc[row][col]
        return cell.decode() if isinstance(cell, bytes) else cell

    def _manhattan(self, state: int) -> float:
        row, col = divmod(state, self.size)
        gr, gc = self._goal_pos
        return abs(row - gr) + abs(col - gc)

    def shape(self, obs: int, next_obs: int, reward: float, done: bool) -> float:
        if not self.enabled:
            return reward

        # Original goal reward — keep it
        if reward == 1.0:
            return self.r_goal

        # Hole penalty
        if done and reward == 0.0:
            tile = self._tile(next_obs)
            if tile == "H":
                return self.r_hole

        # Distance bonus — reward moving closer to goal
        d_before = self._manhattan(obs)
        d_after = self._manhattan(next_obs)
        closer = self.r_closer if d_after < d_before else 0.0

        return self.r_step + closer


# ─────────────────────────────────────────────
#  PLOTTING
# ─────────────────────────────────────────────

# One fixed colour per agent name — consistent across all comparison plots
AGENT_COLORS = {
    "Q-Learning": "#e06b3b",
    "DQN": "#3b8be0",
    "PPO": "#3b9e60",
}


def _agent_color(name: str) -> str:
    return AGENT_COLORS.get(name, "#888780")


class Plotter:
    """
    Accumulates reward histories from multiple agents and plots them
    together for easy comparison.

    Usage:
        plotter = Plotter(cfg)
        plotter.add("Q-Learning", q_rewards)
        plotter.add("DQN",        dqn_rewards)
        plotter.save_and_show("comparison")
    """

    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.window = get(cfg, "plot", "smoothing_window", default=20)
        self.save_dir = Path(get(cfg, "plot", "save_dir", default="results"))
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.agents: dict[str, list] = {}

    def add(self, name: str, rewards: list):
        self.agents[name] = rewards

    def save_and_show(self, filename: str = "comparison"):
        threshold = get(self.cfg, "environment", "solved_threshold", default=195)
        show = get(self.cfg, "plot", "show", default=True)
        env_name = get(self.cfg, "environment", "name", default="")

        fig, ax = plt.subplots(figsize=(10, 5))
        fig.suptitle(f"Agent comparison — {env_name}", fontsize=13, fontweight="bold")

        for name, rewards in self.agents.items():
            color = _agent_color(name)
            eps = len(rewards)
            # raw (faint)
            ax.plot(rewards, color=color, linewidth=0.4, alpha=0.25)
            # smoothed
            if eps >= self.window:
                smoothed = np.convolve(
                    rewards, np.ones(self.window) / self.window, mode="valid"
                )
                ax.plot(
                    range(self.window - 1, eps),
                    smoothed,
                    color=color,
                    linewidth=2,
                    label=name,
                )

        ax.axhline(
            threshold,
            color="#444441",
            linewidth=1,
            linestyle="--",
            label=f"Solved ({threshold})",
        )
        # Perfect score depends on env: 1.0 for FrozenLake, 500 for CartPole
        perfect = 1.0 if "FrozenLake" in env_name else 500
        ax.axhline(
            perfect,
            color="#3b9e60",
            linewidth=0.8,
            linestyle=":",
            label=f"Perfect ({perfect})",
        )

        ax.set_xlabel("Episode")
        ax.set_ylabel("Total reward")
        ax.legend(fontsize=10)
        ax.spines[["top", "right"]].set_visible(False)

        out = self.save_dir / f"{filename}.png"
        plt.tight_layout()
        plt.savefig(out, dpi=150)
        print(f"Saved → {out}")
        if show:
            plt.show()
        plt.close()

    def print_summary(self, eval_results: dict[str, dict]):
        """Print a clean comparison table of eval results."""
        print("\n" + "─" * 52)
        print(f"{'Agent':<15} {'Mean':>8} {'Std':>8} {'Min':>6} {'Max':>6}")
        print("─" * 52)
        threshold = get(self.cfg, "environment", "solved_threshold", default=195)
        for name, res in eval_results.items():
            solved = "✓ solved" if res["mean"] >= threshold else ""
            print(
                f"{name:<15} {res['mean']:>8.1f} {res['std']:>8.1f} "
                f"{res['min']:>6.0f} {res['max']:>6.0f}  {solved}"
            )
        print("─" * 52 + "\n")
