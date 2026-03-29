"""
utils.py — shared utilities for all RL experiments
===================================================
Import anything you need:

    from utils import load_config, set_seed, ReplayBuffer, Plotter, evaluate_agent
"""

import os
import random
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
#  DEVICE SELECTION
# ─────────────────────────────────────────────


def get_device(cfg: dict):
    """
    Resolve the compute device from config.
    system.device: "auto" | "cpu" | "mps" | "cuda"

    "auto" priority: MPS (Apple Silicon) → CUDA → CPU
    """
    try:
        import torch
    except ImportError:
        return None

    preference = get(cfg, "system", "device", default="auto")

    if preference == "auto":
        if torch.backends.mps.is_available():
            return torch.device("mps")
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")

    return torch.device(preference)


# ─────────────────────────────────────────────
#  EPSILON-GREEDY SCHEDULE
# ─────────────────────────────────────────────


class EpsilonSchedule:
    """
    Tracks and decays epsilon.  Keeps all agents on the same schedule.

    For FrozenLake, reads frozenlake.epsilon_decay if present —
    overrides training.epsilon_decay so each env can be tuned independently.

    Usage:
        eps = EpsilonSchedule(cfg)
        action = random_action if eps.explore() else best_action
        eps.step()
    """

    def __init__(self, cfg: dict, per_step: bool = False):
        """
        Args:
            per_step: if True, reads the per-step decay value from config.
                      Set this when step() is called inside the step loop
                      rather than once per episode.
        """
        self.value = get(cfg, "training", "epsilon_start")
        self.minimum = get(cfg, "training", "epsilon_end")
        env_name = get(cfg, "environment", "name", default="")

        if env_name == "FrozenLake-v1":
            if per_step:
                # use the slower per-step decay so epsilon survives long enough
                self.decay = get(
                    cfg,
                    "frozenlake",
                    "epsilon_decay_per_step",
                    default=get(
                        cfg,
                        "frozenlake",
                        "epsilon_decay",
                        default=get(cfg, "training", "epsilon_decay"),
                    ),
                )
            else:
                self.decay = get(
                    cfg,
                    "frozenlake",
                    "epsilon_decay",
                    default=get(cfg, "training", "epsilon_decay"),
                )
        else:
            self.decay = get(cfg, "training", "epsilon_decay")

        print(
            f"  EpsilonSchedule: decay={self.decay}  "
            f"({'per-step' if per_step else 'per-episode'})"
            f"  → ε≈{max(self.minimum, self.decay**5000):.3f} at ep 5000"
        )

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


# ─────────────────────────────────────────────
#  EXPERIMENT LOGGER
#  Saves one row per agent run to a CSV file.
#  Columns: timestamp, env, agent, all config
#  params relevant to that agent, then eval
#  results (mean, std, min, max, solved).
#
#  Every run appends — so you build up a full
#  history of experiments you can sort/filter
#  in Excel or pandas to compare runs.
# ─────────────────────────────────────────────

import csv
import datetime


class ExperimentLogger:
    """
    Appends one row per agent to results/experiments.csv.

    Each row contains:
      - timestamp, env, agent name
      - all hyperparameters relevant to that agent
      - training episode count
      - eval mean / std / min / max / solved flag
      - per-episode reward summary (min, max, final-100 avg)

    Usage:
        logger = ExperimentLogger(cfg)
        logger.log(
            agent_name  = "DQN",
            rewards     = rewards,        # list of per-episode rewards
            eval_result = eval_results["DQN"],
        )
    """

    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.save_dir = Path(get(cfg, "plot", "save_dir", default="results"))
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.path = self.save_dir / "experiments.csv"
        self._ensure_header()

    def _common_fields(self) -> dict:
        """Fields shared by every agent row."""
        cfg = self.cfg
        env = get(cfg, "environment", "name", default="")
        is_fl = "FrozenLake" in env
        return {
            "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "env": env,
            "seed": get(cfg, "training", "seed"),
            "episodes": (
                get(
                    cfg,
                    "frozenlake",
                    "episodes",
                    default=get(cfg, "training", "episodes"),
                )
                if is_fl
                else get(cfg, "training", "episodes")
            ),
            "discount": get(cfg, "training", "discount"),
            "epsilon_start": get(cfg, "training", "epsilon_start"),
            "epsilon_end": get(cfg, "training", "epsilon_end"),
            "epsilon_decay": (
                get(
                    cfg,
                    "frozenlake",
                    "epsilon_decay",
                    default=get(cfg, "training", "epsilon_decay"),
                )
                if is_fl
                else get(cfg, "training", "epsilon_decay")
            ),
            # FrozenLake settings
            "fl_size": get(cfg, "frozenlake", "size") if is_fl else "",
            "fl_slippery": get(cfg, "frozenlake", "is_slippery") if is_fl else "",
            "fl_randomisation": (
                get(cfg, "frozenlake", "domain_randomisation") if is_fl else ""
            ),
            "fl_curriculum": get(cfg, "frozenlake", "curriculum") if is_fl else "",
            "fl_shaping": get(cfg, "frozenlake", "reward_shaping") if is_fl else "",
        }

    def _agent_fields(self, agent_name: str) -> dict:
        """Hyperparameters specific to this agent."""
        cfg = self.cfg
        is_fl = "FrozenLake" in get(cfg, "environment", "name", default="")

        if agent_name == "Q-Learning":
            return {
                "lr": get(cfg, "qlearning", "learning_rate"),
                "batch_size": "",
                "replay_cap": "",
                "hidden_size": "",
                "extra": f"bins={get(cfg, 'qlearning', 'bins')}",
            }
        elif agent_name == "DQN":
            replay = (
                get(
                    cfg,
                    "frozenlake",
                    "dqn_replay_capacity",
                    default=get(cfg, "dqn", "replay_capacity"),
                )
                if is_fl
                else get(cfg, "dqn", "replay_capacity")
            )
            return {
                "lr": get(cfg, "dqn", "learning_rate"),
                "batch_size": get(cfg, "dqn", "batch_size"),
                "replay_cap": replay,
                "hidden_size": get(cfg, "dqn", "hidden_size"),
                "extra": f"target_update={get(cfg, 'dqn', 'target_update')}",
            }
        elif agent_name == "PPO":
            upd = (
                get(
                    cfg,
                    "frozenlake",
                    "ppo_update_every",
                    default=get(cfg, "ppo", "update_every"),
                )
                if is_fl
                else get(cfg, "ppo", "update_every")
            )
            return {
                "lr": get(cfg, "ppo", "learning_rate"),
                "batch_size": get(cfg, "ppo", "batch_size"),
                "replay_cap": "",
                "hidden_size": get(cfg, "ppo", "hidden_size"),
                "extra": f"update_every={upd}|clip={get(cfg,'ppo','clip_epsilon')}|epochs={get(cfg,'ppo','epochs')}",
            }
        return {
            "lr": "",
            "batch_size": "",
            "replay_cap": "",
            "hidden_size": "",
            "extra": "",
        }

    @property
    def _columns(self) -> list[str]:
        return [
            # identity
            "timestamp",
            "env",
            "agent",
            "seed",
            "episodes",
            # shared training
            "discount",
            "epsilon_start",
            "epsilon_end",
            "epsilon_decay",
            # agent hyperparams
            "lr",
            "batch_size",
            "replay_cap",
            "hidden_size",
            "extra",
            # frozenlake settings
            "fl_size",
            "fl_slippery",
            "fl_randomisation",
            "fl_curriculum",
            "fl_shaping",
            # training results
            "train_min",
            "train_max",
            "train_last100_avg",
            # eval results
            "eval_mean",
            "eval_std",
            "eval_min",
            "eval_max",
            "solved",
        ]

    def _ensure_header(self):
        """Write header row if file doesn't exist yet."""
        if not self.path.exists():
            with open(self.path, "w", newline="") as f:
                csv.DictWriter(f, fieldnames=self._columns).writeheader()
            print(f"Created experiment log → {self.path}")

    def log(self, agent_name: str, rewards: list, eval_result: dict):
        """Append one row for this agent run."""
        threshold = get(self.cfg, "environment", "solved_threshold", default=195)
        last100 = (
            float(np.mean(rewards[-100:]))
            if len(rewards) >= 100
            else float(np.mean(rewards))
        )

        row = {
            **self._common_fields(),
            "agent": agent_name,
            **self._agent_fields(agent_name),
            # training summary
            "train_min": round(float(np.min(rewards)), 4),
            "train_max": round(float(np.max(rewards)), 4),
            "train_last100_avg": round(last100, 4),
            # eval
            "eval_mean": round(eval_result["mean"], 4),
            "eval_std": round(eval_result["std"], 4),
            "eval_min": round(eval_result["min"], 4),
            "eval_max": round(eval_result["max"], 4),
            "solved": eval_result["mean"] >= threshold,
        }

        with open(self.path, "a", newline="") as f:
            csv.DictWriter(f, fieldnames=self._columns).writerow(row)

        print(f"  Logged → {self.path}  (row: {agent_name})")
