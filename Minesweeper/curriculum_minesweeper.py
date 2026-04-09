"""
curriculum_minesweeper.py — curriculum MAPPO on cooperative Minesweeper
========================================================================
Trains agents through progressively harder Minesweeper boards.
Network weights are preserved between stages — agents reuse what they
learned on the easier board instead of starting from scratch.

Curriculum stages:
  Stage 1 — 5×5, 3 mines   (easy)
  Stage 2 — 7×7, 7 mines   (medium)
  Stage 3 — 9×9, 10 mines  (hard)

All stages share the same observation size (pad_cells = 9×9 = 81) so
actor/critic weights transfer without any network re-initialisation.
This mirrors the pad_tiles mechanism in the FrozenLake curriculum.

Usage:
    python curriculum_minesweeper.py
    python curriculum_minesweeper.py --compare       # also trains baseline
    python curriculum_minesweeper.py --render
    python curriculum_minesweeper.py --episodes 10
"""

import argparse
import copy
import time
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from envs.minesweeper_marl import env_creator, MinesweeperMARLEnv
from agents.mappo import MAPPOTrainer
from train_minesweeper import MAPPORunner
from utils import load_config

# ── fixed obs size for all stages (9×9 = 81 cells) ──────────────────────────
MAX_CELLS = 9 * 9  # pad all boards to this size
SAVE_BASE = Path("results/curriculum")
EVAL_EPISODES = 50

STAGES = [
    {
        "name": "Stage 1 — 5×5 easy",
        "rows": 5,
        "cols": 5,
        "n_mines": 3,
        "timesteps": 150_000,
    },
    {
        "name": "Stage 2 — 7×7 medium",
        "rows": 7,
        "cols": 7,
        "n_mines": 7,
        "timesteps": 200_000,
    },
    {
        "name": "Stage 3 — 9×9 hard",
        "rows": 9,
        "cols": 9,
        "n_mines": 10,
        "timesteps": 250_000,
    },
]


# ─────────────────────────────────────────────
#  CONFIG HELPERS
# ─────────────────────────────────────────────


def stage_cfg(base_cfg: dict, stage: dict) -> dict:
    """
    Return a config for one curriculum stage.
    pad_cells is fixed to MAX_CELLS so obs_dim stays constant across stages
    — this lets the same network process any board up to 9×9 without resize.
    """
    cfg = copy.deepcopy(base_cfg)
    cfg["minesweeper"]["rows"] = stage["rows"]
    cfg["minesweeper"]["cols"] = stage["cols"]
    cfg["minesweeper"]["n_mines"] = stage["n_mines"]
    cfg["minesweeper"]["pad_cells"] = MAX_CELLS
    return cfg


# ─────────────────────────────────────────────
#  CURRICULUM TRAINER
# ─────────────────────────────────────────────


class CurriculumTrainer:
    """
    Trains a single MAPPOTrainer instance through all curriculum stages.

    Key mechanism:
      - One actor + one critic throughout (weights persist between stages).
      - Only the env changes (board size, mine count) between stages.
      - pad_cells = MAX_CELLS keeps obs_dim constant → no network rebuild.

    Compare with CurriculumTrainer in curriculum_marl.py which does the same
    for FrozenLake.
    """

    def __init__(self, base_cfg: dict):
        self.base_cfg = base_cfg
        self.reward_logs = {}  # stage_name → [mean rewards per 100 episodes]
        self.eval_results = {}  # stage_name → {win_rate, mean_reward, std_reward}
        SAVE_BASE.mkdir(parents=True, exist_ok=True)

        # build trainer once using Stage-1 config (sets obs size to 2 * MAX_CELLS)
        cfg1 = stage_cfg(base_cfg, STAGES[0])
        _probe = MinesweeperMARLEnv(cfg=cfg1)
        obs_dim = _probe.observation_space("agent_0").shape[0]
        action_dim = _probe.action_space("agent_0").n
        _probe.close()

        self.trainer = MAPPOTrainer(cfg1, obs_dim, action_dim, n_agents=2)

    # ── full curriculum run ───────────────────────────────────────

    def run(self) -> dict:
        """Train through all stages; return eval results for each."""
        for stage in STAGES:
            cfg = stage_cfg(self.base_cfg, stage)
            print(f"\n{'━'*52}")
            print(f"  {stage['name']}")
            print(
                f"  Board: {stage['rows']}×{stage['cols']}  |  Mines: {stage['n_mines']}"
            )
            print(f"  Timesteps: {stage['timesteps']:,}")
            print(f"{'━'*52}")

            reward_log = self._train_stage(cfg, stage["timesteps"])
            self.reward_logs[stage["name"]] = reward_log

            result = self._evaluate(cfg)
            self.eval_results[stage["name"]] = result
            print(f"\n  Win rate    : {result['win_rate'] * 100:.1f}%")
            print(
                f"  Mean reward : {result['mean_reward']:.3f} ± {result['std_reward']:.3f}"
            )

            # save weights after each stage
            stage_dir = SAVE_BASE / stage["name"].replace(" ", "_").replace("—", "-")
            stage_dir.mkdir(parents=True, exist_ok=True)
            self.trainer.save(str(stage_dir / "mappo"))

        return self.eval_results

    # ── single stage training loop ────────────────────────────────

    def _train_stage(self, cfg: dict, timesteps: int) -> list[float]:
        """
        Train the existing trainer on a new env config for `timesteps` steps.
        The trainer's network weights carry over from the previous stage.
        """
        env = env_creator(cfg=cfg)
        t_start = time.time()
        ep_count = 0
        env_steps = 0
        ep_rewards_buf: list[float] = []
        reward_log: list[float] = []

        while env_steps < timesteps:
            env.reset()
            ep_reward = 0.0

            for agent in env.agent_iter():
                obs, _, term, trunc, _ = env.last()
                done = term or trunc

                if done:
                    env.step(None)
                    continue

                partner = "agent_1" if agent == "agent_0" else "agent_0"
                global_obs = np.concatenate([obs, env.observe(partner)])

                value = self.trainer.get_value(global_obs)
                action, log_prob = self.trainer.get_action(agent, obs, mask=env.action_mask())

                env.step(action)
                env_steps += 1

                reward = env.last_step_reward
                ep_reward += reward
                is_done = env.terminations.get(agent, False) or env.truncations.get(
                    agent, False
                )
                self.trainer.store(
                    agent,
                    obs,
                    global_obs,
                    action,
                    log_prob,
                    value,
                    reward,
                    is_done,
                )
                self.trainer.maybe_update()

                if env_steps >= timesteps:
                    break

            ep_rewards_buf.append(ep_reward)
            ep_count += 1
            if ep_count % 100 == 0:
                mean_r = float(np.mean(ep_rewards_buf[-100:]))
                reward_log.append(mean_r)
                elapsed = time.time() - t_start
                print(
                    f"  Ep {ep_count:5d}  |  steps {env_steps:7d}  |"
                    f"  mean reward: {mean_r:+.3f}  |  {elapsed:.0f}s"
                )

        env.close()
        return reward_log

    # ── evaluation ───────────────────────────────────────────────

    def _evaluate(self, cfg: dict, n_episodes: int = EVAL_EPISODES) -> dict:
        env = env_creator(cfg=cfg)
        wins = 0
        ep_rewards: list[float] = []

        for _ in range(n_episodes):
            env.reset()
            ep_reward = 0.0
            for agent in env.agent_iter():
                obs, _, term, trunc, _ = env.last()
                if term or trunc:
                    env.step(None)
                else:
                    action, _ = self.trainer.get_action(agent, obs, mask=env.action_mask())
                    env.step(action)
                    ep_reward += env.last_step_reward
            ep_rewards.append(ep_reward)
            if not env.mine_hit and not any(env.truncations.values()):
                wins += 1

        env.close()
        return {
            "win_rate": wins / n_episodes,
            "mean_reward": float(np.mean(ep_rewards)),
            "std_reward": float(np.std(ep_rewards)),
        }

    # ── render ───────────────────────────────────────────────────

    def render(self, n_episodes: int = 5):
        """Render trained agents on the hardest stage."""
        cfg = stage_cfg(self.base_cfg, STAGES[-1])
        env = MinesweeperMARLEnv(cfg=cfg, render_mode="human")
        print(f"\nRendering Stage 3 (9×9) — {n_episodes} episodes")

        for ep in range(n_episodes):
            env.reset()
            ep_reward, steps = 0.0, 0
            for agent in env.agent_iter():
                obs, _, term, trunc, _ = env.last()
                env.render()
                if term or trunc:
                    env.step(None)
                else:
                    action, _ = self.trainer.get_action(agent, obs, mask=env.action_mask())
                    env.step(action)
                    ep_reward += env.last_step_reward
                    steps += 1
            won    = not env.mine_hit and not any(env.truncations.values())
            result = "WIN" if won else ("MINE HIT" if env.mine_hit else "timeout")
            print(
                f"  Episode {ep + 1}: {result}  "
                f"(reward={ep_reward:.2f}, steps={steps})"
            )

        env.close()


# ─────────────────────────────────────────────
#  PLOTS
# ─────────────────────────────────────────────


def plot_curriculum(trainer: CurriculumTrainer):
    """Training curves — one subplot per stage."""
    stage_colors = ["#aaaaee", "#88bbee", "#3355cc"]
    window = 5

    fig, axes = plt.subplots(1, len(STAGES), figsize=(15, 4), sharey=True)
    fig.suptitle(
        "Curriculum learning — Minesweeper MAPPO", fontsize=13, fontweight="bold"
    )

    for ax, stage, color in zip(axes, STAGES, stage_colors):
        rewards = trainer.reward_logs.get(stage["name"], [])
        ax.set_title(stage["name"].split("—")[-1].strip(), fontsize=10)
        ax.set_facecolor("#f8f9ff")
        ax.plot(rewards, color=color, linewidth=0.5, alpha=0.4)
        if len(rewards) >= window:
            smoothed = np.convolve(rewards, np.ones(window) / window, mode="valid")
            ax.plot(
                range(window - 1, len(rewards)),
                smoothed,
                color=color,
                linewidth=2,
                label="MAPPO",
            )
        ax.axhline(0, color="#cccccc", linewidth=0.8, linestyle="--")
        ax.set_xlabel("Checkpoint (×100 ep)")
        ax.spines[["top", "right"]].set_visible(False)

    axes[0].set_ylabel("Mean episode reward")
    axes[0].legend(fontsize=9, loc="lower right")

    out = SAVE_BASE / "curriculum_curves.png"
    plt.tight_layout()
    plt.savefig(out, dpi=150)
    print(f"  Saved → {out}")
    plt.show()
    plt.close()


def plot_comparison(curriculum_results: dict, baseline_result: dict):
    """Bar chart comparing curriculum vs direct training on Stage 3."""
    labels = ["Baseline\n(9×9 direct)", "Curriculum\n(5→7→9×9)"]
    win_r = [
        baseline_result["win_rate"] * 100,
        curriculum_results[STAGES[-1]["name"]]["win_rate"] * 100,
    ]
    rewards = [
        baseline_result["mean_reward"],
        curriculum_results[STAGES[-1]["name"]]["mean_reward"],
    ]
    colors = ["#e06b3b", "#3b9e60"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 4))
    fig.suptitle(
        "Curriculum vs Baseline — Stage 3 (9×9, 10 mines)",
        fontsize=13,
        fontweight="bold",
    )

    b1 = ax1.bar(labels, win_r, color=colors, edgecolor="white")
    ax1.set_ylabel("Win rate (%)")
    ax1.set_ylim(0, 105)
    ax1.bar_label(b1, fmt="%.0f%%", padding=3, fontsize=11)
    ax1.spines[["top", "right"]].set_visible(False)

    b2 = ax2.bar(labels, rewards, color=colors, edgecolor="white")
    ax2.set_ylabel("Mean episode reward")
    ax2.axhline(0, color="#cccccc", linewidth=0.8, linestyle="--")
    ax2.bar_label(b2, fmt="%.2f", padding=3, fontsize=11)
    ax2.spines[["top", "right"]].set_visible(False)

    out = SAVE_BASE / "curriculum_vs_baseline.png"
    plt.tight_layout()
    plt.savefig(out, dpi=150)
    print(f"  Saved → {out}")
    plt.show()
    plt.close()


# ─────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--render", action="store_true", help="Render final stage after training"
    )
    parser.add_argument(
        "--compare",
        action="store_true",
        help="Also train a baseline (9×9 direct) for comparison",
    )
    parser.add_argument(
        "--episodes", type=int, default=5, help="Episodes to render (--render mode)"
    )
    args = parser.parse_args()

    base_cfg = load_config("config.yaml")

    # ── curriculum run ───────────────────────────────────────────
    print("\n" + "═" * 52)
    print("  CURRICULUM MAPPO  (5×5 → 7×7 → 9×9)")
    print("═" * 52)
    trainer = CurriculumTrainer(base_cfg)
    results = trainer.run()
    plot_curriculum(trainer)

    # ── summary table ────────────────────────────────────────────
    print("\n" + "═" * 62)
    print(f"  {'Stage':<26}  {'Win rate':>9}  {'Mean reward':>13}")
    print("═" * 62)
    for stage in STAGES:
        r = results[stage["name"]]
        print(
            f"  {stage['name']:<26}  {r['win_rate']*100:>7.1f}%  {r['mean_reward']:>12.3f}"
        )
    print("═" * 62)

    # ── optional baseline comparison ─────────────────────────────
    if args.compare:
        print("\n" + "═" * 52)
        print("  BASELINE — training directly on 9×9")
        print("═" * 52)
        hard_cfg = stage_cfg(base_cfg, STAGES[-1])
        total_steps = sum(s["timesteps"] for s in STAGES)
        runner = MAPPORunner(
            cfg=hard_cfg,
            total_timesteps=total_steps,
            save_dir=SAVE_BASE / "baseline",
        )
        runner.train()
        baseline_result = runner.evaluate(n_episodes=EVAL_EPISODES)
        plot_comparison(results, baseline_result)

    if args.render:
        trainer.render(n_episodes=args.episodes)


if __name__ == "__main__":
    main()
