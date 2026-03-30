"""
curriculum_marl.py — curriculum learning for 3-agent cooperative FrozenLake
=============================================================================
Trains agents through progressively harder stages, transferring weights
between stages instead of retraining from scratch each time.

Curriculum stages:
  Stage 1 — 4×4, non-slippery   (easy)
  Stage 2 — 6×6, non-slippery   (medium)
  Stage 3 — 8×8, slippery       (hard)

All stages share the same observation size (padded to 8×8 = 64 tiles) so
network weights can be reused across stages.

Compare against baseline (train directly on 8×8 slippery from scratch)
to show curriculum speeds up learning.

Usage:
    python curriculum_marl.py              # full curriculum run
    python curriculum_marl.py --render     # render final stage after training
    python curriculum_marl.py --compare    # also trains baseline for comparison
"""

import argparse
import copy
import time
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from stable_baselines3 import PPO

from utils import load_config, get
from train_marl import IPPOTrainer, make_vec_env
from envs.frozen_lake_marl import env_creator

# ── fixed obs size for all stages (8×8 map = 64 tiles) ──────────────────────
MAX_TILES   = 8 * 8   # pad all observations to this size
PAD_OBS     = 4 * MAX_TILES + 3  # = 259

SAVE_BASE   = Path("results/curriculum")
EVAL_EPISODES = 50

STAGES = [
    {"name": "Stage 1 — 4×4 non-slippery", "size": 4, "is_slippery": False, "timesteps": 200_000},
    {"name": "Stage 2 — 6×6 non-slippery", "size": 6, "is_slippery": False, "timesteps": 200_000},
    {"name": "Stage 3 — 8×8 slippery",     "size": 8, "is_slippery": True,  "timesteps": 200_000},
]


# ─────────────────────────────────────────────
#  BUILD CURRICULUM CONFIG
# ─────────────────────────────────────────────
def stage_cfg(base_cfg: dict, stage: dict) -> dict:
    """Return a config for one curriculum stage with fixed obs padding."""
    cfg = copy.deepcopy(base_cfg)
    cfg["frozenlake"]["size"]         = stage["size"]
    cfg["frozenlake"]["is_slippery"]  = stage["is_slippery"]
    cfg["frozenlake"]["pad_tiles"]    = MAX_TILES   # fixed obs size across all stages
    return cfg


# ─────────────────────────────────────────────
#  CURRICULUM TRAINER
# ─────────────────────────────────────────────
class CurriculumTrainer:
    def __init__(self, base_cfg: dict):
        self.base_cfg = base_cfg
        self.reward_logs = {}   # stage_name → {agent: [rewards]}
        self.eval_results = {}  # stage_name → {success_rate, mean_reward, ...}

        SAVE_BASE.mkdir(parents=True, exist_ok=True)

        # build agents once using Stage-1 config (sets obs size to PAD_OBS)
        cfg1 = stage_cfg(base_cfg, STAGES[0])
        self._trainers = {}
        agent_names = ["agent_0", "agent_1", "agent_2"]
        self.agents = {
            name: PPO(
                policy="MlpPolicy",
                env=make_vec_env(cfg1, n_envs=base_cfg.get("marl", {}).get("n_envs", 4)),
                learning_rate=base_cfg.get("marl", {}).get("learning_rate", 3e-4),
                n_steps=base_cfg.get("marl", {}).get("n_steps", 256),
                batch_size=base_cfg.get("marl", {}).get("batch_size", 64),
                n_epochs=base_cfg.get("marl", {}).get("n_epochs", 4),
                clip_range=base_cfg.get("marl", {}).get("clip_range", 0.2),
                ent_coef=base_cfg.get("marl", {}).get("ent_coef", 0.01),
                gamma=get(base_cfg, "training", "discount", default=0.99),
                gae_lambda=0.95,
                verbose=0,
                policy_kwargs=dict(net_arch=[64, 64]),
            )
            for name in agent_names
        }

    def run(self):
        """Run all curriculum stages, transferring weights between them."""
        from stable_baselines3.common.callbacks import BaseCallback

        class RewardLogger(BaseCallback):
            def __init__(self):
                super().__init__()
                self.episode_rewards = []
                self._buf = []
            def _on_step(self):
                for info in self.locals.get("infos", []):
                    if "episode" in info:
                        self._buf.append(info["episode"]["r"])
                return True
            def _on_rollout_end(self):
                if self._buf:
                    self.episode_rewards.append(float(np.mean(self._buf)))
                    self._buf = []

        for stage in STAGES:
            cfg = stage_cfg(self.base_cfg, stage)
            print(f"\n{'━'*52}")
            print(f"  {stage['name']}")
            print(f"  Timesteps: {stage['timesteps']:,}  |  Map: {stage['size']}×{stage['size']}  |  Slippery: {stage['is_slippery']}")
            print(f"{'━'*52}")

            # swap each agent's env to the new stage's env (weights are preserved)
            new_env = make_vec_env(cfg, n_envs=self.base_cfg.get("marl", {}).get("n_envs", 4))
            logs = {}

            for name, agent in self.agents.items():
                agent.set_env(new_env)
                cb = RewardLogger()
                print(f"  Training {name}...")
                agent.learn(
                    total_timesteps=stage["timesteps"] // len(self.agents),
                    reset_num_timesteps=False,  # keep timestep counter → continuous curves
                    progress_bar=True,
                    callback=cb,
                )
                logs[name] = cb.episode_rewards

            self.reward_logs[stage["name"]] = logs

            # evaluate after this stage
            result = self._evaluate(cfg)
            self.eval_results[stage["name"]] = result
            print(f"\n  Success rate : {result['success_rate']*100:.1f}%")
            print(f"  Mean reward  : {result['mean_reward']:.3f} ± {result['std_reward']:.3f}")

            # save weights after each stage
            stage_dir = SAVE_BASE / stage["name"].replace(" ", "_").replace("—", "-")
            stage_dir.mkdir(parents=True, exist_ok=True)
            for name, agent in self.agents.items():
                agent.save(str(stage_dir / f"{name}.zip"))

        return self.eval_results

    def _evaluate(self, cfg: dict, n_episodes: int = EVAL_EPISODES) -> dict:
        env = env_creator(cfg=cfg)
        successes = 0
        ep_rewards = []

        for _ in range(n_episodes):
            env.reset()
            ep_reward = 0.0
            for agent in env.agent_iter():
                obs, reward, term, trunc, _ = env.last()
                ep_reward += reward
                if term or trunc:
                    env.step(None)
                else:
                    action, _ = self.agents[agent].predict(obs.reshape(1, -1), deterministic=True)
                    env.step(int(action[0]))
            ep_rewards.append(ep_reward)
            if ep_reward >= 1.0:
                successes += 1

        env.close()
        return {
            "success_rate": successes / n_episodes,
            "mean_reward":  float(np.mean(ep_rewards)),
            "std_reward":   float(np.std(ep_rewards)),
        }

    def render(self, n_episodes: int = 5):
        """Render the final trained agents on Stage 3 (hardest map)."""
        from envs.frozen_lake_marl import FrozenLakeMARLEnv
        cfg = stage_cfg(self.base_cfg, STAGES[-1])
        env = FrozenLakeMARLEnv(cfg=cfg, render_mode="human")

        print(f"\nRendering Stage 3 — 8×8 slippery ({n_episodes} episodes)")
        for ep in range(n_episodes):
            env.reset()
            ep_reward = 0.0
            step = 0
            for agent in env.agent_iter():
                obs, reward, term, trunc, _ = env.last()
                ep_reward += reward
                env.render()
                step += 1
                if term or trunc:
                    env.step(None)
                else:
                    action, _ = self.agents[agent].predict(obs.reshape(1, -1), deterministic=False)
                    env.step(int(action[0]))
            result = "SUCCESS" if ep_reward >= 1.0 else "failed"
            print(f"  Episode {ep+1}: {result}  (reward={ep_reward:.2f}, steps={step})")
        env.close()


# ─────────────────────────────────────────────
#  PLOTS
# ─────────────────────────────────────────────
def plot_curriculum(trainer: CurriculumTrainer, baseline_logs: dict = None):
    """Training curves per stage + optional baseline comparison."""
    colors = {"agent_0": "#3b8be0", "agent_1": "#e06b3b", "agent_2": "#3b9e60"}
    stage_colors = ["#aaaaee", "#88bbee", "#3355cc"]
    window = 8

    n_stages = len(STAGES)
    fig, axes = plt.subplots(1, n_stages, figsize=(15, 4), sharey=True)
    fig.suptitle("Curriculum learning — training progress per stage", fontsize=13, fontweight="bold")

    for ax, stage, color in zip(axes, STAGES, stage_colors):
        logs = trainer.reward_logs.get(stage["name"], {})
        ax.set_title(stage["name"].split("—")[-1].strip(), fontsize=10)
        ax.set_facecolor("#f8f9ff")

        for agent_name, rewards in logs.items():
            c = colors.get(agent_name, "#888")
            ax.plot(rewards, color=c, linewidth=0.4, alpha=0.3)
            if len(rewards) >= window:
                smoothed = np.convolve(rewards, np.ones(window) / window, mode="valid")
                ax.plot(range(window - 1, len(rewards)), smoothed,
                        color=c, linewidth=2,
                        label=agent_name if ax == axes[0] else None)

        ax.axhline(0, color="#cccccc", linewidth=0.8, linestyle="--")
        ax.set_xlabel("Rollout")
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
    """Bar chart: curriculum vs baseline on final stage."""
    labels = ["Baseline\n(8×8 direct)", "Curriculum\n(4→6→8×8)"]
    success = [
        baseline_result["success_rate"] * 100,
        curriculum_results[STAGES[-1]["name"]]["success_rate"] * 100,
    ]
    rewards = [
        baseline_result["mean_reward"],
        curriculum_results[STAGES[-1]["name"]]["mean_reward"],
    ]
    colors = ["#e06b3b", "#3b9e60"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 4))
    fig.suptitle("Curriculum vs Baseline — final stage (8×8 slippery)", fontsize=13, fontweight="bold")

    b1 = ax1.bar(labels, success, color=colors, edgecolor="white")
    ax1.set_ylabel("Success rate (%)")
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
    parser.add_argument("--render",  action="store_true", help="Render final stage after training")
    parser.add_argument("--compare", action="store_true", help="Also train baseline (8×8 direct) for comparison")
    parser.add_argument("--episodes", type=int, default=5, help="Episodes to render")
    args = parser.parse_args()

    base_cfg = load_config("config.yaml")

    # ── curriculum run ───────────────────────
    print("\n" + "═"*52)
    print("  CURRICULUM LEARNING  (4×4 → 6×6 → 8×8)")
    print("═"*52)
    trainer = CurriculumTrainer(base_cfg)
    curriculum_results = trainer.run()

    plot_curriculum(trainer)

    # ── print summary table ──────────────────
    print("\n" + "═"*62)
    print(f"  {'Stage':<28} {'Success':>9} {'Mean reward':>13}")
    print("═"*62)
    for stage in STAGES:
        r = curriculum_results[stage["name"]]
        print(f"  {stage['name']:<28}  {r['success_rate']*100:>7.1f}%  {r['mean_reward']:>12.3f}")
    print("═"*62)

    # ── optional baseline comparison ─────────
    if args.compare:
        print("\n" + "═"*52)
        print("  BASELINE — training directly on 8×8 slippery")
        print("═"*52)
        hard_cfg = stage_cfg(base_cfg, STAGES[-1])
        baseline_trainer = IPPOTrainer(
            cfg=hard_cfg,
            total_timesteps=sum(s["timesteps"] for s in STAGES),
            save_dir=str(SAVE_BASE / "baseline"),
        )
        baseline_trainer.train()
        baseline_result = baseline_trainer.evaluate(n_episodes=EVAL_EPISODES)
        plot_comparison(curriculum_results, baseline_result)

    if args.render:
        trainer.render(n_episodes=args.episodes)


if __name__ == "__main__":
    main()
