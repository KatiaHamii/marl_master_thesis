"""
ablation_marl.py — ablation study: how does the number of agents affect performance?
======================================================================================
Conditions:
  1. Random baseline  — no training, agents pick random actions
  2. 1 agent          — single agent
  3. 2 agents         — agent_0 + agent_1
  4. 3 agents         — agent_0 + agent_1 + agent_2

All conditions use the same map config (from config.yaml).
Results are printed as a table and saved as a bar chart.

Usage:
    python ablation_marl.py
"""

import copy
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from utils import load_config
from train_marl import IPPOTrainer, make_vec_env
from envs.frozen_lake_marl import env_creator
from stable_baselines3 import PPO

TIMESTEPS   = 500_000
EVAL_EPISODES = 50
SAVE_BASE   = Path("results/ablation")


# ─────────────────────────────────────────────
#  RANDOM BASELINE
# ─────────────────────────────────────────────
def eval_random(cfg: dict, n_episodes: int) -> dict:
    """Agents pick uniformly random actions — no training."""
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
                env.step(env.action_space(agent).sample())
        ep_rewards.append(ep_reward)
        if ep_reward >= 1.0:
            successes += 1

    env.close()
    return {
        "success_rate": successes / n_episodes,
        "mean_reward":  float(np.mean(ep_rewards)),
        "std_reward":   float(np.std(ep_rewards)),
    }


# ─────────────────────────────────────────────
#  IPPO WITH N AGENTS
# ─────────────────────────────────────────────
def run_n_agents(cfg: dict, n: int) -> dict:
    """Train and evaluate with exactly n agents."""
    agent_names = [f"agent_{i}" for i in range(n)]

    # patch the env to only use n agents
    patched_cfg = copy.deepcopy(cfg)

    # monkeypatch: override possible_agents in the env at creation time
    # by passing n_agents through cfg so FrozenLakeMARLEnv can read it
    patched_cfg["marl_n_agents"] = n

    save_dir = SAVE_BASE / f"{n}_agents"

    trainer = _NAgentTrainer(
        cfg=patched_cfg,
        agent_names=agent_names,
        total_timesteps=TIMESTEPS,
        save_dir=str(save_dir),
    )
    trainer.train()
    trainer.save()
    result = trainer.evaluate(n_episodes=EVAL_EPISODES)
    return result


# ─────────────────────────────────────────────
#  TRAINER SUBCLASS THAT SUPPORTS N AGENTS
# ─────────────────────────────────────────────
class _NAgentTrainer(IPPOTrainer):
    """IPPOTrainer that works with any number of agents (1, 2, or 3)."""

    def __init__(self, cfg, agent_names, total_timesteps, save_dir):
        # bypass IPPOTrainer.__init__ and build manually
        from stable_baselines3.common.vec_env import VecMonitor
        import supersuit as ss
        from pathlib import Path
        from pettingzoo.utils.conversions import aec_to_parallel

        self.cfg = cfg
        self.total_timesteps = total_timesteps
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)

        marl = cfg.get("marl", {})
        self.n_envs     = marl.get("n_envs", 4)
        self.lr         = marl.get("learning_rate", 3e-4)
        self.n_steps    = marl.get("n_steps", 256)
        self.batch_size = marl.get("batch_size", 64)
        self.n_epochs   = marl.get("n_epochs", 4)
        self.clip_range = marl.get("clip_range", 0.2)
        self.ent_coef   = marl.get("ent_coef", 0.01)

        n = len(agent_names)
        print(f"\n{'─'*52}")
        print(f"  IPPO — {n}-agent FrozenLake  (ablation)")
        print(f"  Timesteps : {total_timesteps:,}")
        print(f"{'─'*52}\n")

        self.agents = {
            name: self._make_ppo(name, make_vec_env(cfg, n_envs=self.n_envs))
            for name in agent_names
        }

    def evaluate(self, n_episodes: int = 50) -> dict:
        env = env_creator(cfg=self.cfg)
        # only keep the agents this trainer knows about
        active = set(self.agents.keys())

        successes  = 0
        ep_rewards = []

        for _ in range(n_episodes):
            env.reset()
            ep_reward = 0.0

            for agent in env.agent_iter():
                obs, reward, term, trunc, _ = env.last()
                ep_reward += reward
                if term or trunc:
                    env.step(None)
                elif agent in active:
                    action, _ = self.agents[agent].predict(
                        obs.reshape(1, -1), deterministic=True
                    )
                    env.step(int(action[0]))
                else:
                    # agents not in this condition act randomly
                    env.step(env.action_space(agent).sample())

            ep_rewards.append(ep_reward)
            if ep_reward >= 1.0:
                successes += 1

        env.close()

        result = {
            "success_rate": successes / n_episodes,
            "mean_reward":  float(np.mean(ep_rewards)),
            "std_reward":   float(np.std(ep_rewards)),
        }
        n = len(self.agents)
        print(f"\n── Ablation eval ({n} agents, {n_episodes} episodes) ──")
        print(f"  Success rate : {result['success_rate']*100:.1f}%")
        print(f"  Mean reward  : {result['mean_reward']:.3f} ± {result['std_reward']:.3f}")
        return result


# ─────────────────────────────────────────────
#  TABLE + PLOT
# ─────────────────────────────────────────────
def print_table(conditions, results):
    print("\n" + "═" * 62)
    print(f"  {'Condition':<22} {'Success':>9} {'Mean reward':>13} {'Std':>8}")
    print("═" * 62)
    for cond, res in zip(conditions, results):
        print(
            f"  {cond:<22}"
            f"  {res['success_rate']*100:>7.1f}%"
            f"  {res['mean_reward']:>12.3f}"
            f"  {res['std_reward']:>7.3f}"
        )
    print("═" * 62 + "\n")


def plot_results(conditions, results):
    colors = ["#aaaaaa", "#3b8be0", "#e06b3b", "#3b9e60"]
    success_rates = [r["success_rate"] * 100 for r in results]
    mean_rewards  = [r["mean_reward"]        for r in results]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("Ablation study — effect of number of agents", fontsize=13, fontweight="bold")

    x = range(len(conditions))

    bars1 = ax1.bar(x, success_rates, color=colors, edgecolor="white", linewidth=0.5)
    ax1.set_xticks(x)
    ax1.set_xticklabels(conditions, fontsize=10)
    ax1.set_ylabel("Success rate (%)")
    ax1.set_ylim(0, 105)
    ax1.bar_label(bars1, fmt="%.0f%%", padding=3, fontsize=10)
    ax1.spines[["top", "right"]].set_visible(False)

    bars2 = ax2.bar(x, mean_rewards, color=colors, edgecolor="white", linewidth=0.5)
    ax2.set_xticks(x)
    ax2.set_xticklabels(conditions, fontsize=10)
    ax2.set_ylabel("Mean episode reward")
    ax2.axhline(0, color="#aaaaaa", linewidth=0.8, linestyle="--")
    ax2.bar_label(bars2, fmt="%.2f", padding=3, fontsize=10)
    ax2.spines[["top", "right"]].set_visible(False)

    SAVE_BASE.mkdir(parents=True, exist_ok=True)
    out = SAVE_BASE / "ablation.png"
    plt.tight_layout()
    plt.savefig(out, dpi=150)
    print(f"  Saved plot → {out}")
    plt.show()
    plt.close()


# ─────────────────────────────────────────────
#  RENDER ALL CONDITIONS
# ─────────────────────────────────────────────
def render_condition(cfg: dict, label: str, agents: dict, n_episodes: int = 3):
    """Render one ablation condition in a pygame window."""
    from envs.frozen_lake_marl import FrozenLakeMARLEnv
    import pygame

    env = FrozenLakeMARLEnv(cfg=cfg, render_mode="human")
    active = set(agents.keys())
    # only draw the trained agents — hides random/inactive ones from view
    env.highlight_agents = active if active else None

    print(f"\n  ── Rendering: {label} ──")
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
            elif agent in active:
                action, _ = agents[agent].predict(obs.reshape(1, -1), deterministic=False)
                env.step(int(action[0]))
            else:
                env.step(env.action_space(agent).sample())

            # allow closing the window mid-render
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    env.close()
                    return

        result = "SUCCESS" if ep_reward >= 1.0 else "failed"
        print(f"    Episode {ep+1}: {result}  (reward={ep_reward:.2f}, steps={step})")

    env.close()


def render_all(cfg: dict, n_episodes: int = 3):
    """Load saved agents for each condition and render one after another."""
    from stable_baselines3 import PPO
    from envs.frozen_lake_marl import env_creator

    # Random — no agents to load
    render_condition(cfg, "Random baseline", agents={}, n_episodes=n_episodes)

    # 1, 2, 3 trained agents
    for n in [1, 2, 3]:
        save_dir = SAVE_BASE / f"{n}_agents"
        agents = {}
        for i in range(n):
            path = save_dir / f"agent_{i}.zip"
            if path.exists():
                # build a throwaway env just to get the obs/action space
                dummy_env = make_vec_env(cfg, n_envs=1)
                agents[f"agent_{i}"] = PPO.load(str(path), env=dummy_env)
            else:
                print(f"  Warning: {path} not found — run without --render first.")
                return
        render_condition(cfg, f"{n} agent{'s' if n > 1 else ''}", agents, n_episodes)


# ─────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────
def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--render", action="store_true", help="Render all conditions after training")
    parser.add_argument("--render-only", action="store_true", help="Skip training, just render saved agents")
    parser.add_argument("--episodes", type=int, default=3, help="Episodes to render per condition")
    args = parser.parse_args()

    cfg = load_config("config.yaml")

    if args.render_only:
        render_all(cfg, n_episodes=args.episodes)
        return

    conditions = ["Random", "1 agent", "2 agents", "3 agents"]
    results    = []

    print("\n── Condition: Random baseline ──")
    results.append(eval_random(cfg, EVAL_EPISODES))

    for n in [1, 2, 3]:
        results.append(run_n_agents(cfg, n))

    print_table(conditions, results)
    plot_results(conditions, results)

    if args.render:
        render_all(cfg, n_episodes=args.episodes)


if __name__ == "__main__":
    main()
