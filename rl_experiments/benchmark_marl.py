"""
benchmark_marl.py — train and evaluate IPPO across multiple map configurations
===============================================================================
Runs three scenarios back-to-back and prints a comparison table.

Usage:
    python benchmark_marl.py
"""

import copy
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from utils import load_config
from train_marl import IPPOTrainer

SCENARIOS = [
    {"name": "4×4  non-slippery", "size": 4, "is_slippery": False},
    {"name": "6×6  non-slippery", "size": 6, "is_slippery": False},
    {"name": "8×8  slippery",     "size": 8, "is_slippery": True},
]

TIMESTEPS = 500_000
EVAL_EPISODES = 50


def run_scenario(base_cfg: dict, scenario: dict) -> dict:
    cfg = copy.deepcopy(base_cfg)
    cfg["frozenlake"]["size"] = scenario["size"]
    cfg["frozenlake"]["is_slippery"] = scenario["is_slippery"]

    save_dir = f"results/benchmark/{scenario['name'].replace(' ', '_')}"

    trainer = IPPOTrainer(cfg=cfg, total_timesteps=TIMESTEPS, save_dir=save_dir)
    trainer.train()
    trainer.save()

    result = trainer.evaluate(n_episodes=EVAL_EPISODES)
    result["reward_logs"] = trainer._reward_logs
    return result


def print_table(scenarios, results):
    print("\n" + "═" * 62)
    print(f"  {'Scenario':<22} {'Success':>9} {'Mean reward':>13} {'Std':>8}")
    print("═" * 62)
    for scenario, res in zip(scenarios, results):
        print(
            f"  {scenario['name']:<22}"
            f"  {res['success_rate']*100:>7.1f}%"
            f"  {res['mean_reward']:>12.3f}"
            f"  {res['std_reward']:>7.3f}"
        )
    print("═" * 62 + "\n")


def plot_comparison(scenarios, results):
    colors = ["#3b8be0", "#e06b3b", "#3b9e60"]
    window = 10

    fig, axes = plt.subplots(1, 3, figsize=(15, 4), sharey=True)
    fig.suptitle("IPPO — Training curves across map sizes", fontsize=13, fontweight="bold")

    for ax, scenario, res, color in zip(axes, scenarios, results, colors):
        logs = res["reward_logs"]
        ax.set_title(scenario["name"], fontsize=11)

        for agent_name, rewards in logs.items():
            ax.plot(rewards, color=color, linewidth=0.4, alpha=0.25)
            if len(rewards) >= window:
                smoothed = np.convolve(rewards, np.ones(window) / window, mode="valid")
                ax.plot(range(window - 1, len(rewards)), smoothed,
                        color=color, linewidth=2,
                        label=agent_name if agent_name == "agent_0" else None)

        ax.axhline(0, color="#aaaaaa", linewidth=0.8, linestyle="--")
        ax.set_xlabel("Rollout")
        ax.spines[["top", "right"]].set_visible(False)

    axes[0].set_ylabel("Mean episode reward")

    out = Path("results/benchmark/comparison.png")
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out, dpi=150)
    print(f"  Saved comparison plot → {out}")
    plt.show()
    plt.close()


def main():
    base_cfg = load_config("config.yaml")
    results = []

    for scenario in SCENARIOS:
        print(f"\n{'━'*52}")
        print(f"  Scenario: {scenario['name']}")
        print(f"{'━'*52}")
        results.append(run_scenario(base_cfg, scenario))

    print_table(SCENARIOS, results)
    plot_comparison(SCENARIOS, results)


if __name__ == "__main__":
    main()
