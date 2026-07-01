"""
train.py — IPPO training for OvercookedV2 (7-channel compact obs)
==================================================================
Run from repo root (MARL/):

  # quick smoke-test (few steps, prints progress)
  uv run --project overcooked_v2_rethink python overcooked_v2_rethink/train.py

  # full run
  uv run --project overcooked_v2_rethink python overcooked_v2_rethink/train.py \\
      --layout cramped_room_v2 --steps 2000000 --envs 16 --lr 1e-4

    JAX_PLATFORMS=cpu uv run --project overcooked_v2_rethink \
    python overcooked_v2_rethink/train.py \
    --layout asymm_advantages_recipes_center \
    --steps 5000000 --envs 16


  # partial observability (5×5 window around each agent)
  uv run --project overcooked_v2_rethink python overcooked_v2_rethink/train.py \\
      --layout cramped_room_v2 --view-size 2 --steps 3000000

    JAX_PLATFORMS=cpu uv run --project overcooked_v2_rethink \
    python overcooked_v2_rethink/train.py \
    --layout asymm_advantages_recipes_center \
    --reward-mode shaped \
    --view-size 1 \
    --steps 5000000 --envs 16 \
    --checkpoint-every 100

    JAX_PLATFORMS=cpu uv run --project overcooked_v2_rethink \
    python overcooked_v2_rethink/train.py \
    --layout asymm_advantages_recipes_center \    --reward-mode sparse \
    --steps 5000000 --envs 16 \
    --load-checkpoint overcooked_v2_rethink/results/ippo/asymm_advantages_recipes_center_full_sparse/2026-05-31_18-55-44

    JAX_PLATFORMS=cpu uv run --project overcooked_v2_rethink \
    python overcooked_v2_rethink/train.py \
    --layout asymm_advantages_recipes_center \
    --view-size 2 \
    --steps 5000000 --envs 16

  # auto-curriculum: agents progress through layouts based on performance
  JAX_PLATFORMS=cpu uv run --project overcooked_v2_rethink \
    python overcooked_v2_rethink/train.py \
    --curriculum \
    --curriculum-window 50 \
    --steps 5000000 --envs 16

  # list all layouts
  uv run --project overcooked_v2_rethink python overcooked_v2_rethink/train.py \\
      --list-layouts
"""

import os, sys

os.environ.setdefault("JAX_PLATFORMS", "cpu")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import csv
import json
import pickle
import subprocess
import time
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import jax
import jax.numpy as jnp

from overcooked_v2_rethink import OvercookedV2, overcooked_v2_layouts
from overcooked_v2_rethink.ippo_jax import IPPOTrainer

from overcooked_v2_rethink.curriculum import (
    CurriculumManager,
    make_default_curriculum,
)
from overcooked_v2_rethink.ued_trainer import UEDTrainer
from overcooked_v2_rethink.accel_trainer import ACCELTrainer

DEFAULT_CFG = {
    "n_envs": 16,
    "rollout_len": 256,
    "n_epochs": 4,
    "batch_size": 256,
    "lr": 1e-4,
    "gamma": 0.99,
    "gae_lambda": 0.95,
    "clip_eps": 0.2,
    "vf_coef": 0.5,
    "ent_coef": 0.05,  # higher than before → less premature convergence
    "max_grad_norm": 0.5,
    "shaped_reward_scale": 1.0,
    "step_penalty": 0.0,
    "collision_penalty": 0.0,
    "wrong_ingredient_penalty": 0.0,
    "total_steps": 5_000_000,
    "log_every": 10,
    "hidden_size": 128,
}


def _fmt_steps(n: int) -> str:
    """Format a step count as e.g. '5M', '500K', '1.5M'."""
    if n >= 1_000_000:
        v = n / 1_000_000
        return f"{v:.0f}M" if v == int(v) else f"{v:.1f}M"
    if n >= 1_000:
        v = n / 1_000
        return f"{v:.0f}K" if v == int(v) else f"{v:.1f}K"
    return str(n)


def _hparam_text(cfg: dict) -> str:
    """Format cfg dict as a compact 3-line hyperparameter string for the plot."""
    spu = cfg.get("n_envs", "?") * cfg.get("rollout_len", "?")
    lines = [
        (
            f"steps: {cfg.get('total_steps', '?'):,}   "
            f"envs: {cfg.get('n_envs', '?')}   "
            f"rollout_len: {cfg.get('rollout_len', '?')}   "
            f"samples/upd: {spu:,}   "
            f"reward: {cfg.get('reward_mode', '?')}"
        ),
        (
            f"lr: {cfg.get('lr', '?')}   "
            f"ent_coef: {cfg.get('ent_coef', '?')}   "
            f"clip_eps: {cfg.get('clip_eps', '?')}   "
            f"vf_coef: {cfg.get('vf_coef', '?')}   "
            f"max_grad_norm: {cfg.get('max_grad_norm', '?')}"
        ),
        (
            f"epochs: {cfg.get('n_epochs', '?')}   "
            f"batch_size: {cfg.get('batch_size', '?')}   "
            f"gamma: {cfg.get('gamma', '?')}   "
            f"gae_lambda: {cfg.get('gae_lambda', '?')}   "
            f"hidden: {cfg.get('hidden_size', '?')}"
        ),
    ]
    return "\n".join(lines)


def plot_curves(
    log,
    out_dir: Path,
    title: str,
    filename: str = "training_curve.png",
    cfg: dict = None,
    ued_trainer=None,  # Pass UED trainer for curriculum metrics
):
    """Save training curves (mean return + PPO loss + deliveries + curriculum). Called live during training."""
    steps = [r["steps"] for r in log]
    mean_r = [r["mean_r"] for r in log]
    losses = [r["loss"] for r in log]
    deliveries = [r.get("total_deliveries", 0) for r in log]

    has_hparams = cfg is not None
    has_deliveries = "total_deliveries" in log[0]
    has_curriculum = ued_trainer is not None  # Always show curriculum subplot if UED is enabled

    # Adjust height ratios based on what we have
    if has_hparams and has_deliveries and has_curriculum:
        height_ratios = [3, 2, 1.5, 1.5, 0.7]  # return, loss, deliveries, curriculum, hparams
        n_rows = 5
    elif has_hparams and has_deliveries:
        height_ratios = [3, 2, 1.5, 0.7]  # return, loss, deliveries, hparams
        n_rows = 4
    elif has_hparams and has_curriculum:
        height_ratios = [3, 2, 1.5, 0.7]  # return, loss, curriculum, hparams
        n_rows = 4
    elif has_deliveries and has_curriculum:
        height_ratios = [3, 2, 1.5, 1.5]  # return, loss, deliveries, curriculum
        n_rows = 4
    elif has_deliveries:
        height_ratios = [3, 2, 1.5]  # return, loss, deliveries
        n_rows = 3
    elif has_curriculum:
        height_ratios = [3, 2, 1.5]  # return, loss, curriculum
        n_rows = 3
    elif has_hparams:
        height_ratios = [3, 2, 0.7]  # return, loss, hparams
        n_rows = 3
    else:
        height_ratios = [3, 2]  # return, loss only
        n_rows = 2
    fig, axes = plt.subplots(
        n_rows,
        1,
        figsize=(
            10,
            10 if (has_hparams and has_deliveries) else 9 if has_deliveries else 8,
        ),
        gridspec_kw={"height_ratios": height_ratios},
        sharex=False,
    )

    # Assign axes based on what we're plotting
    ax1 = axes[0]
    ax2 = axes[1]

    ax_deliv = None
    ax_curriculum = None
    ax_hp = None

    if has_deliveries and has_curriculum and has_hparams:
        ax_deliv = axes[2]
        ax_curriculum = axes[3]
        ax_hp = axes[4]
    elif has_deliveries and has_curriculum:
        ax_deliv = axes[2]
        ax_curriculum = axes[3]
    elif has_deliveries and has_hparams:
        ax_deliv = axes[2]
        ax_hp = axes[3]
    elif has_curriculum and has_hparams:
        ax_curriculum = axes[2]
        ax_hp = axes[3]
    elif has_deliveries:
        ax_deliv = axes[2]
    elif has_curriculum:
        ax_curriculum = axes[2]
    elif has_hparams:
        ax_hp = axes[2]

    ax1.plot(steps, mean_r, lw=1.0, color="steelblue", alpha=0.6, label="mean return")
    window = max(1, len(mean_r) // 10)
    smoothed = np.convolve(mean_r, np.ones(window) / window, "same")
    ax1.plot(
        steps, smoothed, lw=2, color="darkorange", label=f"smoothed ({window}-upd avg)"
    )
    ax1.set_ylabel("Mean Return")
    ax1.legend(fontsize=8)
    ax1.grid(alpha=0.3)

    ax2.plot(steps, losses, lw=1.0, color="mediumpurple", label="PPO loss")
    ax2.set_ylabel("Loss")
    ax2.legend(fontsize=8)
    ax2.grid(alpha=0.3)

    # Plot deliveries per update period (as spikes/bars)
    if ax_deliv is not None:
        # Calculate deliveries per update (difference between consecutive cumulative values)
        deliveries_per_update = [0] + [
            deliveries[i] - deliveries[i - 1] for i in range(1, len(deliveries))
        ]

        # Plot as bar chart (spikes)
        ax_deliv.bar(
            steps,
            deliveries_per_update,
            width=max(steps) * 0.015,
            color="green",
            alpha=0.7,
            label="deliveries per update",
        )
        ax_deliv.set_ylabel("Deliveries / Update")
        ax_deliv.set_xlabel("Environment steps")
        ax_deliv.legend(fontsize=8)
        ax_deliv.grid(alpha=0.3, axis="y")

    # Plot curriculum progression (UED)
    if ax_curriculum is not None and ued_trainer is not None:
        # Get curriculum check points
        check_steps = []
        difficulties = []
        learning_speeds = []
        curriculum_levels = []

        for i, m in enumerate(ued_trainer.mutation_history):
            check_steps.append(m['episode'])
            curriculum_levels.append(i + 1)  # Level 1, 2, 3, etc.
            if i < len(ued_trainer.difficulty_history):
                difficulties.append(ued_trainer.difficulty_history[i])
            if i < len(ued_trainer.learning_speed_history):
                learning_speeds.append(ued_trainer.learning_speed_history[i])

        if check_steps:
            # Plot curriculum level (step function)
            ax_curriculum.step(check_steps, curriculum_levels, where='post', lw=2.5, color="darkblue", label="curriculum level", marker='o', markersize=5)
            ax_curriculum.set_ylabel("Curriculum Level", color="darkblue", fontweight="bold")
            ax_curriculum.tick_params(axis='y', labelcolor="darkblue")
            ax_curriculum.grid(alpha=0.3)

            # Plot difficulty and learning_speed on secondary axis
            ax_curr2 = ax_curriculum.twinx()
            ax_curr2.plot(check_steps, difficulties, lw=2.5, color="crimson", label="difficulty (mean return)", marker='^', markersize=5, alpha=0.8)
            if learning_speeds:
                ax_curr2.plot(check_steps, learning_speeds, lw=2.5, color="orange", label="learning_speed", marker='s', markersize=4, alpha=0.8, linestyle='--')
            ax_curr2.set_ylabel("Difficulty / Learning Speed", color="darkred", fontweight="bold")
            ax_curr2.tick_params(axis='y', labelcolor="darkred")

            # Combined legend
            lines1, labels1 = ax_curriculum.get_legend_handles_labels()
            lines2, labels2 = ax_curr2.get_legend_handles_labels()
            ax_curriculum.legend(lines1 + lines2, labels1 + labels2, loc="upper left", fontsize=8)

            ax_curriculum.set_xlabel("Episode")

            # Mark mutations with vertical lines and annotations
            for i, m in enumerate(ued_trainer.mutation_history):
                ax_curriculum.axvline(x=m['episode'], color="red", linestyle="--", alpha=0.3, linewidth=1)
                # Add small annotation for mutation event
                ax_curriculum.text(m['episode'], ax_curriculum.get_ylim()[1] * 0.95, f"M{i+1}",
                                  fontsize=7, ha='center', color='red', alpha=0.6)
        else:
            # No mutations yet - show placeholder
            ax_curriculum.text(0.5, 0.5, "Waiting for first mutation...\n(threshold check every 50 episodes)",
                              ha='center', va='center', transform=ax_curriculum.transAxes,
                              fontsize=10, color='gray', alpha=0.6)
            ax_curriculum.set_ylabel("Curriculum Level", color="darkblue", fontweight="bold")
            ax_curriculum.set_xlabel("Episode")
            ax_curriculum.grid(alpha=0.3)

    # Per-component shaped rewards (only present in shaped mode)
    sr_keys = {
        "sr_dish": ("Dish pickup", "gold"),
        "sr_pot": ("Placement in pot", "tomato"),
        "sr_pot_start": ("Pot start", "mediumseagreen"),
        "sr_plate": ("Plate pickup", "cornflowerblue"),
    }
    if any(sr_keys.keys() & set(log[0].keys())):
        # Rebuild figure to include shaped rewards (and curriculum, deliveries, hparams if available)
        # Layout: return, loss, (deliveries?), (curriculum?), shaped_rewards, (hparams?)

        # Calculate new height ratios - account for curriculum!
        if has_curriculum and has_deliveries and has_hparams:
            hr_new = [3, 2, 1.5, 1.5, 1.5, 0.7]  # return, loss, deliveries, curriculum, sr, hparams
        elif has_curriculum and has_deliveries:
            hr_new = [3, 2, 1.5, 1.5, 1.5]  # return, loss, deliveries, curriculum, sr
        elif has_curriculum and has_hparams:
            hr_new = [3, 2, 1.5, 1.5, 0.7]  # return, loss, curriculum, sr, hparams
        elif has_curriculum:
            hr_new = [3, 2, 1.5, 1.5]  # return, loss, curriculum, sr
        elif has_deliveries and has_hparams:
            hr_new = [3, 2, 1.5, 1.5, 0.7]  # return, loss, deliveries, sr, hparams
        elif has_deliveries:
            hr_new = [3, 2, 1.5, 1.5]  # return, loss, deliveries, sr
        elif has_hparams:
            hr_new = [3, 2, 1.5, 0.7]  # return, loss, sr, hparams
        else:
            hr_new = [3, 2, 1.5]  # return, loss, sr

        # Close and recreate
        plt.close(fig)
        fig, axes = plt.subplots(
            len(hr_new),
            1,
            figsize=(10, sum(hr_new) * 1.3),
            gridspec_kw={"height_ratios": hr_new},
            sharex=False,
        )

        # Assign axes (account for curriculum!)
        ax1 = axes[0]
        ax2 = axes[1]
        ax_deliv = None
        ax_curriculum = None
        ax3_sr = None
        ax_hp = None
        axis_idx = 2

        if has_deliveries:
            ax_deliv = axes[axis_idx]
            axis_idx += 1
        if has_curriculum:
            ax_curriculum = axes[axis_idx]
            axis_idx += 1
        ax3_sr = axes[axis_idx]
        axis_idx += 1
        if has_hparams:
            ax_hp = axes[axis_idx]

        ax1.plot(
            steps, mean_r, lw=1.0, color="steelblue", alpha=0.6, label="mean return"
        )
        ax1.plot(
            steps,
            smoothed,
            lw=2,
            color="darkorange",
            label=f"smoothed ({window}-upd avg)",
        )
        ax1.set_ylabel("Mean Return")
        ax1.legend(fontsize=8)
        ax1.grid(alpha=0.3)

        ax2.plot(steps, losses, lw=1.0, color="mediumpurple", label="PPO loss")
        ax2.set_ylabel("Loss")
        ax2.legend(fontsize=8)
        ax2.grid(alpha=0.3)

        # Plot deliveries per update (if available)
        if ax_deliv is not None:
            deliveries_per_update = [0] + [
                deliveries[i] - deliveries[i - 1] for i in range(1, len(deliveries))
            ]
            ax_deliv.bar(
                steps,
                deliveries_per_update,
                width=max(steps) * 0.015,
                color="green",
                alpha=0.7,
                label="deliveries per update",
            )
            ax_deliv.set_ylabel("Deliveries / Update")
            ax_deliv.legend(fontsize=8)
            ax_deliv.grid(alpha=0.3, axis="y")

        for key, (label, color) in sr_keys.items():
            vals = [r.get(key, 0.0) for r in log]
            if any(v > 0 for v in vals):
                ax3_sr.plot(steps, vals, lw=1.2, label=label, color=color)
        # In shaped mode these signals ARE added to the reward; in
        # delivery/sparse mode they are only measured (not rewarded).
        reward_mode = (cfg or {}).get("reward_mode", "shaped")
        if reward_mode == "shaped":
            ax3_sr.set_ylabel("Shaped reward\n(given, mean/step)")
            ax3_sr.set_title("Shaped reward components (added to reward)", fontsize=9)
        else:
            ax3_sr.set_ylabel("Sub-skill activity\n(measured, mean/step)")
            ax3_sr.set_title(
                f"Sub-skill activity — NOT rewarded ({reward_mode} mode); shown for diagnostics only",
                fontsize=9,
            )
        ax3_sr.set_xlabel("Environment steps")
        ax3_sr.legend(fontsize=7, ncol=2)
        ax3_sr.grid(alpha=0.3)

        # Plot curriculum progression (UED) - after figure reconstruction
        if ax_curriculum is not None and ued_trainer is not None:
            check_steps = []
            difficulties = []
            learning_speeds = []
            curriculum_levels = []

            for i, m in enumerate(ued_trainer.mutation_history):
                check_steps.append(m['episode'])
                curriculum_levels.append(i + 1)
                if i < len(ued_trainer.difficulty_history):
                    difficulties.append(ued_trainer.difficulty_history[i])
                if i < len(ued_trainer.learning_speed_history):
                    learning_speeds.append(ued_trainer.learning_speed_history[i])

            if check_steps:
                ax_curriculum.step(check_steps, curriculum_levels, where='post', lw=2.5, color="darkblue", label="curriculum level", marker='o', markersize=5)
                ax_curriculum.set_ylabel("Curriculum Level", color="darkblue", fontweight="bold")
                ax_curriculum.tick_params(axis='y', labelcolor="darkblue")
                ax_curriculum.grid(alpha=0.3)

                ax_curr2 = ax_curriculum.twinx()
                ax_curr2.plot(check_steps, difficulties, lw=2.5, color="crimson", label="difficulty (mean return)", marker='^', markersize=5, alpha=0.8)
                if learning_speeds:
                    ax_curr2.plot(check_steps, learning_speeds, lw=2.5, color="orange", label="learning_speed", marker='s', markersize=4, alpha=0.8, linestyle='--')
                ax_curr2.set_ylabel("Difficulty / Learning Speed", color="darkred", fontweight="bold")
                ax_curr2.tick_params(axis='y', labelcolor="darkred")

                lines1, labels1 = ax_curriculum.get_legend_handles_labels()
                lines2, labels2 = ax_curr2.get_legend_handles_labels()
                ax_curriculum.legend(lines1 + lines2, labels1 + labels2, loc="upper left", fontsize=8)
                ax_curriculum.set_xlabel("Episode")

                for i, m in enumerate(ued_trainer.mutation_history):
                    ax_curriculum.axvline(x=m['episode'], color="red", linestyle="--", alpha=0.3, linewidth=1)
                    ax_curriculum.text(m['episode'], ax_curriculum.get_ylim()[1] * 0.95, f"M{i+1}",
                                      fontsize=7, ha='center', color='red', alpha=0.6)
            else:
                ax_curriculum.text(0.5, 0.5, "Waiting for first mutation...",
                                  ha='center', va='center', transform=ax_curriculum.transAxes,
                                  fontsize=10, color='gray', alpha=0.6)
                ax_curriculum.set_ylabel("Curriculum Level", color="darkblue", fontweight="bold")
                ax_curriculum.set_xlabel("Episode")
                ax_curriculum.grid(alpha=0.3)

        if ax_hp is not None:
            ax_hp.axis("off")
            ax_hp.text(
                0.5,
                0.5,
                _hparam_text(cfg),
                transform=ax_hp.transAxes,
                ha="center",
                va="center",
                fontsize=7.5,
                fontfamily="monospace",
                bbox=dict(boxstyle="round,pad=0.5", facecolor="#f5f5dc", alpha=0.7),
            )

        fig.suptitle(title, fontsize=11)
        fig.tight_layout()
        fig.savefig(out_dir / filename, dpi=120)
        plt.close(fig)
        return

    if not any(sr_keys.keys() & set(log[0].keys())):
        ax2.set_xlabel("Environment steps")

    if has_hparams:
        ax3 = axes[2]
        ax3.axis("off")
        ax3.text(
            0.5,
            0.5,
            _hparam_text(cfg),
            transform=ax3.transAxes,
            ha="center",
            va="center",
            fontsize=7.5,
            fontfamily="monospace",
            bbox=dict(boxstyle="round,pad=0.5", facecolor="#f5f5dc", alpha=0.7),
        )

    fig.suptitle(title, fontsize=11)
    fig.tight_layout()
    fig.savefig(out_dir / filename, dpi=120)
    plt.close(fig)


def make_live_plot_callback(out_dir: Path, title: str, filename: str, cfg: dict = None, ued_trainer=None):
    """Returns a callback that re-saves the plot after every log entry."""
    out_dir.mkdir(parents=True, exist_ok=True)

    def _callback(log):
        plot_curves(log, out_dir, title, filename, cfg, ued_trainer)

    return _callback


def save_results(
    log, params_0, params_1, out_dir: Path, title: str, filename: str, cfg: dict = None, ued_trainer=None
):
    out_dir.mkdir(parents=True, exist_ok=True)

    plot_curves(log, out_dir, title, filename, cfg, ued_trainer)

    with open(out_dir / "log.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=log[0].keys())
        writer.writeheader()
        writer.writerows(log)

    with open(out_dir / "params_agent0.pkl", "wb") as f:
        pickle.dump(params_0, f)
    with open(out_dir / "params_agent1.pkl", "wb") as f:
        pickle.dump(params_1, f)

    print(f"\nResults saved to {out_dir}/")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layout", default="cramped_room_v2")
    ap.add_argument("--steps", type=int, default=DEFAULT_CFG["total_steps"])
    ap.add_argument("--envs", type=int, default=DEFAULT_CFG["n_envs"])
    ap.add_argument("--lr", type=float, default=DEFAULT_CFG["lr"])
    ap.add_argument("--ent-coef", type=float, default=DEFAULT_CFG["ent_coef"])
    ap.add_argument("--hidden", type=int, default=DEFAULT_CFG["hidden_size"])
    ap.add_argument(
        "--reward-mode",
        default="shaped",
        choices=["shaped", "delivery", "sparse"],
        help="'shaped': dense rewards (default). "
        "'delivery': +1/delivery team reward, no shaping. "
        "'sparse': +20/delivery team reward, no shaping.",
    )
    ap.add_argument(
        "--obs-mode",
        default="rich",
        choices=["rich", "simple"],
        help="'rich': full ObsPreprocessor (51 channels, bit-decoded). "
        "'simple': minimal normalization (7 channels, raw). "
        "Default: rich",
    )
    ap.add_argument(
        "--shaped",
        type=float,
        default=DEFAULT_CFG["shaped_reward_scale"],
        help="Shaped reward scale, only used in shaped mode (0=off, 1=full)",
    )
    ap.add_argument(
        "--view-size",
        type=int,
        default=None,
        help="Agent view radius for partial obs (None=full grid)",
    )
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument(
        "--checkpoint-every",
        type=int,
        default=100,
        help="Save params checkpoint every N updates (default 100, 0 = off)",
    )
    ap.add_argument(
        "--load-checkpoint",
        default=None,
        help="Path to checkpoint directory to resume from (e.g., results/ippo/cramped_room_v2_full_shaped/2026-05-31_10-30-45)",
    )
    ap.add_argument(
        "--policy",
        default="ippo",
        help="Algorithm label used as top-level results sub-folder (default: ippo)",
    )
    ap.add_argument("--out-dir", default=str(Path(__file__).parent / "results"))
    ap.add_argument("--list-layouts", action="store_true")
    ap.add_argument(
        "--curriculum",
        action="store_true",
        help="Enable auto-curriculum (agents progress through layouts based on performance)",
    )
    ap.add_argument(
        "--curriculum-window",
        type=int,
        default=50,
        help="Episode window for averaging performance in curriculum (default 50)",
    )
    ap.add_argument(
        "--ued",
        action="store_true",
        help="Enable UED (Unsupervised Environment Design) with teacher-guided mutations",
    )
    ap.add_argument(
        "--ued-teacher-lr",
        type=float,
        default=5e-4,
        help="Teacher learning rate for UED (default 1e-4)",
    )
    ap.add_argument(
        "--ued-update-freq",
        type=int,
        default=10,
        help="Teacher updates every N student updates (default 10, was 100)",
    )
    ap.add_argument(
        "--ued-plateau-window",
        type=int,
        default=5,
        help="Episodes window for computing learning speed (default 50)",
    )
    ap.add_argument(
        "--ued-curriculum-mode",
        type=str,
        default="progressive",
        choices=["progressive", "plateau"],
        help="Curriculum mode: 'progressive' (episode-based) or 'plateau' (learning-speed-based)",
    )
    ap.add_argument(
        "--ued-mutation-freq",
        type=int,
        default=50,
        help="Check for mutations every N episodes (progressive mode, default 50)",
    )
    ap.add_argument(
        "--ued-success-threshold",
        type=float,
        default=25.0,
        help="Mean return threshold for applying mutations (progressive mode, default 25.0)",
    )
    # ── ACCEL arguments ────────────────────────────────────────────────────────
    ap.add_argument(
        "--accel",
        action="store_true",
        help="Enable ACCEL (Adversarially Compounding Complexity by Editing Levels)",
    )
    ap.add_argument(
        "--accel-buffer-size",
        type=int,
        default=100,
        help="ACCEL level buffer capacity K (default 100)",
    )
    ap.add_argument(
        "--accel-fill-ratio",
        type=float,
        default=0.5,
        help="ACCEL initial buffer fill ratio ρ (default 0.5)",
    )
    ap.add_argument(
        "--accel-replay-prob",
        type=float,
        default=0.5,
        help="ACCEL replay probability P(d=1) (default 0.5)",
    )
    ap.add_argument(
        "--accel-score-threshold",
        type=float,
        default=0.05,
        help="ACCEL minimum PVL score to enter buffer (default 0.05)",
    )
    ap.add_argument(
        "--accel-edit-step",
        type=float,
        default=0.1,
        help="ACCEL Gaussian noise std for EnvParams editing (default 0.1)",
    )
    args = ap.parse_args()

    if args.list_layouts:
        print("Available layouts:")
        for name, layout in overcooked_v2_layouts.items():
            print(
                f"  {name:45s} {layout.height}×{layout.width}  "
                f"{layout.num_ingredients} ingredient(s)"
            )
        return

    # Initialize curriculum if enabled
    curriculum_mgr = None
    initial_layout = args.layout

    # UED/ACCEL defaults: use partial observability
    if args.ued or args.accel:
        if args.view_size is None:
            args.view_size = 2  # Partial observability for UED/ACCEL (default)

    cfg = {
        **DEFAULT_CFG,
        "total_steps": args.steps,
        "n_envs": args.envs,
        "lr": args.lr,
        "ent_coef": args.ent_coef,
        "hidden_size": args.hidden,
        "reward_mode": args.reward_mode,
        "obs_mode": args.obs_mode,
        "shaped_reward_scale": args.shaped,
    }

    if args.curriculum:
        curriculum_mgr = CurriculumManager(
            make_default_curriculum(),
            eval_window=args.curriculum_window,
        )
        initial_layout = curriculum_mgr.current_layout
        print(
            f"🎓 Curriculum enabled: {curriculum_mgr.levels[0].layout_name} → {curriculum_mgr.levels[-1].layout_name}\n"
        )

    env = OvercookedV2(
        layout=initial_layout,
        max_steps=400,
        agent_view_size=args.view_size,
    )

    # Load checkpoint first (if resuming) to extract update number for folder name
    resume_params = None
    resume_tag = ""
    if args.load_checkpoint:
        ckpt_path = Path(args.load_checkpoint)
        if not ckpt_path.exists():
            raise FileNotFoundError(f"Checkpoint dir not found: {ckpt_path}")

        # If path points to a results folder with checkpoints/ subfolder,
        # find the latest checkpoint (prefer "final" if it exists, else latest update_XXXXX)
        if (ckpt_path / "checkpoints").exists():
            # Check for "final" checkpoint first (saved at end of training)
            if (ckpt_path / "checkpoints" / "final").exists():
                ckpt_path = ckpt_path / "checkpoints" / "final"
            else:
                updates = sorted(
                    (ckpt_path / "checkpoints").glob("update_*"),
                    key=lambda p: int(p.name.split("_")[-1]),
                )
                if not updates:
                    raise FileNotFoundError(
                        f"No checkpoints found in {ckpt_path / 'checkpoints'}"
                    )
                ckpt_path = updates[-1]

        p0_file = ckpt_path / "params_agent0.pkl"
        p1_file = ckpt_path / "params_agent1.pkl"
        meta_file = ckpt_path / "checkpoint_metadata.json"

        if not p0_file.exists() or not p1_file.exists():
            raise FileNotFoundError(
                f"Checkpoint missing params files. Expected:\n"
                f"  {p0_file}\n  {p1_file}"
            )

        # Check reward mode consistency and load summary
        if meta_file.exists():
            with open(meta_file, "r") as f:
                ckpt_meta = json.load(f)
                ckpt_reward_mode = ckpt_meta.get("reward_mode", "unknown")
            if ckpt_reward_mode != args.reward_mode:
                raise ValueError(
                    f"❌ Reward mode mismatch!\n"
                    f"  Checkpoint was trained with: {ckpt_reward_mode}\n"
                    f"  You specified:               {args.reward_mode}\n"
                    f"  These must match to continue training.\n"
                    f"  Use: --reward-mode {ckpt_reward_mode}"
                )
            # Display summary of where training left off
            print(f"\n📊 Checkpoint Summary:")
            if "final_update" in ckpt_meta:
                print(f"  Final update: {ckpt_meta['final_update']}")
            if "final_steps" in ckpt_meta:
                print(f"  Final steps:  {ckpt_meta['final_steps']:,}")
            if "final_mean_r" in ckpt_meta:
                print(f"  Final mean_r: {ckpt_meta['final_mean_r']:.3f}")
            if "final_loss" in ckpt_meta:
                print(f"  Final loss:   {ckpt_meta['final_loss']:.4f}")
            if "final_episodes" in ckpt_meta:
                print(f"  Final episodes: {ckpt_meta['final_episodes']}")
            print()

        with open(p0_file, "rb") as f:
            p0 = pickle.load(f)
        with open(p1_file, "rb") as f:
            p1 = pickle.load(f)
        resume_params = {"agent_0": p0, "agent_1": p1}
        # Extract update number from path like ".../checkpoints/update_00500"
        # or read from metadata if it's the "final" checkpoint
        if ckpt_path.name == "final":
            metadata_path = ckpt_path / "checkpoint_metadata.json"
            if metadata_path.exists():
                with open(metadata_path, "r") as f:
                    metadata = json.load(f)
                    update_num = int(metadata.get("final_update", 0))
            else:
                update_num = 0
        else:
            update_num = int(ckpt_path.name.split("_")[-1])
        resume_tag = f"_resumed-from-upd{update_num}"

    obs_tag = f"view{args.view_size}" if args.view_size else "full"

    # Determine folder name based on training mode
    if args.accel:
        layout_for_path = "accel"
    elif args.ued:
        layout_for_path = "ued"
    elif curriculum_mgr:
        layout_for_path = "curriculum"
    else:
        layout_for_path = args.layout

    title = f"OvercookedV2 {args.policy.upper()} — {layout_for_path} ({obs_tag}, {args.reward_mode}, {args.obs_mode})"
    if args.accel:
        args.policy = "accel"

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    out = (
        Path(args.out_dir)
        / args.policy
        / f"{layout_for_path}_{obs_tag}_{args.reward_mode}_{args.obs_mode}"
        / f"{timestamp}{resume_tag}"
    )

    # Update config with curriculum info
    if curriculum_mgr:
        cfg["curriculum_enabled"] = True
        cfg["curriculum_window"] = args.curriculum_window

    samples_per_update = cfg["n_envs"] * cfg["rollout_len"]
    png_name = (
        f"training_curve"
        f"_{_fmt_steps(args.steps)}"
        f"_{cfg['n_envs']}env"
        f"_{samples_per_update}spu"
        f".png"
    )

    print(
        f"\nLayout    : {initial_layout}  {env.height}×{env.width}  obs={env.obs_shape}"
    )
    print(f"Reward    : {args.reward_mode}")
    print(
        f"Envs      : {cfg['n_envs']}  rollout_len={cfg['rollout_len']}  "
        f"-> {samples_per_update:,} samples/update"
    )
    print(
        f"PPO       : lr={cfg['lr']}  ent={cfg['ent_coef']}  "
        f"epochs={cfg['n_epochs']}  batch={cfg['batch_size']}"
    )
    print(
        f"Total     : {args.steps:,} steps  (~{args.steps // samples_per_update} updates)"
    )
    if curriculum_mgr:
        print(
            f"Curriculum: {curriculum_mgr.levels[0].layout_name} -> {curriculum_mgr.levels[-1].layout_name}"
        )
    print(f"Plots     : {out}/{png_name}\n")

    # Choose trainer: ACCEL, UED, or regular IPPO with optional curriculum
    if args.accel:
        # ACCEL mode: level buffer + regret-based editing (Algorithm 1)
        from overcooked_v2_rethink.layouts import overcooked_v2_layouts as _layouts
        _base_layout_obj = _layouts.get(args.layout)
        if _base_layout_obj is None:
            raise ValueError(f"Layout '{args.layout}' not found for ACCEL. "
                             f"Use --list-layouts to see available layouts.")
        # Reconstruct layout string from Layout object's static_objects
        # ACCELTrainer needs the layout string to build ParametrizedOvercooked
        # We use the raw layout string from layouts.py instead
        from overcooked_v2_rethink.overcooked_parametrized_current import _LAYOUTS as _raw_layouts
        _base_layout_str = _raw_layouts.get(args.layout)
        if _base_layout_str is None:
            raise ValueError(
                f"Layout '{args.layout}' not found in parametrized layouts. "
                f"ACCEL requires a layout defined in layouts.py as a string constant."
            )
        trainer = ACCELTrainer(
            env,
            cfg,
            base_layout_str=_base_layout_str,
            buffer_size=args.accel_buffer_size,
            initial_fill_ratio=args.accel_fill_ratio,
            replay_prob=args.accel_replay_prob,
            score_threshold=args.accel_score_threshold,
            edit_step=args.accel_edit_step,
            seed=args.seed,
        )
        print(
            f"[ACCEL] buffer={args.accel_buffer_size}  fill_ratio={args.accel_fill_ratio}  "
            f"replay_prob={args.accel_replay_prob}  score_threshold={args.accel_score_threshold}  "
            f"edit_step={args.accel_edit_step}"
        )
    elif args.ued:
        # UED mode: teacher designs layouts via mutations
        trainer = UEDTrainer(
            env,
            cfg,
            base_layout=env.layout,
            teacher_lr=args.ued_teacher_lr,
            update_frequency=args.ued_update_freq,
            plateau_window=args.ued_plateau_window,
            curriculum_mode=args.ued_curriculum_mode,
            mutation_frequency=args.ued_mutation_freq,
            success_threshold=args.ued_success_threshold,
        )
        print(
            f"UED mode enabled (curriculum={args.ued_curriculum_mode}, "
            f"mutation_freq={args.ued_mutation_freq}, success_threshold={args.ued_success_threshold})"
        )
    else:
        # Regular IPPO, optionally with simple curriculum
        trainer = IPPOTrainer(env, cfg) #curriculum_mgr=curriculum_mgr)

    key = jax.random.PRNGKey(args.seed)
    # Pass UED trainer for curriculum plotting (if enabled, not ACCEL)
    ued_trainer_for_plot = trainer if args.ued else None
    live_plot = make_live_plot_callback(out, title, png_name, cfg, ued_trainer_for_plot)
    ckpt_every = args.checkpoint_every
    ckpt_dir = out if ckpt_every > 0 else None

    if resume_params:
        print(f"Resumed from checkpoint: update {update_num}\n")

    # Track training time
    training_start_time = time.time()
    training_start_datetime = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"Training start: {training_start_datetime}")

    # Start powermetrics for energy monitoring
    # power_log_path = out / "powermetrics.log"
    # powermetrics_proc = None
    # try:
    #     # Start powermetrics in background, piping to results directory
    #     # Note: May prompt for sudo password on first run
    #     with open(power_log_path, "w") as power_log_file:
    #         powermetrics_proc = subprocess.Popen(
    #             ["sudo", "powermetrics", "-n", "1", "-s", "cpu_power", "-i", "1000"],
    #             stdout=power_log_file,
    #             stderr=subprocess.DEVNULL,
    #             text=True,
    #         )
    #     print(f"Power monitoring started → {power_log_path}\n")
    # except FileNotFoundError:
    #     print("Warning: powermetrics not found. Skipping power monitoring.\n")
    # except PermissionError:
    #     print(
    #         "Warning: Unable to run powermetrics (permission denied). Skipping power monitoring.\n"
    #     )

    ts0, ts1, log = trainer.train(
        key,
        log_callback=live_plot,
        checkpoint_dir=ckpt_dir,
        checkpoint_every=ckpt_every,
        resume_params=resume_params,
    )

    # Stop powermetrics
    # if powermetrics_proc is not None:
    #     try:
    #         powermetrics_proc.terminate()
    #         powermetrics_proc.wait(timeout=5)
    #     except Exception:
    #         powermetrics_proc.kill()
    #     print(f"Power monitoring saved to {power_log_path}\n")

    training_elapsed_seconds = time.time() - training_start_time
    training_end_datetime = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    if log:
        # Add timing info to each log entry
        for entry in log:
            entry["elapsed_seconds"] = training_elapsed_seconds
            entry["elapsed_minutes"] = training_elapsed_seconds / 60
            entry["elapsed_hours"] = training_elapsed_seconds / 3600

        save_results(log, ts0.params, ts1.params, out, title, png_name, cfg, ued_trainer_for_plot)

        # Save curriculum state if enabled
        if curriculum_mgr:
            curriculum_path = out / "curriculum_state.json"
            curriculum_mgr.save(curriculum_path)
            print(f"Curriculum state saved to {curriculum_path}")

        # Print training summary
        hours = int(training_elapsed_seconds // 3600)
        minutes = int((training_elapsed_seconds % 3600) // 60)
        secs = int(training_elapsed_seconds % 60)
        print(f"\nTraining completed:")
        print(f"  Start:   {training_start_datetime}")
        print(f"  End:     {training_end_datetime}")
        print(
            f"  Duration: {hours}h {minutes}m {secs}s ({training_elapsed_seconds:.0f}s total)"
        )
        print(f"  Updates:  {len(log)}")
        print(f"  Avg time per update: {training_elapsed_seconds / len(log):.2f}s")

        # Save resumption metadata if this was a resumed run
        if resume_params:
            meta = {
                "resumed_from_update": int(update_num),
                "resumed_at_timestamp": timestamp,
                "original_load_path": str(args.load_checkpoint),
            }
            with open(out / "resumption_metadata.json", "w") as f:
                json.dump(meta, f, indent=2)
            print(f"  Resumption info saved to {out / 'resumption_metadata.json'}")


if __name__ == "__main__":
    main()
