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

# from overcooked_v2_rethink.curriculum import (
#     CurriculumManager,
#     make_default_curriculum,
# )
# from overcooked_v2_rethink.ued_trainer import UEDTrainer

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
):
    """Save training curves (mean return + PPO loss + deliveries + hyperparams). Called live during training."""
    steps = [r["steps"] for r in log]
    mean_r = [r["mean_r"] for r in log]
    losses = [r["loss"] for r in log]
    deliveries = [r.get("total_deliveries", 0) for r in log]
    # difficulties = [r.get("difficulty", None) for r in log]  # UED: teacher difficulty
    # learning_speeds = [r.get("learning_speed", None) for r in log]  # UED: student learning speed

    has_hparams = cfg is not None
    has_deliveries = "total_deliveries" in log[0]
    # has_difficulty = any(d is not None for d in difficulties)  # Check if UED is used

    # Adjust height ratios based on what we have
    # if has_hparams and has_deliveries and has_difficulty:
    #     height_ratios = [3, 2, 1.5, 1.5, 0.7]  # return, loss, deliveries, difficulty, hparams
    #     n_rows = 5
    if has_hparams and has_deliveries:
        height_ratios = [3, 2, 1.5, 0.7]  # return, loss, deliveries, hparams
        n_rows = 4
    # elif has_hparams and has_difficulty:
    #     height_ratios = [3, 2, 1.5, 0.7]  # return, loss, difficulty, hparams
    #     n_rows = 4
    # elif has_deliveries and has_difficulty:
    #     height_ratios = [3, 2, 1.5, 1.5]  # return, loss, deliveries, difficulty
    #     n_rows = 4
    elif has_deliveries:
        height_ratios = [3, 2, 1.5]  # return, loss, deliveries
        n_rows = 3
    # elif has_difficulty:
    #     height_ratios = [3, 2, 1.5]  # return, loss, difficulty
    #     n_rows = 3
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
    # ax_diff = None
    ax_hp = None

    # if has_deliveries and has_difficulty and has_hparams:
    #     ax_deliv = axes[2]
    #     ax_diff = axes[3]
    #     ax_hp = axes[4]
    # elif has_deliveries and has_difficulty:
    #     ax_deliv = axes[2]
    #     ax_diff = axes[3]
    if has_deliveries and has_hparams:
        ax_deliv = axes[2]
        ax_hp = axes[3]
    # elif has_difficulty and has_hparams:
    #     ax_diff = axes[2]
    #     ax_hp = axes[3]
    elif has_deliveries:
        ax_deliv = axes[2]
    # elif has_difficulty:
    #     ax_diff = axes[2]
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

    # Plot difficulty level and learning speed (UED)
    # if ax_diff is not None:
    #     difficulties_clean = [d for d in difficulties if d is not None]
    #     learning_speeds_clean = [ls for ls in learning_speeds if ls is not None]
    #     steps_clean = steps[:len(difficulties_clean)]

    #     # Plot difficulty on left axis
    #     ax_diff.plot(steps_clean, difficulties_clean, lw=2.5, color="crimson", label="difficulty", marker='o', markersize=3)
    #     ax_diff.set_ylabel("Difficulty [0-1]", color="crimson", fontweight="bold")
    #     ax_diff.set_ylim([0, 1])
    #     ax_diff.tick_params(axis='y', labelcolor="crimson")
    #     ax_diff.grid(alpha=0.3)

    #     # Plot learning speed on right axis (secondary y-axis)
    #     ax_diff2 = ax_diff.twinx()
    #     ax_diff2.plot(steps_clean, learning_speeds_clean, lw=2.5, color="steelblue", label="learning speed", marker='s', markersize=3, alpha=0.7)
    #     ax_diff2.set_ylabel("Learning Speed", color="steelblue", fontweight="bold")
    #     ax_diff2.tick_params(axis='y', labelcolor="steelblue")
    #     ax_diff2.axhline(y=0, color="gray", linestyle="--", alpha=0.3, linewidth=1)

    #     # Combined legend
    #     lines1, labels1 = ax_diff.get_legend_handles_labels()
    #     lines2, labels2 = ax_diff2.get_legend_handles_labels()
    #     ax_diff.legend(lines1 + lines2, labels1 + labels2, loc="upper left", fontsize=8)

    #     ax_diff.set_xlabel("Environment steps")

    # Per-component shaped rewards (only present in shaped mode)
    sr_keys = {
        "sr_dish": ("Dish pickup", "gold"),
        "sr_pot": ("Placement in pot", "tomato"),
        "sr_pot_start": ("Pot start", "mediumseagreen"),
        "sr_plate": ("Plate pickup", "cornflowerblue"),
    }
    if any(sr_keys.keys() & set(log[0].keys())):
        # Rebuild figure to include shaped rewards (and deliveries if available)
        # Layout: return, loss, (deliveries?), shaped_rewards, (hparams?)

        # Calculate new height ratios
        if has_hparams and has_deliveries:
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

        # Assign axes
        ax1 = axes[0]
        ax2 = axes[1]
        ax_deliv = None
        ax3_sr = None
        ax_hp = None

        if has_deliveries and has_hparams:
            ax_deliv = axes[2]
            ax3_sr = axes[3]
            ax_hp = axes[4]
        elif has_deliveries:
            ax_deliv = axes[2]
            ax3_sr = axes[3]
        elif has_hparams:
            ax3_sr = axes[2]
            ax_hp = axes[3]
        else:
            ax3_sr = axes[2]

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


def make_live_plot_callback(out_dir: Path, title: str, filename: str, cfg: dict = None):
    """Returns a callback that re-saves the plot after every log entry."""
    out_dir.mkdir(parents=True, exist_ok=True)

    def _callback(log):
        plot_curves(log, out_dir, title, filename, cfg)

    return _callback


def save_results(
    log, params_0, params_1, out_dir: Path, title: str, filename: str, cfg: dict = None
):
    out_dir.mkdir(parents=True, exist_ok=True)

    plot_curves(log, out_dir, title, filename, cfg)

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
    # ap.add_argument(
    #     "--curriculum",
    #     action="store_true",
    #     help="Enable auto-curriculum (agents progress through layouts based on performance)",
    # )
    # ap.add_argument(
    #     "--curriculum-window",
    #     type=int,
    #     default=50,
    #     help="Episode window for averaging performance in curriculum (default 50)",
    # )
    # ap.add_argument(
    #     "--ued",
    #     action="store_true",
    #     help="Enable UED (Unsupervised Environment Design) with teacher-guided mutations",
    # )
    # ap.add_argument(
    #     "--ued-teacher-lr",
    #     type=float,
    #     default=1e-4,
    #     help="Teacher learning rate for UED (default 1e-4)",
    # )
    # ap.add_argument(
    #     "--ued-update-freq",
    #     type=int,
    #     default=100,
    #     help="Teacher updates every N student updates (default 100)",
    # )
    # ap.add_argument(
    #     "--ued-plateau-window",
    #     type=int,
    #     default=50,
    #     help="Episodes window for computing learning speed (default 50)",
    # )
    args = ap.parse_args()

    if args.list_layouts:
        print("Available layouts:")
        for name, layout in overcooked_v2_layouts.items():
            print(
                f"  {name:45s} {layout.height}×{layout.width}  "
                f"{layout.num_ingredients} ingredient(s)"
            )
        return

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

    # Initialize curriculum if enabled
    # curriculum_mgr = None
    initial_layout = args.layout
    # if args.curriculum:
    #     curriculum_mgr = CurriculumManager(
    #         make_default_curriculum(),
    #         eval_window=args.curriculum_window,
    #     )
    #     initial_layout = curriculum_mgr.current_layout
    #     print(
    #         f"🎓 Curriculum enabled: {curriculum_mgr.levels[0].layout_name} → {curriculum_mgr.levels[-1].layout_name}\n"
    #     )

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
    # if args.ued:
    #     layout_for_path = "ued"
    # elif curriculum_mgr:
    #     layout_for_path = "curriculum"
    #else:
    layout_for_path = args.layout

    title = f"OvercookedV2 {args.policy.upper()} — {layout_for_path} ({obs_tag}, {args.reward_mode}, {args.obs_mode})"

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    out = (
        Path(args.out_dir)
        / args.policy
        / f"{layout_for_path}_{obs_tag}_{args.reward_mode}_{args.obs_mode}"
        / f"{timestamp}{resume_tag}"
    )

    # Update config with curriculum info
    # if curriculum_mgr:
    #     cfg["curriculum_enabled"] = True
    #     cfg["curriculum_window"] = args.curriculum_window

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
    # if curriculum_mgr:
    #     print(
    #         f"Curriculum: {curriculum_mgr.levels[0].layout_name} -> {curriculum_mgr.levels[-1].layout_name}"
    #     )
    print(f"Plots     : {out}/{png_name}\n")

    # Choose trainer: UED or regular IPPO with optional curriculum
    # if args.ued:
    #     # UED mode: teacher designs layouts via mutations
    #     # For now, pass the layout object directly (mutations will work on static_objects)
    #     trainer = UEDTrainer(
    #         env,
    #         cfg,
    #         base_layout=env.layout,
    #         teacher_lr=args.ued_teacher_lr,
    #         update_frequency=args.ued_update_freq,
    #         plateau_window=args.ued_plateau_window,
    #     )
    #     print(
    #         f"🎓 UED mode enabled (teacher lr={args.ued_teacher_lr}, update_freq={args.ued_update_freq})"
    #     )
    # else:
    #     # Regular IPPO, optionally with simple curriculum
    trainer = IPPOTrainer(env, cfg) #curriculum_mgr=curriculum_mgr)

    key = jax.random.PRNGKey(args.seed)
    live_plot = make_live_plot_callback(out, title, png_name, cfg)
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

        save_results(log, ts0.params, ts1.params, out, title, png_name, cfg)

        # Save curriculum state if enabled
        # if curriculum_mgr:
        #     curriculum_path = out / "curriculum_state.json"
        #     curriculum_mgr.save(curriculum_path)
        #     print(f"Curriculum state saved to {curriculum_path}")

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
