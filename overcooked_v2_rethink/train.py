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
    --view-size 2 \
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
import pickle
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


DEFAULT_CFG = {
    "n_envs":                   16,
    "rollout_len":              256,
    "n_epochs":                   4,
    "batch_size":               256,
    "lr":                       1e-4,
    "gamma":                    0.99,
    "gae_lambda":               0.95,
    "clip_eps":                  0.2,
    "vf_coef":                   0.5,
    "ent_coef":                 0.05,   # higher than before → less premature convergence
    "max_grad_norm":             0.5,
    "shaped_reward_scale":       1.0,
    "step_penalty":              0.0,
    "collision_penalty":         0.0,
    "wrong_ingredient_penalty":  0.0,
    "total_steps":         5_000_000,
    "log_every":                  10,
    "hidden_size":               128,
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
        (f"steps: {cfg.get('total_steps', '?'):,}   "
         f"envs: {cfg.get('n_envs', '?')}   "
         f"rollout_len: {cfg.get('rollout_len', '?')}   "
         f"samples/upd: {spu:,}   "
         f"reward: {cfg.get('reward_mode', '?')}"),
        (f"lr: {cfg.get('lr', '?')}   "
         f"ent_coef: {cfg.get('ent_coef', '?')}   "
         f"clip_eps: {cfg.get('clip_eps', '?')}   "
         f"vf_coef: {cfg.get('vf_coef', '?')}   "
         f"max_grad_norm: {cfg.get('max_grad_norm', '?')}"),
        (f"epochs: {cfg.get('n_epochs', '?')}   "
         f"batch_size: {cfg.get('batch_size', '?')}   "
         f"gamma: {cfg.get('gamma', '?')}   "
         f"gae_lambda: {cfg.get('gae_lambda', '?')}   "
         f"hidden: {cfg.get('hidden_size', '?')}"),
    ]
    return "\n".join(lines)


def plot_curves(log, out_dir: Path, title: str,
                filename: str = "training_curve.png", cfg: dict = None):
    """Save training curves (mean return + PPO loss + hyperparams). Called live during training."""
    steps  = [r["steps"]  for r in log]
    mean_r = [r["mean_r"] for r in log]
    losses = [r["loss"]   for r in log]

    has_hparams = cfg is not None
    height_ratios = [3, 2, 0.7] if has_hparams else [3, 2]
    n_rows = 3 if has_hparams else 2
    fig, axes = plt.subplots(n_rows, 1, figsize=(10, 8 if has_hparams else 7),
                             gridspec_kw={"height_ratios": height_ratios},
                             sharex=False)
    ax1, ax2 = axes[0], axes[1]

    ax1.plot(steps, mean_r, lw=1.0, color="steelblue", alpha=0.6, label="mean return")
    window = max(1, len(mean_r) // 10)
    smoothed = np.convolve(mean_r, np.ones(window) / window, "same")
    ax1.plot(steps, smoothed, lw=2, color="darkorange", label=f"smoothed ({window}-upd avg)")
    ax1.set_ylabel("Mean Return")
    ax1.legend(fontsize=8)
    ax1.grid(alpha=0.3)

    ax2.plot(steps, losses, lw=1.0, color="mediumpurple", label="PPO loss")
    ax2.set_ylabel("Loss")
    ax2.legend(fontsize=8)
    ax2.grid(alpha=0.3)

    # Per-component shaped rewards (only present in shaped mode)
    sr_keys = {"sr_dish": ("Dish pickup",       "gold"),
               "sr_pot":  ("Placement in pot",  "tomato"),
               "sr_pot_start": ("Pot start",    "mediumseagreen"),
               "sr_plate":     ("Plate pickup", "cornflowerblue")}
    if any(sr_keys.keys() & set(log[0].keys())):
        ax3 = axes[2] if has_hparams else None
        # insert a new panel between loss and hparams
        # rebuild figure with 3 (or 4) rows
        if has_hparams:
            hr_new = [height_ratios[0], height_ratios[1], 1.5, height_ratios[2]]
        else:
            hr_new = [height_ratios[0], height_ratios[1], 1.5]
        # Close and recreate
        plt.close(fig)
        fig, axes = plt.subplots(len(hr_new), 1,
                                 figsize=(10, sum(hr_new) * 1.3),
                                 gridspec_kw={"height_ratios": hr_new},
                                 sharex=False)
        ax1, ax2, ax3_sr = axes[0], axes[1], axes[2]
        ax_hp = axes[3] if has_hparams else None

        ax1.plot(steps, mean_r, lw=1.0, color="steelblue", alpha=0.6, label="mean return")
        ax1.plot(steps, smoothed, lw=2, color="darkorange", label=f"smoothed ({window}-upd avg)")
        ax1.set_ylabel("Mean Return"); ax1.legend(fontsize=8); ax1.grid(alpha=0.3)

        ax2.plot(steps, losses, lw=1.0, color="mediumpurple", label="PPO loss")
        ax2.set_ylabel("Loss"); ax2.legend(fontsize=8); ax2.grid(alpha=0.3)

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
            ax_hp.text(0.5, 0.5, _hparam_text(cfg),
                       transform=ax_hp.transAxes, ha="center", va="center",
                       fontsize=7.5, fontfamily="monospace",
                       bbox=dict(boxstyle="round,pad=0.5", facecolor="#f5f5dc", alpha=0.7))

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
            0.5, 0.5, _hparam_text(cfg),
            transform=ax3.transAxes,
            ha="center", va="center",
            fontsize=7.5, fontfamily="monospace",
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


def save_results(log, params_0, params_1, out_dir: Path, title: str,
                 filename: str, cfg: dict = None):
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
    ap.add_argument("--layout",      default="cramped_room_v2")
    ap.add_argument("--steps",       type=int,   default=DEFAULT_CFG["total_steps"])
    ap.add_argument("--envs",        type=int,   default=DEFAULT_CFG["n_envs"])
    ap.add_argument("--lr",          type=float, default=DEFAULT_CFG["lr"])
    ap.add_argument("--ent-coef",    type=float, default=DEFAULT_CFG["ent_coef"])
    ap.add_argument("--hidden",      type=int,   default=DEFAULT_CFG["hidden_size"])
    ap.add_argument("--reward-mode", default="shaped", choices=["shaped", "delivery", "sparse"],
                    help="'shaped': dense rewards (default). "
                         "'delivery': +1/delivery team reward, no shaping. "
                         "'sparse': +20/delivery team reward, no shaping.")
    ap.add_argument("--shaped",      type=float, default=DEFAULT_CFG["shaped_reward_scale"],
                    help="Shaped reward scale, only used in shaped mode (0=off, 1=full)")
    ap.add_argument("--view-size",   type=int,   default=None,
                    help="Agent view radius for partial obs (None=full grid)")
    ap.add_argument("--seed",             type=int,   default=0)
    ap.add_argument("--checkpoint-every", type=int,   default=100,
                    help="Save params checkpoint every N updates (default 100, 0 = off)")
    ap.add_argument("--policy",      default="ippo",
                    help="Algorithm label used as top-level results sub-folder (default: ippo)")
    ap.add_argument("--out-dir",     default=str(Path(__file__).parent / "results"))
    ap.add_argument("--list-layouts", action="store_true")
    args = ap.parse_args()

    if args.list_layouts:
        print("Available layouts:")
        for name, layout in overcooked_v2_layouts.items():
            print(f"  {name:45s} {layout.height}×{layout.width}  "
                  f"{layout.num_ingredients} ingredient(s)")
        return

    cfg = {**DEFAULT_CFG,
           "total_steps":          args.steps,
           "n_envs":               args.envs,
           "lr":                   args.lr,
           "ent_coef":             args.ent_coef,
           "hidden_size":          args.hidden,
           "reward_mode":          args.reward_mode,
           "shaped_reward_scale":  args.shaped}

    env = OvercookedV2(
        layout=args.layout,
        max_steps=400,
        agent_view_size=args.view_size,
    )

    obs_tag   = f"view{args.view_size}" if args.view_size else "full"
    title     = f"OvercookedV2 {args.policy.upper()} — {args.layout} ({obs_tag}, {args.reward_mode})"
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    out = Path(args.out_dir) / args.policy / f"{args.layout}_{obs_tag}_{args.reward_mode}" / timestamp

    samples_per_update = cfg["n_envs"] * cfg["rollout_len"]
    png_name = (
        f"training_curve"
        f"_{_fmt_steps(args.steps)}"
        f"_{cfg['n_envs']}env"
        f"_{samples_per_update}spu"
        f".png"
    )

    print(f"\nLayout    : {args.layout}  {env.height}×{env.width}  obs={env.obs_shape}")
    print(f"Reward    : {args.reward_mode}")
    print(f"Envs      : {cfg['n_envs']}  rollout_len={cfg['rollout_len']}  "
          f"→ {samples_per_update:,} samples/update")
    print(f"PPO       : lr={cfg['lr']}  ent={cfg['ent_coef']}  "
          f"epochs={cfg['n_epochs']}  batch={cfg['batch_size']}")
    print(f"Total     : {args.steps:,} steps  (~{args.steps // samples_per_update} updates)")
    print(f"Plots     : {out}/{png_name}\n")

    trainer = IPPOTrainer(env, cfg)
    key = jax.random.PRNGKey(args.seed)
    live_plot       = make_live_plot_callback(out, title, png_name, cfg)
    ckpt_every      = args.checkpoint_every
    ckpt_dir        = out if ckpt_every > 0 else None
    ts0, ts1, log   = trainer.train(
        key,
        log_callback     = live_plot,
        checkpoint_dir   = ckpt_dir,
        checkpoint_every = ckpt_every,
    )

    if log:
        save_results(log, ts0.params, ts1.params, out, title, png_name, cfg)


if __name__ == "__main__":
    main()
