"""
train.py — CLI entry point for UED training.

Usage:
    # SFL training
    JAX_PLATFORMS=cpu python -m ued_implementation.train \
        --sfl --grid-size 5x9 --steps 5000000 --envs 16

    # Plain IPPO on a fixed layout
    JAX_PLATFORMS=cpu python -m ued_implementation.train \
        --layout cramped_room --steps 5000000
"""

import os
import sys


os.environ.setdefault("JAX_PLATFORMS", "cpu")

import argparse
import csv
import json
import pickle
import time
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import jax
import jax.numpy as jnp

from algorithms.simple_sfl import SimpleSFLTrainer


# ── Helpers ───────────────────────────────────────────────────────────────────

DEFAULT_CFG = {
    "n_envs": 16,
    "rollout_len": 500,
    "n_epochs": 4,
    "batch_size": 400,
    "lr": 1e-4,
    "gamma": 0.99,
    "gae_lambda": 0.95,
    "clip_eps": 0.2,
    "vf_coef": 0.5,
    "ent_coef": 0.05,
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
    if n >= 1_000_000:
        v = n / 1_000_000
        return f"{v:.0f}M" if v == int(v) else f"{v:.1f}M"
    if n >= 1_000:
        v = n / 1_000
        return f"{v:.0f}K" if v == int(v) else f"{v:.1f}K"
    return str(n)


def _hparam_text(cfg: dict) -> str:
    spu = cfg.get("n_envs", "?") * cfg.get("rollout_len", "?")
    return "\n".join([
        f"steps: {cfg.get('total_steps','?'):,}  envs: {cfg.get('n_envs','?')}  "
        f"rollout: {cfg.get('rollout_len','?')}  spu: {spu:,}  reward: {cfg.get('reward_mode','?')}",
        f"lr: {cfg.get('lr','?')}  ent: {cfg.get('ent_coef','?')}  clip: {cfg.get('clip_eps','?')}  "
        f"vf: {cfg.get('vf_coef','?')}  grad_norm: {cfg.get('max_grad_norm','?')}",
        f"epochs: {cfg.get('n_epochs','?')}  batch: {cfg.get('batch_size','?')}  "
        f"gamma: {cfg.get('gamma','?')}  gae: {cfg.get('gae_lambda','?')}  hidden: {cfg.get('hidden_size','?')}",
    ])


def plot_curves(log, out_dir, title, filename="training_curve.png",
                cfg=None, is_curriculum=False, curriculum_type="sfl"):
    steps = [r["steps"] for r in log]
    mean_r = [r["mean_r"] for r in log]
    has_deliv = "total_deliveries" in log[0]

    n_rows = 1 + has_deliv + is_curriculum + (cfg is not None)
    ratios = [3] + ([1.5] if has_deliv else []) + ([1.5] if is_curriculum else []) + ([0.7] if cfg else [])

    fig, axes = plt.subplots(n_rows, 1, figsize=(10, sum(ratios) * 1.3),
                             gridspec_kw={"height_ratios": ratios})
    if n_rows == 1: axes = [axes]
    idx = 0

    # Reward
    ax = axes[idx]; idx += 1
    ax.plot(steps, mean_r, lw=1, color="steelblue", alpha=0.6, label="mean return")
    w = max(1, len(mean_r) // 10)
    ax.plot(steps, np.convolve(mean_r, np.ones(w)/w, "same"), lw=2, color="darkorange", label=f"smoothed ({w})")
    ax.set_ylabel("Mean Return"); ax.legend(fontsize=8); ax.grid(alpha=0.3)

    # Deliveries
    if has_deliv:
        ax = axes[idx]; idx += 1
        deliv = [r.get("total_deliveries", 0) for r in log]
        dperu = [0] + [deliv[i]-deliv[i-1] for i in range(1, len(deliv))]
        bw = (steps[-1]-steps[0]) / max(len(steps),1) * 0.8 if len(steps)>1 else steps[0]*0.8
        ax.bar(steps, dperu, width=bw, color="green", alpha=0.7)
        ax.set_ylabel("Deliveries / Update"); ax.grid(alpha=0.3, axis="y")

    # Buffer metrics
    if is_curriculum:
        ax = axes[idx]; idx += 1
        if curriculum_type == "accel":
            ax.plot(steps, [r.get("buffer_mean_score",0) for r in log], lw=2, color="darkblue", label="Buffer Mean Regret (PVL)")
            ax.plot(steps, [r.get("buffer_max_score",0) for r in log], lw=1.5, color="crimson", ls="--", label="Buffer Max Regret")
            ax.set_ylabel("ACCEL Regret Score")
        elif curriculum_type == "evosfl":
            ax.plot(steps, [r.get("buffer_mean_score",0) for r in log], lw=2, color="darkorange", label="archive mean p*(1-p)")
            ax.plot(steps, [r.get("buffer_max_score",0) for r in log], lw=1.5, color="crimson", ls="--", label="archive max")
            ax.plot(steps, [r.get("buffer_min_score",0) for r in log], lw=1, color="gray", ls=":", label="archive min")
            ax.set_ylabel("Archive Learnability p*(1-p)")
        else:
            ax.plot(steps, [r.get("buffer_mean_score",0) for r in log], lw=2, color="teal", label="buf mean p*(1-p)")
            ax.plot(steps, [r.get("buffer_max_score",0) for r in log], lw=1.5, color="magenta", ls="--", label="buf max")
            ax.plot(steps, [r.get("buffer_min_score",0) for r in log], lw=1, color="gray", ls=":", label="buf min")
            ax.set_ylabel("SFL Learnability p*(1-p)")
        ax.legend(fontsize=8); ax.grid(alpha=0.3)

    # Hyperparams
    if cfg:
        ax = axes[idx]
        ax.axis("off")
        ax.text(0.5, 0.5, _hparam_text(cfg), transform=ax.transAxes, ha="center", va="center",
                fontsize=7.5, fontfamily="monospace", bbox=dict(boxstyle="round,pad=0.5", fc="#f5f5dc", alpha=0.7))

    fig.suptitle(title, fontsize=11); fig.tight_layout()
    fig.savefig(Path(out_dir) / filename, dpi=120); plt.close(fig)


def save_results(log, params_0, params_1, out_dir, title, filename, cfg=None,
                 is_curriculum=False, curriculum_type="sfl"):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    plot_curves(log, out_dir, title, filename, cfg, is_curriculum, curriculum_type)
    with open(out_dir / "log.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=log[0].keys())
        w.writeheader(); w.writerows(log)
    with open(out_dir / "params_agent0.pkl", "wb") as f: pickle.dump(params_0, f)
    with open(out_dir / "params_agent1.pkl", "wb") as f: pickle.dump(params_1, f)
    print(f"\nResults saved to {out_dir}/")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description="UED training for OvercookedV2")

    ap.add_argument("--layout", default=None, help="Layout name for IPPO (--list-layouts for options)")
    ap.add_argument("--grid-size", default=None, metavar="HxW", help="Grid dims for SFL (e.g. 5x9)")
    ap.add_argument("--steps", type=int, default=DEFAULT_CFG["total_steps"])
    ap.add_argument("--envs", type=int, default=DEFAULT_CFG["n_envs"])
    ap.add_argument("--lr", type=float, default=DEFAULT_CFG["lr"])
    ap.add_argument("--ent-coef", type=float, default=DEFAULT_CFG["ent_coef"])
    ap.add_argument("--hidden", type=int, default=DEFAULT_CFG["hidden_size"])
    ap.add_argument("--reward-mode", default="shaped", choices=["shaped", "delivery", "sparse"])
    ap.add_argument("--obs-mode", default="rich", choices=["rich", "simple"])
    ap.add_argument("--shaped", type=float, default=DEFAULT_CFG["shaped_reward_scale"])
    ap.add_argument("--view-size", type=int, default=None)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--checkpoint-every", type=int, default=100)
    ap.add_argument("--render-every", type=int, default=50000)
    ap.add_argument("--load-checkpoint", default=None)
    ap.add_argument("--out-dir", default=str(Path(__file__).parent / "results"))
    ap.add_argument("--list-layouts", action="store_true")

    # SFL arguments
    ap.add_argument("--sfl", action="store_true", help="Use SFL curriculum")
    ap.add_argument("--sfl-buffer-size", type=int, default=25)
    ap.add_argument("--sfl-pool-size", type=int, default=200)
    ap.add_argument("--sfl-rho", type=float, default=0.5)
    ap.add_argument("--sfl-inner-steps", type=int, default=25)
    ap.add_argument("--sfl-refresh-every", type=int, default=1)
    ap.add_argument("--sfl-rollouts-per-level", type=int, default=4)
    ap.add_argument("--episode-len", type=int, default=150)
    
    # ACCEL arguments
    # ap.add_argument("--accel", action="store_true", help="Use ACCEL curriculum")
    # ap.add_argument("--accel-buffer-size", type=int, default=100)
    # ap.add_argument("--accel-fill-ratio", type=float, default=0.5)
    # ap.add_argument("--accel-replay-prob", type=float, default=0.5)
    # ap.add_argument("--accel-score-threshold", type=float, default=0.05)
    # ap.add_argument("--accel-edit-step", type=float, default=0.1)
    
    args = ap.parse_args()

    # if args.list_layouts:
    #     for name, layout in overcooked_v2_layouts.items():
    #         print(f"  {name:45s} {layout.height}×{layout.width}  {layout.num_ingredients} ingredient(s)")
    #     return

    is_curriculum = args.sfl or args.accel or args.evosfl

    if not is_curriculum:
        ap.error("Plain IPPO training is disabled — pass --sfl (e.g. --sfl --grid-size 5x9).")

    if args.grid_size is None:
        ap.error("--grid-size HxW is required for SFL (e.g. --grid-size 5x9).")
    _h, _w = (int(x) for x in args.grid_size.lower().split("x"))
    grid_size = (_h, _w)

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

    # ── Environment setup ─────────────────────────────────────────────────
    initial_layout = f"Parametrized_{_h}x{_w}"
    policy_name, curriculum_type = "sfl", "sfl"

    # ── Checkpoint loading ────────────────────────────────────────────────
    resume_params = None
    resume_tag = ""
    if args.load_checkpoint:
        ckpt = Path(args.load_checkpoint)
        if (ckpt / "checkpoints").exists():
            ckpt = ckpt / "checkpoints" / "final" if (ckpt / "checkpoints" / "final").exists() \
                else sorted((ckpt / "checkpoints").glob("update_*"), key=lambda p: int(p.name.split("_")[-1]))[-1]
        with open(ckpt / "params.pkl", "rb") as f: resume_params = pickle.load(f)
        resume_tag = f"_resumed"

    # ── Output path ───────────────────────────────────────────────────────
    obs_tag = f"view{args.view_size}" if args.view_size else "full"
    layout_tag = policy_name
    title = f"OvercookedV2 {policy_name.upper()} — {layout_tag} ({obs_tag}, {args.reward_mode}, {args.obs_mode})"
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    out = Path(args.out_dir) / policy_name / f"{layout_tag}_{obs_tag}_{args.reward_mode}_{args.obs_mode}" / f"{timestamp}{resume_tag}"
    batch_size = args.envs
    spu = batch_size * args.episode_len * 2  # 2 agents per level
    png = f"training_curve_{_fmt_steps(args.steps)}_{batch_size}env_{spu}spu.png"

    # ── Trainer ───────────────────────────────────────────────────────────
    trainer = SimpleSFLTrainer(
        height=_h, width=_w, cfg=cfg,
        buffer_size=args.sfl_buffer_size, pool_size=args.sfl_pool_size,
        rho=args.sfl_rho, rollouts_per_level=args.sfl_rollouts_per_level,
        episode_len=args.episode_len, refresh_every=args.sfl_refresh_every,
        seed=args.seed,
    )
    env = trainer.env

    # ── Save config ───────────────────────────────────────────────────────
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "run_config.json", "w") as f:
        json.dump({
            "timestamp": timestamp, "policy": policy_name, "layout": initial_layout,
            "env_height": env.height, "env_width": env.width,
            "obs_shape": list(env.obs_shape), "seed": args.seed,
            **cfg,
            "sfl_buffer_size": args.sfl_buffer_size,
            "sfl_pool_size": args.sfl_pool_size,
            "sfl_rho": args.sfl_rho,
            "sfl_rollouts_per_level": args.sfl_rollouts_per_level,
            "episode_len": args.episode_len,
        }, f, indent=2)

    # ── Print config ──────────────────────────────────────────────────────
    print(f"\n{'='*50}")
    print(f"Layout  : {initial_layout}  obs={env.obs_shape}")
    print(f"Reward  : {args.reward_mode}  obs_mode={args.obs_mode}")
    print(f"Envs    : {batch_size} levels/update  episode_len={args.episode_len}  → {spu:,} spu")
    print(f"PPO     : lr={cfg['lr']}  ent={cfg['ent_coef']}")
    print(f"Total   : {args.steps:,} steps  (~{max(args.steps // spu, 1)} updates)")
    print(f"Output  : {out.resolve()}/{png}")
    print(f"{'='*50}\n")

    # ── Run training ──────────────────────────────────────────────────────
    live_plot = lambda log: plot_curves(log, out, title, png, cfg, is_curriculum, curriculum_type)
    total_updates = max(args.steps // spu, 1)

    params, log = trainer.train(
        total_updates=total_updates, batch_size=batch_size,
        resume_params=resume_params, log_callback=live_plot,
        checkpoint_dir=out,
        checkpoint_every=args.checkpoint_every,
    )

    if log:
        save_results(log, params, params, out, title, png, cfg, is_curriculum, curriculum_type)


if __name__ == "__main__":
    main()
