# run ACCEL training with:

# JAX_PLATFORMS=cpu uv run --project overcooked_v2_rethink \                               
# python train.py \                       
# --layout asymmetric_advantages_v2 \       --accel \                                 
# --accel-replay-prob 0.3 \                 
# --accel-score-threshold 1.2 \
# --accel-edit-step 0.02 \
# --ent-coef 0.1 \
# --lr 0.0003 \
# --shaped 10.0 \
# --steps 5000000 \
# --envs 16




import os, sys


# Set default JAX backend to CPU and append repository root to path
os.environ.setdefault("JAX_PLATFORMS", "cpu")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

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

from overcooked_v2_rethink import OvercookedV2, overcooked_v2_layouts
from overcooked_v2_rethink.ippo_jax import IPPOTrainer
from overcooked_v2_rethink.accel_trainer import ACCELTrainer
from overcooked_v2_rethink.sfl_trainer import SFLTrainer

# Default PPO hyperparameter configurations
DEFAULT_CFG = {
    "n_envs": 16,
    "rollout_len": 400,
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
    """Format step count into a human-readable string (e.g., 5M, 500K)."""
    if n >= 1_000_000:
        v = n / 1_000_000
        return f"{v:.0f}M" if v == int(v) else f"{v:.1f}M"
    if n >= 1_000:
        v = n / 1_000
        return f"{v:.0f}K" if v == int(v) else f"{v:.1f}K"
    return str(n)


def _hparam_text(cfg: dict) -> str:
    """Generate a compact block of hyperparameter text for plot visualization."""
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
    is_curriculum: bool = False,
    curriculum_type: str = "accel"
):
    """Save training metric curves (Mean Return, Loss, Deliveries, and ACCEL Buffer metrics)."""
    steps = [r["steps"] for r in log]
    mean_r = [r["mean_r"] for r in log]
    losses = [r["loss"] for r in log]
    deliveries = [r.get("total_deliveries", 0) for r in log]

    has_hparams = cfg is not None
    has_deliveries = "total_deliveries" in log[0]

    # Calculate subplots and layout structure based on flags
    n_rows = 2
    height_ratios = [3, 2]
    if has_deliveries:
        n_rows += 1
        height_ratios.append(1.5)
    if is_curriculum:
        n_rows += 1
        height_ratios.append(1.5)
    if has_hparams:
        n_rows += 1
        height_ratios.append(0.7)

    fig, axes = plt.subplots(n_rows, 1, figsize=(10, sum(height_ratios) * 1.3), gridspec_kw={"height_ratios": height_ratios})

    ax1 = axes[0]
    ax2 = axes[1]
    
    axis_idx = 2
    ax_deliv = axes[axis_idx] if has_deliveries else None
    if has_deliveries: axis_idx += 1
    
    ax_curriculum = axes[axis_idx] if is_curriculum else None
    if is_curriculum: axis_idx += 1
    
    ax_hp = axes[axis_idx] if has_hparams else None

    # 1. Plot Reward (Mean Return)
    ax1.plot(steps, mean_r, lw=1.0, color="steelblue", alpha=0.6, label="mean return")
    window = max(1, len(mean_r) // 10)
    smoothed = np.convolve(mean_r, np.ones(window) / window, "same")
    ax1.plot(steps, smoothed, lw=2, color="darkorange", label=f"smoothed ({window}-upd avg)")
    ax1.set_ylabel("Mean Return")
    ax1.legend(loc="upper left", fontsize=8)
    ax1.grid(alpha=0.3)

    # 2. Plot Optimization Loss (PPO Loss)
    ax2.plot(steps, losses, lw=1.0, color="mediumpurple", label="PPO loss")
    ax2.set_ylabel("Loss")
    ax2.legend(loc="upper left", fontsize=8)
    ax2.grid(alpha=0.3)

    # 3. Plot Task Deliveries 
    if ax_deliv is not None:
        deliveries_per_update = [0] + [deliveries[i] - deliveries[i - 1] for i in range(1, len(deliveries))]
        ax_deliv.bar(steps, deliveries_per_update, width=max(steps) * 0.015, color="green", alpha=0.7, label="deliveries per update")
        ax_deliv.set_ylabel("Deliveries / Update")
        ax_deliv.legend(loc="upper left", fontsize=8)
        ax_deliv.grid(alpha=0.3, axis="y")

    # 4. Curriculum Specific Plot: Evolution of Buffer Metrics
    if ax_curriculum is not None and is_curriculum:
        buf_mean = [r.get("buffer_mean_score", 0.0) for r in log]
        buf_max = [r.get("buffer_max_score", 0.0) for r in log]
        
        if curriculum_type == "sfl":
            ax_curriculum.plot(steps, buf_mean, lw=2, color="teal", label="Buffer Mean Learnability (p*(1-p))")
            ax_curriculum.plot(steps, buf_max, lw=1.5, color="magenta", linestyle="--", label="Buffer Max Learnability")
            ax_curriculum.set_ylabel("SFL Learnability Score")
        else:
            ax_curriculum.plot(steps, buf_mean, lw=2, color="darkblue", label="Buffer Mean Regret (PVL)")
            ax_curriculum.plot(steps, buf_max, lw=1.5, color="crimson", linestyle="--", label="Buffer Max Regret")
            ax_curriculum.set_ylabel("ACCEL Regret Score")
            
        ax_curriculum.legend(loc="upper left", fontsize=8)
        ax_curriculum.grid(alpha=0.3)

    # Plot hyperparameters card
    if ax_hp is not None:
        ax_hp.axis("off")
        ax_hp.text(0.5, 0.5, _hparam_text(cfg), transform=ax_hp.transAxes, ha="center", va="center", fontsize=7.5, fontfamily="monospace", bbox=dict(boxstyle="round,pad=0.5", facecolor="#f5f5dc", alpha=0.7))

    fig.suptitle(title, fontsize=11)
    fig.tight_layout()
    fig.savefig(out_dir / filename, dpi=120)
    plt.close(fig)


def make_live_plot_callback(out_dir: Path, title: str, filename: str, cfg: dict = None, is_curriculum: bool = False, curriculum_type: str = "accel"):
    """Callback wrapper used to perform live plot updates during training execution."""
    out_dir.mkdir(parents=True, exist_ok=True)
    return lambda log: plot_curves(log, out_dir, title, filename, cfg, is_curriculum, curriculum_type)


def save_results(log, params_0, params_1, out_dir: Path, title: str, filename: str, cfg: dict = None, is_curriculum: bool = False, curriculum_type: str = "accel"):
    """Serialize network weights and export final structured execution logs."""
    out_dir.mkdir(parents=True, exist_ok=True)
    plot_curves(log, out_dir, title, filename, cfg, is_curriculum, curriculum_type)

    with open(out_dir / "log.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=log[0].keys())
        writer.writeheader()
        writer.writerows(log)

    with open(out_dir / "params_agent0.pkl", "wb") as f: pickle.dump(params_0, f)
    with open(out_dir / "params_agent1.pkl", "wb") as f: pickle.dump(params_1, f)
    print(f"\nResults saved to {out_dir}/")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layout", default="cramped_room_v2")
    ap.add_argument("--steps", type=int, default=DEFAULT_CFG["total_steps"])
    ap.add_argument("--envs", type=int, default=DEFAULT_CFG["n_envs"])
    ap.add_argument("--lr", type=float, default=DEFAULT_CFG["lr"])
    ap.add_argument("--ent-coef", type=float, default=DEFAULT_CFG["ent_coef"])
    ap.add_argument("--hidden", type=int, default=DEFAULT_CFG["hidden_size"])
    ap.add_argument("--reward-mode", default="shaped", choices=["shaped", "delivery", "sparse"])
    ap.add_argument("--obs-mode", default="rich", choices=["rich", "simple"])
    ap.add_argument("--shaped", type=float, default=DEFAULT_CFG["shaped_reward_scale"])
    ap.add_argument("--view-size", type=int, default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--checkpoint-every", type=int, default=100)
    ap.add_argument("--load-checkpoint", default=None)
    ap.add_argument("--policy", default="ippo")
    ap.add_argument("--out-dir", default=str(Path(__file__).parent / "results"))
    ap.add_argument("--list-layouts", action="store_true")
    
    # ACCEL arguments
    ap.add_argument("--accel", action="store_true")
    ap.add_argument("--accel-buffer-size", type=int, default=100)
    ap.add_argument("--accel-fill-ratio", type=float, default=0.5)
    ap.add_argument("--accel-replay-prob", type=float, default=0.5)
    ap.add_argument("--accel-score-threshold", type=float, default=0.05)
    ap.add_argument("--accel-edit-step", type=float, default=0.1)
    
    # SFL arguments 
    ap.add_argument("--sfl", action="store_true", help="Use Sampling For Learnability curriculum")
    ap.add_argument("--sfl-buffer-size", type=int, default=50, help="Buffer capacity K")
    ap.add_argument("--sfl-pool-size", type=int, default=200, help="Random candidate pool size N")
    ap.add_argument("--sfl-rho", type=float, default=0.7, help="Ratio of buffer levels in training batch")
    ap.add_argument("--sfl-inner-steps", type=int, default=10, help="Inner training steps T per buffer update")
    ap.add_argument("--sfl-refresh-every", type=int, default=1, help="Refresh buffer only every N outer iterations (reduces eval overhead)")
    
    args = ap.parse_args()
    
    if args.list_layouts:
        for name, layout in overcooked_v2_layouts.items():
            print(f"  {name:45s} {layout.height}×{layout.width}  {layout.num_ingredients} ingredient(s)")
        return

    initial_layout = args.layout

    # ACCEL automatically enforces partial observability (view_size = 2) if not specified
    #if args.accel and args.view_size is None:
    #    args.view_size = 2  

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

    env = OvercookedV2(layout=initial_layout, max_steps=400, agent_view_size=args.view_size)

    # Checkpoint loading and state restoration logic
    resume_params = None
    resume_tag = ""
    if args.load_checkpoint:
        ckpt_path = Path(args.load_checkpoint)
        if (ckpt_path / "checkpoints").exists():
            ckpt_path = ckpt_path / "checkpoints" / "final" if (ckpt_path / "checkpoints" / "final").exists() else sorted((ckpt_path / "checkpoints").glob("update_*"), key=lambda p: int(p.name.split("_")[-1]))[-1]

        with open(ckpt_path / "params_agent0.pkl", "rb") as f: p0 = pickle.load(f)
        with open(ckpt_path / "params_agent1.pkl", "rb") as f: p1 = pickle.load(f)
        resume_params = {"agent_0": p0, "agent_1": p1}
        update_num = int(ckpt_path.name.split("_")[-1]) if ckpt_path.name != "final" else 0
        resume_tag = f"_resumed-from-upd{update_num}"

    obs_tag = f"view{args.view_size}" if args.view_size else "full"
    layout_for_path = "accel" if args.accel else args.layout

    if args.accel: 
        args.policy = "accel" 
    elif args.sfl:
        args.policy = "sfl"
    
    layout_for_path = args.policy if (args.accel or args.sfl) else args.layout 
    title = f"OvercookedV2 {args.policy.upper()} — {layout_for_path} ({obs_tag}, {args.reward_mode}, {args.obs_mode})"

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    out = Path(args.out_dir) / args.policy / f"{layout_for_path}_{obs_tag}_{args.reward_mode}_{args.obs_mode}" / f"{timestamp}{resume_tag}"

    samples_per_update = cfg["n_envs"] * cfg["rollout_len"]
    png_name = f"training_curve_{_fmt_steps(args.steps)}_{cfg['n_envs']}env_{samples_per_update}spu.png"

    # Trainer Initialization: Branch between ACCEL and standard IPPO
    if args.accel:
        from overcooked_v2_rethink.overcooked_parametrized_current import _LAYOUTS as _raw_layouts
        _base_layout_str = _raw_layouts.get(args.layout)
        trainer = ACCELTrainer(
            env, cfg, base_layout_str=_base_layout_str,
            buffer_size=args.accel_buffer_size, initial_fill_ratio=args.accel_fill_ratio,
            replay_prob=args.accel_replay_prob, score_threshold=args.accel_score_threshold,
            edit_step=args.accel_edit_step, seed=args.seed
        )
    elif args.sfl:
        from overcooked_v2_rethink.overcooked_parametrized_current import _LAYOUTS as _raw_layouts
        _base_layout_str = _raw_layouts.get(args.layout)
        trainer = SFLTrainer(
            env=env, cfg=cfg, base_layout_str=_base_layout_str,
            buffer_size=args.sfl_buffer_size, n_random_pool=args.sfl_pool_size,
            rho=args.sfl_rho, seed=args.seed,
            buffer_refresh_every=args.sfl_refresh_every,
        )
    else:
        trainer = IPPOTrainer(env, cfg)
        
    # Save full run configuration before training starts (survives crashes)
    out.mkdir(parents=True, exist_ok=True)
    run_config = {
        # metadata
        "timestamp": timestamp,
        "policy": args.policy,
        "layout": initial_layout,
        "env_height": env.height,
        "env_width": env.width,
        "obs_shape": list(env.obs_shape),
        "seed": args.seed,
        # PPO hyperparameters
        **cfg,
        # observation / reward settings
        "view_size": args.view_size,
        # ACCEL settings (None when not used)
        "accel": args.accel,
        "accel_buffer_size":      args.accel_buffer_size      if args.accel else None,
        "accel_fill_ratio":       args.accel_fill_ratio        if args.accel else None,
        "accel_replay_prob":      args.accel_replay_prob       if args.accel else None,
        "accel_score_threshold":  args.accel_score_threshold   if args.accel else None,
        "accel_edit_step":        args.accel_edit_step         if args.accel else None,
        # SFL settings (None when not used)
        "sfl": args.sfl,
        "sfl_buffer_size": args.sfl_buffer_size if args.sfl else None,
        "sfl_pool_size": args.sfl_pool_size if args.sfl else None,
        "sfl_rho": args.sfl_rho if args.sfl else None,
        "sfl_inner_steps": args.sfl_inner_steps if args.sfl else None,
        "sfl_refresh_every": args.sfl_refresh_every if args.sfl else None,
        # output
        "out_dir": str(out.resolve()),
        "checkpoint_every": args.checkpoint_every,
        "resumed_from": args.load_checkpoint,
    }
    with open(out / "run_config.json", "w") as f:
        json.dump(run_config, f, indent=2)

    # === PRINT TRAINING CONFIGURATION CARD ===
    print(f"\n" + "="*50)
    print(f"Layout    : {initial_layout}  {env.height}×{env.width}  obs={env.obs_shape}")
    print(f"View Size : {args.view_size if args.view_size else 'full'}  obs_mode={args.obs_mode}")
    print(f"Reward    : {args.reward_mode}")
    print(f"Envs      : {cfg['n_envs']}  rollout_len={cfg['rollout_len']}  -> {samples_per_update:,} samples/update")
    print(f"PPO       : lr={cfg['lr']}  ent={cfg['ent_coef']}  epochs={cfg['n_epochs']}  batch={cfg['batch_size']}")
    print(f"Total     : {args.steps:,} steps  (~{args.steps // samples_per_update} updates)")
    print(f"Plots     : {out.resolve()}/{png_name}")
    print("="*50 + "\n")

    key = jax.random.PRNGKey(args.seed)
    
    # callback for live plotting during training
    is_curriculum = args.accel or args.sfl
    curriculum_type = "sfl" if args.sfl else "accel"
    live_plot = make_live_plot_callback(out, title, png_name, cfg, is_curriculum=is_curriculum, curriculum_type=curriculum_type)
    
    # Starting the appropriate training package
    train_kwargs = {
        "key": key,
        "resume_params": resume_params,
        "log_callback": live_plot,
        "checkpoint_dir": out if args.checkpoint_every > 0 else None,
        "checkpoint_every": args.checkpoint_every
    }
    # For SFL, add the SFL-specific named parameter T for the inner loop.   
    if args.sfl:
        train_kwargs["T_steps"] = args.sfl_inner_steps

    # Launch training pipeline with explicit keyword arguments mapping
    ts0, ts1, log = trainer.train(**train_kwargs)

    if log:
        save_results(log, ts0.params, ts1.params, out, title, png_name, cfg, is_curriculum=is_curriculum, curriculum_type=curriculum_type)


if __name__ == "__main__":
    main()