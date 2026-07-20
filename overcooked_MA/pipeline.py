"""
pipeline.py — training pipeline for overcooked_MA.

Five stages, called in a straight line by run_pipeline(). No framework, no
hooks, no stage registry — just named functions so the flow reads top to
bottom and each stage can be tested or swapped in isolation.

    load_config       — merge config.yaml + CLI overrides into one dict
    build_environment  — construct the OvercookedEnvironment from cfg
    build_trainer       — construct SimpleSFLTrainer from cfg + env
    run_training          — call trainer.train(), return (params, log)
    save_outputs           — plot_curves + log.csv + checkpoints

run_pipeline() also writes run_config.json into out_dir right after it's
created (before training starts), so every run directory records exactly
the settings that produced it — independent of whatever config.yaml
happens to contain later.

Standalone and read-only with respect to train.py — it imports
plot_curves/save_results/_fmt_steps from there rather than duplicating
them, but doesn't modify or get called from train.py yet.

Known seam: build_environment() and build_trainer() each end up
constructing their own OvercookedEnvironment, since SimpleSFLTrainer.
__init__ still takes (height, width, episode_len) rather than an env
object. Left as-is here rather than changing simple_sfl.py's public API
in this pass — worth revisiting if that constructor is ever cleaned up.
"""

import json
import pickle
from datetime import datetime
from pathlib import Path

import yaml

from algorithms.simple_sfl import SimpleSFLTrainer
from environment.environment import OvercookedEnvironment
from train import DEFAULT_CFG, plot_curves, save_results, _fmt_steps


def _display_cfg(cfg: dict) -> dict:
    """Translate config.yaml's schema into the legacy DEFAULT_CFG shape that
    train.py's plot_curves/_hparam_text expect for the hyperparameter panel.
    Display-only — the real training values come straight from cfg."""
    return {
        **DEFAULT_CFG,
        "total_steps": cfg["steps"],
        "n_envs": cfg["envs"],
        "lr": cfg["lr"],
        "ent_coef": cfg["ent_coef"],
        "obs_mode": cfg["obs_mode"],
        "reward_mode": cfg.get("reward_mode", "shaped"),
    }


# ── Stage 1 ──────────────────────────────────────────────────────────────────

def load_config(config_path: str, overrides: dict | None = None) -> dict:
    """Load config.yaml and apply any non-None CLI overrides on top of it."""
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    for key, value in (overrides or {}).items():
        if value is not None:
            cfg[key] = value
    return cfg


# ── Stage 2 ──────────────────────────────────────────────────────────────────

def build_environment(cfg: dict) -> OvercookedEnvironment:
    """Construct the OvercookedEnvironment described by cfg.

    view_size and reward_mode are real config fields (config.yaml / CLI),
    but environment.py doesn't implement either yet — refuse to silently
    ignore a non-default value rather than train with different settings
    than what was asked for.
    """
    if cfg.get("view_size") is not None:
        raise NotImplementedError(
            f"cfg['view_size'] = {cfg['view_size']!r}, but partial observability isn't "
            f"implemented in environment.py yet. Set view_size: null until it's added."
        )
    reward_mode = cfg.get("reward_mode", "shaped")
    if reward_mode != "shaped":
        raise NotImplementedError(
            f"cfg['reward_mode'] = {reward_mode!r}, but environment.py's reward "
            f"calculation is shaped-only today — sparse mode isn't implemented yet."
        )

    h, w = (int(x) for x in cfg["grid_size"].lower().split("x"))
    return OvercookedEnvironment(height=h, width=w, max_steps=cfg["episode_len"])


# ── Stage 3 ──────────────────────────────────────────────────────────────────

def build_trainer(cfg: dict, env: OvercookedEnvironment) -> SimpleSFLTrainer:
    """Construct SimpleSFLTrainer from cfg['sfl'] and the environment's dims."""
    algo = cfg.get("algo", "sfl")
    if algo != "sfl":
        raise ValueError(
            f"cfg['algo'] = {algo!r}, but build_trainer() only implements 'sfl' today "
            f"(accel/evosfl trainers were never ported to this pipeline)."
        )
    sfl_cfg = cfg["sfl"]
    return SimpleSFLTrainer(
        height=env.height, width=env.width, cfg=cfg,
        buffer_size=sfl_cfg["buffer_size"], pool_size=sfl_cfg["pool_size"],
        rho=sfl_cfg["rho"], rollouts_per_level=sfl_cfg["rollouts_per_level"],
        episode_len=env.max_steps, refresh_every=sfl_cfg["refresh_every"],
        seed=cfg["seed"],
    )


# ── Stage 4 ──────────────────────────────────────────────────────────────────

def run_training(trainer: SimpleSFLTrainer, cfg: dict, batch_size: int, steps_per_update: int, out_dir: Path, title: str, png: str):
    """Run trainer.train(...), live-plotting to out_dir as it goes. Returns (params, log)."""
    resume_params = None
    if cfg.get("load_checkpoint"):
        with open(Path(cfg["load_checkpoint"]) / "params.pkl", "rb") as f:
            resume_params = pickle.load(f)

    #batch_size = cfg["envs"]
    # steps_per_update = batch_size * cfg["episode_len"] * 2  # 2 agents per level
    total_updates = max(cfg["steps"] // steps_per_update, 1)

    display_cfg = _display_cfg(cfg)
    algo = cfg.get("algo", "sfl")
    live_plot = lambda log: plot_curves(log, out_dir, title, png, display_cfg, True, algo)
    return trainer.train(
        total_updates=total_updates, batch_size=batch_size,
        resume_params=resume_params, log_callback=live_plot,
        checkpoint_dir=out_dir, checkpoint_every=cfg["checkpoint_every"],
    )


def _save_run_config(cfg: dict, env: OvercookedEnvironment, out_dir: Path, timestamp: str) -> None:
    """Record exactly what produced this run, independent of config.yaml's later contents."""
    with open(out_dir / "run_config.json", "w") as f:
        json.dump({
            "timestamp": timestamp,
            "env_height": env.height, "env_width": env.width,
            "obs_shape": list(env.obs_shape),
            **cfg,
        }, f, indent=2)


def _print_run_summary(cfg: dict, env: OvercookedEnvironment, algo: str, obs_tag: str,
                        reward_mode: str, batch_size: int, steps_per_update: int, out_dir: Path) -> None:
    """Same idea as the old train.py's "Print config" block, adapted to config.yaml's schema."""
    sfl_cfg = cfg["sfl"]
    total_updates = max(cfg["steps"] // steps_per_update, 1)
    print(f"\n{'=' * 50}")
    print(f"Algo    : {algo}  grid={cfg['grid_size']}  obs={env.obs_shape}")
    print(f"Reward  : {reward_mode}  obs_mode={cfg['obs_mode']}  view={obs_tag}")
    print(f"Envs    : {batch_size} levels/update  episode_len={cfg['episode_len']}  → {steps_per_update:,} spu")
    print(f"SFL     : buffer={sfl_cfg['buffer_size']}  pool={sfl_cfg['pool_size']}  "
          f"rho={sfl_cfg['rho']}  refresh_every={sfl_cfg['refresh_every']}")
    print(f"PPO     : lr={cfg['lr']}  ent={cfg['ent_coef']}  vf={cfg['vf_coef']}  gamma={cfg['gamma']}")
    print(f"Total   : {cfg['steps']:,} steps  (~{total_updates:,} updates)")
    print(f"Output  : {out_dir.resolve()}")
    print(f"{'=' * 50}\n")


# ── Stage 5 ──────────────────────────────────────────────────────────────────

def save_outputs(log, params, cfg: dict, out_dir: Path, title: str, png: str) -> None:
    """Save the final plot, log.csv, and params — same artifacts save_results always wrote."""
    if log:
        algo = cfg.get("algo", "sfl")
        save_results(log, params, params, out_dir, title, png, _display_cfg(cfg), True, algo)


# ── Orchestration ────────────────────────────────────────────────────────────

def run_pipeline(config_path: str, overrides: dict | None = None):
    cfg = load_config(config_path, overrides)
    env = build_environment(cfg)
    trainer = build_trainer(cfg, env)

    # algo/obs_tag/reward_mode/obs_mode are four *different* axes, matching
    # the old train.py convention (f"{layout_tag}_{obs_tag}_{reward_mode}_{obs_mode}"):
    #   algo        — which trainer (only "sfl" today; build_trainer() enforces this)
    #   obs_tag     — field of view: "full" or "view{N}", derived from view_size.
    #                 Practically always "full" right now since build_environment()
    #                 raises on a non-null view_size, but computed dynamically so
    #                 this is honest once partial observability is implemented.
    #   reward_mode — "shaped" or "sparse"; always "shaped" today for the same reason.
    #   obs_mode    — "rich" or "simple", the channel encoding — orthogonal to the
    #                 other three, already fully implemented in networks.ActorCritic.
    
    # path format: out_dir/algo/obs_tag/reward_mode/obs_mode/timestamp/ for logging and results
    algo = cfg.get("algo", "sfl")
    view_size = cfg.get("view_size")
    obs_tag = "full" if view_size is None else f"view{view_size}"
    reward_mode = cfg.get("reward_mode", "shaped")
    variant = f"{algo}_{obs_tag}_{reward_mode}_{cfg['obs_mode']}"

    batch_size = cfg["envs"]
    steps_per_update = batch_size * cfg["episode_len"] * 2 # steps per update = 2 agents per level
    png = f"training_curve_{_fmt_steps(cfg['steps'])}_{batch_size}env_{steps_per_update}spu.png"
    title = f"OvercookedV2 SFL — {cfg['grid_size']} ({cfg['obs_mode']})"

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    out_dir = Path(cfg["out_dir"]) / algo / variant / timestamp
    out_dir.mkdir(parents=True, exist_ok=True)
    _save_run_config(cfg, env, out_dir, timestamp)
    _print_run_summary(cfg, env, algo, obs_tag, reward_mode, batch_size, steps_per_update, out_dir)

    params, log = run_training(trainer, cfg, batch_size, steps_per_update, out_dir, title, png)
    save_outputs(log, params, cfg, out_dir, title, png)
    return params, log
