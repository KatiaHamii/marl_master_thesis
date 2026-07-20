"""
pipeline.py — training pipeline for overcooked_MA.

Five stages, called in a straight line by run_pipeline(). No framework, no
hooks, no stage registry — just named functions so the flow reads top to
bottom and each stage can be tested or swapped in isolation.

    load_config       — merge config.yaml + CLI overrides into one dict
    build_environment  — construct the env described by cfg
    build_trainer       — construct the trainer from cfg + env
    run_training          — call trainer.train(), return (params0, params1, log)
    save_outputs           — plot_curves + log.csv + checkpoints

Two algorithms are supported via cfg["algo"]:

    "sfl"      — the lightweight NumPy path (SimpleSFLTrainer + the plain-Python
                 OvercookedEnvironment). One shared policy, vanilla policy
                 gradient, sequential rollouts. reward_mode/view_size are NOT
                 implemented on this path (build_environment raises).
    "sfl_jax"  — the real JAX-native path (SFLTrainer + IPPOTrainer + the
                 vmapped OvercookedV2 engine). Two independent PPO policies,
                 GAE, jitted/vmapped rollouts. Supports partial observability
                 (view_size) and reward_mode, since the JAX engine implements
                 both.

run_pipeline() writes run_config.json into out_dir right after it's created
(before training starts), so every run directory records exactly the settings
that produced it — independent of whatever config.yaml contains later.
"""

import json
import pickle
from datetime import datetime
from pathlib import Path

import numpy as np
import yaml

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


def _jax_cfg(cfg: dict) -> dict:
    """Build the IPPO/SFL-compatible cfg the JAX trainers read.

    IPPOTrainer reads n_envs/total_steps/rollout_len/clip_eps/n_epochs/etc.
    from its cfg dict; config.yaml uses friendlier names (envs/steps/...).
    Start from train.py's DEFAULT_CFG (which has every key with a sane value)
    and override with the config.yaml values that have a home."""
    return {
        **DEFAULT_CFG,
        "n_envs": cfg["envs"],
        "total_steps": cfg["steps"],
        "rollout_len": cfg["episode_len"],  # SFLTrainer forces this up to env.max_steps anyway
        "lr": cfg["lr"],
        "ent_coef": cfg["ent_coef"],
        "vf_coef": cfg["vf_coef"],
        "gamma": cfg["gamma"],
        "obs_mode": cfg["obs_mode"],
        "reward_mode": cfg.get("reward_mode", "shaped"),
        "shaped_reward_scale": cfg.get("shaped_reward_scale", DEFAULT_CFG["shaped_reward_scale"]),
        "hidden_size": cfg.get("hidden", DEFAULT_CFG["hidden_size"]),
        "log_every": cfg.get("log_every", DEFAULT_CFG["log_every"]),
    }


def _parse_grid(cfg: dict):
    h, w = (int(x) for x in cfg["grid_size"].lower().split("x"))
    return h, w


# ── Stage 1 ──────────────────────────────────────────────────────────────────

def load_config(config_path: str, overrides: dict | None = None) -> dict:
    """Load config.yaml and apply any non-None CLI overrides on top of it."""
    with open(config_path) as f:
        print(f"[pipeline] loading config from {config_path} with overrides {overrides}")
        cfg = yaml.safe_load(f)
    for key, value in (overrides or {}).items():
        if value is not None:
            cfg[key] = value
    return cfg


# ── Stage 2 ──────────────────────────────────────────────────────────────────

def _build_env_numpy(cfg: dict):
    """The lightweight NumPy OvercookedEnvironment (algo=sfl).

    view_size and reward_mode are real config fields but environment.py
    doesn't implement either — refuse to silently ignore a non-default value
    rather than train with different settings than what was asked for."""
    from environment.environment import OvercookedEnvironment

    if cfg.get("view_size") is not None:
        raise NotImplementedError(
            f"cfg['view_size'] = {cfg['view_size']!r}, but partial observability isn't "
            f"implemented in the NumPy environment.py. Use algo=sfl_jax for partial "
            f"observability, or set view_size: null."
        )
    reward_mode = cfg.get("reward_mode", "shaped")
    if reward_mode != "shaped":
        raise NotImplementedError(
            f"cfg['reward_mode'] = {reward_mode!r}, but the NumPy environment.py's reward "
            f"calculation is shaped-only. Use algo=sfl_jax for sparse/delivery rewards."
        )

    h, w = _parse_grid(cfg)
    return OvercookedEnvironment(height=h, width=w, max_steps=cfg["episode_len"])


def _build_env_jax(cfg: dict):
    """The JAX-native OvercookedV2 engine (algo=sfl_jax).

    Built with a placeholder all-open Layout; SFLTrainer overrides the grid on
    every reset with a freshly generated level, so the layout here only fixes
    the shape/agent-count/ingredient-count. Mirrors OvercookedUEDEnv's
    _create_student_env pattern in ued.py. view_size maps to agent_view_size
    (partial observability is genuinely supported here)."""
    from environment.overcooked_env import OvercookedV2
    from environment.layouts import Layout
    from environment.common import StaticObject

    h, w = _parse_grid(cfg)
    static = np.full((h, w), int(StaticObject.WALL), dtype=int)
    static[1:h - 1, 1:w - 1] = int(StaticObject.EMPTY)
    layout = Layout(
        agent_positions=[(1, 1), (w - 2, h - 2)],
        static_objects=static,
        num_ingredients=2,
        possible_recipes=[[0, 0, 0], [1, 1, 1]],
    )
    return OvercookedV2(
        layout=layout,
        max_steps=cfg["episode_len"],
        agent_view_size=cfg.get("view_size"),
    )


def build_environment(cfg: dict):
    """Construct the env described by cfg, branching on the algorithm."""
    algo = cfg.get("algo", "sfl")
    if algo == "sfl_jax":
        print("[pipeline] building JAX-native OvercookedV2 environment (algo=sfl_jax)")
        return _build_env_jax(cfg)
    if algo == "sfl":
        print("[pipeline] building NumPy OvercookedEnvironment (algo=sfl)")
        return _build_env_numpy(cfg)
    raise ValueError(f"cfg['algo'] = {algo!r} is not a known algorithm (expected 'sfl' or 'sfl_jax').")


# ── Stage 3 ──────────────────────────────────────────────────────────────────

def build_trainer(cfg: dict, env):
    """Construct the trainer for cfg['algo'] from cfg + env."""
    algo = cfg.get("algo", "sfl")
    sfl_cfg = cfg["sfl"]

    if algo == "sfl":
        print("[pipeline] building SimpleSFLTrainer (NumPy path, algo=sfl)")
        from algorithms.simple_sfl import SimpleSFLTrainer
        return SimpleSFLTrainer(
            height=env.height, width=env.width, cfg=cfg,
            buffer_size=sfl_cfg["buffer_size"], pool_size=sfl_cfg["pool_size"],
            rho=sfl_cfg["rho"], rollouts_per_level=sfl_cfg["rollouts_per_level"],
            episode_len=env.max_steps, refresh_every=sfl_cfg["refresh_every"],
            seed=cfg["seed"],
        )

    if algo == "sfl_jax":
        print("[pipeline] building SFLTrainer (JAX path, algo=sfl_jax)")
        from algorithms.sfl_jax import SFLTrainer
        h, w = _parse_grid(cfg)
        return SFLTrainer(
            env=env, cfg=_jax_cfg(cfg), grid_size=(h, w),
            buffer_size=sfl_cfg["buffer_size"], n_random_pool=sfl_cfg["pool_size"],
            rho=sfl_cfg["rho"], seed=cfg["seed"],
            buffer_refresh_every=sfl_cfg["refresh_every"],
        )

    raise ValueError(f"cfg['algo'] = {algo!r} is not a known algorithm (expected 'sfl' or 'sfl_jax').")


# ── Stage 4 ──────────────────────────────────────────────────────────────────

def _load_resume_params(cfg: dict, algo: str):
    """Load params to resume from, if cfg['load_checkpoint'] is set."""
    ckpt = cfg.get("load_checkpoint")
    if not ckpt:
        return None
    ckpt = Path(ckpt)
    if algo == "sfl_jax":
        with open(ckpt / "params_agent0.pkl", "rb") as f:
            p0 = pickle.load(f)
        with open(ckpt / "params_agent1.pkl", "rb") as f:
            p1 = pickle.load(f)
        return {"agent_0": p0, "agent_1": p1}
    with open(ckpt / "params.pkl", "rb") as f:
        return pickle.load(f)


def run_training(trainer, cfg: dict, batch_size: int, steps_per_update: int, out_dir: Path, title: str, png: str):
    """Run trainer.train(...), live-plotting to out_dir as it goes.
    Returns (params_agent0, params_agent1, log) — for the shared-policy NumPy
    path both params are the same object."""
    algo = cfg.get("algo", "sfl")
    resume_params = _load_resume_params(cfg, algo)
    display_cfg = _display_cfg(cfg)
    live_plot = lambda log: plot_curves(log, out_dir, title, png, display_cfg, True, algo)

    if algo == "sfl":
        total_updates = max(cfg["steps"] // steps_per_update, 1)
        params, log = trainer.train(
            total_updates=total_updates, batch_size=batch_size,
            resume_params=resume_params, log_callback=live_plot,
            checkpoint_dir=out_dir, checkpoint_every=cfg["checkpoint_every"],
        )
        return params, params, log

    # sfl_jax
    import jax
    T_steps = cfg["sfl"].get("inner_steps", 10)
    ts0, ts1, log = trainer.train(
        key=jax.random.PRNGKey(cfg["seed"]),
        resume_params=resume_params, log_callback=live_plot,
        checkpoint_dir=out_dir, checkpoint_every=cfg["checkpoint_every"],
        T_steps=T_steps, render_every=cfg.get("render_every", 0),
    )
    return ts0.params, ts1.params, log


def _save_run_config(cfg: dict, env, out_dir: Path, timestamp: str) -> None:
    """Record exactly what produced this run, independent of config.yaml's later contents."""
    with open(out_dir / "run_config.json", "w") as f:
        json.dump({
            "timestamp": timestamp,
            "env_height": env.height, "env_width": env.width,
            "obs_shape": list(env.obs_shape),
            **cfg,
        }, f, indent=2)


def _print_run_summary(cfg: dict, env, algo: str, obs_tag: str,
                        reward_mode: str, batch_size: int, steps_per_update: int, out_dir: Path) -> None:
    """Config summary printed before training starts."""
    sfl_cfg = cfg["sfl"]
    total_updates = max(cfg["steps"] // steps_per_update, 1)
    print(f"\n{'=' * 50}")
    print(f"Algo    : {algo}  grid={cfg['grid_size']}  obs={env.obs_shape}")
    print(f"Reward  : {reward_mode}  obs_mode={cfg['obs_mode']}  view={obs_tag}")
    print(f"Envs    : {batch_size} levels/update  episode_len={cfg['episode_len']}  → {steps_per_update:,} spu")
    print(f"SFL     : buffer={sfl_cfg['buffer_size']}  pool={sfl_cfg['pool_size']}  "
          f"rho={sfl_cfg['rho']}  refresh_every={sfl_cfg['refresh_every']} rollouts_per_level={sfl_cfg['rollouts_per_level']}  inner_steps={sfl_cfg.get('inner_steps', 10)}")
    print(f"PPO     : lr={cfg['lr']}  ent={cfg['ent_coef']}  vf={cfg['vf_coef']}  gamma={cfg['gamma']}")
    print(f"Total   : {cfg['steps']:,} steps  (~{total_updates:,} updates)")
    print(f"Output  : {out_dir.resolve()}")
    print(f"{'=' * 50}\n")


# ── Stage 5 ──────────────────────────────────────────────────────────────────

def save_outputs(log, params0, params1, cfg: dict, out_dir: Path, title: str, png: str) -> None:
    """Save the final plot, log.csv, and both agents' params."""
    if log:
        algo = cfg.get("algo", "sfl")
        save_results(log, params0, params1, out_dir, title, png, _display_cfg(cfg), True, algo)


# ── Orchestration ────────────────────────────────────────────────────────────

def run_pipeline(config_path: str, overrides: dict | None = None):
    cfg = load_config(config_path, overrides)
    env = build_environment(cfg)
    trainer = build_trainer(cfg, env)

    # Four independent axes, matching train.py's {algo}_{obs_tag}_{reward_mode}_{obs_mode}:
    #   algo        — sfl (NumPy) | sfl_jax (JAX)
    #   obs_tag     — full | view{N}, from view_size (real for sfl_jax; sfl raises on non-null)
    #   reward_mode — shaped | sparse | delivery (real for sfl_jax; sfl raises on non-shaped)
    #   obs_mode    — rich | simple, the channel encoding
    algo = cfg.get("algo", "sfl")
    view_size = cfg.get("view_size")
    obs_tag = "full" if view_size is None else f"view{view_size}"
    reward_mode = cfg.get("reward_mode", "shaped")
    variant = f"{algo}_{obs_tag}_{reward_mode}_{cfg['obs_mode']}"

    batch_size = cfg["envs"]
    # NumPy path counts both agents' steps into the sample budget; JAX path
    # counts n_envs * rollout_len agent-shared env-steps per update.
    per_agent_mult = 2 if algo == "sfl" else 1
    steps_per_update = batch_size * cfg["episode_len"] * per_agent_mult
    png = f"training_curve_{_fmt_steps(cfg['steps'])}_{batch_size}env_{steps_per_update}spu.png"
    title = f"OvercookedV2 {algo.upper()} — {cfg['grid_size']} ({cfg['obs_mode']})"

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    out_dir = Path(cfg["out_dir"]) / algo / variant / timestamp
    out_dir.mkdir(parents=True, exist_ok=True)
    _save_run_config(cfg, env, out_dir, timestamp)
    _print_run_summary(cfg, env, algo, obs_tag, reward_mode, batch_size, steps_per_update, out_dir)

    params0, params1, log = run_training(trainer, cfg, batch_size, steps_per_update, out_dir, title, png)
    save_outputs(log, params0, params1, cfg, out_dir, title, png)
    return params0, params1, log
