"""
accel.py — ACCEL: Adversarially Compounding Complexity by Editing Levels

Adapted for ued_implementation package structure.
Uses UED utilities from ued.py for grid conversion.
"""

import csv
import json
import pickle
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, List, Optional, Tuple

import jax
import jax.numpy as jnp
import numpy as np

from ..environment.common import StaticObject
from .ippo import IPPOTrainer, Transition, calculate_gae
from ..environment.settings import DELIVERY_REWARD
from ..level_generator import EnvParams, ParametrizedOvercooked, MAX_OBS_FRAC, MAX_RES_FRAC
from ..ued import (
    OvercookedLevel,
    GridCodes,
    get_static_lookup,
    grid_to_static_jax,
    extract_agent_positions_jax,
    grid_to_layout_numpy,
)

_G = ParametrizedOvercooked.codes


# ── Level buffer ──────────────────────────────────────────────────────────────

class ACCELLevelBuffer:
    """Fixed-capacity replay buffer ranked by PVL score."""

    def __init__(self, capacity: int, score_threshold: float = 0.05):
        self.capacity = capacity
        self.score_threshold = score_threshold
        self._levels: List[OvercookedLevel] = []

    def add(self, level: OvercookedLevel) -> bool:
        if level.score < self.score_threshold:
            return False
        if len(self._levels) < self.capacity:
            self._levels.append(level)
            return True
        min_idx = min(range(len(self._levels)), key=lambda i: self._levels[i].score)
        if level.score > self._levels[min_idx].score:
            self._levels[min_idx] = level
            return True
        return False

    def sample(self, rng: np.random.Generator) -> Optional[OvercookedLevel]:
        if not self._levels:
            return None
        scores = np.array([l.score for l in self._levels], dtype=np.float32)
        scores = scores - scores.min() + 1e-8
        probs = scores / scores.sum()
        idx = rng.choice(len(self._levels), p=probs)
        return self._levels[idx]

    def __len__(self):
        return len(self._levels)

    def mean_score(self):
        return float(np.mean([l.score for l in self._levels])) if self._levels else 0.0

    def max_score(self):
        return float(max(l.score for l in self._levels)) if self._levels else 0.0


# ── ACCEL Trainer ─────────────────────────────────────────────────────────────

class ACCELTrainer:
    """ACCEL trainer for OvercookedV2 with UED level management."""

    def __init__(
        self,
        env,
        cfg: dict,
        grid_size: Tuple[int, int],
        buffer_size: int = 100,
        initial_fill_ratio: float = 0.5,
        replay_prob: float = 0.1,
        score_threshold: float = 0.20,
        edit_step: float = 0.1,
        n_edit_attempts: int = 50,
        seed: int = 42,
    ):
        self.env = env
        self.cfg = cfg
        self.ippo = IPPOTrainer(env, cfg)

        if self.ippo.rollout_len < env.max_steps:
            print(
                f"[ACCEL] WARNING: rollout_len={self.ippo.rollout_len} < "
                f"env.max_steps={env.max_steps}. Adjusting → {env.max_steps}."
            )
            self.ippo.rollout_len = env.max_steps

        self.param_env = ParametrizedOvercooked.from_dims(*grid_size, seed=seed)
        self.buffer = ACCELLevelBuffer(capacity=buffer_size, score_threshold=score_threshold)

        self.initial_fill_ratio = initial_fill_ratio
        self.replay_prob = replay_prob
        self.edit_step = edit_step
        self.n_edit_attempts = n_edit_attempts
        self.rng = np.random.default_rng(seed)

        self.iterations = 0
        self.updates = 0
        self.new_count = 0
        self.replay_count = 0
        self.edit_count = 0

        self.env_param_log: List[Tuple] = []
        self._reset_id: int = 0

    # ── Logging ───────────────────────────────────────────────────────────

    def _log_reset(self, level: OvercookedLevel, branch: str, step: int,
                   writer=None) -> None:
        p = level.params
        if p is not None and hasattr(p, 'obs_density'):
            vec = [p.obs_density, p.obs_skew_x, p.res_density, p.res_skew_x]
        else:
            vec = [0.0, 0.0, 0.0, 0.0]
        rid = self._reset_id
        self.env_param_log.append((vec, rid, level.grid.copy(), level.seed))
        self._reset_id += 1
        if writer is not None:
            writer.writerow({
                "reset_id":    rid,
                "step":        step,
                "branch":      branch,
                "gen_counter": level.seed,
                "obs_density": f"{vec[0]:.4f}",
                "obs_skew_x":  f"{vec[1]:.4f}",
                "res_density": f"{vec[2]:.4f}",
                "res_skew_x":  f"{vec[3]:.4f}",
            })

    # ── Environment interaction ───────────────────────────────────────────

    def _prepare_level(self, grid: np.ndarray):
        """Convert grid to JAX arrays for env.reset."""
        g = jnp.array(grid, dtype=jnp.int32)
        static = get_static_lookup()[g]
        agents = extract_agent_positions_jax(g)
        return static, agents

    def _fresh_reset(self, key, static_objects, agent_positions):
        """Reset all parallel envs with the given layout."""
        key, k_r = jax.random.split(key)
        reset_keys = jax.random.split(k_r, self.ippo.n_envs)
        obs_dict, env_states = jax.vmap(
            lambda k: self.env.reset(k, static_objects, agent_positions)
        )(reset_keys)
        h0 = jnp.zeros((self.ippo.n_envs, self.ippo.hidden_size))
        h1 = jnp.zeros((self.ippo.n_envs, self.ippo.hidden_size))
        return obs_dict, env_states, h0, h1, key

    def _bootstrap_values(self, ts0, ts1, obs_dict, h0, h1):
        def _v(ts, obs, h):
            return jax.vmap(lambda o, hi: ts.apply_fn(ts.params, o, hi))(obs, h)[1]
        return _v(ts0, obs_dict["agent_0"], h0), _v(ts1, obs_dict["agent_1"], h1)

    # ── PVL scoring ───────────────────────────────────────────────────────

    def _compute_pvl(self, trs0, trs1, last_v0, last_v1) -> float:
        adv0, _ = calculate_gae(trs0, last_v0, self.ippo.gamma, self.ippo.lam)
        adv1, _ = calculate_gae(trs1, last_v1, self.ippo.gamma, self.ippo.lam)
        pvl0 = float(jnp.mean(jnp.maximum(adv0, 0.0)))
        pvl1 = float(jnp.mean(jnp.maximum(adv1, 0.0)))
        return (pvl0 + pvl1) / 2.0

    def _score_level(self, ts0, ts1, grid: np.ndarray, key):
        """Collect a stop-gradient rollout and return PVL score."""
        static, agents = self._prepare_level(grid)
        obs_dict, env_states, h0, h1, key = self._fresh_reset(key, static, agents)
        key, k_r = jax.random.split(key)
        (trs0, trs1, (next_obs, _), (nh0, nh1),
         _, _, _, _, key) = self.ippo._collect_rollout(
            ts0, ts1, (obs_dict, env_states), (h0, h1), k_r)
        last_v0, last_v1 = self._bootstrap_values(ts0, ts1, next_obs, nh0, nh1)
        pvl = self._compute_pvl(trs0, trs1, last_v0, last_v1)
        return pvl, key

    # ── Level generation and editing ──────────────────────────────────────

    def _generate_level(self, key, max_tries=50):
        for _ in range(max_tries):
            key, k_gen = jax.random.split(key)
            self.param_env._counter += 1
            ckey = jax.random.fold_in(
                jax.random.PRNGKey(self.param_env.seed), self.param_env._counter
            )
            grid, params = self.param_env._generate_from_key(ckey)
            if ParametrizedOvercooked.validate(grid):
                level = OvercookedLevel(
                    grid=np.asarray(grid), params=params,
                    seed=self.param_env._counter,
                )
                return level, key
        return None, key

    def _edit_level(self, level: OvercookedLevel, key):
        """Perturb level params and regenerate layout."""
        p = level.params
        if p is None:
            return None, key

        for _ in range(self.n_edit_attempts):
            noise = self.rng.normal(0.0, self.edit_step, 4).astype(np.float32)
            new_params = EnvParams(
                obs_density=float(np.clip(p.obs_density + noise[0], 0.0, MAX_OBS_FRAC)),
                obs_skew_x=float(np.clip(p.obs_skew_x + noise[1], -1.0, 1.0)),
                res_density=float(np.clip(p.res_density + noise[2], 0.0, MAX_RES_FRAC)),
                res_skew_x=float(np.clip(p.res_skew_x + noise[3], -1.0, 1.0)),
            )
            key, k_layout = jax.random.split(key)
            k_ng, k_np = jax.random.split(k_layout)
            num_goals = int(jax.random.randint(k_ng, (), 1, self.param_env.max_goals + 1))
            num_pots = int(jax.random.randint(k_np, (), 1, max(self.param_env.num_pots, 1) + 1))

            grid = np.asarray(ParametrizedOvercooked.make_layout(
                self.param_env.H, self.param_env.W, new_params, k_layout,
                num_agents=self.param_env.num_agents,
                num_goals=num_goals,
                num_pots=num_pots,
                num_plates=self.param_env.num_plates,
                ingredient_types=self.param_env.ingredient_types,
            ))
            if ParametrizedOvercooked.validate(grid):
                edited = OvercookedLevel(grid=grid, params=new_params, seed=-1)
                return edited, key
        return None, key

    # ── Main training loop ────────────────────────────────────────────────

    def train(
        self,
        key: jnp.ndarray,
        resume_params: dict = None,
        log_callback: Optional[Callable] = None,
        checkpoint_dir=None,
        checkpoint_every: int = 100,
    ) -> Tuple:
        key, k_init = jax.random.split(key)
        ts0, ts1 = self.ippo._make_train_state(k_init, resume_params)

        log, all_ep_returns, all_ep_deliveries = [], [], []
        t0 = time.time()
        total_collected = 0
        steps_per_rollout = self.ippo.rollout_len * self.ippo.n_envs

        # IO logging
        episode_log_file = episode_log_writer = None
        env_param_log_file = env_param_log_writer = None
        if checkpoint_dir:
            ep_path = Path(checkpoint_dir) / "episodes.csv"
            ep_path.parent.mkdir(parents=True, exist_ok=True)
            episode_log_file = open(ep_path, "w", newline="")
            episode_log_writer = csv.DictWriter(episode_log_file,
                fieldnames=["episode", "deliveries", "return", "level_score", "branch"])
            episode_log_writer.writeheader()
            episode_log_file.flush()

            param_path = Path(checkpoint_dir) / "env_params.csv"
            env_param_log_file = open(param_path, "w", newline="")
            env_param_log_writer = csv.DictWriter(env_param_log_file,
                fieldnames=["reset_id", "step", "branch", "gen_counter",
                            "obs_density", "obs_skew_x", "res_density", "res_skew_x"])
            env_param_log_writer.writeheader()
            env_param_log_file.flush()

        # Phase 0: Fill buffer
        n_initial = max(1, int(self.buffer.capacity * self.initial_fill_ratio))
        print(f"\n[ACCEL] Filling buffer with {n_initial} levels...")
        while len(self.buffer) < n_initial:
            level, key = self._generate_level(key)
            if level is None:
                continue
            pvl, key = self._score_level(ts0, ts1, level.grid, key)
            level.score = pvl
            self._log_reset(level, "buffer_init", total_collected, env_param_log_writer)
            if self.buffer.add(level):
                n = len(self.buffer)
                if n % max(1, n_initial // 5) == 0 or n == n_initial:
                    print(f"  [{n}/{n_initial}] pvl={pvl:.4f} mean={self.buffer.mean_score():.4f}")
        print(f"[ACCEL] Buffer ready: {len(self.buffer)} levels\n")

        # Main loop
        total_target = self.ippo.total_steps
        print(f"[ACCEL] Target: {total_target:,} steps\n")

        while total_collected < total_target:
            self.iterations += 1
            d = (self.rng.random() < self.replay_prob) and (len(self.buffer) > 0)

            if not d:
                # ── Generate new level, score it, add to buffer ───────────
                level, key = self._generate_level(key)
                if level is None:
                    continue
                pvl, key = self._score_level(ts0, ts1, level.grid, key)
                level.score = pvl
                self._log_reset(level, "new", total_collected, env_param_log_writer)
                self.buffer.add(level)
                self.new_count += 1
                total_collected += steps_per_rollout
                branch = "new"
                log_loss = 0.0
                log_level_score = pvl

                ep_returns_np = np.zeros((self.ippo.rollout_len, self.ippo.n_envs))
                ep_dones_np = np.zeros((self.ippo.rollout_len, self.ippo.n_envs), dtype=bool)

            else:
                # ── Replay → train → edit → score ────────────────────────
                level = self.buffer.sample(self.rng)
                static, agents = self._prepare_level(level.grid)

                self._log_reset(level, "replay", total_collected, env_param_log_writer)
                obs_dict, env_states, h0, h1, key = self._fresh_reset(key, static, agents)
                key, k_r = jax.random.split(key)
                (trs0, trs1, (next_obs, _), (nh0, nh1),
                 ep_returns_arr, _, ep_dones_arr, shaped_comps, key) = self.ippo._collect_rollout(
                    ts0, ts1, (obs_dict, env_states), (h0, h1), k_r)

                last_v0, last_v1 = self._bootstrap_values(ts0, ts1, next_obs, nh0, nh1)
                adv0, ret0 = calculate_gae(trs0, last_v0, self.ippo.gamma, self.ippo.lam)
                adv1, ret1 = calculate_gae(trs1, last_v1, self.ippo.gamma, self.ippo.lam)

                key, ks0, ks1 = jax.random.split(key, 3)
                ts0, loss0 = self.ippo._update_agent(ts0, trs0, adv0, ret0, seed=int(np.asarray(ks0)[0]))
                ts1, loss1 = self.ippo._update_agent(ts1, trs1, adv1, ret1, seed=int(np.asarray(ks1)[0]))
                log_loss = (float(loss0) + float(loss1)) / 2.0

                # Re-score parent
                pvl_after, key = self._score_level(ts0, ts1, level.grid, key)
                level.score = pvl_after

                # Edit → score
                edited, key = self._edit_level(level, key)
                if edited is not None:
                    pvl_e, key = self._score_level(ts0, ts1, edited.grid, key)
                    edited.score = pvl_e
                    self._log_reset(edited, "edit", total_collected, env_param_log_writer)
                    if self.buffer.add(edited):
                        self.edit_count += 1
                    log_level_score = pvl_e
                else:
                    log_level_score = level.score

                self.replay_count += 1
                self.updates += 1
                total_collected += steps_per_rollout
                branch = "replay"

                # Episode tracking
                ep_returns_np = np.asarray(ep_returns_arr)
                ep_dones_np = np.asarray(ep_dones_arr)
                step_r_np = np.asarray(trs0.reward)

                if self.ippo.reward_mode == "delivery":
                    deliv_per_step = (step_r_np >= 1.0).astype(np.int32)
                else:
                    deliv_per_step = (step_r_np >= DELIVERY_REWARD).astype(np.int32)

                ep_deliv_buf = np.zeros(self.ippo.n_envs, dtype=np.int32)
                for t in range(self.ippo.rollout_len):
                    ep_deliv_buf += deliv_per_step[t]
                    for env_i in np.where(ep_dones_np[t])[0]:
                        dc = int(ep_deliv_buf[env_i])
                        ep_ret = float(ep_returns_np[t, env_i])
                        all_ep_deliveries.append(dc)
                        all_ep_returns.append(ep_ret)
                        if episode_log_writer:
                            episode_log_writer.writerow({
                                "episode": len(all_ep_returns) - 1,
                                "deliveries": dc,
                                "return": f"{ep_ret:.2f}",
                                "level_score": f"{level.score:.4f}",
                                "branch": branch,
                            })
                            episode_log_file.flush()
                        ep_deliv_buf[env_i] = 0

            # ── Logging ───────────────────────────────────────────────────
            if self.iterations % self.ippo.log_every == 0:
                mean_r = float(np.mean(all_ep_returns[-100:])) if all_ep_returns else 0.0
                sps = total_collected / max(time.time() - t0, 1e-9)

                entry = {
                    "steps": total_collected,
                    "iteration": self.iterations,
                    "update": self.updates,
                    "mean_r": mean_r,
                    "loss": log_loss,
                    "buffer_size": len(self.buffer),
                    "buffer_mean_score": self.buffer.mean_score(),
                    "buffer_max_score": self.buffer.max_score(),
                    "level_score": log_level_score,
                    "branch": 1 if d else 0,
                    "replay_count": self.replay_count,
                    "new_count": self.new_count,
                    "total_deliveries": sum(all_ep_deliveries),
                }
                log.append(entry)
                if log_callback:
                    log_callback(log)

                if self.iterations % (self.ippo.log_every * 5) == 0:
                    print(
                        f"[ACCEL] iter={self.iterations:5d}  upd={self.updates:5d}  "
                        f"steps={total_collected:9,}  mean_r={mean_r:6.2f}  "
                        f"loss={log_loss:.4f}  buf={len(self.buffer)}/{self.buffer.capacity}  "
                        f"score={self.buffer.mean_score():.4f}  "
                        f"replays={self.replay_count}  edits={self.edit_count}  "
                        f"sps={sps:.0f}",
                        flush=True,
                    )

            # ── Checkpoint ────────────────────────────────────────────────
            if (checkpoint_dir and checkpoint_every > 0
                    and self.updates > 0 and self.updates % checkpoint_every == 0):
                ckpt = Path(checkpoint_dir) / "checkpoints" / f"update_{self.updates:05d}"
                ckpt.mkdir(parents=True, exist_ok=True)
                with open(ckpt / "params_agent0.pkl", "wb") as f: pickle.dump(ts0.params, f)
                with open(ckpt / "params_agent1.pkl", "wb") as f: pickle.dump(ts1.params, f)

        # Final checkpoint
        if checkpoint_dir:
            final = Path(checkpoint_dir) / "checkpoints" / "final"
            final.mkdir(parents=True, exist_ok=True)
            with open(final / "params_agent0.pkl", "wb") as f: pickle.dump(ts0.params, f)
            with open(final / "params_agent1.pkl", "wb") as f: pickle.dump(ts1.params, f)

        if episode_log_file: episode_log_file.close()
        if env_param_log_file: env_param_log_file.close()
        if checkpoint_dir:
            with open(Path(checkpoint_dir) / "env_params.pkl", "wb") as f:
                pickle.dump(self.env_param_log, f)

        print(
            f"\n[ACCEL] Done: {self.updates} updates, {self.replay_count} replays, "
            f"{self.edit_count} edits, buf={len(self.buffer)}, "
            f"mean_score={self.buffer.mean_score():.4f}"
        )
        return ts0, ts1, log
