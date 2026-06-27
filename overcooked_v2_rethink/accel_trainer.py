"""
accel_trainer.py — ACCEL: Adversarially Compounding Complexity by Editing Levels

Implements Algorithm 1 from:
  "Evolving Curricula with Regret-Based Environment Design"
  Parker-Holder et al., NeurIPS 2022 — https://accelagent.github.io/

Algorithm 1 (ACCEL):
  Input:  Buffer size K, initial fill ratio ρ, level generator
  Init:   policy π(φ), level buffer Λ.  Sample K*ρ initial levels.
  while not converged:
    Sample replay decision d ~ P_D(d)
    if d = 0:  (generate new)
      θ  ← level generator
      τ  ← collect trajectory on θ  [stop-gradient φ_⊥]
      S  ← PVL(τ)
      Add θ to Λ if S ≥ threshold
    else:       (replay + edit)
      θ  ~ Λ
      τ  ← collect trajectory on θ
      Update π with R(τ)
      θ' ← edit(θ)
      τ' ← collect trajectory on θ'  [stop-gradient φ_⊥]
      S  ← PVL(τ')
      Add θ' to Λ if S ≥ threshold
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

from .common import StaticObject
from .ippo_jax import IPPOTrainer, Transition, compute_gae
from .overcooked_parametrized_current import EnvParams, ParametrizedOvercooked
from .settings import DELIVERY_REWARD

# Convenience aliases so the rest of this file stays readable
_G = ParametrizedOvercooked.codes   # GridCodes namespace

# ── Object-code mapping ────────────────────────────────────────────────────────
_TO_STATIC = {
    _G.EMPTY:        int(StaticObject.EMPTY),
    _G.GOAL:         int(StaticObject.GOAL),
    _G.POT:          int(StaticObject.POT),
    _G.OBSTACLE:     int(StaticObject.WALL),
    _G.PLATE_PILE:   int(StaticObject.PLATE_PILE),
    _G.INGREDIENT_0: int(StaticObject.INGREDIENT_PILE_BASE) + 0,
    _G.INGREDIENT_1: int(StaticObject.INGREDIENT_PILE_BASE) + 1,
    _G.AGENT_0:      int(StaticObject.EMPTY),
    _G.AGENT_1:      int(StaticObject.EMPTY),
}

# ── Level dataclass ────────────────────────────────────────────────────────────

@dataclass
class Level:
    """A single curriculum level stored in the ACCEL buffer."""
    params: EnvParams       # 4-parameter density encoding (for editing)
    grid:   np.ndarray      # H×W numpy grid in ParametrizedOvercooked format
    score:  float = 0.0     # PVL regret score (higher = more learnable)


# ── Level buffer ───────────────────────────────────────────────────────────────

class LevelBuffer:
    """Fixed-capacity replay buffer ranked by PVL score."""

    def __init__(self, capacity: int, score_threshold: float = 0.05):
        self.capacity = capacity
        self.score_threshold = score_threshold
        self._levels: List[Level] = []

    def add(self, level: Level) -> bool:
        """Try to insert *level*. Returns True if it was accepted."""
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

    def sample(self, rng: np.random.Generator) -> Optional[Level]:
        """Score-proportional sample. Returns None when buffer is empty."""
        if not self._levels:
            return None
        scores = np.array([l.score for l in self._levels], dtype=np.float32)
        scores = scores - scores.min() + 1e-8          # ensure positive
        probs  = scores / scores.sum()
        idx    = rng.choice(len(self._levels), p=probs)
        return self._levels[idx]

    def __len__(self) -> int:
        return len(self._levels)

    def mean_score(self) -> float:
        return float(np.mean([l.score for l in self._levels])) if self._levels else 0.0

    def max_score(self) -> float:
        return float(max(l.score for l in self._levels)) if self._levels else 0.0


# ── ACCEL Trainer ──────────────────────────────────────────────────────────────

class ACCELTrainer:
    """ACCEL trainer wrapper over IPPOTrainer for JAX-native OvercookedV2."""

    def __init__(
        self,
        env,
        cfg: dict,
        base_layout_str: str,
        buffer_size: int = 100,
        initial_fill_ratio: float = 0.5,
        replay_prob: float = 0.1,
        score_threshold: float = 0.20,
        edit_step: float = 0.1,
        n_edit_attempts: int = 50,
        seed: int = 42,
    ):
        self.env  = env
        self.cfg  = cfg
        self.ippo = IPPOTrainer(env, cfg)

        # ACCEL calls _fresh_reset before every rollout (state.time resets to 0).
        # If rollout_len < env.max_steps, the episode never terminates within the
        # rollout window → done is always False → all_ep_returns stays empty →
        # mean_r = 0.00 forever.  Extend rollout_len to cover at least one full episode.
        if self.ippo.rollout_len < env.max_steps:
            print(
                f"[ACCEL] WARNING: rollout_len={self.ippo.rollout_len} < "
                f"env.max_steps={env.max_steps}.  Episodes never complete; "
                f"adjusting rollout_len → {env.max_steps}."
            )
            self.ippo.rollout_len = env.max_steps

        base_grid        = ParametrizedOvercooked.from_string(base_layout_str)
        self.param_env   = ParametrizedOvercooked(base_grid=base_grid, seed=seed)
        self.buffer      = LevelBuffer(capacity=buffer_size, score_threshold=score_threshold)

        self.initial_fill_ratio = initial_fill_ratio
        self.replay_prob        = replay_prob
        self.edit_step          = edit_step
        self.n_edit_attempts    = n_edit_attempts
        self.rng                = np.random.default_rng(seed)

        # Diagnostics
        self.iterations   = 0   
        self.updates      = 0   
        self.new_count    = 0
        self.replay_count = 0
        self.edit_count   = 0

    def _grid_to_layout(self, grid: np.ndarray):
        H, W = grid.shape
        static_objects = np.vectorize(_TO_STATIC.get)(grid).astype(int)

        agent_positions = []
        for code in [_G.AGENT_0, _G.AGENT_1]:
            rows, cols = np.where(grid == code)
            if len(rows):
                r, c = int(rows[0]), int(cols[0])
                agent_positions.append((c, r))   # (x=col, y=row)

        return static_objects, agent_positions

    def _set_env_layout(self, grid: np.ndarray):
        static_objects, agent_positions = self._grid_to_layout(grid)
        self.env.layout.static_objects = static_objects
        if agent_positions:
            self.env.layout.agent_positions = agent_positions

    def _fresh_reset(self, key: jnp.ndarray):
        key, k_r = jax.random.split(key)
        reset_keys = jax.random.split(k_r, self.ippo.n_envs)
        obs_dict, env_states = jax.vmap(self.env.reset)(reset_keys)
        h0 = jnp.zeros((self.ippo.n_envs, self.ippo.hidden_size))
        h1 = jnp.zeros((self.ippo.n_envs, self.ippo.hidden_size))
        return obs_dict, env_states, h0, h1, key

    def _bootstrap_values(self, ts0, ts1, obs_dict, h0, h1):
        def _v(ts, obs, h):
            return jax.vmap(lambda o, hi: ts.apply_fn(ts.params, o, hi))(obs, h)[1]
        return _v(ts0, obs_dict["agent_0"], h0), _v(ts1, obs_dict["agent_1"], h1)

    def _compute_pvl(
        self,
        trs0: Transition,
        trs1: Transition,
        last_v0: jnp.ndarray,
        last_v1: jnp.ndarray,
    ) -> float:
        adv0, _ = compute_gae(trs0, last_v0, self.ippo.gamma, self.ippo.lam)
        adv1, _ = compute_gae(trs1, last_v1, self.ippo.gamma, self.ippo.lam)
        pvl0 = float(jnp.mean(jnp.maximum(adv0, 0.0)))
        pvl1 = float(jnp.mean(jnp.maximum(adv1, 0.0)))
        return (pvl0 + pvl1) / 2.0

    def _score_level(self, ts0, ts1, grid: np.ndarray, key: jnp.ndarray) -> Tuple[float, jnp.ndarray]:
        """Collect a stop-gradient rollout (inherent to JAX forward pass) and return PVL score."""
        self._set_env_layout(grid)
        obs_dict, env_states, h0, h1, key = self._fresh_reset(key)
        key, k_r = jax.random.split(key)
        
        (trs0, trs1, (next_obs, _), (nh0, nh1),
         _ep_rets, _ep_lens, _dones, _comps, key) = \
            self.ippo._collect_rollout(ts0, ts1, (obs_dict, env_states), (h0, h1), k_r)
            
        last_v0, last_v1 = self._bootstrap_values(ts0, ts1, next_obs, nh0, nh1)
        pvl = self._compute_pvl(trs0, trs1, last_v0, last_v1)
        return pvl, key

    def _generate_level(self, key: jnp.ndarray, max_tries: int = 50) -> Tuple[Optional[Level], jnp.ndarray]:
        for _ in range(max_tries):
            key, k_gen = jax.random.split(key)
            self.param_env._counter += 1
            ckey = jax.random.fold_in(
                jax.random.PRNGKey(self.param_env.seed), self.param_env._counter
            )
            grid, params = self.param_env._generate_from_key(ckey)
            if ParametrizedOvercooked.validate(grid):
                return Level(params=params, grid=grid), key
        return None, key

    def _edit_level(self, level: Level, key: jnp.ndarray) -> Tuple[Optional[Level], jnp.ndarray]:
        p = level.params
        for _ in range(self.n_edit_attempts):
            noise = self.rng.normal(0.0, self.edit_step, 4).astype(np.float32)
            new_params = EnvParams(
                obstacles_left  = float(np.clip(p.obstacles_left  + noise[0], 0.0, _G.MAX_OBS_FRAC)),
                obstacles_right = float(np.clip(p.obstacles_right + noise[1], 0.0, _G.MAX_OBS_FRAC)),
                resources_left  = float(np.clip(p.resources_left  + noise[2], 0.0, _G.MAX_RES_FRAC)),
                resources_right = float(np.clip(p.resources_right + noise[3], 0.0, _G.MAX_RES_FRAC)),
            )
            key, k_layout = jax.random.split(key)
            k_ng, k_np = jax.random.split(k_layout)
            num_goals = int(jax.random.randint(k_ng, (), 1, self.param_env.max_goals + 1))
            num_pots  = int(jax.random.randint(k_np, (), 1, max(self.param_env.num_pots, 1) + 1))
            grid = ParametrizedOvercooked.make_layout(
                self.param_env.H, self.param_env.W, new_params, k_layout,
                num_agents=self.param_env.num_agents,
                num_goals=num_goals,
                num_pots=num_pots,
                num_plates=self.param_env.num_plates,
                ingredient_types=self.param_env.ingredient_types,
            )
            if ParametrizedOvercooked.validate(grid):
                return Level(params=new_params, grid=grid), key
        return None, key

    # ── Main training loop — Algorithm 1 ───────────────────────────────────────

    def train(
        self,
        key: jnp.ndarray,
        resume_params: dict = None,
        log_callback: Optional[Callable] = None,
        checkpoint_dir=None,
        checkpoint_every: int = 100,
    ) -> Tuple:
        # Initialise student policy TrainStates
        key, k_init = jax.random.split(key)
        ts0, ts1 = self.ippo._make_train_state(k_init, resume_params)

        log: list = []
        all_ep_returns: list = []
        all_ep_deliveries: list = []
        t0 = time.time()
        total_collected = 0
        steps_per_rollout = self.ippo.rollout_len * self.ippo.n_envs

        # IO Logging setup
        episode_log_file   = None
        episode_log_writer = None
        if checkpoint_dir:
            ep_log_path = Path(checkpoint_dir) / "episodes.csv"
            ep_log_path.parent.mkdir(parents=True, exist_ok=True)
            episode_log_file   = open(ep_log_path, "w", newline="")
            episode_log_writer = csv.DictWriter(
                episode_log_file,
                fieldnames=["episode", "deliveries", "return", "level_score", "branch"],
            )
            episode_log_writer.writeheader()
            episode_log_file.flush()

            ckpt_root = Path(checkpoint_dir) / "checkpoints"
            ckpt_root.mkdir(parents=True, exist_ok=True)

        # ── Phase 0: Initialise buffer (Guaranteed K*ρ fill) ─────────────────
        n_initial = max(1, int(self.buffer.capacity * self.initial_fill_ratio))
        print(f"\n[ACCEL] Initialising buffer with {n_initial} valid levels...")
        
        while len(self.buffer) < n_initial:
            level, key = self._generate_level(key)
            if level is None:
                continue
            pvl, key = self._score_level(ts0, ts1, level.grid, key)
            level.score = pvl # Positive Value Loss / Regret
            if self.buffer.add(level):
                curr_len = len(self.buffer)
                if curr_len % max(1, n_initial // 5) == 0 or curr_len == n_initial:
                    print(f"  [{curr_len}/{n_initial}] buf={curr_len}  "
                          f"pvl={pvl:.4f}  mean_score={self.buffer.mean_score():.4f}")
                          
        print(f"[ACCEL] Buffer ready: {len(self.buffer)} levels\n")

        # ── Main loop ──────────────────────────────────────────────────────────
        total_target = self.ippo.total_steps
        print(f"[ACCEL] Main loop: target {total_target:,} steps\n")

        while total_collected < total_target:
            self.iterations += 1

            # Sample replay decision d
            d = (self.rng.random() < self.replay_prob) and (len(self.buffer) > 0)

            if not d:
                # ── d = 0: Generate new level (Evaluation Branch, stop-grad) ──
                level, key = self._generate_level(key)
                if level is None:
                    continue

                pvl, key   = self._score_level(ts0, ts1, level.grid, key)
                level.score = pvl
                self.buffer.add(level)
                self.new_count   += 1
                total_collected  += steps_per_rollout
                branch            = "new"
                log_loss          = 0.0
                log_level_score   = pvl

                # Dummy structures to skip stats
                ep_returns_np  = np.zeros((self.ippo.rollout_len, self.ippo.n_envs))
                ep_dones_np    = np.zeros((self.ippo.rollout_len, self.ippo.n_envs), dtype=bool)

            else:
                # ── d = 1: Replay → Update Policy → Edit → Score ──────────────
                level = self.buffer.sample(self.rng)
                self._set_env_layout(level.grid)

                # Collect full rollout on θ
                obs_dict, env_states, h0, h1, key = self._fresh_reset(key)
                key, k_r = jax.random.split(key)
                (trs0, trs1, (next_obs, _), (nh0, nh1),
                 ep_returns_arr, ep_lens_arr, ep_dones_arr,
                 shaped_comps, key) = self.ippo._collect_rollout(
                    ts0, ts1, (obs_dict, env_states), (h0, h1), k_r
                )

                # Bootstrap values using ending hidden states (nh0, nh1)
                last_v0, last_v1 = self._bootstrap_values(ts0, ts1, next_obs, nh0, nh1)
                adv0, ret0 = compute_gae(trs0, last_v0, self.ippo.gamma, self.ippo.lam)
                adv1, ret1 = compute_gae(trs1, last_v1, self.ippo.gamma, self.ippo.lam)

                # Update Flax TrainStates (Replaces immutable objects with optimized parameters)
                key, ks0, ks1 = jax.random.split(key, 3)
                ts0, loss0 = self.ippo._update_agent(ts0, trs0, adv0, ret0, seed=int(np.asarray(ks0)[0]))
                ts1, loss1 = self.ippo._update_agent(ts1, trs1, adv1, ret1, seed=int(np.asarray(ks1)[0]))
                log_loss = (float(loss0) + float(loss1)) / 2.0

                # CRITICAL: Re-evaluate parent level score now that policy π is smarter!
                pvl_after, key = self._score_level(ts0, ts1, level.grid, key)
                level.score = pvl_after 

                # Edit θ → θ'
                edited_level, key = self._edit_level(level, key)
                if edited_level is not None:
                    pvl_e, key    = self._score_level(ts0, ts1, edited_level.grid, key)
                    edited_level.score = pvl_e
                    if self.buffer.add(edited_level):
                        self.edit_count += 1
                    log_level_score = pvl_e
                else:
                    log_level_score = level.score

                self.replay_count  += 1
                self.updates       += 1
                total_collected    += steps_per_rollout
                branch              = "replay"

                # Parse logging and tracking stats
                ep_returns_np = np.asarray(ep_returns_arr)
                ep_dones_np   = np.asarray(ep_dones_arr)
                step_r_np      = np.asarray(trs0.reward)

                if self.ippo.reward_mode == "delivery":
                    deliv_per_step = (step_r_np >= 1.0).astype(np.int32)
                else:
                    deliv_per_step = (step_r_np >= DELIVERY_REWARD).astype(np.int32)
                    
                ep_deliv_buf = np.zeros(self.ippo.n_envs, dtype=np.int32)
                for t in range(self.ippo.rollout_len):
                    ep_deliv_buf += deliv_per_step[t]
                    for env_i in np.where(ep_dones_np[t])[0]:
                        ep_num = len(all_ep_returns)
                        dc     = int(ep_deliv_buf[env_i])
                        ep_ret = float(ep_returns_np[t, env_i])
                        all_ep_deliveries.append(dc)
                        all_ep_returns.append(ep_ret)
                        if episode_log_writer:
                            episode_log_writer.writerow({
                                "episode":     ep_num,
                                "deliveries":  dc,
                                "return":      f"{ep_ret:.2f}",
                                "level_score": f"{level.score:.4f}",
                                "branch":      branch,
                            })
                            episode_log_file.flush()
                        ep_deliv_buf[env_i] = 0

            # ── Logging ────────────────────────────────────────────────────────
            if self.iterations % self.ippo.log_every == 0:
                mean_r   = float(np.mean(all_ep_returns[-100:])) if all_ep_returns else 0.0
                elapsed  = time.time() - t0
                sps      = total_collected / max(elapsed, 1e-9)

                entry = {
                    "steps":             total_collected,
                    "iteration":         self.iterations,
                    "update":            self.updates,
                    "mean_r":            mean_r,
                    "loss":              log_loss,
                    "buffer_size":       len(self.buffer),
                    "buffer_mean_score": self.buffer.mean_score(),
                    "buffer_max_score":  self.buffer.max_score(),
                    "level_score":       log_level_score,
                    "branch":            1 if d else 0,
                    "replay_count":      self.replay_count,
                    "new_count":         self.new_count,
                    "total_deliveries":  sum(all_ep_deliveries),
                }
                log.append(entry)
                if log_callback:
                    log_callback(log)

                if self.iterations % (self.ippo.log_every * 5) == 0:
                    print(
                        f"[ACCEL] iter={self.iterations:5d}  upd={self.updates:5d}  "
                        f"steps={total_collected:9,}  "
                        f"mean_r={mean_r:6.2f}  loss={log_loss:.4f}  "
                        f"buf={len(self.buffer)}/{self.buffer.capacity}  "
                        f"score={self.buffer.mean_score():.4f}  "
                        f"replays={self.replay_count}  edits={self.edit_count}  "
                        f"sps={sps:.0f}"
                    )

            # ── Checkpoint ─────────────────────────────────────────────────────
            if (checkpoint_dir and checkpoint_every > 0
                    and self.updates > 0 and self.updates % checkpoint_every == 0):
                ckpt = Path(checkpoint_dir) / "checkpoints" / f"update_{self.updates:05d}"
                ckpt.mkdir(parents=True, exist_ok=True)
                with open(ckpt / "params_agent0.pkl", "wb") as f:
                    pickle.dump(ts0.params, f)
                with open(ckpt / "params_agent1.pkl", "wb") as f:
                    pickle.dump(ts1.params, f)
                with open(ckpt / "checkpoint_metadata.json", "w") as f:
                    json.dump({
                        "final_update":      self.updates,
                        "final_steps":       total_collected,
                        "final_mean_r":      float(np.mean(all_ep_returns[-100:])) if all_ep_returns else 0.0,
                        "final_loss":        log_loss,
                        "buffer_size":       len(self.buffer),
                        "buffer_mean_score": self.buffer.mean_score(),
                        "reward_mode":       self.ippo.reward_mode,
                    }, f, indent=2)

        # ── Final checkpoint ───────────────────────────────────────────────────
        if checkpoint_dir:
            final = Path(checkpoint_dir) / "checkpoints" / "final"
            final.mkdir(parents=True, exist_ok=True)
            with open(final / "params_agent0.pkl", "wb") as f:
                pickle.dump(ts0.params, f)
            with open(final / "params_agent1.pkl", "wb") as f:
                pickle.dump(ts1.params, f)
            with open(final / "checkpoint_metadata.json", "w") as f:
                json.dump({
                    "final_update":  self.updates,
                    "final_steps":   total_collected,
                    "final_mean_r":  float(np.mean(all_ep_returns[-100:])) if all_ep_returns else 0.0,
                    "replay_count":  self.replay_count,
                    "edit_count":    self.edit_count,
                    "reward_mode":   self.ippo.reward_mode,
                }, f, indent=2)

        if episode_log_file:
            episode_log_file.close()

        print(
            f"\n[ACCEL] Done: {self.updates} policy updates, "
            f"{self.replay_count} replays, {self.edit_count} edits added, "
            f"buffer={len(self.buffer)} levels, "
            f"mean_score={self.buffer.mean_score():.4f}"
        )
        return ts0, ts1, log