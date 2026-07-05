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
from .ippo_jax import IPPOTrainer, Transition, calculate_gae
from .overcooked_parametrized_current import EnvParams, ParametrizedOvercooked
from .settings import DELIVERY_REWARD

# Grid codes mapping configuration
_G = ParametrizedOvercooked.codes
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

# JAX lookup array: index with a ParametrizedOvercooked grid code → StaticObject code
_MAX_GRID_CODE = 13
_STATIC_LOOKUP = jnp.array(
    [_TO_STATIC.get(i, int(StaticObject.EMPTY)) for i in range(_MAX_GRID_CODE)],
    dtype=jnp.int32,
)


def _extract_agent_pos_jax(grid_hw: jnp.ndarray) -> jnp.ndarray:
    """Return (2, 2) int32 array [[x0,y0],[x1,y1]] from a (H,W) param-env grid. JAX-compatible."""
    H, W = grid_hw.shape
    flat     = grid_hw.reshape(-1)
    row_idx  = jnp.repeat(jnp.arange(H), W)
    col_idx  = jnp.tile(jnp.arange(W), H)
    idx0 = jnp.argmax(flat == _G.AGENT_0)  # first match; 0 if not found (invalid grid)
    idx1 = jnp.argmax(flat == _G.AGENT_1)
    pos0 = jnp.stack([col_idx[idx0], row_idx[idx0]])  # [x, y]
    pos1 = jnp.stack([col_idx[idx1], row_idx[idx1]])
    return jnp.stack([pos0, pos1]).astype(jnp.int32)  # (2, 2)

@dataclass
class Level:
    """Dataclass to store a curriculum level within the SFL buffer."""
    params:      EnvParams
    grid:        np.ndarray
    score:       float = 0.0  # Learnability score: p * (1 - p)
    gen_counter: int   = -1   # param_env._counter used to generate this level


class SFLTrainer:
    """Trainer implementing 'Sampling For Learnability' (SFL) optimized natively for JAX."""

    def __init__(
        self,
        env,
        cfg: dict,
        grid_size: Tuple[int, int],
        buffer_size: int = 100,         # K (buffer D capacity)
        n_random_pool: int = 400,      # N (random pool size for evaluation)
        rho: float = 0.5,              # Mixing coefficient for levels "ro"
        seed: int = 42,
        buffer_refresh_every: int = 1, # Refresh buffer only every N outer iterations
    ):
        self.env  = env
        self.cfg  = cfg
        self.ippo = IPPOTrainer(env, cfg)

        # Ensure rollout length covers at least one full episode for accurate metrics
        if self.ippo.rollout_len < env.max_steps:
            self.ippo.rollout_len = env.max_steps

        self.param_env   = ParametrizedOvercooked.from_dims(*grid_size, seed=seed)
        
        self.buffer_capacity = buffer_size
        self.buffer: List[Level] = []

        self.n_random_pool       = n_random_pool
        self.rho                 = rho # 
        self.buffer_refresh_every = buffer_refresh_every
        self.rng                 = np.random.default_rng(seed)

        # Diagnostic counters
        self.iterations       = 0
        self.updates          = 0
        self.total_deliveries = 0

        # Env-parameter log: list of ([obs_left, obs_right, res_left, res_right], reset_id)
        self.env_param_log: List[Tuple[List[float], int]] = []
        self._reset_id: int = 0

        # JIT-compiled level evaluator — compiled once, reused for all N levels
        self._jit_eval_level = self._make_jit_eval()

    def _log_reset(self, level: "Level", branch: str, step: int,
                   writer=None) -> None:
        p = level.params
        vec = [p.obstacles_left, p.obstacles_right, p.resources_left, p.resources_right]
        rid = self._reset_id
        self.env_param_log.append((vec, rid, level.grid.copy(), level.gen_counter))
        self._reset_id += 1
        if writer is not None:
            writer.writerow({
                "reset_id":    rid,
                "step":        step,
                "branch":      branch,
                "gen_counter": level.gen_counter,
                "obs_left":    f"{vec[0]:.4f}",
                "obs_right":   f"{vec[1]:.4f}",
                "res_left":    f"{vec[2]:.4f}",
                "res_right":   f"{vec[3]:.4f}",
            })

    def _make_jit_eval(self):
        """Build a JIT-compiled closure for single-level learnability evaluation.
        Captures env/ippo refs at construction — compiled once across all N levels."""
        env  = self.env
        ippo = self.ippo

        def _eval(ts0, ts1, static_objects, agent_positions_xy, key):
            key, k_r = jax.random.split(key)
            reset_keys = jax.random.split(k_r, ippo.n_envs)
            obs_dict, env_states = jax.vmap(
                lambda k: env.reset(k, static_objects, agent_positions_xy)
            )(reset_keys)
            h0 = jnp.zeros((ippo.n_envs, ippo.hidden_size))
            h1 = jnp.zeros((ippo.n_envs, ippo.hidden_size))
            key, k_roll = jax.random.split(key)
            (_, _, _, _, _, _, _, shaped_comps, key) = ippo._collect_rollout(
                ts0, ts1, (obs_dict, env_states), (h0, h1), k_roll
            )
            # shaped_comps: (rollout_len, n_envs) — > 0 when an ingredient was delivered
            p = jnp.mean(jnp.any(shaped_comps > 0, axis=0).astype(jnp.float32))
            return p * (1.0 - p), key

        return jax.jit(_eval)

    def _grid_to_layout(self, grid: np.ndarray):
        H, W = grid.shape
        static_objects = np.vectorize(_TO_STATIC.get)(grid).astype(int)
        agent_positions = []
        for code in [_G.AGENT_0, _G.AGENT_1]:
            rows, cols = np.where(grid == code)
            if len(rows):
                agent_positions.append((int(cols[0]), int(rows[0])))
        return static_objects, agent_positions

    # def _set_env_layout(self, grid: np.ndarray):
    #     static_objects, agent_positions = self._grid_to_layout(grid)
    #     self.env.layout.static_objects = static_objects
    #     if agent_positions:
    #         self.env.layout.agent_positions = agent_positions
    
    def _set_env_layout(self, grids_batch: np.ndarray):
        """
        Updates the environment layout to handle a batch of grids (N_L, H, W).
        This ensures that each parallel environment in the vectorized rollout 
        gets its specific layout from the D_t mixture.
        """
        # If we only receive a single grid (H, W), add a batch dimension to avoid breaking old code
        if grids_batch.ndim == 2:
            grids_batch = np.expand_dims(grids_batch, axis=0)
            
        batch_static_objects = []
        batch_agent_positions = []
        
        # Process each grid in the batch to extract static objects and agent spawn points
        for grid in grids_batch:
            static_objects, agent_positions = self._grid_to_layout(grid)
            batch_static_objects.append(static_objects)
            
            if agent_positions:
                batch_agent_positions.append(agent_positions)

        # Stack into batched numpy/JAX arrays of shape (N_L, ...)
        # The underlying vectorized environment reset/step functions must be 
        # configured to vmap over this new leading dimension.
        self.env.layout.static_objects = np.stack(batch_static_objects)
        
        if batch_agent_positions:
            self.env.layout.agent_positions = np.stack(batch_agent_positions)
    
    def _prepare_layouts_batch(self, grids_batch: np.ndarray) -> Tuple[jnp.ndarray, Optional[jnp.ndarray]]:
        """
        Converts a batch of grids into stacked static_objects and agent_positions.
        Replaces the old mutating _set_env_layout function.
        """
        if grids_batch.ndim == 2:
            grids_batch = np.expand_dims(grids_batch, axis=0)
            
        batch_static_objects = []
        batch_agent_positions = []
        
        for grid in grids_batch:
            static_objects, agent_positions = self._grid_to_layout(grid)
            batch_static_objects.append(static_objects)
            if agent_positions:
                batch_agent_positions.append(agent_positions)

        static_out = jnp.array(batch_static_objects)
        agent_out = jnp.array(batch_agent_positions) if batch_agent_positions else None
        
        return static_out, agent_out
    
    def _fresh_reset(self, key: jnp.ndarray, batched_static_objects: jnp.ndarray, batched_agent_positions: Optional[jnp.ndarray] = None):
        """
        Resets the vectorized environment using dynamically passed batched layouts.
        """
        key, k_r = jax.random.split(key)
        reset_keys = jax.random.split(k_r, self.ippo.n_envs)
        
        # We must tell vmap to map over the 0-th axis of keys, static_objects, and agent_positions.
        # If agent_positions is None, adjust the in_axes accordingly.
        if batched_agent_positions is not None:
            # Assumes self.env.reset signature is: reset(key, static_objects, agent_positions)
            reset_vmap = jax.vmap(self.env.reset, in_axes=(0, 0, 0))
            obs_dict, env_states = reset_vmap(reset_keys, batched_static_objects, batched_agent_positions)
        else:
            # Assumes self.env.reset signature is: reset(key, static_objects)
            reset_vmap = jax.vmap(self.env.reset, in_axes=(0, 0))
            obs_dict, env_states = reset_vmap(reset_keys, batched_static_objects)

        h0 = jnp.zeros((self.ippo.n_envs, self.ippo.hidden_size))
        h1 = jnp.zeros((self.ippo.n_envs, self.ippo.hidden_size))
        
        return obs_dict, env_states, h0, h1, key

    def _bootstrap_values(self, ts0, ts1, obs_dict, h0, h1):
        def _v(ts, obs, h):
            return jax.vmap(lambda o, hi: ts.apply_fn(ts.params, o, hi))(obs, h)[1]
        return _v(ts0, obs_dict["agent_0"], h0), _v(ts1, obs_dict["agent_1"], h1)

    def _generate_level(self, key: jnp.ndarray) -> Tuple[Optional[Level], jnp.ndarray]:
        for _ in range(50):
            key, k_gen = jax.random.split(key)
            self.param_env._counter += 1
            ckey = jax.random.fold_in(jax.random.PRNGKey(self.param_env.seed), self.param_env._counter)
            grid, params = self.param_env._generate_from_key(ckey)
            if ParametrizedOvercooked.validate(grid):
                return Level(params=params, grid=grid, gen_counter=self.param_env._counter), key
        return None, key

    # ── JAX NATIVE ACCELERATED ALGORITHM 2 ─────────────────────────────────────

    def get_learnability_set(self, ts0, ts1, key: jnp.ndarray) -> Tuple[List[Level], jnp.ndarray]:
        """
        SFL Algorithm 2 with JAX batch generation and JIT-compiled evaluation.

        - N layouts generated in one vmap call (no Python loop)
        - env.reset receives static_objects as a dynamic JAX arg (no layout mutation)
        - _jit_eval_level compiled once, reused for all N levels without retracing
        """
        # 1. Generate N layout keys and batch-generate all layouts via vmap
        key, k_gen = jax.random.split(key)
        gen_keys = jax.random.split(k_gen, self.n_random_pool) # B <- N  random levels
        grids, params_vecs = jax.vmap(self.param_env._generate_from_key_jax)(gen_keys)
        # grids: (N, H, W) param-env codes; params_vecs: (N, 4) floats

        # 2. Convert param-env codes → StaticObject codes
        static_objects_batch = _STATIC_LOOKUP[grids]  # (N, H, W)

        # 3. Extract agent positions for each layout
        agent_pos_batch = jax.vmap(_extract_agent_pos_jax)(grids)  # (N, 2, 2)

        # 4. Evaluate each level — same JIT-compiled function, no retracing across levels
        scores = []
        # evaluate policy performance on each level and compute learnability score p*(1-p)
        for i in range(self.n_random_pool):
            score, key = self._jit_eval_level( #policy rollout
                ts0, ts1,
                static_objects_batch[i],
                agent_pos_batch[i],
                key,
            )
            scores.append(float(score))

        # 5. Build Level objects and rank by learnability score p*(1-p)
        grids_np       = np.asarray(grids)
        params_vecs_np = np.asarray(params_vecs)
        levels = []
        for i in range(self.n_random_pool):
            p_vec = params_vecs_np[i]
            params = EnvParams(
                obstacles_left=float(p_vec[0]),
                obstacles_right=float(p_vec[1]),
                resources_left=float(p_vec[2]),
                resources_right=float(p_vec[3]),
            )
            levels.append(Level(params=params, grid=grids_np[i], score=scores[i]))

        levels.sort(key=lambda l: l.score * (1.0 - l.score), reverse=True)
        return levels[:self.buffer_capacity], key # Top K levels with highest learnability scores

    # ── ALGORITHM 1: Sampling For Learnability (Main Loop) ─────────────────────

    def train(
        self,
        key: jnp.ndarray,
        resume_params: dict = None,
        log_callback: Optional[Callable] = None,
        checkpoint_dir=None,
        checkpoint_every: int = 100,
        T_steps: int = 10,
    ) -> Tuple:
        # Initialize student policy TrainStates
        key, k_init = jax.random.split(key)
        ts0, ts1 = self.ippo._make_train_state(k_init, resume_params)

        log: list = []
        all_ep_returns: list = []
        t0 = time.time()
        total_collected = 0
        
        N_L = self.ippo.n_envs # the number of parallel environments
        steps_per_rollout = self.ippo.rollout_len * N_L
        total_target = self.ippo.total_steps

        # IO File logging setup
        episode_log_file = None
        episode_log_writer = None
        env_param_log_file   = None
        env_param_log_writer = None
        if checkpoint_dir:
            ep_log_path = Path(checkpoint_dir) / "episodes.csv"
            ep_log_path.parent.mkdir(parents=True, exist_ok=True)
            episode_log_file = open(ep_log_path, "w", newline="")
            episode_log_writer = csv.DictWriter(
                episode_log_file,
                fieldnames=[
                    "episode", "deliveries", "return",
                    "level_score", "branch",
                    "buf_mean_score", "buf_max_score", "buf_min_score",
                ],
            )
            episode_log_writer.writeheader()
            episode_log_file.flush()

            ep_log_path = Path(checkpoint_dir) / "env_params.csv"
            env_param_log_file   = open(ep_log_path, "w", newline="")
            env_param_log_writer = csv.DictWriter(
                env_param_log_file,
                fieldnames=["reset_id", "step", "branch", "gen_counter", "obs_left", "obs_right", "res_left", "res_right"],
            )
            env_param_log_writer.writeheader()
            env_param_log_file.flush()

        print(f"[SFL] Main training loop activated (Target: {total_target:,} steps)\n")

        while total_collected < total_target:
            self.iterations += 1

            # --- Step 1: Update the level buffer D using Accelerated Algorithm 2 ---
            # --- D <- collect_learnable_levels(pi) Using Alg. 2 ---
            t_eval_start = time.time()
            self.buffer, key = self.get_learnability_set(ts0, ts1, key)
            t_eval_elapsed = time.time() - t_eval_start
            print(f"[SFL] Buffer D updated with {len(self.buffer)} levels in {t_eval_elapsed:.1f}s")

            # Record diagnostic metrics for logging pipelines
            buf_scores = [l.score for l in self.buffer]
            log_buf_mean = float(np.mean(buf_scores)) if buf_scores else 0.0
            log_buf_max  = float(np.max(buf_scores))  if buf_scores else 0.0
            log_buf_min  = float(np.min(buf_scores))  if buf_scores else 0.0

            # --- Step 2: Inner training loop execution (for t = 1, ..., T) ---
            # --- Inner loop: for t = 1, ..., T do ---
            for t in range(T_steps):
                if total_collected >= total_target:
                    break

                n_buffer_levels = int(self.rho * N_L) # 1. D_t <- \rho * N_L levels sampled uniformly from D
                n_random_levels = N_L - n_buffer_levels # 2. D_t <- D_t U (1 - \rho) * N_L randomly generated levels
                
                batch_grids  = []
                batch_params = []

                # Sample from Buffer D
                if len(self.buffer) > 0 and n_buffer_levels > 0:
                    # Uniformly sample with replacement (in case buffer < n_buffer_levels)
                    sampled_indices = np.random.choice(len(self.buffer), size=n_buffer_levels, replace=True)
                    for i in sampled_indices:
                        batch_grids.append(self.buffer[i].grid)
                        batch_params.append(self.buffer[i])  # full Level for logging
                else:
                    # Fallback if buffer is empty on very first iterations
                    n_random_levels = N_L

                # Generate Random levels
                for _ in range(n_random_levels):
                    level, key = self._generate_level(key)
                    if level is not None:
                        batch_grids.append(level.grid)
                        batch_params.append(level)  # full Level for logging
                    else:
                        # Fallback for generation failure (append a dummy/zero grid)
                        fallback = self.buffer[0] if self.buffer else batch_params[0]
                        batch_grids.append(fallback.grid)
                        batch_params.append(fallback)

                # Stack into a batched array of shape (N_L, H, W)
                D_t_grids = np.stack(batch_grids)

                # Load the mixed batch of D_t layouts into the vectorized simulator
                # NOTE: _set_env_layout must be updated to accept a batch of grids (N_L, H, W)
                self._set_env_layout(D_t_grids)
                
                batched_static_objs, batched_agent_pos = self._prepare_layouts_batch(D_t_grids)

                # --- Collect pi's trajectory on D_t and update phi ---
                obs_dict, env_states, h0, h1, key = self._fresh_reset(key, batched_static_objs, batched_agent_pos)
                for lv in batch_params:
                    self._log_reset(lv, "train", total_collected, env_param_log_writer)
                key, k_r = jax.random.split(key)
                
                (trs0, trs1, (next_obs, _), (nh0, nh1),
                 ep_returns_arr, _, ep_dones_arr, shaped_comps, key) = self.ippo._collect_rollout(
                    ts0, ts1, (obs_dict, env_states), (h0, h1), k_r
                )

                # Calculate critic values and GAE advantages
                last_v0, last_v1 = self._bootstrap_values(ts0, ts1, next_obs, nh0, nh1)
                adv0, ret0 = calculate_gae(trs0, last_v0, self.ippo.gamma, self.ippo.lam)
                adv1, ret1 = calculate_gae(trs1, last_v1, self.ippo.gamma, self.ippo.lam)

                # Perform optimization and policy parameter updates
                key, ks0, ks1 = jax.random.split(key, 3)
                ts0, loss0 = self.ippo._update_agent(ts0, trs0, adv0, ret0, seed=int(np.asarray(ks0)[0]))
                ts1, loss1 = self.ippo._update_agent(ts1, trs1, adv1, ret1, seed=int(np.asarray(ks1)[0]))
                log_loss = (float(loss0) + float(loss1)) / 2.0

                total_collected += steps_per_rollout
                self.updates += 1

                # Parse logging and tracking statistics
                ep_returns_np = np.asarray(ep_returns_arr)
                ep_dones_np   = np.asarray(ep_dones_arr)
                step_r_np     = np.asarray(trs0.reward)

                # Compute task deliveries
                if self.ippo.reward_mode == "delivery":
                    deliv_per_step = (step_r_np >= 1.0).astype(np.int32)
                else:
                    deliv_per_step = (step_r_np >= DELIVERY_REWARD).astype(np.int32)

                self.total_deliveries += int(np.sum(deliv_per_step))

                ep_deliv_buf = np.zeros(N_L, dtype=np.int32)
                for step_idx in range(self.ippo.rollout_len):
                    ep_deliv_buf += deliv_per_step[step_idx]
                    for env_i in np.where(ep_dones_np[step_idx])[0]:
                        ep_num = len(all_ep_returns)
                        dc     = int(ep_deliv_buf[env_i])
                        ep_ret = float(ep_returns_np[step_idx, env_i])
                        all_ep_returns.append(ep_ret)

                        if episode_log_writer:
                            episode_log_writer.writerow({
                                "episode":        ep_num,
                                "deliveries":     dc,
                                "return":         f"{ep_ret:.2f}",
                                "buf_mean_score": f"{log_buf_mean:.4f}",
                                "buf_max_score":  f"{log_buf_max:.4f}",
                                "buf_min_score":  f"{log_buf_min:.4f}",
                            })
                            episode_log_file.flush()
                        ep_deliv_buf[env_i] = 0

                # --- Metrics evaluation and logging ---
                if self.updates % self.ippo.log_every == 0:
                    mean_r = float(np.mean(all_ep_returns[-100:])) if all_ep_returns else 0.0
                    elapsed = time.time() - t0
                    sps = total_collected / max(elapsed, 1e-9)

                    entry = {
                        "steps":             total_collected,
                        "iteration":         self.iterations,
                        "update":            self.updates,
                        "mean_r":            mean_r,
                        "loss":              log_loss,
                        "buffer_size":       len(self.buffer),
                        "buffer_mean_score": log_buf_mean,
                        "buffer_max_score":  log_buf_max,
                        "buffer_min_score":  log_buf_min,
                        "total_deliveries":  self.total_deliveries,
                    }
                    log.append(entry)
                    if log_callback:
                        log_callback(log)
                        
                    print(
                        f"[SFL] iter={self.iterations:4d} | "
                        f"upd={self.updates:5d} | "
                        f"steps={total_collected:9,} | "
                        f"mean_r={mean_r:6.2f} | "
                        f"loss={log_loss:.4f} | "
                        f"buf_mean={log_buf_mean:.4f} | "
                        f"buf_max={log_buf_max:.4f} | "
                        f"deliv={self.total_deliveries} | "
                        f"sps={sps:.0f}| " , 
                        flush=True
                    )

                # --- Serialize periodic checkpoints ---
                # (Same logic as before)
                if checkpoint_dir and checkpoint_every > 0 and self.updates % checkpoint_every == 0:
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
                            "buffer_size":       len(self.buffer),
                            "buffer_mean_score": log_buf_mean,
                        }, f, indent=2)

        # --- Serialize final trained model parameters ---
        if checkpoint_dir:
            final = Path(checkpoint_dir) / "checkpoints" / "final"
            final.mkdir(parents=True, exist_ok=True)
            with open(final / "params_agent0.pkl", "wb") as f:
                pickle.dump(ts0.params, f)
            with open(final / "params_agent1.pkl", "wb") as f:
                pickle.dump(ts1.params, f)

        if episode_log_file:
            episode_log_file.close()
        if env_param_log_file:
            env_param_log_file.close()
        if checkpoint_dir:
            with open(Path(checkpoint_dir) / "env_params.pkl", "wb") as f:
                pickle.dump(self.env_param_log, f)

        print(f"\n[SFL] Training pipeline execution completed. Steps reached: {total_collected:,}")
        return ts0, ts1, log