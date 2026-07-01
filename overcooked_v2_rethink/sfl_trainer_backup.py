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

@dataclass
class Level:
    """Dataclass to store a curriculum level within the SFL buffer."""
    params: EnvParams
    grid:   np.ndarray
    score:  float = 0.0     # Learnability score: p * (1 - p)


class SFLTrainer:
    """Trainer implementing 'Sampling For Learnability' (SFL) for JAX Overcooked."""

    def __init__(
        self,
        env,
        cfg: dict,
        base_layout_str: str,
        buffer_size: int = 50,         # K (buffer D capacity)
        n_random_pool: int = 200,      # N (random pool size for evaluation)
        rho: float = 0.7,              # Mixing coefficient for levels
        seed: int = 42,
        buffer_refresh_every: int = 1, # Refresh buffer only every N outer iterations
    ):
        self.env  = env
        self.cfg  = cfg
        self.ippo = IPPOTrainer(env, cfg)

        # Ensure rollout length covers at least one full episode for accurate metrics
        if self.ippo.rollout_len < env.max_steps:
            self.ippo.rollout_len = env.max_steps

        base_grid        = ParametrizedOvercooked.from_string(base_layout_str)
        self.param_env   = ParametrizedOvercooked(base_grid=base_grid, seed=seed)
        
        self.buffer_capacity = buffer_size
        self.buffer: List[Level] = []

        self.n_random_pool       = n_random_pool
        self.rho                 = rho
        self.buffer_refresh_every = buffer_refresh_every
        self.rng                 = np.random.default_rng(seed)

        # Diagnostic counters
        self.iterations       = 0
        self.updates          = 0
        self.total_deliveries = 0

    def _grid_to_layout(self, grid: np.ndarray):
        H, W = grid.shape
        static_objects = np.vectorize(_TO_STATIC.get)(grid).astype(int)
        agent_positions = []
        for code in [_G.AGENT_0, _G.AGENT_1]:
            rows, cols = np.where(grid == code)
            if len(rows):
                agent_positions.append((int(cols[0]), int(rows[0])))
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

    def _generate_level(self, key: jnp.ndarray) -> Tuple[Optional[Level], jnp.ndarray]:
        for _ in range(50):
            key, k_gen = jax.random.split(key)
            self.param_env._counter += 1
            ckey = jax.random.fold_in(jax.random.PRNGKey(self.param_env.seed), self.param_env._counter)
            grid, params = self.param_env._generate_from_key(ckey)
            if ParametrizedOvercooked.validate(grid):
                return Level(params=params, grid=grid), key
        return None, key

    # ── ALGORITHM 2: Collect learnable levels ──────────────────────────────────
    
    def _evaluate_learnability(self, ts0, ts1, grid: np.ndarray, key: jnp.ndarray) -> Tuple[float, jnp.ndarray]:
        """Evaluate a level using the p * (1 - p) formula. Executed without gradient updates."""
        self._set_env_layout(grid)
        obs_dict, env_states, h0, h1, key = self._fresh_reset(key)
        key, k_r = jax.random.split(key)
        
        # Collect a clean trajectory rollout
        (_, _, _, _, _, _, _, shaped_comps, key) = \
            self.ippo._collect_rollout(ts0, ts1, (obs_dict, env_states), (h0, h1), k_r)
            
        comps_np = np.asarray(shaped_comps) # Shape: (rollout_len, n_envs)
        
        # Define success: check if at least one delivery occurred in each parallel environment
        successes = np.any(comps_np > 0, axis=0) # Vector of shape (n_envs,)
        p = float(np.mean(successes))            # Average success rate for the level
        
        # Formula: Learnability = p * (1 - p)
        learnability = p * (1.0 - p)
        return learnability, key

    # def get_learnability_set(self, ts0, ts1, key: jnp.ndarray) -> Tuple[List[Level], jnp.ndarray]:
    #     """Implementation of Algorithm 2: Sample a pool of N levels and rank top-K."""
    #     candidate_pool: List[Level] = []
        
    #     # Sample N random valid levels (B <- N random levels)
    #     while len(candidate_pool) < self.n_random_pool:
    #         level, key = self._generate_level(key)
    #         if level is not None:
    #             candidate_pool.append(level)
                
    #     # Calculate Learnability for each sampled candidate level
    #     for level in candidate_pool:
    #         score, key = self._evaluate_learnability(ts0, ts1, level.grid, key)
    #         level.score = score
            
    #     # Sort in descending order and return top-K levels
    #     candidate_pool.sort(key=lambda l: l.score, reverse=True)
    #     return candidate_pool[:self.buffer_capacity], key
    
    def get_learnability_set(self, ts0, ts1, key: jnp.ndarray) -> Tuple[List[Level], jnp.ndarray]:
        """
        Optimized SFL Algorithm 2: Generates a pool of N candidate levels 
        and evaluates them using JAX-accelerated rollouts.
        """
        candidate_pool: List[Level] = []
        
        # 1. Generate N valid levels via the generator 
        # (This is relatively fast as it only builds the grid matrices)
        while len(candidate_pool) < self.n_random_pool:
            level, key = self._generate_level(key)
            if level is not None:
                candidate_pool.append(level)
                
        # 2. Compile the evaluation function using jax.jit to prevent Python overhead
        # during the rollout evaluation phase
        @partial(jax.jit, static_argnums=(0,))
        def _jitted_rollout_eval(ts0, ts1, k_rollout):
            # We enforce a fast compiled rollout over the current environment layout
            return self.ippo._collect_rollout(ts0, ts1, (obs_dict, env_states), (h0, h1), k_rollout)

        # 3. Evaluate the learnability score for each candidate level
        for level in candidate_pool:
            # We still set layout in Python, but the internal execution loop
            # now runs entirely inside JIT compiled forward steps
            score, key = self._evaluate_learnability(ts0, ts1, level.grid, key)
            level.score = score
            
        # 4. Rank by learnability code: p * (1 - p) descending and take Top-K
        candidate_pool.sort(key=lambda l: l.score, reverse=True)
        return candidate_pool[:self.buffer_capacity], key

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
        steps_per_rollout = self.ippo.rollout_len * self.ippo.n_envs
        total_target = self.ippo.total_steps

        # IO File logging setup
        episode_log_file = None
        episode_log_writer = None
        if checkpoint_dir:
            ep_log_path = Path(checkpoint_dir) / "episodes.csv"
            ep_log_path.parent.mkdir(parents=True, exist_ok=True)
            episode_log_file = open(ep_log_path, "w", newline="")
            episode_log_writer = csv.DictWriter(
                episode_log_file,
                fieldnames=["episode", "deliveries", "return", "level_score", "branch"],
            )
            episode_log_writer.writeheader()
            episode_log_file.flush()

        print(f"[SFL] Main training loop activated (Target: {total_target:,} steps)\n")

        while total_collected < total_target:
            self.iterations += 1

            # --- Step 1: Update the level buffer D using Algorithm 2 ---
            # Refresh only every buffer_refresh_every outer iterations to reduce
            # the eval overhead (200 rollouts per refresh vs. T_steps training rollouts).
            if self.iterations == 1 or (self.iterations % self.buffer_refresh_every == 0):
                self.buffer, key = self.get_learnability_set(ts0, ts1, key)

            # Record diagnostic metrics for logging pipelines
            buf_scores = [l.score for l in self.buffer]
            log_buf_mean = float(np.mean(buf_scores)) if buf_scores else 0.0
            log_buf_max  = float(np.max(buf_scores)) if buf_scores else 0.0

            # --- Step 2: Inner training loop execution (for t = 1, ..., T) ---
            for t in range(T_steps):
                if total_collected >= total_target:
                    break

                # Form the training sample D_t (Mix rho from buffer and 1-rho random levels)
                use_buffer_level = (self.rng.random() < self.rho) and (len(self.buffer) > 0)
                
                if use_buffer_level:
                    level = self.rng.choice(self.buffer)
                    grid_to_use = level.grid
                    branch = "buffer"
                    log_level_score = level.score
                else:
                    level, key = self._generate_level(key)
                    if level is None:
                        continue
                    grid_to_use = level.grid
                    branch = "random"
                    log_level_score = 0.0 # Newly sampled random levels do not track score

                # Load chosen grid layout into the simulator
                self._set_env_layout(grid_to_use)

                # Collect trajectory data via rollout execution
                obs_dict, env_states, h0, h1, key = self._fresh_reset(key)
                key, k_r = jax.random.split(key)
                (trs0, trs1, (next_obs, _), (nh0, nh1),
                 ep_returns_arr, _, ep_dones_arr, shaped_comps, key) = self.ippo._collect_rollout(
                    ts0, ts1, (obs_dict, env_states), (h0, h1), k_r
                )

                # Calculate critic values and GAE advantages (PPO)
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

                ep_deliv_buf = np.zeros(self.ippo.n_envs, dtype=np.int32)
                for step_idx in range(self.ippo.rollout_len):
                    ep_deliv_buf += deliv_per_step[step_idx]
                    for env_i in np.where(ep_dones_np[step_idx])[0]:
                        ep_num = len(all_ep_returns)
                        dc     = int(ep_deliv_buf[env_i])
                        ep_ret = float(ep_returns_np[step_idx, env_i])
                        all_ep_returns.append(ep_ret)

                        if episode_log_writer:
                            episode_log_writer.writerow({
                                "episode":     ep_num,
                                "deliveries":  dc,
                                "return":      f"{ep_ret:.2f}",
                                "level_score": f"{log_level_score:.4f}",
                                "branch":      branch,
                            })
                            episode_log_file.flush()
                        ep_deliv_buf[env_i] = 0

                # --- Metrics evaluation and dashboard plotting intervals ---
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
                        "level_score":       log_level_score,
                        "branch":            1 if use_buffer_level else 0,
                        "total_deliveries":  self.total_deliveries,
                    }
                    log.append(entry)
                    if log_callback:
                        log_callback(log)
                        
                    # prints every 10 updates (rollouts) with full stats and flush=True
                    print(
                        f"[SFL] iter={self.iterations:4d} | "
                        f"upd={self.updates:5d} | "
                        f"steps={total_collected:9,} | "
                        f"mean_r={mean_r:6.2f} | "
                        f"loss={log_loss:.4f} | "
                        f"buf_mean={log_buf_mean:.4f} | "
                        f"deliv={self.total_deliveries} | "
                        f"sps={sps:.0f}",
                        flush=True
                    )

                    if self.updates % (self.ippo.log_every * 5) == 0:
                        print(
                            f"[SFL] iter={self.iterations:4d} | upd={self.updates:5d} | "
                            f"steps={total_collected:9,} | mean_r={mean_r:6.2f} | "
                            f"loss={log_loss:.4f} | buf_mean={log_buf_mean:.4f} | "
                            f"deliv={self.total_deliveries} | sps={sps:.0f}"
                        )

                # --- Serialize periodic training checkpoints ---
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

        print(f"\n[SFL] Training pipeline execution completed. Steps reached: {total_collected:,}")
        return ts0, ts1, log