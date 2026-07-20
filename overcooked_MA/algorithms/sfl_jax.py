"""
sfl.py — Sampling For Learnability trainer.

Uses the unified UED interface from ued.py.
"""

import csv
import json
import pickle
import time
from pathlib import Path
from typing import Callable, List, Optional, Tuple

import jax
import jax.numpy as jnp
import numpy as np

from algorithms.ippo_jax import IPPOTrainer, calculate_gae
from environment.settings import DELIVERY_REWARD

# UED components
from ued import (
    OvercookedLevel,
    LevelBuffer,
    LevelGenerator,
    LevelScorer,
    GridCodes,
    get_static_lookup,
    grid_to_static_jax,
    grid_batch_to_static_jax,
    extract_agent_positions_jax,
    grid_to_layout_numpy,
)

# Parametrized env (for JAX-native vmap generation)
from environment.level_generator import EnvParams, ParametrizedOvercooked, is_valid_layout


# ── Visualization helpers ─────────────────────────────────────────────────────

def _save_layout_snapshot(grid: np.ndarray, step: int, out_dir: Path) -> None:
    from environment.level_generator import render_env as _render_env
    snap_dir = out_dir / "layout_snapshots"
    snap_dir.mkdir(exist_ok=True)
    img = _render_env(grid, tile_size=48)
    img.save(snap_dir / f"step_{step:09d}.png")


def _save_agent_gif(grid: np.ndarray, network, params_0, params_1,
                    step: int, out_dir: Path, max_steps: int = 200) -> None:
    import imageio
    from environment.layouts import Layout
    from environment.overcooked_env import OvercookedV2
    from environment.common import StaticObject

    H, W = grid.shape
    static_objects, agent_positions = grid_to_layout_numpy(grid)

    if len(agent_positions) < 2:
        for r in range(1, H - 1):
            for c in range(1, W - 1):
                if static_objects[r, c] == int(StaticObject.EMPTY) and (c, r) not in agent_positions:
                    agent_positions.append((c, r))
                if len(agent_positions) == 2:
                    break
            if len(agent_positions) == 2:
                break

    layout = Layout(agent_positions=agent_positions, static_objects=static_objects,
                    num_ingredients=2, possible_recipes=[[0, 0, 0], [1, 1, 1]])
    env = OvercookedV2(layout=layout, max_steps=max_steps)

    key = jax.random.PRNGKey(step % 100000)
    key, k_reset = jax.random.split(key)
    obs, state = env.reset(k_reset)
    states = [state]
    deliveries = 0

    @jax.jit
    def _act(params, single_obs):
        logits, _ = network.apply(params, single_obs[None])
        return jnp.argmax(logits[0])

    for _ in range(max_steps):
        key, k_step = jax.random.split(key)
        a0 = _act(params_0, obs["agent_0"])
        a1 = _act(params_1, obs["agent_1"])
        obs, state, _, dones, _ = env.step(k_step, state, {"agent_0": a0, "agent_1": a1})
        deliveries += int(state.new_correct_delivery)
        states.append(state)
        if dones["__all__"]:
            break

    try:
        from environment.viz.overcooked_v2_visualizer import OvercookedV2Visualizer
        viz = OvercookedV2Visualizer(tile_size=64)
        state_seq = jax.tree_util.tree_map(lambda *xs: jnp.stack(xs), *states)
        frame_seq = np.array(viz.render_sequence(state_seq), dtype=np.uint8)

        gif_dir = out_dir / "agent_gifs"
        gif_dir.mkdir(exist_ok=True)
        imageio.mimsave(str(gif_dir / f"step_{step:09d}.gif"), frame_seq,
                        format="GIF", duration=150, loop=0)
        print(f"[SFL JAX] GIF saved: step={step:,}  deliveries={deliveries}", flush=True)
    except ImportError:
        print(f"[SFL JAX] GIF skipped (visualizer not available): step={step:,}", flush=True)


# ── SFL Trainer ───────────────────────────────────────────────────────────────

class SFLTrainer:
    """
    Sampling For Learnability trainer.

    Outer loop:  generate N random levels → evaluate → keep top-K in buffer
    Inner loop:  sample ρ from buffer + (1-ρ) fresh → rollout → PPO update
    """

    def __init__(
        self,
        env,
        cfg: dict,
        grid_size: Tuple[int, int],
        buffer_size: int = 50,
        n_random_pool: int = 200,
        rho: float = 0.5,
        seed: int = 42,
        buffer_refresh_every: int = 1,
    ):
        self.env = env
        self.cfg = cfg
        self.ippo = IPPOTrainer(env, cfg)

        if self.ippo.rollout_len < env.max_steps:
            self.ippo.rollout_len = env.max_steps

        # UED components
        self.generator = LevelGenerator(
            H=grid_size[0], W=grid_size[1],
            max_goals=3, max_pots=3, max_plates=2,
            ingredient_types=(GridCodes.INGREDIENT_0, GridCodes.INGREDIENT_1),
        )
        self.buffer = LevelBuffer(capacity=buffer_size)
        self.scorer = LevelScorer()

        # JAX-native vmap generation
        self.param_env = ParametrizedOvercooked.from_dims(*grid_size, seed=seed)

        self.n_random_pool = n_random_pool
        self.rho = rho
        self.buffer_refresh_every = buffer_refresh_every
        self.rng = np.random.default_rng(seed)

        self.iterations = 0
        self.updates = 0
        self.total_deliveries = 0
        self.env_param_log: List[Tuple] = []
        self._reset_id: int = 0

        # Record of every candidate scored during buffer refresh (the full
        # pool, not just the top-K that made it into the buffer), matching
        # simple_sfl.py's pool_scores.csv logging.
        self.pool_log: List[dict] = []
        self._refresh_id: int = 0

        self._jit_eval_level = self._make_jit_eval()

    # ── Logging ───────────────────────────────────────────────────────────

    def _log_reset(self, level: OvercookedLevel, branch: str, step: int,
                   writer=None) -> None:
        p = level.params
        if p is None:
            vec = [0.0, 0.0, 0.0, 0.0]
        elif hasattr(p, 'obs_density'):
            vec = [p.obs_density, p.obs_skew_x, p.res_density, p.res_skew_x]
        else:
            vec = [0.0, 0.0, 0.0, 0.0]

        rid = self._reset_id
        self.env_param_log.append((vec, rid, level.grid.copy(), level.seed))
        self._reset_id += 1

        if writer is not None:
            writer.writerow({
                "reset_id": rid, "step": step, "branch": branch,
                "gen_counter": level.seed,
                "obs_density": f"{vec[0]:.4f}", "obs_skew_x": f"{vec[1]:.4f}",
                "res_density": f"{vec[2]:.4f}", "res_skew_x": f"{vec[3]:.4f}",
            })

    # ── JIT-compiled level evaluator ──────────────────────────────────────

    def _make_jit_eval(self):
        env = self.env
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
            p = jnp.mean(jnp.any(shaped_comps > 0, axis=0).astype(jnp.float32))
            return p * (1.0 - p), key

        return jax.jit(_eval)

    # ── Batch reset ───────────────────────────────────────────────────────

    def _fresh_reset(self, key, batched_static, batched_agents):
        key, k_r = jax.random.split(key)
        reset_keys = jax.random.split(k_r, self.ippo.n_envs)

        if batched_agents is not None:
            obs_dict, env_states = jax.vmap(
                self.env.reset, in_axes=(0, 0, 0)
            )(reset_keys, batched_static, batched_agents)
        else:
            obs_dict, env_states = jax.vmap(
                self.env.reset, in_axes=(0, 0)
            )(reset_keys, batched_static)

        h0 = jnp.zeros((self.ippo.n_envs, self.ippo.hidden_size))
        h1 = jnp.zeros((self.ippo.n_envs, self.ippo.hidden_size))
        return obs_dict, env_states, h0, h1, key

    def _bootstrap_values(self, ts0, ts1, obs_dict, h0, h1):
        def _v(ts, obs, h):
            return jax.vmap(lambda o, hi: ts.apply_fn(ts.params, o, hi))(obs, h)[1]
        return _v(ts0, obs_dict["agent_0"], h0), _v(ts1, obs_dict["agent_1"], h1)

    # ── Level generation ──────────────────────────────────────────────────

    def _generate_single_level(self, key):
        key, k_gen = jax.random.split(key)
        level = self.generator.generate(k_gen, validate=True)
        return level, key

    # ── Validated pool generation ─────────────────────────────────────────

    def _generate_valid_pool(self, key, n, max_batches=30):
        """Generate exactly n VALID levels for the learnability pool.

        The JAX vmap generator (_generate_from_key_jax) does not validate — it
        can produce unsolvable layouts (blocked paths, unreachable objects).
        is_valid_layout is a NumPy function (BFS) that can't run inside vmap, so
        validity has to be applied *after* generation: generate a full batch via
        vmap, filter by is_valid_layout, and regenerate fresh batches until n
        valid grids are collected. This makes is_valid_layout actually control
        pool generation, matching how the fresh-level path already validates.

        Candidates are filtered by is_valid_layout (validity) AND by a grid hash
        (uniqueness — no duplicate layouts in the pool). If the grid/density is so
        constrained that n unique valid levels can't be found — either the valid
        space is exhausted (several batches add nothing new) or max_batches is hit —
        the pool ADAPTS: it returns however many unique valid levels exist and warns
        loudly, rather than padding with duplicates (which would defeat uniqueness)
        or silently inflating a tiny set into a fake n.

        Returns (list_of_unique_valid_grids, list_of_param_vectors, key)."""
        grids_out, params_out = [], []
        seen = set()                # grid.tobytes() of every level already collected
        n_generated = 0             # total candidates drawn (to report the yield)
        attempts = 0
        stalled = 0                 # consecutive batches that added zero new unique levels
        while len(grids_out) < n and attempts < max_batches:
            key, k_gen = jax.random.split(key)
            gen_keys = jax.random.split(k_gen, n)  # fixed batch size n → no vmap retracing
            grids, params_vecs = jax.vmap(self.param_env._generate_from_key_jax)(gen_keys)
            grids_np, params_np = np.asarray(grids), np.asarray(params_vecs)
            n_generated += len(grids_np)
            added = 0
            for g, p in zip(grids_np, params_np):
                if len(grids_out) >= n:
                    break
                h = g.tobytes()
                if h in seen:               # duplicate layout — skip
                    continue
                if not is_valid_layout(g):  # unsolvable layout — skip
                    continue
                seen.add(h)
                grids_out.append(g)
                params_out.append(p)
                added += 1
            attempts += 1
            # If a few whole batches in a row yield no NEW unique valid level, the
            # valid+unique space is effectively exhausted for this grid/density —
            # stop rather than spinning through the full max_batches budget.
            stalled = stalled + 1 if added == 0 else 0
            if stalled >= 3:
                break

        n_valid = len(grids_out)
        if n_valid < n:
            print(f"[SFL JAX] WARNING: only {n_valid} UNIQUE valid levels found "
                  f"(requested {n}) from {n_generated} generated over {attempts} batch(es). "
                  f"The grid/density is likely too constrained — using an adaptive pool of "
                  f"{n_valid}. Lower obs_density/res_density or enlarge grid_size for a bigger pool.",
                  flush=True)
        else:
            print(f"[SFL JAX] valid pool: {n} unique valid levels from {n_generated} generated "
                  f"({100 * n / max(n_generated, 1):.0f}% yield) over {attempts} batch(es)", flush=True)
        return grids_out, params_out, key

    # ── Algorithm 2: get_learnability_set ─────────────────────────────────

    def get_learnability_set(self, ts0, ts1, key, writer=None):
        print(f"[SFL JAX] Creating learnability set. Generating {self.n_random_pool} valid random levels for evaluation...")
        pool_grids, pool_params, key = self._generate_valid_pool(key, self.n_random_pool)

        # The pool adapts to how many unique valid levels actually exist, so use
        # its real length everywhere below rather than the requested n_random_pool.
        n_pool = len(pool_grids)
        if n_pool == 0:
            raise RuntimeError(
                "No valid layout could be generated for this grid_size/density — "
                "the configuration is unsolvable. Lower obs_density/res_density or "
                "enlarge grid_size."
            )

        grids = jnp.stack([jnp.asarray(g, dtype=jnp.int32) for g in pool_grids])
        static_objects_batch = grid_batch_to_static_jax(grids)
        agent_pos_batch = jax.vmap(extract_agent_positions_jax)(grids)

        scores = []
        for i in range(n_pool):
            score, key = self._jit_eval_level(
                ts0, ts1, static_objects_batch[i], agent_pos_batch[i], key,
            )
            scores.append(float(score))

        scored_levels = []
        for i in range(n_pool):
            p_vec = pool_params[i]
            params = EnvParams(
                obs_density=float(p_vec[0]), obs_skew_x=float(p_vec[1]),
                res_density=float(p_vec[2]), res_skew_x=float(p_vec[3]),
            )
            level = OvercookedLevel(
                grid=pool_grids[i], params=params, seed=i, score=scores[i],
            )
            scored_levels.append(level)

        scored_levels.sort(key=lambda lv: lv.score, reverse=True)

        # Record every scored candidate — not just the top-K that survives into
        # the buffer — so the full learnability landscape is inspectable.
        cap = self.buffer.capacity
        for rank, lv in enumerate(scored_levels):
            p = lv.params
            entry = {
                "refresh_id": self._refresh_id,
                "rank": rank,
                "selected": rank < cap,
                "obs_density": float(p.obs_density), "obs_skew_x": float(p.obs_skew_x),
                "res_density": float(p.res_density), "res_skew_x": float(p.res_skew_x),
                "score": lv.score,
            }
            # pool_log (and its pickle) additionally carries the actual grid,
            # so candidates can be re-rendered exactly as scored.
            self.pool_log.append({**entry, "grid": lv.grid.copy()})
            if writer:
                writer.writerow(entry)
        self._refresh_id += 1

        new_buffer = LevelBuffer(capacity=cap)
        for lv in scored_levels[:cap]:
            new_buffer.insert(lv)
        return new_buffer, key

    # ── Prepare batch layouts ─────────────────────────────────────────────

    def _prepare_training_batch(self, batch_levels):
        grids = jnp.stack([jnp.array(lv.grid) for lv in batch_levels])
        static_batch = grid_batch_to_static_jax(grids)
        agents_batch = jax.vmap(extract_agent_positions_jax)(grids)
        return static_batch, agents_batch

    # ── Main training loop ────────────────────────────────────────────────

    def train(
        self,
        key: jnp.ndarray,
        resume_params: dict = None,
        log_callback: Optional[Callable] = None,
        checkpoint_dir=None,
        checkpoint_every: int = 100,
        T_steps: int = 10,
        render_every: int = 0,
    ) -> Tuple:
        key, k_init = jax.random.split(key)
        ts0, ts1 = self.ippo._make_train_state(k_init, resume_params)

        log, all_ep_returns = [], []
        t0 = time.time()
        total_collected = 0

        N_L = self.ippo.n_envs
        steps_per_rollout = self.ippo.rollout_len * N_L
        total_target = self.ippo.total_steps

        # IO logging
        episode_log_file = episode_log_writer = None
        env_param_log_file = env_param_log_writer = None
        pool_log_file = pool_log_writer = None

        if checkpoint_dir:
            ep_path = Path(checkpoint_dir) / "episodes.csv"
            ep_path.parent.mkdir(parents=True, exist_ok=True)
            episode_log_file = open(ep_path, "w", newline="")
            episode_log_writer = csv.DictWriter(episode_log_file, fieldnames=[
                "episode", "deliveries", "return",
                "buf_mean_score", "buf_max_score", "buf_min_score",
            ])
            episode_log_writer.writeheader()
            episode_log_file.flush()

            param_path = Path(checkpoint_dir) / "env_params.csv"
            env_param_log_file = open(param_path, "w", newline="")
            env_param_log_writer = csv.DictWriter(env_param_log_file, fieldnames=[
                "reset_id", "step", "branch", "gen_counter",
                "obs_density", "obs_skew_x", "res_density", "res_skew_x",
            ])
            env_param_log_writer.writeheader()
            env_param_log_file.flush()

            pool_path = Path(checkpoint_dir) / "pool_scores.csv"
            pool_log_file = open(pool_path, "w", newline="")
            pool_log_writer = csv.DictWriter(pool_log_file, fieldnames=[
                "refresh_id", "rank", "selected",
                "obs_density", "obs_skew_x", "res_density", "res_skew_x", "score",
            ])
            pool_log_writer.writeheader()
            pool_log_file.flush()

        # Plain-text workflow trace: every _log() call below goes to this file
        # only (not the console), so the run can be inspected without flooding
        # the terminal.
        workflow_log_file = None
        if checkpoint_dir:
            workflow_log_path = Path(checkpoint_dir) / "training_log.txt"
            workflow_log_path.parent.mkdir(parents=True, exist_ok=True)
            workflow_log_file = open(workflow_log_path, "a")

        def _log(msg: str) -> None:
            if workflow_log_file:
                workflow_log_file.write(msg + "\n")
                workflow_log_file.flush()

        _log(f"[SFL JAX] Training started (target: {total_target:,} steps)\n")

        while total_collected < total_target:
            self.iterations += 1
            _log(f"\n[SFL JAX] ── iteration {self.iterations} " + "─" * 40)

            # Step 1: Update buffer
            _log(f"[SFL JAX] [1/5] refreshing level buffer ({self.n_random_pool} candidates)...")
            t_eval = time.time()
            self.buffer, key = self.get_learnability_set(ts0, ts1, key, writer=pool_log_writer)
            if pool_log_file:
                pool_log_file.flush()
            _log(f"[SFL JAX] [1/5] buffer updated: {self.buffer.size} levels in {time.time()-t_eval:.1f}s")

            buf_stats = self.buffer.stats()
            log_buf_mean = buf_stats["buffer_mean_score"]
            log_buf_max = buf_stats["buffer_max_score"]
            log_buf_min = buf_stats["buffer_min_score"]

            # Step 2: Inner training loop
            for t in range(T_steps):
                if total_collected >= total_target:
                    break

                n_buffer = int(self.rho * N_L)
                n_random = N_L - n_buffer
                batch_levels: List[OvercookedLevel] = []

                if self.buffer.size > 0 and n_buffer > 0:
                    key, k_sample = jax.random.split(key)
                    sampled = self.buffer.sample_uniform(k_sample, n=min(n_buffer, self.buffer.size))
                    while len(sampled) < n_buffer:
                        key, k_extra = jax.random.split(key)
                        sampled.extend(self.buffer.sample_uniform(k_extra, n=1))
                    batch_levels.extend(sampled[:n_buffer])
                else:
                    n_random = N_L

                for _ in range(n_random):
                    level, key = self._generate_single_level(key)
                    if level is not None:
                        batch_levels.append(level)
                    else:
                        fallback = self.buffer.levels[0] if self.buffer.size > 0 else batch_levels[0]
                        batch_levels.append(fallback.copy())

                _log(
                    f"[SFL JAX] [2/5] upd {self.updates + 1} (t={t + 1}/{T_steps}) | "
                    f"batch: {n_buffer} from buffer + {n_random} fresh"
                )

                batched_static, batched_agents = self._prepare_training_batch(batch_levels)

                for lv in batch_levels:
                    self._log_reset(lv, "train", total_collected, env_param_log_writer)

                obs_dict, env_states, h0, h1, key = self._fresh_reset(key, batched_static, batched_agents)
                key, k_r = jax.random.split(key)

                _log(f"[SFL JAX] [3/5] collecting rollout ({self.ippo.rollout_len} steps x {N_L} envs)...")
                t_roll = time.time()
                (trs0, trs1, (next_obs, _), (nh0, nh1),
                 ep_returns_arr, _, ep_dones_arr, shaped_comps, key) = self.ippo._collect_rollout(
                    ts0, ts1, (obs_dict, env_states), (h0, h1), k_r
                )
                _log(f"[SFL JAX] [3/5] rollout done in {time.time()-t_roll:.1f}s")

                _log("[SFL JAX] [4/5] computing GAE + PPO update...")
                t_upd = time.time()
                last_v0, last_v1 = self._bootstrap_values(ts0, ts1, next_obs, nh0, nh1)
                adv0, ret0 = calculate_gae(trs0, last_v0, self.ippo.gamma, self.ippo.lam)
                adv1, ret1 = calculate_gae(trs1, last_v1, self.ippo.gamma, self.ippo.lam)

                key, ks0, ks1 = jax.random.split(key, 3)
                ts0, loss0 = self.ippo._update_agent(ts0, trs0, adv0, ret0, seed=int(np.asarray(ks0)[0]))
                ts1, loss1 = self.ippo._update_agent(ts1, trs1, adv1, ret1, seed=int(np.asarray(ks1)[0]))
                log_loss = (float(loss0) + float(loss1)) / 2.0
                _log(f"[SFL JAX] [4/5] update done in {time.time()-t_upd:.1f}s | loss={log_loss:.4f}")

                total_collected += steps_per_rollout
                self.updates += 1

                # Episode tracking
                ep_returns_np = np.asarray(ep_returns_arr)
                ep_dones_np = np.asarray(ep_dones_arr)
                step_r_np = np.asarray(trs0.reward)

                if self.ippo.reward_mode == "delivery":
                    deliv_per_step = (step_r_np >= 1.0).astype(np.int32)
                else:
                    deliv_per_step = (step_r_np >= DELIVERY_REWARD).astype(np.int32)

                self.total_deliveries += int(np.sum(deliv_per_step))

                ep_deliv_buf = np.zeros(N_L, dtype=np.int32)
                for step_idx in range(self.ippo.rollout_len):
                    ep_deliv_buf += deliv_per_step[step_idx]
                    for env_i in np.where(ep_dones_np[step_idx])[0]:
                        ep_ret = float(ep_returns_np[step_idx, env_i])
                        all_ep_returns.append(ep_ret)
                        if episode_log_writer:
                            episode_log_writer.writerow({
                                "episode": len(all_ep_returns) - 1,
                                "deliveries": int(ep_deliv_buf[env_i]),
                                "return": f"{ep_ret:.2f}",
                                "buf_mean_score": f"{log_buf_mean:.4f}",
                                "buf_max_score": f"{log_buf_max:.4f}",
                                "buf_min_score": f"{log_buf_min:.4f}",
                            })
                            episode_log_file.flush()
                        ep_deliv_buf[env_i] = 0

                # Step 5: Summary logging — every update, so the console trace and
                # the live plot both stay populated from the first update onward
                # rather than only every log_every updates.
                mean_r = float(np.mean(all_ep_returns[-100:])) if all_ep_returns else 0.0
                sps = total_collected / max(time.time() - t0, 1e-9)

                entry = {
                    "steps": total_collected, "iteration": self.iterations,
                    "update": self.updates, "mean_r": mean_r, "loss": log_loss,
                    "buffer_size": self.buffer.size,
                    "buffer_mean_score": log_buf_mean,
                    "buffer_max_score": log_buf_max,
                    "buffer_min_score": log_buf_min,
                    "total_deliveries": self.total_deliveries,
                }
                log.append(entry)
                if log_callback:
                    log_callback(log)

                summary_line = (
                    f"[SFL JAX] [5/5] iter={self.iterations:4d} | upd={self.updates:5d} | "
                    f"steps={total_collected:9,} | mean_r={mean_r:6.2f} | "
                    f"loss={log_loss:.4f} | buf={self.buffer.size} "
                    f"mean={log_buf_mean:.4f} max={log_buf_max:.4f} | "
                    f"deliv={self.total_deliveries} | sps={sps:.0f}"
                )
                print(summary_line, flush=True)  # only console line printed every update
                _log(summary_line)
                if checkpoint_dir and batch_levels and self.updates % self.ippo.log_every == 0:
                    _save_layout_snapshot(batch_levels[0].grid, total_collected, Path(checkpoint_dir))

                # GIF
                if render_every > 0 and checkpoint_dir and batch_levels:
                    prev = total_collected - steps_per_rollout
                    if total_collected // render_every > prev // render_every:
                        _save_agent_gif(batch_levels[0].grid, self.ippo.network,
                                        ts0.params, ts1.params, total_collected, Path(checkpoint_dir))

                # Checkpoint
                if checkpoint_dir and checkpoint_every > 0 and self.updates % checkpoint_every == 0:
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
        if pool_log_file: pool_log_file.close()
        if checkpoint_dir:
            with open(Path(checkpoint_dir) / "env_params.pkl", "wb") as f:
                pickle.dump(self.env_param_log, f)
            with open(Path(checkpoint_dir) / "pool_scores.pkl", "wb") as f:
                pickle.dump(self.pool_log, f)

        _log(f"\n[SFL JAX] Training complete. Steps: {total_collected:,}")
        if workflow_log_file:
            workflow_log_file.close()
        return ts0, ts1, log
