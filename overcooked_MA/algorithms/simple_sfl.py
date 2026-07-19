"""
simple_sfl.py — lightweight Sampling For Learnability trainer for the
overcooked_MA prototype.

Unlike ued_implementation/training/sfl.py (JAX-vmapped, requires the full
OvercookedV2 JAX engine), this trainer drives the plain NumPy
OvercookedEnvironment directly with Python-level rollouts. Both agents share
one ActorCritic (networks.ActorCritic), trained with vanilla policy gradient
+ value baseline (no PPO clipping, no GAE — deliberately simple).

Outer loop:  generate N random levels -> estimate learnability -> keep top-K
Inner loop:  sample rho*buffer + (1-rho)*fresh levels -> rollout -> PG update
"""

import csv
import pickle
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import optax

from environment.environment import OvercookedEnvironment, NUM_ACTIONS
from environment.generator import LevelGenerator
from environment.settings import SHAPED_REWARDS
from networks import ActorCritic


class SimpleSFLTrainer:
    def __init__(
        self,
        height,
        width,
        cfg,
        buffer_size=20,
        pool_size=40,
        rho=0.5,
        rollouts_per_level=4,
        episode_len=150,
        refresh_every=5,
        seed=42,
    ):
        self.height, self.width = height, width
        self.cfg = cfg
        self.episode_len = episode_len
        self.refresh_every = refresh_every

        self.env = OvercookedEnvironment(height=height, width=width, max_steps=episode_len)
        self.generator = LevelGenerator(height=height, width=width)
        self.network = ActorCritic(
            n_actions=NUM_ACTIONS,
            use_rnn=False,
            obs_mode=cfg.get("obs_mode", "rich"),
        )

        self.buffer_size = buffer_size
        self.pool_size = pool_size
        self.rho = rho
        self.rollouts_per_level = rollouts_per_level
        # One full pre-delivery shaped-reward cycle: 3x ingredient placement +
        # pot-start + plate pickup + dish pickup (see environment/settings.py).
        _default_reward_norm = 3 * SHAPED_REWARDS["PLACEMENT_IN_POT"] + SHAPED_REWARDS["POT_START_COOKING"] \
            + SHAPED_REWARDS["PLATE_PICKUP"] + SHAPED_REWARDS["DISH_PICKUP"]
        self.max_reward_norm = cfg.get("max_reward_norm", float(_default_reward_norm))

        self.gamma = cfg.get("gamma", 0.99)
        self.ent_coef = cfg.get("ent_coef", 0.01)
        self.vf_coef = cfg.get("vf_coef", 0.5)
        self.lr = cfg.get("lr", 3e-4)

        self.rng = np.random.default_rng(seed)
        self.key = jax.random.PRNGKey(seed)
        self.buffer = []  # list of {"grid": np.ndarray, "params": np.ndarray, "score": float}
        self.total_deliveries = 0

        # Per-iteration record of every level (grid + UED param vector) actually
        # used for training, so the curriculum's level history can be inspected
        # or re-plotted later — mirrors ued_implementation's env_params.csv.
        self.env_param_log = []
        self._reset_id = 0

        # Record of every candidate scored during buffer refresh (the full
        # pool, not just the top-K that made it into the buffer), so the
        # learnability landscape the curriculum picked from is inspectable.
        self.pool_log = []
        self._refresh_id = 0

        self._act_fn = self._make_act_fn()

    # ── Level generation ────────────────────────────────────────────────────

    def _random_level(self):
        """Returns (grid, params_vec) where params_vec = [obs_density, obs_skew_x, res_density, res_skew_x]."""
        obs_d, obs_s, res_d, res_s = self.rng.uniform(0.0, 1.0, size=4)
        obs_s = obs_s * 2.0 - 1.0
        res_s = res_s * 2.0 - 1.0
        elems = self.generator.create_level_elements(obs_d, obs_s, res_d, res_s)
        grid = self.generator.calculate_element_coords(elems)
        params_vec = np.array([obs_d, obs_s, res_d, res_s], dtype=np.float32)
        return grid, params_vec

    # ── Rollout ──────────────────────────────────────────────────────────────

    def _make_act_fn(self):
        @jax.jit
        def _act(params, obs, key):
            logits, value = self.network.apply(params, obs[None])
            action = jax.random.categorical(key, logits[0])
            return action, value[0]

        return _act

    def _select_action(self, params, obs, key):
        key, k_sample = jax.random.split(key)
        action, value = self._act_fn(params, obs, k_sample)
        return int(action), float(value), key

    def _rollout(self, params, grid, key, collect=False):
        self.env.set_level_template(grid)
        obs_dict, _ = self.env.reset()
        traj = {"agent_0": [], "agent_1": []}
        deliveries = 0
        episode_reward = 0.0

        for _ in range(self.episode_len):
            actions, step_info = {}, {}
            for aid in ("agent_0", "agent_1"):
                a, v, key = self._select_action(params, obs_dict[aid], key)
                actions[aid] = a
                step_info[aid] = (obs_dict[aid], a, v)

            obs_dict, rewards, terminated, truncated, info = self.env.step(actions)
            deliveries += info["deliveries"]
            episode_reward += rewards["agent_0"] + rewards["agent_1"]

            if collect:
                for aid in ("agent_0", "agent_1"):
                    o, a, v = step_info[aid]
                    traj[aid].append((o, a, v, rewards[aid]))

            if terminated or truncated:
                break

        return deliveries, episode_reward, traj, key

    # ── Learnability scoring + buffer ───────────────────────────────────────

    def _score_level(self, params, grid, key):
        # Deliveries are too rare early in training to differentiate levels (a
        # full delivery needs a long chained sequence — 3 ingredients into the
        # pot, ~POT_COOK_TIME steps cooking, plate, dish, goal — that a
        # not-yet-competent policy essentially never completes within
        # episode_len). Shaped reward (ingredient pickup, pot placement,
        # pot-start, plate pickup) gives a continuous, non-zero signal from
        # the first update, so learnability can differentiate levels long
        # before any of them are solvable end-to-end.
        outcomes = []
        for _ in range(self.rollouts_per_level):
            _, episode_reward, _, key = self._rollout(params, grid, key, collect=False)
            outcomes.append(episode_reward)
        p = float(np.clip(np.mean(outcomes) / max(self.max_reward_norm, 1e-8), 0.0, 1.0))
        return p * (1.0 - p), key

    def refresh_buffer(self, params, writer=None):
        t_start = time.time()
        heartbeat = max(1, self.pool_size // 10)
        candidates = []
        for i in range(self.pool_size):
            grid, params_vec = self._random_level()
            score, self.key = self._score_level(params, grid, self.key)
            candidates.append({"grid": grid, "params": params_vec, "score": score})
            if (i + 1) % heartbeat == 0 or i + 1 == self.pool_size:
                print(
                    f"[SimpleSFL] scoring buffer candidates: {i + 1}/{self.pool_size} "
                    f"({time.time() - t_start:.1f}s elapsed)",
                    flush=True,
                )
        candidates.sort(key=lambda c: c["score"], reverse=True)
        self.buffer = candidates[: self.buffer_size]

        # Record every scored candidate — not just the top-K that survives
        # into the buffer — so the full learnability landscape is inspectable.
        for rank, c in enumerate(candidates):
            csv_row = {
                "refresh_id": self._refresh_id,
                "rank": rank,
                "selected": rank < self.buffer_size,
                "obs_density": float(c["params"][0]), "obs_skew_x": float(c["params"][1]),
                "res_density": float(c["params"][2]), "res_skew_x": float(c["params"][3]),
                "score": c["score"],
            }
            # pool_log (and its pickle) additionally carries the actual grid,
            # so candidates can be re-rendered exactly as scored — the CSV
            # stays vector-only since it can't hold a 2D array per row.
            self.pool_log.append({**csv_row, "grid": c["grid"]})
            if writer:
                writer.writerow(csv_row)
        self._refresh_id += 1

    def _sample_batch_levels(self, batch_size):
        """Returns a list of {"grid", "params", "source"} — the K levels used for this iteration."""
        n_buffer = int(self.rho * batch_size) if self.buffer else 0
        levels = []
        if n_buffer:
            idx = self.rng.integers(0, len(self.buffer), size=n_buffer)
            levels.extend(
                {"grid": self.buffer[i]["grid"], "params": self.buffer[i]["params"], "source": "buffer"}
                for i in idx
            )
        for _ in range(batch_size - len(levels)):
            grid, params_vec = self._random_level()
            levels.append({"grid": grid, "params": params_vec, "source": "fresh"})
        return levels

    # ── Policy gradient update ──────────────────────────────────────────────

    @staticmethod
    def _discounted_returns(rewards, gamma):
        returns = np.zeros(len(rewards), dtype=np.float32)
        running = 0.0
        for t in reversed(range(len(rewards))):
            running = rewards[t] + gamma * running
            returns[t] = running
        return returns

    def _make_update_step(self, optimizer):
        @jax.jit
        def _step(params, opt_state, obs, actions, returns):
            def loss_fn(p):
                logits, values = self.network.apply(p, obs)
                log_probs = jax.nn.log_softmax(logits)
                logp_a = log_probs[jnp.arange(actions.shape[0]), actions]
                adv = returns - jax.lax.stop_gradient(values)
                adv_n = (adv - adv.mean()) / (adv.std() + 1e-8)
                entropy = -jnp.sum(jnp.exp(log_probs) * log_probs, axis=-1).mean()
                actor_loss = -(logp_a * adv_n).mean()
                critic_loss = jnp.mean((values - returns) ** 2)
                return actor_loss + self.vf_coef * critic_loss - self.ent_coef * entropy

            loss, grads = jax.value_and_grad(loss_fn)(params)
            updates, opt_state = optimizer.update(grads, opt_state, params)
            params = optax.apply_updates(params, updates)
            return params, opt_state, loss

        return _step

    # ── Main training loop ───────────────────────────────────────────────────

    def train(
        self,
        total_updates,
        batch_size=8,
        resume_params=None,
        log_callback=None,
        checkpoint_dir=None,
        checkpoint_every=0,
        **_ignored,
    ):
        if resume_params is not None:
            params = resume_params
        else:
            dummy_obs = jnp.zeros((1, self.height, self.width, 7))
            self.key, k_init = jax.random.split(self.key)
            params = self.network.init(k_init, dummy_obs)
        optimizer = optax.adam(self.lr)
        opt_state = optimizer.init(params)
        update_step = self._make_update_step(optimizer)

        env_param_log_file = env_param_log_writer = None
        pool_log_file = pool_log_writer = None
        if checkpoint_dir:
            param_path = Path(checkpoint_dir) / "env_params.csv"
            param_path.parent.mkdir(parents=True, exist_ok=True)
            env_param_log_file = open(param_path, "w", newline="")
            env_param_log_writer = csv.DictWriter(env_param_log_file, fieldnames=[
                "reset_id", "update", "source",
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

        log = []
        t0 = time.time()
        total_steps = 0

        for update in range(1, total_updates + 1):
            if update == 1 or update % self.refresh_every == 0:
                self.refresh_buffer(params, writer=pool_log_writer)
                if pool_log_file:
                    pool_log_file.flush()

            levels = self._sample_batch_levels(batch_size)

            all_obs, all_actions, all_returns, ep_returns = [], [], [], []
            for level in levels:
                grid, params_vec, source = level["grid"], level["params"], level["source"]
                rid = self._reset_id
                self._reset_id += 1
                self.env_param_log.append({
                    "reset_id": rid, "update": update, "source": source,
                    "obs_density": float(params_vec[0]), "obs_skew_x": float(params_vec[1]),
                    "res_density": float(params_vec[2]), "res_skew_x": float(params_vec[3]),
                })
                if env_param_log_writer:
                    env_param_log_writer.writerow(self.env_param_log[-1])
                    env_param_log_file.flush()

                deliveries, _, traj, self.key = self._rollout(params, grid, self.key, collect=True)
                self.total_deliveries += deliveries
                for aid in ("agent_0", "agent_1"):
                    steps = traj[aid]
                    if not steps:
                        continue
                    rewards = [s[3] for s in steps]
                    returns = self._discounted_returns(rewards, self.gamma)
                    all_obs.extend(s[0] for s in steps)
                    all_actions.extend(s[1] for s in steps)
                    all_returns.extend(returns.tolist())
                    ep_returns.append(float(np.sum(rewards)))
                    total_steps += len(steps)

            batch_obs = jnp.asarray(np.stack(all_obs))
            batch_actions = jnp.asarray(all_actions)
            batch_returns = jnp.asarray(all_returns, dtype=jnp.float32)

            params, opt_state, loss = update_step(
                params, opt_state, batch_obs, batch_actions, batch_returns
            )
            loss = float(loss)

            buf_scores = [c["score"] for c in self.buffer] if self.buffer else [0.0]
            entry = {
                "steps": total_steps,
                "update": update,
                "mean_r": float(np.mean(ep_returns)) if ep_returns else 0.0,
                "loss": loss,
                "buffer_size": len(self.buffer),
                "buffer_mean_score": float(np.mean(buf_scores)),
                "buffer_max_score": float(np.max(buf_scores)),
                "buffer_min_score": float(np.min(buf_scores)),
                "total_deliveries": self.total_deliveries,
            }
            log.append(entry)
            if log_callback:
                log_callback(log)

            sps = total_steps / max(time.time() - t0, 1e-9)
            print(
                f"[SimpleSFL] upd={update:4d}/{total_updates} steps={total_steps:8,} "
                f"mean_r={entry['mean_r']:6.2f} loss={loss:.4f} "
                f"buf={len(self.buffer)} mean={entry['buffer_mean_score']:.4f} "
                f"deliv={self.total_deliveries} sps={sps:.0f}",
                flush=True,
            )

            if checkpoint_dir and checkpoint_every > 0 and update % checkpoint_every == 0:
                ckpt = Path(checkpoint_dir) / "checkpoints" / f"update_{update:05d}"
                ckpt.mkdir(parents=True, exist_ok=True)
                with open(ckpt / "params.pkl", "wb") as f:
                    pickle.dump(params, f)

        if checkpoint_dir:
            final = Path(checkpoint_dir) / "checkpoints" / "final"
            final.mkdir(parents=True, exist_ok=True)
            with open(final / "params.pkl", "wb") as f:
                pickle.dump(params, f)

        if env_param_log_file:
            env_param_log_file.close()
        if pool_log_file:
            pool_log_file.close()
        if checkpoint_dir:
            with open(Path(checkpoint_dir) / "env_params.pkl", "wb") as f:
                pickle.dump(self.env_param_log, f)
            with open(Path(checkpoint_dir) / "pool_scores.pkl", "wb") as f:
                pickle.dump(self.pool_log, f)

        return params, log
