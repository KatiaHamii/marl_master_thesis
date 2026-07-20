"""
ippo_jax.py — JAX-native Independent PPO for OvercookedV2
==========================================================
Implements IPPO (Independent PPO) as used in
"OvercookedV2: Rethinking Overcooked for Zero-Shot Coordination"
(Gessler et al., ICLR 2025)  arXiv 2503.17821

Key design
----------
- N_ENVS parallel environments via jax.vmap over reset/step
- T-step rollout via jax.lax.scan (single XLA graph)
- Auto-reset on episode end inside scan
- Per-agent ActorCritic (recurrent, conv+GRU) from networks.py
- GAE via a reverse lax.scan
- Minibatch PPO with clipped surrogate + entropy bonus
- Flax TrainState (params + optax opt_state) per agent

Exposed API
-----------
  IPPOTrainer(env, cfg) — create
  IPPOTrainer.train(key) — run training loop, log to stdout
"""

import functools
import json
import time
from pathlib import Path
from typing import Tuple, NamedTuple, Dict, Any

import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax.training.train_state import TrainState

from networks import ActorCritic
from environment.settings import DELIVERY_REWARD

# ── Rollout buffer types ───────────────────────────────────────────────────────


class Transition(NamedTuple):
    obs: jnp.ndarray  # (n_envs, H, W, C)
    action: jnp.ndarray  # (n_envs,)
    log_prob: jnp.ndarray  # (n_envs,)
    value: jnp.ndarray  # (n_envs,)
    reward: jnp.ndarray  # (n_envs,)
    done: jnp.ndarray  # (n_envs,) bool
    hidden: jnp.ndarray  # (n_envs, gru_hidden)  — hidden BEFORE this step


# ── GAE ─(Generalized Advantage Estimation)──────────────────────────────────────────────────────────────────────
#  how good each action was.
def calculate_gae(
    transitions: Transition,
    last_value: jnp.ndarray,  # (n_envs,)
    gamma: float = 0.99,
    lam: float = 0.95,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Compute GAE advantages and returns via reverse scan.
    Returns:
        advantages : (T, n_envs)
        returns    : (T, n_envs)
    """
    rewards = transitions.reward  # (T, n_envs)
    values = transitions.value  # (T, n_envs) what the agent predicted
    dones = transitions.done.astype(jnp.float32)  # (T, n_envs) episode ended? (0=yes, 1=no)
    T = rewards.shape[0] #timesteps

    # We need value at t+1. Append last_value at the end.
    values_ext = jnp.concatenate([values, last_value[None]], axis=0)  # (T+1, n_envs)

    def _scan_body(carry, t):
        gae = carry
        not_done = 1.0 - dones[t] #if episode is done, stop accumulating advantage
        delta = rewards[t] + gamma * values_ext[t + 1] * not_done - values_ext[t] #actual reward + discounted next value - current value
        gae = delta + gamma * lam * not_done * gae
        return gae, gae

    _, advantages_rev = jax.lax.scan(
        _scan_body,
        jnp.zeros_like(last_value),
        jnp.arange(T - 1, -1, -1),
    )
    advantages = jnp.flip(advantages_rev, axis=0)  # (T, n_envs)
    returns = advantages + values
    return advantages, returns


# ── PPO loss ──────────────────────────────────────────────────────────────────


def ppo_loss(
    params: Any,
    apply_fn: Any,  # net.apply
    obs: jnp.ndarray,  # (B, H, W, C)
    hiddens: jnp.ndarray,  # (B, gru_hidden)
    actions: jnp.ndarray,  # (B,)
    old_log_probs: jnp.ndarray,  # (B,)
    old_values: jnp.ndarray,  # (B,)  — value estimates from rollout
    advantages: jnp.ndarray,  # (B,)
    returns: jnp.ndarray,  # (B,)
    clip_eps: float,
    vf_coef: float,
    ent_coef: float,
) -> jnp.ndarray:
    """Combined actor-critic PPO loss with value-function clipping (PPO2)."""

    def _forward_one(obs_i, h_i):
        out = apply_fn(params, obs_i, h_i)
        logits, value = out[0], out[1]
        return logits, value

    logits, values = jax.vmap(_forward_one)(obs, hiddens)  # (B, n_actions), (B,)
    log_probs = jax.nn.log_softmax(logits)  # (B, n_actions)
    new_lp = log_probs[jnp.arange(len(actions)), actions]  # (B,)
    entropy = -jnp.sum(jnp.exp(log_probs) * log_probs, axis=-1)  # (B,)

    ratio = jnp.exp(new_lp - old_log_probs)
    adv_n = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
    surr1 = ratio * adv_n
    surr2 = jnp.clip(ratio, 1 - clip_eps, 1 + clip_eps) * adv_n
    actor_l = -jnp.minimum(surr1, surr2).mean()

    # Value function clipping (PPO2): prevents the critic from making
    # large jumps that cause the loss spikes seen in training
    v_clipped = old_values + jnp.clip(values - old_values, -clip_eps, clip_eps)
    critic_l = jnp.maximum(
        (values - returns) ** 2,
        (v_clipped - returns) ** 2,
    ).mean()

    return actor_l + vf_coef * critic_l - ent_coef * entropy.mean()


# ── Trainer ───────────────────────────────────────────────────────────────────


class IPPOTrainer:
    """
    IPPO trainer for OvercookedV2.

    Config keys (passed as a dict):
        n_envs          int    number of parallel environments
        rollout_len     int    steps per rollout
        n_epochs        int    PPO epochs per update
        batch_size      int    minibatch size
        lr              float  Adam learning rate
        gamma           float  discount
        gae_lambda      float  GAE lambda
        clip_eps        float  PPO clip epsilon
        vf_coef         float  value function coefficient
        ent_coef        float  entropy bonus coefficient
        max_grad_norm   float  global gradient clipping
        total_steps     int    total environment steps
        log_every       int    print every N updates
        hidden_size     int    GRU hidden size (default 128)
        obs_mode        str    observation mode: "rich" (51-ch) or "simple" (7-ch)
    """

    def __init__(self, env, cfg: dict, curriculum_mgr=None):
        self.env = env
        self.cfg = cfg
        self.curriculum_mgr = curriculum_mgr

        self.n_envs = cfg.get("n_envs", 8)
        self.rollout_len = cfg.get("rollout_len", 512)
        self.n_epochs = cfg.get("n_epochs", 4)
        self.batch_size = cfg.get("batch_size", 64)
        self.lr = cfg.get("lr", 3e-4)
        self.gamma = cfg.get("gamma", 0.99)
        self.lam = cfg.get("gae_lambda", 0.95)
        self.clip_eps = cfg.get("clip_eps", 0.2)
        self.vf_coef = cfg.get("vf_coef", 0.5)
        self.ent_coef = cfg.get("ent_coef", 0.01)
        self.reward_mode = cfg.get("reward_mode", "shaped")  # "shaped" | "delivery"
        self.obs_mode = cfg.get("obs_mode", "rich")  # "rich" | "simple"
        self.shaped_reward_scale = cfg.get("shaped_reward_scale", 10.0)
        self.step_penalty = cfg.get("step_penalty", 0.0)
        self.collision_penalty = cfg.get("collision_penalty", 0.0)
        self.wrong_ingredient_penalty = cfg.get("wrong_ingredient_penalty", 0.0)
        self.max_grad = cfg.get("max_grad_norm", 0.5)
        self.total_steps = cfg.get("total_steps", 200_000)
        self.log_every = cfg.get("log_every", 10)
        self.hidden_size = cfg.get("hidden_size", 128)

        if env.num_agents != 2:
            raise ValueError(
                f"IPPOTrainer requires exactly 2 agents, but layout has {env.num_agents}. "
                f"Use a 2-agent layout (all layouts except 'long_room')."
            )
        self.n_actions = env.num_actions
        # Get actual obs shape from a probe reset
        _obs_probe, _ = env.reset(jax.random.PRNGKey(99))
        self.obs_shape = _obs_probe["agent_0"].shape  # (H, W, C)

        self.network = ActorCritic(
            n_actions=self.n_actions,
            gru_hidden=self.hidden_size,
            obs_mode=self.obs_mode,
        )
        self._mb_update = self._make_mb_update()  # JIT-compiled minibatch updater

    def _make_train_state(
        self,
        key: jnp.ndarray,
        resume_params: dict = None,  # {"agent_0": params, "agent_1": params}
    ) -> Tuple[TrainState, TrainState]:
        """Initialise one TrainState per agent, optionally from a checkpoint."""
        dummy_obs = jnp.zeros(self.obs_shape)
        dummy_h = ActorCritic.init_hidden(gru_hidden=self.hidden_size)
        optimizer = optax.chain(
            optax.clip_by_global_norm(self.max_grad),
            optax.adam(self.lr),
        )

        def _init(k, init_params=None, init_opt_state=None):
            params = (
                init_params
                if init_params is not None
                else self.network.init(k, dummy_obs, dummy_h)
            )
            ts = TrainState.create(
                apply_fn=self.network.apply, params=params, tx=optimizer
            )
            if init_opt_state is not None:
                # Restore Adam momentum/variance so training continues smoothly
                ts = ts.replace(opt_state=init_opt_state)
            return ts

        k0, k1 = jax.random.split(key)
        p0 = resume_params["agent_0"] if resume_params else None
        p1 = resume_params["agent_1"] if resume_params else None
        os0 = resume_params.get("opt_state_0") if resume_params else None
        os1 = resume_params.get("opt_state_1") if resume_params else None
        return _init(k0, p0, os0), _init(k1, p1, os1)

    # ── Vectorised reset/step ─────────────────────────────────────────────────

    def _vmap_reset(self, keys: jnp.ndarray):
        """Reset N envs in parallel. keys: (N, 2)."""
        return jax.vmap(self.env.reset)(keys)

    def _vmap_step(self, keys, states, actions_0, actions_1):
        """Step N envs in parallel."""
        act_dicts = jax.vmap(lambda a0, a1: {"agent_0": a0, "agent_1": a1})(
            actions_0, actions_1
        )
        return jax.vmap(self.env.step)(keys, states, act_dicts)

    # ── Rollout collection ────────────────────────────────────────────────────

    def _collect_rollout(
        self,
        ts0: TrainState,
        ts1: TrainState,
        init_obs_states,  # (obs_dict, env_states)
        init_hiddens,  # (h0, h1)  each (n_envs, hidden)
        key: jnp.ndarray,
    ):
        """
        Collect T-step rollout across N_ENVS environments using lax.scan.
        Returns (transitions_0, transitions_1, final_obs_states, final_hiddens, ep_returns, ep_lens).
        """
        obs_dict, env_states = init_obs_states
        h0, h1 = init_hiddens

        def _scan_step(carry, _):
            obs_dict, env_states, h0, h1, key, ep_ret, ep_len = carry
            obs0 = obs_dict["agent_0"]  # (n_envs, H, W, C)
            obs1 = obs_dict["agent_1"]

            key, k0, k1, k_step = jax.random.split(key, 4)

            # Forward pass agent 0
            def _fwd0(o, h):
                out = ts0.apply_fn(ts0.params, o, h)
                return out if len(out) == 3 else (*out, h)

            logits0, val0, new_h0 = jax.vmap(_fwd0)(obs0, h0)

            # Forward pass agent 1
            def _fwd1(o, h):
                out = ts1.apply_fn(ts1.params, o, h)
                return out if len(out) == 3 else (*out, h)

            logits1, val1, new_h1 = jax.vmap(_fwd1)(obs1, h1)

            # Sample actions
            keys0 = jax.random.split(k0, self.n_envs)
            keys1 = jax.random.split(k1, self.n_envs)
            a0 = jax.vmap(lambda k, lg: jax.random.categorical(k, lg))(keys0, logits0)
            a1 = jax.vmap(lambda k, lg: jax.random.categorical(k, lg))(keys1, logits1)
            lp0 = jax.vmap(lambda lg, a: jax.nn.log_softmax(lg)[a])(logits0, a0)
            lp1 = jax.vmap(lambda lg, a: jax.nn.log_softmax(lg)[a])(logits1, a1)

            # Environment step (auto-reset handled inside env.step)
            k_steps = jax.random.split(k_step, self.n_envs)
            new_obs, new_states, rewards, dones, infos = jax.vmap(
                lambda k, s, a0_, a1_: self.env.step(
                    k, s, {"agent_0": a0_, "agent_1": a1_}
                )
            )(k_steps, env_states, a0, a1)

            if self.reward_mode == "delivery":
                # Pure team delivery reward: +1 per correct delivery, shared,
                # no shaping, no penalties. Both agents learn to maximise
                # total deliveries regardless of who makes them.
                delivery = rewards["agent_0"] / float(
                    DELIVERY_REWARD
                )  # +1 per delivery
                r0 = r1 = delivery
            elif self.reward_mode == "sparse":
                # Sparse reward: only the raw delivery reward (+DELIVERY_REWARD per
                # correct delivery), shared between agents, no shaping at all.
                r0 = r1 = rewards["agent_0"]
            else:
                # Shaped mode: delivery + dense shaped signals per agent
                shaped0 = infos["shaped_reward"]["agent_0"]
                shaped1 = infos["shaped_reward"]["agent_1"]
                r0 = rewards["agent_0"] + self.shaped_reward_scale * shaped0
                r1 = rewards["agent_1"] + self.shaped_reward_scale * shaped1

                r0 = r0 - self.step_penalty
                r1 = r1 - self.step_penalty

                coll0 = infos["collision"]["agent_0"].astype(jnp.float32)
                coll1 = infos["collision"]["agent_1"].astype(jnp.float32)
                r0 = r0 - self.collision_penalty * coll0
                r1 = r1 - self.collision_penalty * coll1

                wi0 = infos["wrong_ingredient_pickup"]["agent_0"].astype(jnp.float32)
                wi1 = infos["wrong_ingredient_pickup"]["agent_1"].astype(jnp.float32)
                r0 = r0 - self.wrong_ingredient_penalty * wi0
                r1 = r1 - self.wrong_ingredient_penalty * wi1

            done = dones["__all__"]  # (n_envs,)

            # Episode tracking (use agent_0's reward as team signal for logging)
            ep_ret = ep_ret + r0
            ep_len = ep_len + 1

            # Reset hidden on episode end
            new_h0 = jnp.where(done[:, None], jnp.zeros_like(new_h0), new_h0)
            new_h1 = jnp.where(done[:, None], jnp.zeros_like(new_h1), new_h1)

            tr0 = Transition(
                obs=obs0,
                action=a0,
                log_prob=lp0,
                value=val0,
                reward=r0,
                done=done,
                hidden=h0,
            )
            tr1 = Transition(
                obs=obs1,
                action=a1,
                log_prob=lp1,
                value=val1,
                reward=r1,
                done=done,
                hidden=h1,
            )

            # Per-component shaped rewards for logging (zeros in delivery mode)
            comp0 = infos.get("shaped_reward_components", {}).get(
                "agent_0", jnp.zeros((self.n_envs, 4))
            )

            new_carry = (
                new_obs,
                new_states,
                new_h0,
                new_h1,
                key,
                ep_ret * (1 - done),
                ep_len * (1 - done),
            )
            return new_carry, (tr0, tr1, ep_ret * done, ep_len * done, done, comp0)

        init_carry = (
            obs_dict,
            env_states,
            h0,
            h1,
            key,
            jnp.zeros(self.n_envs),
            jnp.zeros(self.n_envs, dtype=jnp.int32),
        )

        final_carry, (trs0, trs1, ep_returns, ep_lens, dones, shaped_comps) = (
            jax.lax.scan(_scan_step, init_carry, None, length=self.rollout_len)
        )

        final_obs, final_states, final_h0, final_h1, key, _, _ = final_carry
        return (
            trs0,
            trs1,
            (final_obs, final_states),
            (final_h0, final_h1),
            ep_returns,
            ep_lens,
            dones,
            shaped_comps,
            key,
        )

    # ── PPO update ────────────────────────────────────────────────────────────

    def _make_mb_update(self):
        """Return a JIT-compiled single-minibatch gradient update fn."""

        @jax.jit
        def _mb_update(
            ts, obs, hidds, acts, lps, old_vals, advs, rets, clip_eps, vf_coef, ent_coef
        ):
            loss_fn = lambda p: ppo_loss(
                p,
                ts.apply_fn,
                obs,
                hidds,
                acts,
                lps,
                old_vals,
                advs,
                rets,
                clip_eps,
                vf_coef,
                ent_coef,
            )
            loss, grads = jax.value_and_grad(loss_fn)(ts.params)
            ts = ts.apply_gradients(grads=grads)
            return ts, loss

        return _mb_update

    def _update_agent(
        self,
        ts: TrainState,
        transitions: Transition,
        advantages: jnp.ndarray,
        returns: jnp.ndarray,
        seed: int,  # plain Python int — no JAX tracing needed
    ) -> Tuple[TrainState, float]:
        """Run n_epochs of minibatch PPO for one agent. Returns updated TrainState + mean loss."""

        # Flatten time and env dims: (T, N, ...) → (T*N, ...)
        def _flat(x):
            s = x.shape
            return x.reshape(s[0] * s[1], *s[2:])

        obs = _flat(transitions.obs)  # (T*N, H, W, C)
        acts = _flat(transitions.action)  # (T*N,)
        lps = _flat(transitions.log_prob)  # (T*N,)
        hidds = _flat(transitions.hidden)  # (T*N, gru_hidden)
        old_vals = _flat(transitions.value)  # (T*N,)
        advs = _flat(advantages)  # (T*N,)
        rets = _flat(returns)  # (T*N,)
        B = obs.shape[0]

        rng = np.random.default_rng(seed % (2**31))
        total_loss = 0.0
        n_mb = 0

        for _ in range(self.n_epochs):
            idx = rng.permutation(B)
            for start in range(0, B, self.batch_size):
                b = idx[start : start + self.batch_size]
                if len(b) < 2:
                    continue
                ts, loss = self._mb_update(
                    ts,
                    obs[b],
                    hidds[b],
                    acts[b],
                    lps[b],
                    old_vals[b],
                    advs[b],
                    rets[b],
                    self.clip_eps,
                    self.vf_coef,
                    self.ent_coef,
                )
                total_loss += float(loss)
                n_mb += 1

        return ts, total_loss / max(n_mb, 1)

    # ── Main train loop ───────────────────────────────────────────────────────

    def train(
        self,
        key: jnp.ndarray,
        resume_params: dict = None,
        log_callback=None,
        checkpoint_dir=None,
        checkpoint_every: int = 100,
    ):
        """Run full IPPO training. Prints progress to stdout.

        Args:
            key:              JAX PRNGKey
            resume_params:    optional dict {"agent_0": params, "agent_1": params}
            log_callback:     called after every log entry with the full log list
            checkpoint_dir:   if set, save params here every checkpoint_every updates
            checkpoint_every: how many updates between checkpoints (default 100)
        """
        key, k_init = jax.random.split(key)
        ts0, ts1 = self._make_train_state(k_init, resume_params)

        # Initial reset for all envs
        key, k_reset = jax.random.split(key)
        reset_keys = jax.random.split(k_reset, self.n_envs)
        obs_dict, env_states = jax.vmap(self.env.reset)(reset_keys)

        h0 = jnp.zeros((self.n_envs, self.hidden_size))
        h1 = jnp.zeros((self.n_envs, self.hidden_size))

        total_collected = 0
        update_i = 0
        total_updates = self.total_steps // (self.rollout_len * self.n_envs)
        all_ep_returns = []
        all_ep_deliveries = []  # Track deliveries per episode
        log = []
        t0 = time.time()
        last_logged_ep_count = 0  # Track how many episodes were logged last time

        # Open episode log file
        import csv
        episode_log_path = Path(checkpoint_dir) / "episodes.csv" if checkpoint_dir else None
        episode_log_file = None
        episode_log_writer = None
        if episode_log_path:
            episode_log_path.parent.mkdir(parents=True, exist_ok=True)
            episode_log_file = open(episode_log_path, "w", newline="")
            episode_log_writer = csv.DictWriter(
                episode_log_file,
                fieldnames=["episode", "deliveries", "return"]
            )
            episode_log_writer.writeheader()
            episode_log_file.flush()

        if checkpoint_dir is not None:
            import pathlib, pickle as _pickle

            _ckpt_root = pathlib.Path(checkpoint_dir) / "checkpoints"
            _ckpt_root.mkdir(parents=True, exist_ok=True)

        while total_collected < self.total_steps:
            key, k_rollout = jax.random.split(key)
            (
                trs0,
                trs1,
                (obs_dict, env_states),
                (h0, h1),
                ep_returns,
                ep_lens,
                ep_dones,
                shaped_comps,
                key,
            ) = self._collect_rollout(
                ts0,
                ts1,
                (obs_dict, env_states),
                (h0, h1),
                k_rollout,
            )

            total_collected += self.rollout_len * self.n_envs

            # Bootstrap last value for GAE
            obs0_last = obs_dict["agent_0"]  # (n_envs, H, W, C)
            obs1_last = obs_dict["agent_1"]

            def _val(ts_, o, h):
                out = jax.vmap(lambda oi, hi: ts_.apply_fn(ts_.params, oi, hi))(o, h)
                return out[1]

            last_v0 = _val(ts0, obs0_last, h0)  # (n_envs,)
            last_v1 = _val(ts1, obs1_last, h1)

            adv0, ret0 = calculate_gae(trs0, last_v0, self.gamma, self.lam)
            adv1, ret1 = calculate_gae(trs1, last_v1, self.gamma, self.lam)

            # Collect episode stats — record any completed episodes this rollout
            done_np = np.asarray(ep_dones)  # (T, n_envs) bool
            ret_np = np.asarray(ep_returns)  # (T, n_envs)
            # (note: episodes are added in the delivery tracking loop below)

            # Track deliveries per episode (delivery = reward of DELIVERY_REWARD=20)
            from environment.settings import DELIVERY_REWARD
            step_r_np = np.asarray(trs0.reward)  # (T, n_envs)
            ep_lens_np = np.asarray(ep_lens)  # (T, n_envs)
            # In shaped mode reward = DELIVERY_REWARD + shaping (never exactly 20),
            # in delivery mode reward is normalised to 1.0 — use mode-aware threshold.
            if self.reward_mode == "delivery":
                deliveries_per_step = (step_r_np >= 1.0).astype(np.int32)
            else:  # shaped / sparse: delivery spike is always >= 20
                deliveries_per_step = (step_r_np >= DELIVERY_REWARD).astype(np.int32)
            # Accumulate deliveries per episode (reset on episode boundary)
            ep_deliveries = np.zeros(self.n_envs, dtype=np.int32)
            for t in range(self.rollout_len):
                ep_deliveries += deliveries_per_step[t]
                if done_np[t].any():
                    for env_i in np.where(done_np[t])[0]:
                        episode_num = len(all_ep_returns)
                        delivery_count = int(ep_deliveries[env_i])
                        ep_return = float(ret_np[t, env_i])
                        ep_length = int(ep_lens_np[t, env_i])

                        all_ep_deliveries.append(delivery_count)
                        all_ep_returns.append(ep_return)

                        # Track in curriculum if enabled
                        if self.curriculum_mgr:
                            self.curriculum_mgr.record_episode(
                                self.curriculum_mgr.current_layout,
                                delivery_count
                            )

                        # Write to episode log (only if there were deliveries)
                        if episode_log_writer:
                            episode_log_writer.writerow({
                                "episode": episode_num,
                                "deliveries": delivery_count,
                                "return": f"{ep_return:.2f}",
                            })
                            episode_log_file.flush()

                        ep_deliveries[env_i] = 0

            # Also track mean per-step reward for updates with no episode completions
            mean_step_r = float(step_r_np.mean())

            # Per-component shaped reward means: (T, n_envs, 4) → (4,)
            # Order: [dish_pickup, placement_in_pot, pot_start_cooking, plate_pickup]
            comp_means = np.asarray(shaped_comps).mean(axis=(0, 1))  # (4,)

            # Update both agents (pass plain int seeds — no JAX tracing)
            key, k0, k1 = jax.random.split(key, 3)
            seed0 = int(np.asarray(k0)[0])
            seed1 = int(np.asarray(k1)[0])
            ts0, loss0 = self._update_agent(ts0, trs0, adv0, ret0, seed0)
            ts1, loss1 = self._update_agent(ts1, trs1, adv1, ret1, seed1)

            update_i += 1

            # Checkpoint
            if checkpoint_dir is not None and update_i % checkpoint_every == 0:
                _ckpt_path = _ckpt_root / f"update_{update_i:05d}"
                _ckpt_path.mkdir(exist_ok=True)
                with open(_ckpt_path / "params_agent0.pkl", "wb") as _f:
                    _pickle.dump(ts0.params, _f)
                with open(_ckpt_path / "params_agent1.pkl", "wb") as _f:
                    _pickle.dump(ts1.params, _f)
                # Save metadata (reward_mode) to check consistency on load
                _meta = {"reward_mode": self.reward_mode}
                with open(_ckpt_path / "checkpoint_metadata.json", "w") as _f:
                    json.dump(_meta, _f)

            if update_i % self.log_every == 0:
                # Use episode returns if available, else per-step mean
                if all_ep_returns:
                    mean_r = float(np.mean(all_ep_returns[-50:]))
                    r_tag = "ep"
                else:
                    mean_r = mean_step_r
                    r_tag = "step"
                mean_l = (float(loss0) + float(loss1)) / 2.0
                sps = total_collected / max(time.time() - t0, 1)
                pct = 100.0 * total_collected / self.total_steps
                n_ep = len(all_ep_returns)

                # Deliveries: in last log period and total cumulative
                total_deliveries = sum(all_ep_deliveries) if all_ep_deliveries else 0
                # Deliveries in episodes completed since last log
                recent_deliveries = sum(all_ep_deliveries[last_logged_ep_count:]) if all_ep_deliveries else 0
                last_logged_ep_count = len(all_ep_deliveries)

                status_str = (
                    f"  upd {update_i:4d}  |  "
                    f"{total_collected:>8,}/{self.total_steps:,} ({pct:.0f}%)  |  "
                    f"r_{r_tag}={mean_r:+.3f}  loss={mean_l:.4f}  "
                    f"eps={n_ep}  deliveries={recent_deliveries} (total: {total_deliveries})  sps={sps:.0f}"
                )

                # Check curriculum promotion
                if self.curriculum_mgr:
                    promoted = self.curriculum_mgr.check_promotion(update_i)
                    curr_status = self.curriculum_mgr.get_status()
                    status_str += f"  |  lvl={curr_status['current_level']}  layout={curr_status['current_layout']}"
                    if promoted:
                        status_str += f"  ✓PROMOTED"

                print(status_str, flush=True)
                log.append(
                    {
                        "update": update_i,
                        "steps": total_collected,
                        "mean_r": mean_r,
                        "loss": mean_l,
                        "episodes": n_ep,
                        "sps": sps,
                        "sr_dish": float(comp_means[0]),
                        "sr_pot": float(comp_means[1]),
                        "sr_pot_start": float(comp_means[2]),
                        "sr_plate": float(comp_means[3]),
                        "total_deliveries": total_deliveries,
                    }
                )
                if log_callback is not None:
                    log_callback(log)

        # Close episode log file
        if episode_log_file:
            episode_log_file.close()

        # Save curriculum state if enabled
        if self.curriculum_mgr and checkpoint_dir:
            curriculum_path = Path(checkpoint_dir) / "curriculum_state.json"
            self.curriculum_mgr.save(curriculum_path)

        total_deliveries = sum(all_ep_deliveries) if all_ep_deliveries else 0
        print(
            f"\nDone. Updates={update_i}  episodes={len(all_ep_returns)}  "
            f"Final mean_r={float(np.mean(all_ep_returns[-50:])) if all_ep_returns else 0:.3f}  "
            f"Total deliveries={total_deliveries}",
            flush=True,
        )

        # Save final checkpoint and summary
        if checkpoint_dir is not None:
            final_ckpt_path = _ckpt_root / "final"
            final_ckpt_path.mkdir(exist_ok=True)
            with open(final_ckpt_path / "params_agent0.pkl", "wb") as _f:
                _pickle.dump(ts0.params, _f)
            with open(final_ckpt_path / "params_agent1.pkl", "wb") as _f:
                _pickle.dump(ts1.params, _f)
            # Save metadata with final stats
            if log:
                final_log = log[-1]
                _meta = {
                    "reward_mode": self.reward_mode,
                    "final_update": update_i,
                    "final_steps": final_log.get("steps", 0),
                    "final_mean_r": final_log.get("mean_r", 0.0),
                    "final_loss": final_log.get("loss", 0.0),
                    "final_episodes": final_log.get("episodes", 0),
                }
            else:
                _meta = {"reward_mode": self.reward_mode, "final_update": update_i}
            with open(final_ckpt_path / "checkpoint_metadata.json", "w") as _f:
                json.dump(_meta, _f)

        return ts0, ts1, log
