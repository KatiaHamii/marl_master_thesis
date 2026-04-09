"""
agents/mappo.py — Multi-Agent PPO (MAPPO)
==========================================
MAPPO implements the CTDE paradigm (Centralized Training, Decentralized Execution):

  Actor  (shared):      π(a | o_i)           same weights for all agents
  Critic (centralized): V(o_0, o_1, …, o_n)  sees ALL agents' observations

Why a centralized critic?
  A per-agent critic V(o_i) cannot reason about what the other agent observes
  or is about to do. V(o_0, o_1) has full information during training — this
  reduces variance in the advantage estimate and speeds up convergence.
  At execution time only the actor (local obs only) is needed, so no
  inter-agent communication is required.

Key differences from IPPO (train_marl.py):
  IPPO  — independent V(o_i) per agent, separate network weights
  MAPPO — shared actor weights + one centralized V(o_0, o_1, …)
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical


# ─────────────────────────────────────────────
#  NETWORKS
# ─────────────────────────────────────────────

class _Actor(nn.Module):
    """
    Shared actor: local_obs → action logits.

    The same network instance is used by ALL agents (parameter sharing).
    Each agent's local observation is a different input, but the mapping
    obs → policy is identical — this halves the parameter count and means
    every agent's experience trains the policy.

    Tanh activations match the existing PPO agent style.
    """

    def __init__(self, obs_dim: int, action_dim: int, hidden: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.Tanh(),
            nn.Linear(hidden,  hidden), nn.Tanh(),
            nn.Linear(hidden,  action_dim),
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.net(obs)

    def get_action(self, obs: torch.Tensor, mask: torch.Tensor | None = None):
        """Sample one action. Returns (action, log_prob, entropy).
        mask: BoolTensor — True = valid action. Invalid logits → -inf before sampling."""
        logits = self.forward(obs)
        if mask is not None:
            logits = logits.masked_fill(~mask, float("-inf"))
        dist   = Categorical(logits=logits)
        action = dist.sample()
        return action, dist.log_prob(action), dist.entropy()

    def evaluate(self, obs: torch.Tensor, actions: torch.Tensor):
        """Re-evaluate stored actions under the current policy."""
        dist = Categorical(logits=self.forward(obs))
        return dist.log_prob(actions), dist.entropy()


class _CentralCritic(nn.Module):
    """
    Centralized critic: global_obs (all agents concatenated) → V(s).

    Input dimension = obs_dim * n_agents.
    Only used during training — not needed at execution time.
    """

    def __init__(self, global_obs_dim: int, hidden: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(global_obs_dim, hidden), nn.Tanh(),
            nn.Linear(hidden,          hidden), nn.Tanh(),
            nn.Linear(hidden,          1),
        )

    def forward(self, global_obs: torch.Tensor) -> torch.Tensor:
        return self.net(global_obs).squeeze(-1)


# ─────────────────────────────────────────────
#  ROLLOUT BUFFER
#  One per agent.  All are merged into a single
#  batch at update time because the actor is
#  shared — all transitions train the same net.
# ─────────────────────────────────────────────

class RolloutBuffer:
    def __init__(self):
        self.clear()

    def clear(self):
        self.obs        = []   # local obs       → actor update
        self.global_obs = []   # concatenated obs → critic update
        self.actions    = []
        self.log_probs  = []
        self.values     = []
        self.rewards    = []
        self.dones      = []

    def push(self, obs, global_obs, action, log_prob, value, reward, done):
        self.obs.append(obs)
        self.global_obs.append(global_obs)
        self.actions.append(action)
        self.log_probs.append(log_prob)
        self.values.append(value)
        self.rewards.append(reward)
        self.dones.append(done)

    def __len__(self):
        return len(self.rewards)


# ─────────────────────────────────────────────
#  MAPPO TRAINER
# ─────────────────────────────────────────────

class MAPPOTrainer:
    """
    MAPPO trainer for N cooperative agents.

    Typical usage inside a training loop:
        trainer = MAPPOTrainer(cfg, obs_dim, action_dim, n_agents=2)

        # per env step (agent's turn):
        action, log_prob = trainer.get_action(agent, local_obs)
        value            = trainer.get_value(global_obs)
        trainer.store(agent, local_obs, global_obs,
                      action, log_prob, value, reward, done)
        trainer.maybe_update()
    """

    def __init__(self, cfg: dict, obs_dim: int, action_dim: int, n_agents: int = 2):
        mp = cfg.get("mappo", {})
        self.n_agents   = n_agents
        self.obs_dim    = obs_dim
        self.action_dim = action_dim

        # device: honours system.device like the existing PPO agent
        dev_pref = cfg.get("system", {}).get("device", "auto")
        if dev_pref == "auto":
            if torch.backends.mps.is_available():
                self.device = torch.device("mps")
            elif torch.cuda.is_available():
                self.device = torch.device("cuda")
            else:
                self.device = torch.device("cpu")
        else:
            self.device = torch.device(dev_pref)
        print(f"  MAPPO running on: {self.device}")

        # hyperparameters
        self.lr           = mp.get("learning_rate", 3e-4)
        self.gamma        = cfg.get("training", {}).get("discount", 0.99)
        self.gae_lambda   = mp.get("gae_lambda",    0.95)
        self.clip_eps     = mp.get("clip_epsilon",   0.2)
        self.value_coef   = mp.get("value_coef",     0.5)
        self.entropy_coef = mp.get("entropy_coef",   0.01)
        self.n_epochs     = mp.get("n_epochs",       4)
        self.batch_size   = mp.get("batch_size",     64)
        self.update_every = mp.get("update_every",   512)
        hidden            = mp.get("hidden_size",    128)

        # networks
        self.actor  = _Actor(obs_dim, action_dim, hidden).to(self.device)
        self.critic = _CentralCritic(obs_dim * n_agents, hidden).to(self.device)

        # separate optimisers — allows different lr per network if needed later
        self.actor_opt  = optim.Adam(self.actor.parameters(),  lr=self.lr)
        self.critic_opt = optim.Adam(self.critic.parameters(), lr=self.lr)

        # one rollout buffer per agent
        self.buffers      = {f"agent_{i}": RolloutBuffer() for i in range(n_agents)}
        self._total_steps = 0

    # ── action / value selection ─────────────────────────────────

    def get_action(self, agent: str, obs: np.ndarray,
                   mask: np.ndarray | None = None) -> tuple:
        """
        Sample an action from the shared actor.
        mask: boolean array (n_cells,) — True = cell is valid to click.
              Passing the env's action_mask() prevents wasted moves on
              already-revealed cells, which dramatically speeds up learning.
        Returns (action: int, log_prob: float).
        """
        t = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
        m = torch.BoolTensor(mask).unsqueeze(0).to(self.device) if mask is not None else None
        with torch.no_grad():
            action, log_prob, _ = self.actor.get_action(t, m)
        return action.item(), log_prob.item()

    def get_value(self, global_obs: np.ndarray) -> float:
        """
        Estimate V(s) via the centralized critic.
        global_obs = concatenation of all agents' local observations.
        """
        t = torch.FloatTensor(global_obs).unsqueeze(0).to(self.device)
        with torch.no_grad():
            value = self.critic(t)
        return value.item()

    # ── store transition ──────────────────────────────────────────

    def store(self, agent: str, obs: np.ndarray, global_obs: np.ndarray,
              action: int, log_prob: float, value: float,
              reward: float, done: bool):
        self.buffers[agent].push(obs, global_obs, action, log_prob, value, reward, done)
        self._total_steps += 1

    # ── update ────────────────────────────────────────────────────

    def maybe_update(self) -> float | None:
        """Trigger a MAPPO update once enough transitions have been collected."""
        total = sum(len(b) for b in self.buffers.values())
        if total < self.update_every:
            return None
        return self._mappo_update()

    def _compute_gae(self, buf: RolloutBuffer, next_value: float = 0.0):
        """
        Generalised Advantage Estimation.

        Uses the CENTRALIZED critic's value estimates → better advantage signal
        than per-agent V(o_i) because the critic has global information.

        next_value: bootstrapped V for the state after the last stored transition
                    (0.0 works well when the buffer ends at episode termination).
        """
        n          = len(buf)
        advantages = np.zeros(n, dtype=np.float32)
        last_gae   = 0.0

        for t in reversed(range(n)):
            nv       = next_value if t == n - 1 else buf.values[t + 1]
            not_done = 1.0 - float(buf.dones[t])
            delta    = buf.rewards[t] + self.gamma * nv * not_done - buf.values[t]
            last_gae = delta + self.gamma * self.gae_lambda * not_done * last_gae
            advantages[t] = last_gae

        returns = advantages + np.array(buf.values, dtype=np.float32)
        return advantages, returns

    def _mappo_update(self) -> float:
        """
        Core MAPPO PPO update.

        1. Merge transitions from ALL agent buffers into one joint batch.
           The shared actor learns from every agent's experience simultaneously.
        2. Compute GAE advantages using the centralized critic's values.
        3. Run n_epochs of mini-batch PPO on both actor and critic.
        """
        # ── gather data from all agent buffers ────────────────────
        all_obs, all_gobs        = [], []
        all_actions, all_old_lp  = [], []
        all_adv, all_ret         = [], []

        for buf in self.buffers.values():
            if len(buf) == 0:
                continue
            adv, ret = self._compute_gae(buf)
            # normalize advantages per-buffer before merging
            adv = (adv - adv.mean()) / (adv.std() + 1e-8)

            all_obs.extend(buf.obs)
            all_gobs.extend(buf.global_obs)
            all_actions.extend(buf.actions)
            all_old_lp.extend(buf.log_probs)
            all_adv.extend(adv.tolist())
            all_ret.extend(ret.tolist())
            buf.clear()

        # ── convert to tensors ────────────────────────────────────
        obs_t    = torch.FloatTensor(np.array(all_obs)).to(self.device)
        gobs_t   = torch.FloatTensor(np.array(all_gobs)).to(self.device)
        act_t    = torch.LongTensor(all_actions).to(self.device)
        old_lp_t = torch.FloatTensor(all_old_lp).to(self.device)
        adv_t    = torch.FloatTensor(all_adv).to(self.device)
        ret_t    = torch.FloatTensor(all_ret).to(self.device)

        n          = len(all_obs)
        total_loss = 0.0

        # ── PPO epochs ────────────────────────────────────────────
        for _ in range(self.n_epochs):
            idx = np.random.permutation(n)
            for start in range(0, n, self.batch_size):
                b = idx[start : start + self.batch_size]

                # actor loss — shared weights train on all agents' experience
                new_lp, entropy = self.actor.evaluate(obs_t[b], act_t[b])
                ratio  = torch.exp(new_lp - old_lp_t[b])
                adv    = adv_t[b]
                surr1  = ratio * adv
                surr2  = torch.clamp(ratio, 1 - self.clip_eps, 1 + self.clip_eps) * adv
                actor_loss = -torch.min(surr1, surr2).mean()

                # critic loss — centralized: uses global obs
                value       = self.critic(gobs_t[b])
                critic_loss = nn.MSELoss()(value, ret_t[b])

                # entropy bonus — prevents the shared policy from collapsing
                entropy_loss = -entropy.mean()

                loss = (actor_loss
                        + self.value_coef   * critic_loss
                        + self.entropy_coef * entropy_loss)

                self.actor_opt.zero_grad()
                self.critic_opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.actor.parameters(),  0.5)
                nn.utils.clip_grad_norm_(self.critic.parameters(), 0.5)
                self.actor_opt.step()
                self.critic_opt.step()
                total_loss += loss.item()

        return total_loss

    # ── persistence ───────────────────────────────────────────────

    def save(self, prefix: str):
        """Save actor and critic weights to {prefix}_actor.pt and {prefix}_critic.pt."""
        torch.save(self.actor.state_dict(),  f"{prefix}_actor.pt")
        torch.save(self.critic.state_dict(), f"{prefix}_critic.pt")
        print(f"  Saved → {prefix}_actor.pt + {prefix}_critic.pt")

    def load(self, prefix: str):
        self.actor.load_state_dict(
            torch.load(f"{prefix}_actor.pt",  map_location=self.device))
        self.critic.load_state_dict(
            torch.load(f"{prefix}_critic.pt", map_location=self.device))
        print(f"  Loaded ← {prefix}_actor.pt + {prefix}_critic.pt")
