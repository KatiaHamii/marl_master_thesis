"""
agents/ppo.py — Proximal Policy Optimisation agent
===================================================
Key differences from DQN:

  Policy-based:   learns π(a|s) directly, not Q(s,a)
  On-policy:      no replay buffer — collects a fresh batch each update
  Actor-Critic:   two heads on one network — actor (policy) + critic (value)
  Clipped loss:   prevents policy from changing too much in one step
  GAE:            Generalised Advantage Estimation — smoother advantage signal

All hyperparameters come from config.yaml.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical

from utils import get, get_device, EpsilonSchedule


# ─────────────────────────────────────────────
#  NETWORK  — shared trunk + two heads
# ─────────────────────────────────────────────
class _ActorCritic(nn.Module):
    """
    One network, two outputs:
      actor_head  → logits → softmax → action probabilities
      critic_head → single float → V(s)

    Shared trunk lets both heads learn useful state representations together.
    """

    def __init__(self, state_dim: int, action_dim: int, hidden: int):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(state_dim, hidden),
            nn.Tanh(),  # Tanh works better than ReLU for PPO —
            nn.Linear(hidden, hidden),  # smoother gradients through the policy
            nn.Tanh(),
        )
        self.actor_head = nn.Linear(hidden, action_dim)  # → logits
        self.critic_head = nn.Linear(hidden, 1)  # → V(s)

    def forward(self, x):
        features = self.trunk(x)
        logits = self.actor_head(features)
        value = self.critic_head(features).squeeze(-1)
        return logits, value

    def get_action(self, x):
        """Sample an action and return (action, log_prob, value)."""
        logits, value = self.forward(x)
        dist = Categorical(logits=logits)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        return action, log_prob, value

    def evaluate(self, x, actions):
        """Re-evaluate stored actions under the current policy."""
        logits, value = self.forward(x)
        dist = Categorical(logits=logits)
        log_prob = dist.log_prob(actions)
        entropy = dist.entropy()
        return log_prob, value, entropy


# ─────────────────────────────────────────────
#  ROLLOUT BUFFER
#  PPO collects a full batch of on-policy
#  experience BEFORE doing any gradient updates.
#  (opposite of DQN which updates every step)
# ─────────────────────────────────────────────
class RolloutBuffer:
    def __init__(self):
        self.clear()

    def clear(self):
        self.states = []
        self.actions = []
        self.log_probs = []
        self.rewards = []
        self.values = []
        self.dones = []

    def push(self, state, action, log_prob, reward, value, done):
        self.states.append(state)
        self.actions.append(action)
        self.log_probs.append(log_prob)
        self.rewards.append(reward)
        self.values.append(value)
        self.dones.append(done)

    def __len__(self):
        return len(self.rewards)


# ─────────────────────────────────────────────
#  PPO AGENT
# ─────────────────────────────────────────────
class PPOAgent:
    """
    Proximal Policy Optimisation.
    Compatible with run_experiments.py — same interface as Q-Learning and DQN.
    """

    def __init__(self, cfg: dict, state_dim: int, action_dim: int):
        self.cfg = cfg
        self.action_dim = action_dim
        self.discount = get(cfg, "training", "discount")
        self.device = get_device(cfg)
        print(f"  PPO running on: {self.device}")

        # PPO-specific hyperparameters
        self.lr = get(cfg, "ppo", "learning_rate")
        self.epochs = get(cfg, "ppo", "epochs")  # update passes per batch
        self.batch_size = get(cfg, "ppo", "batch_size")
        self.clip_eps = get(cfg, "ppo", "clip_epsilon")  # the "proximal" in PPO
        self.value_coef = get(cfg, "ppo", "value_coef")  # critic loss weight
        self.entropy_coef = get(cfg, "ppo", "entropy_coef")  # exploration bonus
        self.gae_lambda = get(cfg, "ppo", "gae_lambda")  # GAE smoothing
        self.update_freq = get(cfg, "ppo", "update_every")  # steps between updates

        hidden = get(cfg, "ppo", "hidden_size")
        self.net = _ActorCritic(state_dim, action_dim, hidden).to(self.device)
        self.optimizer = optim.Adam(self.net.parameters(), lr=self.lr)

        self.buffer = RolloutBuffer()
        self._steps = 0

        # PPO doesn't use epsilon-greedy (it's stochastic by nature)
        # but run_experiments.py expects self.epsilon — we give it a dummy
        self.epsilon = _DummyEpsilon()

    # ── action selection ─────────────────────────────────────
    def select_action(self, obs: np.ndarray) -> int:
        """
        Sample from the policy distribution.
        PPO explores naturally through stochasticity — no epsilon needed.
        """
        t = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
        with torch.no_grad():
            action, log_prob, value = self.net.get_action(t)
        # stash for buffer — retrieved in store()
        self._last_log_prob = log_prob
        self._last_value = value
        return action.item()

    # ── store transition ─────────────────────────────────────
    def store(self, obs, action, reward, next_obs, done):
        self.buffer.push(
            obs,
            action,
            self._last_log_prob.item(),
            reward,
            self._last_value.item(),
            float(done),
        )
        self._steps += 1

    # ── learning update ──────────────────────────────────────
    def update(self) -> float | None:
        """
        Called every step by the runner, but PPO only actually updates
        every `update_every` steps — it needs a full batch first.
        """
        if self._steps % self.update_freq != 0 or len(self.buffer) == 0:
            return None
        return self._ppo_update()

    def _compute_gae(self, next_value: float) -> tuple:
        """
        Generalised Advantage Estimation (GAE).

        Plain advantage = G_t - V(s_t)  where G_t is the discounted return.
        GAE smooths this with an exponential weighted average, controlled
        by lambda.  lambda=1 → full Monte Carlo (high variance, low bias).
                    lambda=0 → one-step TD (low variance, high bias).
        0.95 is the sweet spot used in most PPO papers.
        """
        rewards = self.buffer.rewards
        values = self.buffer.values
        dones = self.buffer.dones
        n = len(rewards)
        advantages = np.zeros(n, dtype=np.float32)
        last_gae = 0.0

        for t in reversed(range(n)):
            next_val = next_value if t == n - 1 else values[t + 1]
            not_done = 1.0 - dones[t]
            delta = rewards[t] + self.discount * next_val * not_done - values[t]
            last_gae = delta + self.discount * self.gae_lambda * not_done * last_gae
            advantages[t] = last_gae

        returns = advantages + np.array(values, dtype=np.float32)
        return advantages, returns

    def _ppo_update(self) -> float:
        """
        The core PPO update — run multiple epochs over the collected batch.

        Clipped objective:
            L = min( r_t * A_t,  clip(r_t, 1-ε, 1+ε) * A_t )
            where r_t = π_new(a|s) / π_old(a|s)

        This prevents the new policy from moving too far from the old one.
        """
        # Bootstrap value for the last state
        last_state = (
            torch.FloatTensor(self.buffer.states[-1]).unsqueeze(0).to(self.device)
        )
        with torch.no_grad():
            _, last_value, _ = self.net.evaluate(
                last_state, torch.zeros(1, dtype=torch.long).to(self.device)
            )
        next_value = last_value.item()

        advantages, returns = self._compute_gae(next_value)

        # Normalise advantages — reduces variance, stabilises training
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # Convert buffer to tensors
        states_t = torch.FloatTensor(np.array(self.buffer.states)).to(self.device)
        actions_t = torch.LongTensor(self.buffer.actions).to(self.device)
        old_lp_t = torch.FloatTensor(self.buffer.log_probs).to(self.device)
        advantages_t = torch.FloatTensor(advantages).to(self.device)
        returns_t = torch.FloatTensor(returns).to(self.device)

        total_loss = 0.0
        n = len(self.buffer)

        # Multiple epochs over the same batch — the "proximal" constraint
        # keeps this safe (without clipping, multiple epochs would overfit)
        for _ in range(self.epochs):
            # Mini-batch updates within each epoch
            indices = np.random.permutation(n)
            for start in range(0, n, self.batch_size):
                idx = indices[start : start + self.batch_size]

                new_lp, value, entropy = self.net.evaluate(
                    states_t[idx], actions_t[idx]
                )

                # Probability ratio: how much has the policy changed?
                ratio = torch.exp(new_lp - old_lp_t[idx])

                # Clipped surrogate objective
                adv = advantages_t[idx]
                surr1 = ratio * adv
                surr2 = torch.clamp(ratio, 1 - self.clip_eps, 1 + self.clip_eps) * adv
                actor_loss = -torch.min(surr1, surr2).mean()

                # Critic loss — mean squared error between predicted and actual returns
                critic_loss = nn.MSELoss()(value, returns_t[idx])

                # Entropy bonus — encourages exploration by penalising certainty
                entropy_loss = -entropy.mean()

                loss = (
                    actor_loss
                    + self.value_coef * critic_loss
                    + self.entropy_coef * entropy_loss
                )

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.net.parameters(), 0.5)
                self.optimizer.step()
                total_loss += loss.item()

        self.buffer.clear()
        return total_loss

    def end_episode(self):
        pass  # epsilon decay not needed — PPO is naturally stochastic


# ─────────────────────────────────────────────
#  DUMMY EPSILON
#  run_experiments.py calls float(agent.epsilon)
#  and agent.epsilon.value for logging.
#  PPO doesn't use epsilon, but we need the
#  attribute to exist for the shared runner.
# ─────────────────────────────────────────────
class _DummyEpsilon:
    value = 0.0

    def explore(self):
        return False

    def step(self):
        pass

    def __float__(self):
        return 0.0
