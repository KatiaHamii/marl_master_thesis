"""
agents/dqn.py — Deep Q-Network agent
All hyperparameters are from config.yaml via utils.load_config()
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from utils import (
    get_device,
    load_config,
    get,
    EpsilonSchedule,
    ReplayBuffer,
    get_device,
)


class _QNetwork(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, hidden: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, action_dim),
        )

    def forward(self, x):
        return self.net(x)


class DQNAgent:
    """
    Deep Q-Network with experience replay and a target network.
    Reads all params from the shared config.
    """

    def __init__(self, cfg: dict, state_dim: int, action_dim: int):
        self.cfg = cfg
        self.action_dim = action_dim
        self.discount = get(cfg, "training", "discount")

        # FrozenLake-specific overrides if present
        env_name = get(cfg, "environment", "name", default="")
        if env_name == "FrozenLake-v1":
            self.batch_size = get(cfg, "dqn", "batch_size")
            self.target_upd = get(
                cfg,
                "frozenlake",
                "dqn_target_update",
                default=get(cfg, "dqn", "target_update"),
            )
            replay_cap = get(
                cfg,
                "frozenlake",
                "dqn_replay_capacity",
                default=get(cfg, "dqn", "replay_capacity"),
            )
        else:
            self.batch_size = get(cfg, "dqn", "batch_size")
            self.target_upd = get(cfg, "dqn", "target_update")
            replay_cap = get(cfg, "dqn", "replay_capacity")

        self.epsilon = EpsilonSchedule(cfg)
        self._ep_count = 0

        hidden = get(cfg, "dqn", "hidden_size")

        # device MUST be set before networks are created and moved
        self.device = get_device(cfg)
        print(f"  DQN running on: {self.device}")

        self.online = _QNetwork(state_dim, action_dim, hidden).to(self.device)
        self.target = _QNetwork(state_dim, action_dim, hidden).to(self.device)
        self.target.load_state_dict(self.online.state_dict())
        self.target.eval()

        self.optimizer = optim.Adam(
            self.online.parameters(), lr=get(cfg, "dqn", "learning_rate")
        )
        self.replay = ReplayBuffer(replay_cap)
        self.loss_fn = nn.MSELoss()

    # ── action selection ─────────────────────────────────────
    def select_action(self, obs: np.ndarray) -> int:
        if self.epsilon.explore():
            return np.random.randint(self.action_dim)
        with torch.no_grad():
            t = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
            return self.online(t).argmax().item()

    # ── store transition ─────────────────────────────────────
    def store(self, obs, action, reward, next_obs, done):
        self.replay.push(obs, action, reward, next_obs, float(done))

    # ── learning update ──────────────────────────────────────
    def update(self) -> float | None:
        if not self.replay.ready(self.batch_size):
            return None
        states, actions, rewards, next_states, dones = self.replay.sample(
            self.batch_size
        )

        # move ALL tensors to device right after sampling
        states = states.to(self.device)
        actions = actions.to(self.device)
        rewards = rewards.to(self.device)
        next_states = next_states.to(self.device)
        dones = dones.to(self.device)

        current_q = self.online(states).gather(1, actions.unsqueeze(1)).squeeze()
        with torch.no_grad():
            max_next = self.target(next_states).max(1)[0]
            target_q = rewards + self.discount * max_next * (1 - dones)

        loss = self.loss_fn(current_q, target_q)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def end_episode(self):
        self.epsilon.step()
        self._ep_count += 1
        if self._ep_count % self.target_upd == 0:
            self.target.load_state_dict(self.online.state_dict())
