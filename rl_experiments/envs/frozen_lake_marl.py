"""
envs/frozen_lake_marl.py — 2-agent cooperative FrozenLake
==========================================================
Implements the PettingZoo AEC (Agent Environment Cycle) API.

Two agents must BOTH reach the goal tile to get a reward.
Each agent observes:
  - its own position      (one-hot, n_tiles)
  - partner's position    (one-hot, n_tiles)
  - goal position         (one-hot, n_tiles)
  → total obs size: 3 * n_tiles

Reward structure:
  +1.0   both agents on goal tile
  -0.3   either agent falls in hole
  -0.005 every step (encourages efficiency)
  +0.02  moving closer to goal

Compatible with SuperSuit's ss.pettingzoo_env_to_vec_env_v1()
so Stable-Baselines3 PPO can consume it directly.
"""

import numpy as np
import pygame
from gymnasium import spaces
from gymnasium.envs.toy_text.frozen_lake import generate_random_map
from pettingzoo import AECEnv
from pettingzoo.utils import wrappers
from pettingzoo.utils.agent_selector import agent_selector


def env_creator(cfg: dict = None, render_mode: str = None):
    """
    Factory function — returns a wrapped PettingZoo env.
    Pass your config dict to read all params from config.yaml.
    """
    raw = FrozenLakeMARLEnv(cfg=cfg, render_mode=render_mode)
    # OrderEnforcingWrapper ensures agents take turns correctly
    raw = wrappers.OrderEnforcingWrapper(raw)
    return raw


class FrozenLakeMARLEnv(AECEnv):
    """
    2-agent cooperative FrozenLake following PettingZoo AEC API.

    AEC = Agent Environment Cycle:
      agents take turns one at a time (not simultaneous).
      After both have acted, the environment steps forward.
    """

    metadata = {
        "render_modes": ["human", "rgb_array"],
        "name": "frozen_lake_marl_v0",
        "is_parallelizable": True,
    }

    def __init__(self, cfg: dict = None, render_mode: str = None):
        super().__init__()

        # ── read config ──────────────────────────────────────
        cfg = cfg or {}
        fl = cfg.get("frozenlake", {})

        self.size = fl.get("size", 4)
        self.is_slippery = fl.get("is_slippery", False)
        self.randomise = fl.get("domain_randomisation", True)
        self.r_goal = fl.get("reward_goal", 1.0)
        self.r_hole = fl.get("reward_hole", -0.3)
        self.r_step = fl.get("reward_step", -0.005)
        self.r_closer = fl.get("reward_closer", 0.02)
        self.render_mode = render_mode

        # ── agent setup ──────────────────────────────────────
        self.agents = ["agent_0", "agent_1", "agent_2"]
        self.possible_agents = self.agents[:]
        self.n_tiles = self.size * self.size

        # obs: [own_pos | partner1_pos | partner2_pos | goal_pos | agent_id] all one-hot
        # agent_id: agent_0=[1,0,0], agent_1=[0,1,0], agent_2=[0,0,1]
        # pad_tiles: fixes obs size across map sizes for curriculum weight transfer.
        # defaults to n_tiles (no padding) → no change for non-curriculum training.
        self.pad_tiles = fl.get("pad_tiles", self.n_tiles)
        obs_size = 4 * self.pad_tiles + 3
        self.observation_spaces = {
            a: spaces.Box(0.0, 1.0, shape=(obs_size,), dtype=np.float32)
            for a in self.agents
        }
        # 4 actions: left, down, right, up
        self.action_spaces = {a: spaces.Discrete(4) for a in self.agents}

        # ── map ──────────────────────────────────────────────
        self._map = None
        self._goal = None
        self._holes = None
        self._regenerate_map()

        # ── rendering ────────────────────────────────────────
        self.window = None
        self.clock = None
        self.cell_px = 80

    # ─────────────────────────────────────────────
    #  MAP HELPERS
    # ─────────────────────────────────────────────
    def _regenerate_map(self):
        desc = (
            generate_random_map(size=self.size)
            if self.randomise
            else self._default_map()
        )
        self._map = [list(row) for row in desc]
        self._holes = set()
        for r in range(self.size):
            for c in range(self.size):
                cell = self._map[r][c]
                if isinstance(cell, bytes):
                    cell = cell.decode()
                if cell == "H":
                    self._holes.add(r * self.size + c)
                elif cell == "G":
                    self._goal = r * self.size + c

    def _default_map(self):
        return ["SFFF", "FHFH", "FFFH", "HFFG"]

    def _tile(self, pos: int) -> str:
        r, c = divmod(pos, self.size)
        cell = self._map[r][c]
        return cell.decode() if isinstance(cell, bytes) else cell

    def _manhattan(self, pos: int) -> float:
        r, c = divmod(pos, self.size)
        gr, gc = divmod(self._goal, self.size)
        return abs(r - gr) + abs(c - gc)

    def _step_pos(self, pos: int, action: int) -> int:
        """Move pos by action (0=L,1=D,2=R,3=U), clamp to grid."""
        r, c = divmod(pos, self.size)
        if action == 0:
            c = max(0, c - 1)  # left
        elif action == 1:
            r = min(self.size - 1, r + 1)  # down
        elif action == 2:
            c = min(self.size - 1, c + 1)  # right
        elif action == 3:
            r = max(0, r - 1)  # up
        return r * self.size + c

    # ─────────────────────────────────────────────
    #  PETTINGZOO API
    # ─────────────────────────────────────────────
    def observation_space(self, agent):
        return self.observation_spaces[agent]

    def action_space(self, agent):
        return self.action_spaces[agent]

    def _onehot(self, pos: int) -> np.ndarray:
        # pad_tiles >= n_tiles: extra zeros fill the unused tile slots
        v = np.zeros(self.pad_tiles, dtype=np.float32)
        v[pos] = 1.0
        return v

    def _make_obs(self, agent: str) -> np.ndarray:
        """
        Each agent sees:
          own position      (one-hot, n_tiles)
          partner1 pos      (one-hot, n_tiles)
          partner2 pos      (one-hot, n_tiles)
          goal position     (one-hot, n_tiles)
          agent identity    (one-hot, 3)  →  agent_0=[1,0,0], etc.
        """
        own = self._pos[agent]
        others = self._others(agent)
        idx = self.possible_agents.index(agent)
        agent_id = np.zeros(3, dtype=np.float32)
        agent_id[idx] = 1.0
        return np.concatenate(
            [self._onehot(own)]
            + [self._onehot(self._pos[o]) for o in others]
            + [self._onehot(self._goal), agent_id]
        )

    def _others(self, agent: str) -> list:
        return [a for a in self.possible_agents if a != agent]

    def observe(self, agent: str) -> np.ndarray:
        return self._make_obs(agent)

    def reset(self, seed=None, options=None):
        if self.randomise:
            self._regenerate_map()

        self.agents = self.possible_agents[:]
        self._agent_selector = agent_selector(self.agents)
        self.agent_selection = self._agent_selector.next()

        # pick 3 distinct safe (non-hole, non-goal) tiles spread across the map
        safe_tiles = [
            t for t in range(self.n_tiles)
            if t not in self._holes and t != self._goal
        ]
        # spread them out: pick from beginning, middle, and end of safe tile list
        n = len(safe_tiles)
        start_positions = [
            safe_tiles[0],
            safe_tiles[n // 2],
            safe_tiles[-1],
        ]
        self._pos = {a: start_positions[i] for i, a in enumerate(self.agents)}
        self._prev_dist = {a: self._manhattan(self._pos[a]) for a in self.agents}

        self.rewards = {a: 0.0 for a in self.agents}
        self._cumulative_rewards = {a: 0.0 for a in self.agents}
        self.terminations = {a: False for a in self.agents}
        self.truncations = {a: False for a in self.agents}
        self.infos = {a: {} for a in self.agents}

        self._step_count = 0
        self._max_steps = self.n_tiles * 4  # generous per-episode cap

        observations = {a: self._make_obs(a) for a in self.agents}
        return observations, self.infos

    def step(self, action: int):
        agent = self.agent_selection

        if self.terminations[agent] or self.truncations[agent]:
            self._was_dead_step(None)  # SuperSuit may pass a real action for dead agents
            return

        # ── move ─────────────────────────────────────────────
        old_pos = self._pos[agent]
        new_pos = self._step_pos(old_pos, action)
        self._pos[agent] = new_pos
        self._step_count += 1

        # ── reward shaping ───────────────────────────────────
        reward = self.r_step

        # distance bonus
        new_dist = self._manhattan(new_pos)
        if new_dist < self._prev_dist[agent]:
            reward += self.r_closer
        self._prev_dist[agent] = new_dist

        # hole
        tile = self._tile(new_pos)
        if tile == "H":
            reward += self.r_hole
            self.terminations = {a: True for a in self.agents}

        # agent reaches goal → episode ends jointly, only this agent gets goal reward
        elif new_pos == self._goal:
            reward += self.r_goal
            self.terminations = {a: True for a in self.agents}

        # max steps
        if self._step_count >= self._max_steps:
            self.truncations = {a: True for a in self.agents}

        # only the acting agent gets its own reward (individual, not shared)
        self.rewards = {a: 0.0 for a in self.agents}
        self.rewards[agent] = reward
        self._cumulative_rewards[agent] += reward

        # ── advance turn ─────────────────────────────────────
        self.agent_selection = self._agent_selector.next()
        self._accumulate_rewards()

    # ─────────────────────────────────────────────
    #  RENDERING  (optional, human mode)
    # ─────────────────────────────────────────────
    def render(self):
        if self.render_mode is None:
            return
        if self.window is None:
            pygame.init()
            sz = self.size * self.cell_px
            self.window = pygame.display.set_mode((sz, sz))
            self.clock = pygame.time.Clock()

        COLORS = {
            "F": (200, 220, 255),
            "S": (180, 255, 180),
            "H": (80, 80, 80),
            "G": (255, 215, 0),
        }
        self.window.fill((255, 255, 255))
        for r in range(self.size):
            for c in range(self.size):
                cell = self._map[r][c]
                cell = cell.decode() if isinstance(cell, bytes) else cell
                color = COLORS.get(cell, (200, 200, 200))
                rect = pygame.Rect(
                    c * self.cell_px,
                    r * self.cell_px,
                    self.cell_px - 2,
                    self.cell_px - 2,
                )
                pygame.draw.rect(self.window, color, rect, border_radius=6)

        # draw agents — skip any not in highlight_agents (if set)
        active = getattr(self, "highlight_agents", None)
        for i, (agent, color) in enumerate(
            [("agent_0", (50, 100, 220)), ("agent_1", (220, 80, 50)), ("agent_2", (80, 180, 80))]
        ):
            if active is not None and agent not in active:
                continue
            pos = self._pos[agent]
            r, c = divmod(pos, self.size)
            cx = c * self.cell_px + self.cell_px // 2 + (i - 1) * 10
            cy = r * self.cell_px + self.cell_px // 2
            pygame.draw.circle(self.window, color, (cx, cy), 16)

        pygame.display.flip()
        self.clock.tick(4)

    def close(self):
        if self.window:
            pygame.quit()
            self.window = None
