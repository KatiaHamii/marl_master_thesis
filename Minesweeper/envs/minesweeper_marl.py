"""
envs/minesweeper_marl.py — 2-agent cooperative Minesweeper
===========================================================
PettingZoo AEC API. Two agents take turns revealing cells on a shared board.
Cooperative goal: reveal all safe cells without hitting a mine.

Observation per agent (flat vector):
  board_state          (pad_cells floats): hidden=-1.0, revealed=count/8.0
  partner_last_action  (pad_cells one-hot)
  → total obs_size = 2 * pad_cells

Action space: Discrete(n_cells) — choose any cell index to reveal

Reward structure (shared — both agents receive the same reward each step):
  +0.5   reveal a new safe cell
  +2.0   win bonus (all safe cells revealed, episode ends)
  -1.0   hit a mine (episode ends)
  -0.05  reveal an already-revealed cell (wasted move)
  -0.01  per step (encourages efficiency)

Compatible with the same PettingZoo / SuperSuit / SB3 stack used for FrozenLake.
"""

import numpy as np
import pygame
from gymnasium import spaces
from pettingzoo import AECEnv
from pettingzoo.utils import wrappers
from pettingzoo.utils.agent_selector import agent_selector

# classic Minesweeper number colours
_NUM_COLORS = {
    1: (30,  80,  220),   # blue
    2: (30,  150,  30),   # green
    3: (200,  30,  30),   # red
    4: (0,    0,  130),   # dark blue
    5: (130,   0,   0),   # dark red
    6: (0,   130, 130),   # teal
    7: (20,   20,  20),   # near-black
    8: (120, 120, 120),   # grey
}


def env_creator(cfg: dict = None, render_mode: str = None):
    """Factory — returns a wrapped PettingZoo AEC env."""
    raw = MinesweeperMARLEnv(cfg=cfg, render_mode=render_mode)
    raw = wrappers.OrderEnforcingWrapper(raw)
    return raw


class MinesweeperMARLEnv(AECEnv):
    """
    2-agent cooperative Minesweeper following PettingZoo AEC API.

    Board encoding (internal _board array):
      -1  = hidden (not yet revealed)
       0  = revealed, 0 adjacent mines
      1–8 = revealed, N adjacent mines
      -2  = mine that was hit (terminal marker)

    AEC = Agent Environment Cycle:
      agents take turns one at a time.
      Turn order: agent_0 → agent_1 → agent_0 → ...

    Curriculum support:
      pad_cells fixes the observation vector size across board sizes so
      network weights can be transferred between curriculum stages without
      re-initialisation (same pattern as FrozenLake pad_tiles).
    """

    metadata = {
        "render_modes": ["human", "ansi"],
        "name": "minesweeper_marl_v0",
        "is_parallelizable": True,
    }

    def __init__(self, cfg: dict = None, render_mode: str = None):
        super().__init__()
        cfg = cfg or {}
        ms = cfg.get("minesweeper", {})

        self.rows        = ms.get("rows",         5)
        self.cols        = ms.get("cols",         5)
        self.n_mines     = ms.get("n_mines",      3)
        self.max_steps   = ms.get("max_steps",  200)
        self.r_safe      = ms.get("reward_safe",  0.5)
        self.r_win       = ms.get("reward_win",   2.0)
        self.r_mine      = ms.get("reward_mine", -1.0)
        self.r_waste     = ms.get("reward_waste",-0.05)
        self.r_step      = ms.get("reward_step", -0.01)
        self.render_mode = render_mode

        self.n_cells = self.rows * self.cols
        # pad_cells: fixed obs size for curriculum weight transfer across board sizes
        self.pad_cells = ms.get("pad_cells", self.n_cells)

        self.agents          = ["agent_0", "agent_1"]
        self.possible_agents = self.agents[:]

        # obs = board_state (pad_cells) + partner_last_action one-hot (pad_cells)
        obs_size = 2 * self.pad_cells
        self.observation_spaces = {
            a: spaces.Box(-1.0, 1.0, shape=(obs_size,), dtype=np.float32)
            for a in self.agents
        }
        self.action_spaces = {
            a: spaces.Discrete(self.n_cells)
            for a in self.agents
        }

        # internal state — initialised properly in reset()
        self._mines: set         = set()
        self._board: np.ndarray  = None   # shape (n_cells,), dtype int8
        self._last_action        = {"agent_0": 0, "agent_1": 0}
        self.last_step_reward: float = 0.0  # immediate reward, read by trainer after step()
        self.safe_revealed: int  = 0        # public: safe cells uncovered so far
        self.mine_hit: bool      = False    # public: True when episode ends on a mine

        # pygame rendering
        self.cell_px      = 64
        self.render_fps   = 4    # settable from outside: env.render_fps = 10
        self.episode_num  = 0    # current episode index, set by runner
        self.episode_total = 0   # total episodes to render, set by runner
        self.window       = None
        self.clock        = None
        self._font        = None

    # ── board helpers ───────────────────────────────────────────

    def _reveal_cascade(self, start: int) -> int:
        """
        Reveal `start` and, if it has 0 adjacent mines, flood-fill all
        connected hidden safe cells (standard Minesweeper auto-reveal).
        Returns the number of newly revealed cells.
        """
        queue   = [start]
        visited = set()
        newly   = 0

        while queue:
            cell = queue.pop()
            if cell in visited or self._board[cell] >= 0 or cell in self._mines:
                continue
            visited.add(cell)
            count = self._adjacent_mines(cell)
            self._board[cell]  = count
            self.safe_revealed += 1
            newly += 1

            if count == 0:   # no adjacent mines → auto-reveal all hidden neighbours
                r, c = divmod(cell, self.cols)
                for dr in (-1, 0, 1):
                    for dc in (-1, 0, 1):
                        if dr == 0 and dc == 0:
                            continue
                        nr, nc = r + dr, c + dc
                        if 0 <= nr < self.rows and 0 <= nc < self.cols:
                            nb = nr * self.cols + nc
                            if nb not in visited and self._board[nb] == -1:
                                queue.append(nb)
        return newly

    def action_mask(self) -> np.ndarray:
        """
        Boolean mask of valid actions: True = cell is still hidden (can be revealed).
        Passed to the actor so it never wastes a move on an already-revealed cell.
        Shape: (n_cells,)
        """
        return np.array([self._board[c] == -1 for c in range(self.n_cells)], dtype=bool)

    def _adjacent_mines(self, cell: int) -> int:
        """Count mines in the 8 neighbours of `cell`."""
        r, c = divmod(cell, self.cols)
        count = 0
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                if dr == 0 and dc == 0:
                    continue
                nr, nc = r + dr, c + dc
                if 0 <= nr < self.rows and 0 <= nc < self.cols:
                    if (nr * self.cols + nc) in self._mines:
                        count += 1
        return count

    def _board_obs(self) -> np.ndarray:
        """
        Encode board as a float array of length pad_cells.
        Hidden cells = -1.0, revealed cells = adjacent_mine_count / 8.0 ∈ [0, 1].
        Cells beyond n_cells (padding) stay at -1.0 (treated as hidden / unknown).
        """
        obs = np.full(self.pad_cells, -1.0, dtype=np.float32)
        for cell in range(self.n_cells):
            if self._board[cell] >= 0:
                obs[cell] = self._board[cell] / 8.0
        return obs

    def _make_obs(self, agent: str) -> np.ndarray:
        """
        Observation for `agent`:
          [ board_state | partner_last_action_onehot ]
        """
        partner  = "agent_1" if agent == "agent_0" else "agent_0"
        board    = self._board_obs()
        last_act = np.zeros(self.pad_cells, dtype=np.float32)
        last_act[self._last_action[partner]] = 1.0
        return np.concatenate([board, last_act])

    # ── PettingZoo AEC API ──────────────────────────────────────

    def observation_space(self, agent: str):
        return self.observation_spaces[agent]

    def action_space(self, agent: str):
        return self.action_spaces[agent]

    def observe(self, agent: str) -> np.ndarray:
        return self._make_obs(agent)

    def reset(self, seed=None, options=None):
        rng       = np.random.default_rng(seed)
        mine_idx  = rng.choice(self.n_cells, size=self.n_mines, replace=False)
        self._mines       = set(int(m) for m in mine_idx)
        self._board       = np.full(self.n_cells, -1, dtype=np.int8)
        self._last_action = {"agent_0": 0, "agent_1": 0}

        self._step_count      = 0
        self.safe_revealed    = 0
        self._total_safe      = self.n_cells - self.n_mines
        self.last_step_reward = 0.0
        self.mine_hit         = False

        self.agents        = self.possible_agents[:]
        self._agent_selector   = agent_selector(self.agents)
        self.agent_selection   = self._agent_selector.next()

        self.rewards             = {a: 0.0 for a in self.agents}
        self._cumulative_rewards = {a: 0.0 for a in self.agents}
        self.terminations        = {a: False for a in self.agents}
        self.truncations         = {a: False for a in self.agents}
        self.infos               = {a: {} for a in self.agents}

        return {a: self._make_obs(a) for a in self.agents}, self.infos

    def step(self, action: int):
        agent = self.agent_selection

        if self.terminations[agent] or self.truncations[agent]:
            self._was_dead_step(None)
            return

        # reset this agent's cumulative so last() returns only this step's reward
        self._cumulative_rewards[agent] = 0

        self._step_count       += 1
        self._last_action[agent] = action
        reward = self.r_step

        if action in self._mines:
            # ── hit a mine ───────────────────────────────────────
            reward += self.r_mine
            self._board[action] = -2                         # mark exploded cell
            self.mine_hit       = True
            self.terminations   = {a: True for a in self.agents}

        elif self._board[action] >= 0:
            # ── already revealed — wasted move ───────────────────
            reward += self.r_waste

        else:
            # ── new safe cell revealed (with auto-cascade on 0-cells) ────────
            newly = self._reveal_cascade(action)   # reveals 1+ cells
            reward += self.r_safe * newly          # reward scales with cells cleared

            if self.safe_revealed == self._total_safe:
                # all safe cells uncovered → win
                reward += self.r_win
                self.terminations = {a: True for a in self.agents}

        if self._step_count >= self.max_steps:
            self.truncations = {a: True for a in self.agents}

        # shared reward: both agents receive exactly the same signal
        self.last_step_reward           = reward
        self.rewards                    = {a: reward for a in self.agents}
        self.infos                      = {a: {"step_reward": reward} for a in self.agents}

        self.agent_selection = self._agent_selector.next()
        self._accumulate_rewards()

    # ── rendering ───────────────────────────────────────────────

    def render(self):
        if self.render_mode == "human":
            self._render_pygame()
        elif self.render_mode == "ansi":
            self._render_ansi()

    def _render_ansi(self):
        symbols = {-1: "·", -2: "X"}
        header  = "   " + " ".join(f"{c}" for c in range(self.cols))
        lines   = [header]
        for r in range(self.rows):
            row = [f"{r} "]
            for c in range(self.cols):
                v = int(self._board[r * self.cols + c])
                row.append(symbols.get(v, str(v)))
            lines.append(" ".join(row))
        print("\n".join(lines))
        safe_left = self._total_safe - self.safe_revealed
        print(f"  mines={self.n_mines}  safe_left={safe_left}  step={self._step_count}\n")

    def _render_pygame(self):
        px    = self.cell_px
        bar_h = 52                            # bottom info bar height (two text rows)
        win_w = max(self.cols * px, 440)      # minimum 440 px so info text never clips

        if self.window is None:
            pygame.init()
            pygame.font.init()
            h = self.rows * px + bar_h
            self.window = pygame.display.set_mode((win_w, h))
            pygame.display.set_caption("Minesweeper MARL — 2 agents")
            self.clock  = pygame.time.Clock()
            self._font  = pygame.font.SysFont("monospace", int(px * 0.42), bold=True)
            self._sfont = pygame.font.SysFont("monospace", 11)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.close()
                return

        self.window.fill((25, 25, 25))

        # ── draw cells ───────────────────────────────────────────
        for r in range(self.rows):
            for c in range(self.cols):
                cell = r * self.cols + c
                v    = int(self._board[cell])
                x, y = c * px, r * px

                if v == -2:                          # exploded mine
                    bg = (190, 50, 50)
                elif v == -1:                        # hidden
                    bg = (90, 95, 100)
                else:                                # revealed
                    bg = (195, 210, 215) if v == 0 else (185, 200, 205)

                rect = pygame.Rect(x + 1, y + 1, px - 2, px - 2)
                pygame.draw.rect(self.window, bg, rect, border_radius=4)

                if v > 0:
                    color = _NUM_COLORS.get(v, (20, 20, 20))
                    txt = self._font.render(str(v), True, color)
                    tw, th = txt.get_size()
                    self.window.blit(txt, (x + (px - tw) // 2, y + (px - th) // 2))
                elif v == -2:
                    txt = self._font.render("✕", True, (255, 240, 240))
                    tw, th = txt.get_size()
                    self.window.blit(txt, (x + (px - tw) // 2, y + (px - th) // 2))

                # show mine positions when episode is over
                if (any(self.terminations.values()) or any(self.truncations.values())):
                    if cell in self._mines and v != -2:
                        pygame.draw.rect(self.window, (160, 60, 60), rect, border_radius=4)
                        txt = self._font.render("M", True, (255, 200, 200))
                        tw, th = txt.get_size()
                        self.window.blit(txt, (x + (px - tw) // 2, y + (px - th) // 2))

        # ── draw agent last-action markers ────────────────────────
        _AGENT_COLORS = {"agent_0": (60, 120, 230), "agent_1": (230, 120, 50)}
        for agent, color in _AGENT_COLORS.items():
            last = self._last_action[agent]
            ar, ac = divmod(last, self.cols)
            cx = ac * px + px // 2
            cy = ar * px + px // 2
            pygame.draw.circle(self.window, color,   (cx, cy), px // 7 + 3)
            pygame.draw.circle(self.window, (255, 255, 255), (cx, cy), px // 7 + 3, 2)

        # ── info bar (two rows) ───────────────────────────────────
        bar_y    = self.rows * px
        bar_rect = pygame.Rect(0, bar_y, win_w, bar_h)
        pygame.draw.rect(self.window, (40, 42, 46), bar_rect)

        safe_left = self._total_safe - self.safe_revealed
        if self.mine_hit:
            status       = "LOST  —  mine hit"
            status_color = (230, 100, 100)
        elif safe_left == 0:
            status       = "WIN!  —  all safe cells cleared"
            status_color = (100, 220, 100)
        elif any(self.truncations.values()):
            status       = "LOST  —  out of steps"
            status_color = (220, 170, 60)
        else:
            status       = ""
            status_color = (190, 195, 200)

        # row 1: episode + board stats
        ep_part   = f"Ep {self.episode_num}/{self.episode_total}  |  " if self.episode_total > 0 else ""
        row1 = f"  {ep_part}mines:{self.n_mines}  safe left:{safe_left}  step:{self._step_count}"
        self.window.blit(self._sfont.render(row1, True, (190, 195, 200)), (4, bar_y + 6))

        # row 2: agent legend + result status
        row2 = "  \u25cf blue (agent_0)   \u25cf orange (agent_1)"
        self.window.blit(self._sfont.render(row2, True, (190, 195, 200)), (4, bar_y + 24))
        if status:
            stxt = self._sfont.render(status, True, status_color)
            self.window.blit(stxt, (win_w - stxt.get_width() - 8, bar_y + 24))

        pygame.display.flip()
        self.clock.tick(self.render_fps)

    def close(self):
        if self.window is not None:
            pygame.quit()
            self.window = None
