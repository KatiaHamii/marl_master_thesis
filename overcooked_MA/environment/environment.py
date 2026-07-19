import numpy as np
from config import render_full_map, GridCodes
from .generator import LevelGenerator, compute_min_cycle
from .settings import DELIVERY_REWARD, POT_COOK_TIME, SHAPED_REWARDS

# Actions
UP, DOWN, LEFT, RIGHT, STAY, INTERACT = range(6)
NUM_ACTIONS = 6
_DIRS = {UP: (-1, 0), DOWN: (1, 0), LEFT: (0, -1), RIGHT: (0, 1)}

# Inventory codes (bit-decomposed by ObsPreprocessor, so any small int works)
_INV_CODE = {None: 0, "plate": 1, "ingredient_0": 2, "ingredient_1": 3, "dish": 4}

# 7-channel compact obs, matching networks.ActorCritic's expected layout:
#   ch0 self dir+1, ch1 self inventory, ch2 other dir+1, ch3 other inventory,
#   ch4 static cell type, ch5 dynamic items (pot ingredient count), ch6 extra (pot timer)
NUM_OBS_CHANNELS = 7


class OvercookedEnvironment:
    """Overcooked Environment with support for UED vector-based level generation and template-based level setting."""
    def __init__(self, height=9, width=11, max_steps=200):
        self.height = height
        self.width = width
        self.max_steps = max_steps
        self.grid = None
        self.static_grid = None
        self.current_template = None
        self.num_agents = 2
        self.num_actions = NUM_ACTIONS
        self.obs_shape = (height, width, NUM_OBS_CHANNELS)

        self.agent_pos = {}
        self.agent_dir = {}
        self.agent_inventory = {}
        self.pots = {}
        self.t = 0
        self.total_deliveries = 0

    def set_level_template(self, level_des):
        """Set the level template directly (used for testing or fixed layouts)."""
        self.current_template = level_des

    def reset(self, param_vector=None):
        """Reset the environment and update the map according to the template or UED vector."""
        if param_vector is not None:
            # Variant 2 from the screenshot: on-the-fly generation through passing the vector to reset
            gen = LevelGenerator(height=self.height, width=self.width)
            level_elems = gen.create_level_elements(
                param_vector[0], param_vector[1], param_vector[2], param_vector[3]
            )
            self.current_template = gen.calculate_element_coords(level_elems)

        if self.current_template is None:
            raise ValueError("Map not set! Call set_level_template or pass a param_vector.")

        self.height, self.width = self.current_template.shape
        self.obs_shape = (self.height, self.width, NUM_OBS_CHANNELS)

        self.static_grid = self.current_template.copy()
        self.agent_pos = {}
        self.agent_dir = {}
        self.agent_inventory = {}
        for aid, code in (("agent_0", GridCodes.AGENT_0), ("agent_1", GridCodes.AGENT_1)):
            rows, cols = np.where(self.static_grid == code)
            self.agent_pos[aid] = (int(rows[0]), int(cols[0]))
            self.agent_dir[aid] = DOWN
            self.agent_inventory[aid] = None
            self.static_grid[rows[0], cols[0]] = GridCodes.EMPTY

        self.pots = {
            (int(r), int(c)): {"ingredients": 0, "timer": 0, "cooked": False}
            for r, c in zip(*np.where(self.static_grid == GridCodes.POT))
        }

        self.t = 0
        self.total_deliveries = 0
        self.grid = self._render_grid()

        obs = self._get_obs()
        info = {"cycle_length": compute_min_cycle(self.current_template)}
        return obs, info

    def _in_bounds(self, r, c):
        return 0 <= r < self.height and 0 <= c < self.width

    def _render_grid(self):
        """Static layout with the agents' current positions overlaid, for rendering/inspection."""
        g = self.static_grid.copy()
        g[self.agent_pos["agent_0"]] = GridCodes.AGENT_0
        g[self.agent_pos["agent_1"]] = GridCodes.AGENT_1
        return g

    def _get_obs(self):
        H, W = self.height, self.width
        static_ch = self.static_grid.astype(np.float32)

        dir_ch, inv_ch = {}, {}
        for aid in ("agent_0", "agent_1"):
            d = np.zeros((H, W), dtype=np.float32)
            d[self.agent_pos[aid]] = self.agent_dir[aid] + 1
            dir_ch[aid] = d

            v = np.zeros((H, W), dtype=np.float32)
            v[self.agent_pos[aid]] = _INV_CODE[self.agent_inventory[aid]]
            inv_ch[aid] = v

        dyn_ch = np.zeros((H, W), dtype=np.float32)
        extra_ch = np.zeros((H, W), dtype=np.float32)
        for (r, c), pot in self.pots.items():
            dyn_ch[r, c] = pot["ingredients"]
            extra_ch[r, c] = pot["timer"]

        obs = {}
        for aid, other in (("agent_0", "agent_1"), ("agent_1", "agent_0")):
            obs[aid] = np.stack(
                [dir_ch[aid], inv_ch[aid], dir_ch[other], inv_ch[other], static_ch, dyn_ch, extra_ch],
                axis=-1,
            )
        return obs

    def step(self, actions):
        """Step both agents. `actions` is {"agent_0": int, "agent_1": int} in [0, NUM_ACTIONS)."""
        tentative = {}
        for aid in ("agent_0", "agent_1"):
            a = actions[aid]
            r, c = self.agent_pos[aid]
            if a in _DIRS:
                self.agent_dir[aid] = a
                dr, dc = _DIRS[a]
                nr, nc = r + dr, c + dc
                if self._in_bounds(nr, nc) and self.static_grid[nr, nc] == GridCodes.EMPTY:
                    tentative[aid] = (nr, nc)
                else:
                    tentative[aid] = (r, c)
            else:
                tentative[aid] = (r, c)

        t0, t1 = tentative["agent_0"], tentative["agent_1"]
        collide = t0 == t1
        swap = t0 == self.agent_pos["agent_1"] and t1 == self.agent_pos["agent_0"]
        if collide or swap:
            tentative["agent_0"] = self.agent_pos["agent_0"]
            tentative["agent_1"] = self.agent_pos["agent_1"]
        self.agent_pos = tentative

        rewards = {"agent_0": 0.0, "agent_1": 0.0}
        deliveries_this_step = 0

        for aid in ("agent_0", "agent_1"):
            if actions[aid] != INTERACT:
                continue
            r, c = self.agent_pos[aid]
            dr, dc = _DIRS[self.agent_dir[aid]]
            fr, fc = r + dr, c + dc
            if not self._in_bounds(fr, fc):
                continue
            cell = self.static_grid[fr, fc]
            inv = self.agent_inventory[aid]

            if cell in (GridCodes.INGREDIENT_0, GridCodes.INGREDIENT_1) and inv is None:
                self.agent_inventory[aid] = "ingredient_0" if cell == GridCodes.INGREDIENT_0 else "ingredient_1"
            elif cell == GridCodes.PLATE_PILE and inv is None:
                self.agent_inventory[aid] = "plate"
                rewards[aid] += SHAPED_REWARDS["PLATE_PICKUP"]
            elif cell == GridCodes.POT:
                pot = self.pots[(fr, fc)]
                if inv in ("ingredient_0", "ingredient_1") and pot["timer"] == 0 and not pot["cooked"] and pot["ingredients"] < 3:
                    pot["ingredients"] += 1
                    rewards[aid] += SHAPED_REWARDS["PLACEMENT_IN_POT"]
                    self.agent_inventory[aid] = None
                    if pot["ingredients"] == 3:
                        pot["timer"] = POT_COOK_TIME
                        rewards[aid] += SHAPED_REWARDS["POT_START_COOKING"]
                elif inv == "plate" and pot["cooked"]:
                    self.agent_inventory[aid] = "dish"
                    rewards[aid] += SHAPED_REWARDS["DISH_PICKUP"]
                    pot["ingredients"] = 0
                    pot["cooked"] = False
            elif cell == GridCodes.GOAL and inv == "dish":
                rewards[aid] += DELIVERY_REWARD
                self.agent_inventory[aid] = None
                self.total_deliveries += 1
                deliveries_this_step += 1

        for pot in self.pots.values():
            if pot["timer"] > 0:
                pot["timer"] -= 1
                if pot["timer"] == 0:
                    pot["cooked"] = True

        self.t += 1
        terminated = False
        truncated = self.t >= self.max_steps
        self.grid = self._render_grid()

        obs = self._get_obs()
        info = {"deliveries": deliveries_this_step}
        return obs, rewards, terminated, truncated, info

    def render(self, save_path=None, show=False):
        """Visualizes the current map using PIL"""
        if self.grid is None:
            return None

        img = render_full_map(self.grid)

        if save_path:
            img.save(save_path)

        if show:
            img.show()

        return img
