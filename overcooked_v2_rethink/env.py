"""
OvercookedV2 — 7-channel compact observation.

Observation encoding (H × W × 7 per agent):
  ch 0  self direction+1 at own cell (1=UP 2=DOWN 3=RIGHT 4=LEFT), 0 elsewhere
  ch 1  self inventory (bit-packed DynamicObject) at own cell, 0 elsewhere
  ch 2  other-agent direction+1 at their cell, 0 elsewhere
  ch 3  other-agent inventory at their cell, 0 elsewhere
  ch 4  static cell type (StaticObject int: 0=empty 1=wall 4=goal 5=pot …)
  ch 5  dynamic items (bit-packed DynamicObject per cell)
  ch 6  extra: pot cook-timer at pot cells; recipe int at recipe-indicator cell

References:
  - OvercookedV2 environment: arXiv 2503.17821
  - 7-channel grid encoding inspired by: arXiv 2401.05860
"""

from functools import partial
from typing import Dict, List, Optional, Tuple, Union

import chex
import jax
import jax.numpy as jnp
from flax import struct
from jax import lax

from .common import (
    ACTION_TO_DIRECTION,
    Actions,
    Agent,
    Direction,
    DynamicObject,
    Position,
    StaticObject,
)
from .environments import spaces
from .environments.multi_agent_env import MultiAgentEnv
from .layouts import Layout, overcooked_v2_layouts
from .settings import (
    DELIVERY_REWARD,
    INDICATOR_ACTIVATION_COST,
    INDICATOR_ACTIVATION_TIME,
    POT_COOK_TIME,
    SHAPED_REWARDS,
)
from .utils import compute_enclosed_spaces, tree_select


@chex.dataclass
class State:
    agents: Agent
    # H × W × 3:  [:,:,0] static objects  [:,:,1] dynamic items  [:,:,2] extra info
    grid: chex.Array
    time: chex.Array
    terminal: bool
    recipe: int
    new_correct_delivery: bool = False


NUM_OBS_CHANNELS = 7


class OvercookedV2(MultiAgentEnv):
    """
    OvercookedV2 with a fixed 7-channel compact observation.

    Each agent receives an H×W×7 tensor (or a cropped view if agent_view_size
    is set).  All game mechanics from arXiv 2503.17821 are preserved.
    """

    def __init__(
        self,
        layout: Union[str, Layout] = "cramped_room",
        max_steps: int = 400,
        agent_view_size: Optional[int] = None,
        random_reset: bool = False,
        random_agent_positions: bool = False,
        start_cooking_interaction: bool = False,
        negative_rewards: bool = False,
        sample_recipe_on_delivery: bool = False,
    ):
        if isinstance(layout, str):
            if layout not in overcooked_v2_layouts:
                raise ValueError(
                    f"Unknown layout '{layout}'. Available: {list(overcooked_v2_layouts)}"
                )
            layout = overcooked_v2_layouts[layout]

        super().__init__(num_agents=len(layout.agent_positions))

        self.layout = layout
        self.height = layout.height
        self.width = layout.width
        self.agents = [f"agent_{i}" for i in range(self.num_agents)]
        self.action_set = jnp.array(list(Actions))
        self.max_steps = max_steps
        self.agent_view_size = agent_view_size
        self.possible_recipes = jnp.array(layout.possible_recipes, dtype=jnp.int32)
        self.random_reset = random_reset
        self.random_agent_positions = random_agent_positions
        self.start_cooking_interaction = jnp.array(start_cooking_interaction, dtype=jnp.bool_)
        self.negative_rewards = negative_rewards
        self.sample_recipe_on_delivery = jnp.array(sample_recipe_on_delivery, dtype=jnp.bool_)
        self.enclosed_spaces = compute_enclosed_spaces(
            layout.static_objects == StaticObject.EMPTY
        )

        if agent_view_size is not None:
            view = agent_view_size * 2 + 1
            self.obs_shape = (
                min(self.height, view),
                min(self.width, view),
                NUM_OBS_CHANNELS,
            )
        else:
            self.obs_shape = (self.height, self.width, NUM_OBS_CHANNELS)

    def set_layout(self, layout_dict: Dict) -> None:
        """
        Update the environment's layout (for UED curriculum).

        Applies mutations to the current layout by updating walls and object positions.

        Args:
            layout_dict: Dictionary with keys like 'walls', 'goal', 'pot', etc.
        """
        # Update walls
        if 'walls' in layout_dict:
            self.layout.walls = set(layout_dict['walls'])

        # Update object positions (if provided)
        if 'goal' in layout_dict:
            self.layout.goal_pos = [layout_dict['goal']]
        if 'pot' in layout_dict:
            self.layout.pot_pos = [layout_dict['pot']]
        if 'onion_pile' in layout_dict:
            self.layout.onion_pile_pos = [layout_dict['onion_pile']]
        if 'plate_pile' in layout_dict:
            self.layout.plate_pile_pos = [layout_dict['plate_pile']]

    # ------------------------------------------------------------------ reset

    def reset(self, key: chex.PRNGKey, static_objects=None, agent_positions_xy=None) -> Tuple[Dict[str, chex.Array], State]:
        so = self.layout.static_objects if static_objects is None else static_objects
        grid = jnp.stack(
            [
                so,
                jnp.zeros_like(so),
                jnp.zeros_like(so),
            ],
            axis=-1,
            dtype=jnp.int32,
        )

        if agent_positions_xy is None:
            xs, ys = map(jnp.array, zip(*self.layout.agent_positions))
        else:
            xs = agent_positions_xy[:, 0]
            ys = agent_positions_xy[:, 1]
        agents = Agent(
            pos=Position(x=xs, y=ys),
            dir=jnp.full((self.num_agents,), Direction.UP),
            inventory=jnp.zeros((self.num_agents,), dtype=jnp.int32),
        )

        key, subkey = jax.random.split(key)
        recipe = self._sample_recipe(subkey)

        state = State(
            agents=agents,
            grid=grid,
            time=0,
            terminal=False,
            recipe=recipe,
            new_correct_delivery=False,
        )

        key, key_rnd = jax.random.split(key)
        if self.random_reset:
            state = self._randomize_state(state, key_rnd)
        elif self.random_agent_positions:
            state = self._randomize_agent_positions(state, key_rnd)

        return lax.stop_gradient(self.get_obs(state)), lax.stop_gradient(state)

    # ---------------------------------------------------------------- step_env

    def step_env(
        self,
        key: chex.PRNGKey,
        state: State,
        actions: Dict[str, chex.Array],
    ) -> Tuple[Dict[str, chex.Array], State, Dict[str, float], Dict[str, bool], Dict]:
        acts = self.action_set.take(
            jnp.array([actions[f"agent_{i}"] for i in range(self.num_agents)])
        )

        state, reward, shaped_rewards, collision_mask, wrong_pickups = self.step_agents(
            key, state, acts
        )
        correct_delivery = state.new_correct_delivery  # Extract from state
        state = state.replace(time=state.time + 1)
        done = self.is_terminal(state)
        state = state.replace(terminal=done)

        obs = self.get_obs(state)
        rewards = {f"agent_{i}": reward for i in range(self.num_agents)}
        dones = {f"agent_{i}": done for i in range(self.num_agents)}
        dones["__all__"] = done

        return (
            lax.stop_gradient(obs),
            lax.stop_gradient(state),
            rewards,
            dones,
            {
                # shaped_rewards[i]: shape (4,) = [dish_pickup, placement_in_pot, pot_start, plate_pickup]
                "shaped_reward":            {f"agent_{i}": shaped_rewards[i].sum()    for i in range(self.num_agents)},
                "shaped_reward_components": {f"agent_{i}": shaped_rewards[i]          for i in range(self.num_agents)},
                "collision":                {f"agent_{i}": collision_mask[i]          for i in range(self.num_agents)},
                "wrong_ingredient_pickup":  {f"agent_{i}": wrong_pickups[i]           for i in range(self.num_agents)},
                "correct_delivery":         correct_delivery,  # True if correct delivery happened this step
            },
        )

    # ------------------------------------------------------------ observation

    def get_obs(self, state: State) -> Dict[str, chex.Array]:
        H, W = self.height, self.width

        # Shared grid channels (same for all agents)
        static_ch  = state.grid[:, :, 0].astype(jnp.float32)
        dynamic_ch = state.grid[:, :, 1].astype(jnp.float32)
        timer_ch   = state.grid[:, :, 2].astype(jnp.float32)
        recipe_mask = state.grid[:, :, 0] == StaticObject.RECIPE_INDICATOR
        extra_ch   = timer_ch + jnp.where(recipe_mask, state.recipe.astype(jnp.float32), 0.0)
        grid_chs   = jnp.stack([static_ch, dynamic_ch, extra_ch], axis=-1)  # (H, W, 3)

        def _place(val, y, x):
            return jnp.zeros((H, W), dtype=jnp.float32).at[y, x].set(val)

        # All-agent direction / inventory grids for the "other" channels, Inventory = what the agent is currently carrying/holding
        all_dirs = jax.vmap(_place)(
            (state.agents.dir + 1).astype(jnp.float32),
            state.agents.pos.y,
            state.agents.pos.x,
        )  # (num_agents, H, W)
        all_invs = jax.vmap(_place)(
            state.agents.inventory.astype(jnp.float32), # What agent is holding (0=empty, 1=plate, 2/3=ingredients, 6/7=dish)
            state.agents.pos.y,
            state.agents.pos.x,
        )  # (num_agents, H, W)
        sum_dirs = jnp.sum(all_dirs, axis=0)
        sum_invs = jnp.sum(all_invs, axis=0)

        def _agent_obs(agent_id):
            sy = state.agents.pos.y[agent_id]
            sx = state.agents.pos.x[agent_id]
            s_dir = (state.agents.dir[agent_id] + 1).astype(jnp.float32)
            s_inv = state.agents.inventory[agent_id].astype(jnp.float32)

            self_dir_ch  = _place(s_dir, sy, sx)           # ch 0
            self_inv_ch  = _place(s_inv, sy, sx)           # ch 1
            other_dir_ch = sum_dirs - self_dir_ch          # ch 2
            other_inv_ch = sum_invs - self_inv_ch          # ch 3

            agent_chs = jnp.stack([self_dir_ch, self_inv_ch, other_dir_ch, other_inv_ch], axis=-1)
            obs = jnp.concatenate([agent_chs, grid_chs], axis=-1)  # (H, W, 7)
            return obs

        all_obs = jax.vmap(_agent_obs)(jnp.arange(self.num_agents))  # (num_agents, H, W, 7)

        if self.agent_view_size is not None:
            view_size = self.agent_view_size

            def _crop(obs, agent):
                padded = jnp.pad(
                    obs,
                    ((view_size, view_size), (view_size, view_size), (0, 0)),
                    mode="constant",
                    constant_values=0,
                )
                return lax.dynamic_slice(padded, (agent.pos.y, agent.pos.x, 0), self.obs_shape)

            all_obs = jax.vmap(_crop)(all_obs, state.agents)

        return {f"agent_{i}": all_obs[i] for i in range(self.num_agents)}

    # --------------------------------------------------------- step mechanics

    def step_agents(self, key, state, actions):
        grid = state.grid

        def _move(agent, action):
            direction = ACTION_TO_DIRECTION[action]

            def _do_move(agent, dir):
                new_pos = agent.pos.move_in_bounds(dir, self.width, self.height)
                new_pos = tree_select(
                    grid[new_pos.y, new_pos.x, 0] == StaticObject.EMPTY, new_pos, agent.pos
                )
                return agent.replace(pos=new_pos, dir=direction)

            return lax.cond(direction != -1, _do_move, lambda a, _: a, agent, direction)

        new_agents = jax.vmap(_move)(state.agents, actions)

        # Collision resolution
        def _collisions(mask):
            positions = jax.tree_util.tree_map(
                lambda n, o: jax.lax.select(mask, o, n),
                new_agents.pos, state.agents.pos,
            )
            cg = jnp.zeros((self.height, self.width))
            cg, _ = lax.scan(
                lambda g, p: (g.at[p.y, p.x].add(1), None), cg, positions
            )
            return jax.vmap(lambda p: cg[p.y, p.x] > 1)(positions)

        mask = lax.while_loop(
            lambda m: jnp.any(_collisions(m)),
            lambda m: m | _collisions(m),
            jnp.zeros((self.num_agents,), dtype=bool),
        )
        new_agents = new_agents.replace(
            pos=jax.tree_util.tree_map(
                lambda n, o: jax.lax.select(mask, o, n), new_agents.pos, state.agents.pos
            )
        )
        collision_mask = mask

        # Swap prevention
        def _swapped(orig, new):
            o = orig.to_array()
            n = new.to_array()
            swap = (jnp.expand_dims(o, 0) == jnp.expand_dims(n, 1)).all(-1)
            swap = jnp.fill_diagonal(swap, False, inplace=False)
            return jnp.any(swap & swap.T, axis=0)

        swap_mask = _swapped(state.agents.pos, new_agents.pos)
        new_agents = new_agents.replace(
            pos=jax.tree_util.tree_map(
                lambda n, o: jax.lax.select(swap_mask, o, n), new_agents.pos, state.agents.pos
            )
        )
        collision_mask = collision_mask | swap_mask

        # Interact
        def _interact_wrapper(carry, x):
            agent, action = x
            is_interact = action == Actions.interact

            def _interact(carry, agent):
                g, correct, rew = carry
                new_g, new_agent, new_correct, i_rew, s_rew, wrong = self.process_interact(
                    g, agent, new_agents.inventory, state.recipe
                )
                return (new_g, correct | new_correct, rew + i_rew), (new_agent, s_rew, wrong)

            return lax.cond(
                is_interact,
                _interact,
                lambda c, a: (c, (a, jnp.zeros(4, dtype=jnp.float32), jnp.array(False))),
                carry, agent,
            )

        carry = (grid, False, 0.0)
        (new_grid, new_correct_delivery, reward), (new_agents, shaped_rewards, wrong_pickups) = (
            lax.scan(_interact_wrapper, carry, (new_agents, actions))
        )

        # Advance timers
        def _tick(cell):
            def _cook(cell):
                cooking = cell[2] > 0
                new_extra = lax.select(cooking, cell[2] - 1, cell[2])
                done_cooking = cooking & (new_extra == 0)
                return jnp.array([cell[0], cell[1] | (done_cooking * DynamicObject.COOKED), new_extra])

            def _indicator(cell):
                return cell.at[2].set(jnp.clip(cell[2] - 1, min=0))

            branches = jnp.array([StaticObject.POT, StaticObject.BUTTON_RECIPE_INDICATOR]) == cell[0]
            branch_idx = lax.select(jnp.any(branches), jnp.argmax(branches) + 1, 0)
            return lax.switch(branch_idx, [lambda x: x, _cook, _indicator], cell)

        new_grid = jax.vmap(jax.vmap(_tick))(new_grid)

        key, subkey = jax.random.split(key)
        new_recipe = lax.cond(
            new_correct_delivery & self.sample_recipe_on_delivery,
            lambda _, k: self._sample_recipe(k),
            lambda r, _: r,
            state.recipe, subkey,
        )

        return (
            state.replace(
                agents=new_agents, grid=new_grid,
                recipe=new_recipe, new_correct_delivery=new_correct_delivery,
            ),
            reward, shaped_rewards, collision_mask, wrong_pickups,
        )

    def process_interact(self, grid, agent, all_inventories, recipe):
        inventory = agent.inventory
        fwd_pos = agent.get_fwd_pos()
        cell = grid[fwd_pos.y, fwd_pos.x]
        item, ingredients, extra = cell[0], cell[1], cell[2]
        plated_recipe = recipe | DynamicObject.PLATE | DynamicObject.COOKED

        is_plate_pile   = item == StaticObject.PLATE_PILE
        is_ing_pile     = StaticObject.is_ingredient_pile(item)
        is_pile         = is_plate_pile | is_ing_pile
        is_pot          = item == StaticObject.POT
        is_goal         = item == StaticObject.GOAL
        is_wall         = item == StaticObject.WALL
        is_btn          = item == StaticObject.BUTTON_RECIPE_INDICATOR

        no_ingredients  = ingredients == 0
        inv_empty       = inventory == 0
        inv_ingredient  = DynamicObject.is_ingredient(inventory)
        inv_plate       = inventory == DynamicObject.PLATE
        inv_dish        = (inventory & DynamicObject.COOKED) != 0

        merged = ingredients + inventory
        pot_cooking = is_pot & (extra > 0)
        pot_cooked  = is_pot & ((ingredients & DynamicObject.COOKED) != 0)
        pot_idle    = is_pot & ~pot_cooking & ~pot_cooked

        successful_dish_pickup = pot_cooked & inv_plate
        is_dish_useful = merged == plated_recipe
        sr_dish = (
            successful_dish_pickup * is_dish_useful * SHAPED_REWARDS["DISH_PICKUP"]
        )

        successful_pickup = (
            is_pile * inv_empty
            + successful_dish_pickup
            + is_wall * ~no_ingredients * inv_empty
        )
        successful_indicator = is_btn * inv_empty * no_ingredients

        pot_full = DynamicObject.ingredient_count(ingredients) == 3
        successful_pot_drop = pot_idle & inv_ingredient & ~pot_full
        ing_sel = inventory | (inventory << 1)
        is_drop_useful = (ingredients & ing_sel) < (recipe & ing_sel)
        sr_pot = (
            successful_pot_drop
            * is_drop_useful
            * lax.select(is_drop_useful, 1, -1 if self.negative_rewards else 0)
            * SHAPED_REWARDS["PLACEMENT_IN_POT"]
        )

        successful_drop = (
            is_wall * no_ingredients * ~inv_empty
            + successful_pot_drop
        )
        successful_delivery = is_goal & inv_dish
        no_effect = ~successful_pickup & ~successful_drop & ~successful_delivery

        pile_item = (
            is_plate_pile * DynamicObject.PLATE
            + is_ing_pile * StaticObject.get_ingredient(item)
        )

        new_ingredients = successful_drop * merged + no_effect * ingredients
        pot_full_after = DynamicObject.ingredient_count(new_ingredients) == 3

        successful_start = (
            pot_idle & ~no_ingredients & inv_empty & self.start_cooking_interaction
        )
        is_start_useful = ingredients == recipe
        sr_start = (
            successful_start * is_start_useful * SHAPED_REWARDS["POT_START_COOKING"]
        )
        auto_cook = pot_idle & pot_full_after & ~self.start_cooking_interaction
        use_extra = successful_start | auto_cook
        new_extra = (
            use_extra * POT_COOK_TIME
            + successful_indicator * INDICATOR_ACTIVATION_TIME
            + ~use_extra * ~successful_indicator * extra
        )

        new_grid = grid.at[fwd_pos.y, fwd_pos.x].set(
            jnp.array([item, new_ingredients, new_extra])
        )
        new_inventory = (
            successful_pickup * (pile_item + merged)
            + no_effect * inventory
        )
        new_agent = agent.replace(inventory=new_inventory)

        is_correct = inventory == plated_recipe
        reward = jnp.array(0, dtype=float)
        reward += (
            successful_delivery
            * lax.select(is_correct, 1, -1 if self.negative_rewards else 0)
            * DELIVERY_REWARD
        )
        reward -= successful_indicator * INDICATOR_ACTIVATION_COST

        new_inv_is_plate = new_inventory == DynamicObject.PLATE
        successful_plate_pickup = successful_pickup & new_inv_is_plate
        num_plates = jnp.sum(all_inventories == DynamicObject.PLATE)
        num_nonempty_pots = jnp.sum(
            (grid[:, :, 0] == StaticObject.POT) & (grid[:, :, 1] != 0)
        )
        plate_useful = num_plates < num_nonempty_pots
        no_plates_on_counters = jnp.sum(grid[:, :, 1] == DynamicObject.PLATE) == 0
        sr_plate = (
            no_plates_on_counters * plate_useful * successful_plate_pickup * SHAPED_REWARDS["PLATE_PICKUP"]
        )

        # Stack components: [dish_pickup, placement_in_pot, pot_start, plate_pickup]
        shaped_components = jnp.array(
            [sr_dish, sr_pot, sr_start, sr_plate], dtype=jnp.float32
        )

        correct_delivery = successful_delivery & is_correct

        ing_pile_idx = jnp.clip(item - StaticObject.INGREDIENT_PILE_BASE, 0, None)
        ing_bit = DynamicObject.BASE_INGREDIENT << (2 * ing_pile_idx)
        ing_mask = ing_bit | (ing_bit << 1)
        ing_in_recipe = (recipe & ing_mask) > 0
        wrong_pickup = successful_pickup & is_ing_pile & ~ing_in_recipe

        return new_grid, new_agent, correct_delivery, reward, shaped_components, wrong_pickup

    # ----------------------------------------------------------------- helpers

    def _sample_recipe(self, key):
        idx = jax.random.randint(key, (), 0, self.possible_recipes.shape[0])
        return DynamicObject.get_recipe_encoding(self.possible_recipes[idx])

    def _randomize_agent_positions(self, state, key):
        agents = state.agents

        def _pick(taken, x):
            pos, k = x
            allowed = (
                (self.enclosed_spaces == self.enclosed_spaces[pos.y, pos.x]) & ~taken
            ).flatten()
            p = allowed / jnp.sum(allowed)
            idx = jax.random.choice(k, allowed.size, (), p=p)
            new_pos = Position(x=idx % self.width, y=idx // self.width)
            return taken.at[new_pos.y, new_pos.x].set(True), new_pos

        taken = jnp.zeros_like(self.enclosed_spaces, dtype=jnp.bool_)
        key, subkey = jax.random.split(key)
        _, new_positions = lax.scan(_pick, taken, (agents.pos, jax.random.split(subkey, self.num_agents)))

        key, subkey = jax.random.split(key)
        dirs = jax.random.randint(subkey, (self.num_agents,), 0, len(Direction))
        return state.replace(agents=agents.replace(pos=new_positions, dir=dirs))

    def _randomize_state(self, state, key):
        key, subkey = jax.random.split(key)
        state = self._randomize_agent_positions(state, subkey)

        def _sample_inventory(k):
            k_dish, k_ing, k_inv = jax.random.split(k, 3)
            ing_idx = jax.random.randint(k_ing, (), 0, self.layout.num_ingredients)
            recipe_idx = jax.random.randint(k_dish, (), 0, len(self.possible_recipes))
            dish = (
                DynamicObject.get_recipe_encoding(self.possible_recipes[recipe_idx])
                | DynamicObject.COOKED
                | DynamicObject.PLATE
            )
            choices = jnp.array(
                [DynamicObject.EMPTY, DynamicObject.PLATE, DynamicObject.ingredient(ing_idx), dish]
            )
            return jax.random.choice(k_inv, choices, (), p=jnp.array([0.5, 0.1, 0.25, 0.15]))

        key, subkey = jax.random.split(key)
        inventories = jax.vmap(_sample_inventory)(jax.random.split(subkey, self.num_agents))

        def _sample_cell(cell, k):
            def _pot(k):
                k, ki, kn, kt = jax.random.split(k, 4)
                raw = jax.vmap(DynamicObject.ingredient)(
                    jax.random.randint(ki, (3,), 0, self.layout.num_ingredients)
                )
                n = jax.random.randint(kn, (), 1, 4)
                masked_ing = jnp.sum(raw * (jnp.arange(3) < n))
                full_ing = jnp.sum(raw)
                timer = jax.random.randint(kt, (), 0, POT_COOK_TIME) + 1
                options = jnp.array([
                    cell,
                    jnp.array([cell[0], masked_ing, 0]),
                    jnp.array([cell[0], full_ing, timer]),
                    jnp.array([cell[0], full_ing | DynamicObject.COOKED, 0]),
                ])
                return jax.random.choice(k, options, p=jnp.array([0.4, 0.35, 0.15, 0.1]))

            def _wall(k):
                k, ki, kd = jax.random.split(k, 3)
                ing_idx = jax.random.randint(ki, (), 0, self.layout.num_ingredients)
                dish_idx = jax.random.randint(kd, (), 0, len(self.possible_recipes))
                dish = (
                    DynamicObject.get_recipe_encoding(self.possible_recipes[dish_idx])
                    | DynamicObject.COOKED | DynamicObject.PLATE
                )
                opts = jnp.array([
                    DynamicObject.EMPTY, DynamicObject.PLATE,
                    DynamicObject.ingredient(ing_idx), dish,
                ])
                return cell.at[1].set(
                    jax.random.choice(k, opts, p=jnp.array([0.5, 0.1, 0.3, 0.1]))
                )

            idx = lax.select(
                jnp.any(jnp.array([StaticObject.POT, StaticObject.WALL]) == cell[0]),
                jnp.argmax(jnp.array([StaticObject.POT, StaticObject.WALL]) == cell[0]) + 1,
                0,
            )
            return lax.switch(idx, [lambda _: cell, _pot, _wall], k)

        key, subkey = jax.random.split(key)
        key_grid = jax.random.split(subkey, (self.height, self.width))
        new_grid = jax.vmap(jax.vmap(_sample_cell))(state.grid, key_grid)

        return state.replace(agents=state.agents.replace(inventory=inventories), grid=new_grid)

    def is_terminal(self, state):
        return state.time >= self.max_steps

    # ------------------------------------------------------- spaces / metadata

    @property
    def name(self):
        return "OvercookedV2"

    @property
    def num_actions(self):
        return len(self.action_set)

    def action_space(self, agent_id=""):
        return spaces.Discrete(len(self.action_set), dtype=jnp.uint32)

    def observation_space(self):
        return spaces.Box(0, 255, self.obs_shape)
