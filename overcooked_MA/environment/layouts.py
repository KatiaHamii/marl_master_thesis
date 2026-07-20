from .common import StaticObject
import numpy as np
from typing import List, Tuple, Optional
from dataclasses import dataclass
import itertools

# Layouts from Overcooked-AI
@dataclass
class Layout:
    # agent positions list of positions, tuples (x, y)
    agent_positions: List[Tuple[int, int]]

    # width x height grid with static items
    static_objects: np.ndarray

    num_ingredients: int

    # If recipe is none, recipes will be sampled from the possible_recipes
    # If possible_recipes is none, all possible recipes with the available ingredients will be considered
    possible_recipes: Optional[List[List[int]]]

    def __post_init__(self):
        if len(self.agent_positions) == 0:
            raise ValueError("At least one agent position must be provided")
        if self.num_ingredients < 1:
            raise ValueError("At least one ingredient must be available")
        if self.possible_recipes is None:
            self.possible_recipes = self._get_all_possible_recipes(self.num_ingredients)

    @property
    def height(self):
        return self.static_objects.shape[0]

    @property
    def width(self):
        return self.static_objects.shape[1]

    @staticmethod
    def _get_all_possible_recipes(num_ingredients: int) -> List[List[int]]:
        """
        Get all possible recipes given the number of ingredients.
        """
        available_ingredients = list(range(num_ingredients)) * 3
        raw_combinations = itertools.combinations(available_ingredients, 3)
        unique_recipes = set(
            tuple(sorted(combination)) for combination in raw_combinations
        )

        return [list(recipe) for recipe in unique_recipes]

    @staticmethod
    def from_string(grid, possible_recipes=None, swap_agents=False):
        """Assumes `grid` is string representation of the layout, with 1 line per row, and the following symbols:
        W: wall
        A: agent
        X: goal
        B: plate (bowl) pile
        P: pot location
        R: recipe of the day indicator
        L: button recipe indicator
        0-9: Ingredient x pile
        ' ' (space) : empty cell

        Depricated:
        O: onion pile - will be interpreted as ingredient 0


        If `recipe` is provided, it should be a list of ingredient indices, max 3 ingredients per recipe
        If `recipe` is not provided, the recipe will be randomized on reset.
        If the layout does not have a recipe indicator, a fixed `recipe` must be provided.

        If `possible_recipes` is provided, it should be a list of lists of ingredient indices, 3 ingredients per recipe.

        Swap agents will swap the positions of the agents in the layout. This is only used for compatibility with the old Overcooked-AI layouts.
        """

        rows = grid.split("\n")

        if len(rows[0]) == 0:
            rows = rows[1:]
        if len(rows[-1]) == 0:
            rows = rows[:-1]

        row_lens = [len(row) for row in rows]
        static_objects = np.zeros((len(rows), max(row_lens)), dtype=int)

        char_to_static_item = {
            " ": StaticObject.EMPTY,
            "W": StaticObject.WALL,
            "X": StaticObject.GOAL,
            "B": StaticObject.PLATE_PILE,
            "P": StaticObject.POT,
            "R": StaticObject.RECIPE_INDICATOR,
            "L": StaticObject.BUTTON_RECIPE_INDICATOR,
        }

        for r in range(10):
            char_to_static_item[f"{r}"] = StaticObject.INGREDIENT_PILE_BASE + r

        agent_positions = []

        num_ingredients = 0
        includes_recipe_indicator = False
        for r, row in enumerate(rows):
            c = 0
            while c < len(row):
                char = row[c]

                if char == "O":
                    char = "0"

                if char == "A":
                    agent_pos = (c, r)
                    agent_positions.append(agent_pos)

                obj = char_to_static_item.get(char, StaticObject.EMPTY)
                static_objects[r, c] = obj

                if StaticObject.is_ingredient_pile(obj):
                    ingredient_idx = obj - StaticObject.INGREDIENT_PILE_BASE
                    num_ingredients = max(num_ingredients, ingredient_idx + 1)

                if obj == StaticObject.RECIPE_INDICATOR:
                    includes_recipe_indicator = True

                c += 1

        # TODO: add some sanity checks - e.g. agent must exist, surrounded by walls, etc.

        if possible_recipes is not None:
            if not isinstance(possible_recipes, list):
                raise ValueError("possible_recipes must be a list")
            if not all(isinstance(recipe, list) for recipe in possible_recipes):
                raise ValueError("possible_recipes must be a list of lists")
            if not all(len(recipe) == 3 for recipe in possible_recipes):
                raise ValueError("All recipes must be of length 3")
        elif not includes_recipe_indicator:
            raise ValueError(
                "Layout does not include a recipe indicator, a fixed recipe must be provided"
            )

        if swap_agents:
            agent_positions = agent_positions[::-1]

        layout = Layout(
            agent_positions=agent_positions,
            static_objects=static_objects,
            num_ingredients=num_ingredients,
            possible_recipes=possible_recipes,
        )

        return layout
