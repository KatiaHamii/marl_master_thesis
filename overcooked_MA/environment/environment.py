import numpy as np
import random
from config import render_full_map
from .generator import LevelGenerator, compute_min_cycle

class OvercookedEnvironment:
    """Overcooked Environment with support for UED vector-based level generation and template-based level setting."""
    def __init__(self, height=9, width=11):
        self.height = height
        self.width = width
        self.grid = None
        self.current_template = None

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

        if self.current_template is not None:
            self.grid = self.current_template.copy()
        else:
            raise ValueError("Map not set! Call set_level_template or pass a param_vector.")

        obs = self.grid
        info = {"cycle_length": compute_min_cycle(self.grid)}
        return obs, info

    def step(self, action):
        """Step in the game (placeholder for simulation)"""
        obs = self.grid
        reward = 0.0
        terminated = False
        truncated = False
        info = {}
        return obs, reward, terminated, truncated, info

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