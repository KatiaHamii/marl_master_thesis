"""
Interactive visualization for parametrized environments.

Uses Overcooked-style rendering with black background and proper cell styling.
Shows base layout, generated mutations, and their parameter encodings.
Click "Generate Mutation" button to create new random variant.
"""

import os
os.environ['JAX_PLATFORMS'] = 'cpu'

from PIL import Image, ImageDraw
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button
import jax
from parametrized_env import (
    ParametrizedOvercookedEnv,
    EnvParameters,
    extract_layout_features,
    asymm_advantages_recipes_center,
    asymm_advantages
)

# Overcooked colors
COLORS = {
    "red": (255, 0, 0),
    "blue": (0, 0, 255),
    "green": (0, 255, 0),
    "yellow": (255, 255, 0),
    "grey": (100, 100, 100),
    "white": (255, 255, 255),
    "black": (25, 25, 25),
    "orange": (230, 180, 0),
    "dark_green": (0, 150, 0),
    "brown": (139, 69, 19),
}

TILE_SIZE = 64  # Pixel size per tile


class OvercookedRenderer:
    """Render Overcooked layouts with black background (matching game style)."""

    @staticmethod
    def draw_empty_cell(draw, x, y):
        """Draw empty cell (black background)."""
        x1 = x * TILE_SIZE
        y1 = y * TILE_SIZE
        x2 = x1 + TILE_SIZE
        y2 = y1 + TILE_SIZE
        draw.rectangle([x1, y1, x2-1, y2-1], fill=COLORS["black"], outline=COLORS["grey"], width=1)

    @staticmethod
    def draw_wall_cell(draw, x, y):
        """Draw wall cell (grey filled)."""
        x1 = x * TILE_SIZE
        y1 = y * TILE_SIZE
        x2 = x1 + TILE_SIZE
        y2 = y1 + TILE_SIZE
        draw.rectangle([x1, y1, x2-1, y2-1], fill=COLORS["grey"], outline=COLORS["black"], width=2)

    @staticmethod
    def draw_agent_cell(draw, x, y, agent_color):
        """Draw agent cell with triangle."""
        x1 = x * TILE_SIZE
        y1 = y * TILE_SIZE
        x2 = x1 + TILE_SIZE
        y2 = y1 + TILE_SIZE

        # Black background
        draw.rectangle([x1, y1, x2-1, y2-1], fill=COLORS["black"], outline=COLORS["grey"], width=1)

        # Agent triangle
        cx = (x1 + x2) // 2
        cy = (y1 + y2) // 2
        r = TILE_SIZE // 3
        points = [
            (cx, cy - r),      # top
            (cx - r, cy + r),  # bottom-left
            (cx + r, cy + r),  # bottom-right
        ]
        draw.polygon(points, fill=agent_color, outline=COLORS["black"], width=2)

    @staticmethod
    def draw_ingredient_cell(draw, x, y):
        """Draw ingredient pile cell (yellow circles)."""
        x1 = x * TILE_SIZE
        y1 = y * TILE_SIZE
        x2 = x1 + TILE_SIZE
        y2 = y1 + TILE_SIZE

        # Grey background
        draw.rectangle([x1, y1, x2-1, y2-1], fill=COLORS["grey"], outline=COLORS["black"], width=2)

        # Yellow circles at various positions
        positions = [
            (0.3, 0.3), (0.7, 0.3), (0.5, 0.65),
            (0.3, 0.7), (0.7, 0.7)
        ]
        for px, py in positions:
            cx = int(x1 + px * TILE_SIZE)
            cy = int(y1 + py * TILE_SIZE)
            r = TILE_SIZE // 5
            draw.ellipse([cx-r, cy-r, cx+r, cy+r], fill=COLORS["yellow"], outline=COLORS["black"], width=1)

    @staticmethod
    def draw_pot_cell(draw, x, y):
        """Draw pot cell (orange circle)."""
        x1 = x * TILE_SIZE
        y1 = y * TILE_SIZE
        x2 = x1 + TILE_SIZE
        y2 = y1 + TILE_SIZE

        # Grey background
        draw.rectangle([x1, y1, x2-1, y2-1], fill=COLORS["grey"], outline=COLORS["black"], width=2)

        # Orange pot
        cx = (x1 + x2) // 2
        cy = (y1 + y2) // 2
        r = TILE_SIZE // 3
        draw.ellipse([cx-r, cy-r, cx+r, cy+r], fill=COLORS["orange"], outline=COLORS["black"], width=2)

    @staticmethod
    def draw_goal_cell(draw, x, y):
        """Draw goal cell (green background)."""
        x1 = x * TILE_SIZE
        y1 = y * TILE_SIZE
        x2 = x1 + TILE_SIZE
        y2 = y1 + TILE_SIZE

        # Green background
        draw.rectangle([x1, y1, x2-1, y2-1], fill=COLORS["dark_green"], outline=COLORS["black"], width=2)

    @staticmethod
    def draw_plate_pile_cell(draw, x, y):
        """Draw plate pile cell (white circles on grey background)."""
        x1 = x * TILE_SIZE
        y1 = y * TILE_SIZE
        x2 = x1 + TILE_SIZE
        y2 = y1 + TILE_SIZE

        # Grey background
        draw.rectangle([x1, y1, x2-1, y2-1], fill=COLORS["grey"], outline=COLORS["black"], width=2)

        # White plate circles
        positions = [
            (0.3, 0.3), (0.75, 0.42), (0.4, 0.75)
        ]
        for px, py in positions:
            cx = int(x1 + px * TILE_SIZE)
            cy = int(y1 + py * TILE_SIZE)
            r = TILE_SIZE // 5
            draw.ellipse([cx-r, cy-r, cx+r, cy+r], fill=COLORS["white"], outline=COLORS["black"], width=1)

    @staticmethod
    def render_layout(features, width=9, height=5):
        """Render layout to PIL Image with Overcooked style."""
        img_width = width * TILE_SIZE
        img_height = height * TILE_SIZE
        img = Image.new("RGB", (img_width, img_height), color=COLORS["black"])
        draw = ImageDraw.Draw(img)

        # Initialize grid
        grid = {(x, y): "empty" for x in range(width) for y in range(height)}

        # Mark features
        for x, y in features['walls']:
            if 0 <= x < width and 0 <= y < height:
                grid[(x, y)] = "wall"
        for x, y in features['ingredients']:
            if 0 <= x < width and 0 <= y < height:
                grid[(x, y)] = "ingredient"
        for x, y in features['pots']:
            if 0 <= x < width and 0 <= y < height:
                grid[(x, y)] = "pot"
        for x, y in features['goals']:
            if 0 <= x < width and 0 <= y < height:
                grid[(x, y)] = "goal"
        for x, y in features['plates']:
            if 0 <= x < width and 0 <= y < height:
                grid[(x, y)] = "plate"

        # Draw all cells
        for x in range(width):
            for y in range(height):
                tile_type = grid[(x, y)]
                if tile_type == "wall":
                    OvercookedRenderer.draw_wall_cell(draw, x, y)
                elif tile_type == "ingredient":
                    OvercookedRenderer.draw_ingredient_cell(draw, x, y)
                elif tile_type == "pot":
                    OvercookedRenderer.draw_pot_cell(draw, x, y)
                elif tile_type == "goal":
                    OvercookedRenderer.draw_goal_cell(draw, x, y)
                elif tile_type == "plate":
                    OvercookedRenderer.draw_plate_pile_cell(draw, x, y)
                else:
                    OvercookedRenderer.draw_empty_cell(draw, x, y)

        # Draw agents (on top)
        agent_positions = [(1, 3), (3, 3)]
        agent_colors = [COLORS["red"], COLORS["blue"]]
        for pos, color in zip(agent_positions, agent_colors):
            if 0 <= pos[0] < width and 0 <= pos[1] < height:
                OvercookedRenderer.draw_agent_cell(draw, pos[0], pos[1], color)

        return img


class InteractiveEnvVisualizer:
    """Interactive visualization using Overcooked-style rendering."""

    def __init__(self):
        self.seed = 42
        self.key = jax.random.PRNGKey(np.uint32(self.seed))

        self.param_env = ParametrizedOvercookedEnv(seed=self.seed)

        # History
        self.history = []
        self.base_params = EnvParameters()
        self.history.append({
            'name': 'Base Layout',
            'params': self.base_params,
            'vector': self.base_params.to_vector(),
            'seed': self.seed,
        })

        # Create figure
        self.fig = plt.figure(figsize=(18, 10))
        self.fig.suptitle('Parametrized Environment Visualization (Overcooked Style)',
                         fontsize=16, fontweight='bold')

        # Create layout
        self.ax_base = plt.subplot(2, 3, 1)
        self.ax_current = plt.subplot(2, 3, 2)
        self.ax_history = plt.subplot(2, 3, 3)

        self.ax_info_base = plt.subplot(2, 3, 4)
        self.ax_info_current = plt.subplot(2, 3, 5)
        self.ax_legend = plt.subplot(2, 3, 6)

        # Render base layout
        self.render_base_layout()
        self.draw_info_panel(self.ax_info_base, self.base_params, self.seed, "Base Layout")
        self.draw_legend()

        # Button
        ax_button = plt.axes([0.45, 0.08, 0.1, 0.04])
        self.btn_generate = Button(ax_button, 'Generate Mutation')
        self.btn_generate.on_clicked(self.on_generate_clicked)

        plt.tight_layout(rect=[0, 0.12, 1, 0.96])

    def render_base_layout(self):
        """Render base layout."""
        features = extract_layout_features(asymm_advantages)
        img = OvercookedRenderer.render_layout(features)

        self.ax_base.clear()
        self.ax_base.imshow(np.array(img))
        self.ax_base.set_title('Base Layout',
                              fontweight='bold', fontsize=12)
        self.ax_base.axis('off')

    def render_current_layout(self, params: EnvParameters):
        """Render mutated layout."""
        features = extract_layout_features(asymm_advantages)
        img = OvercookedRenderer.render_layout(features)

        self.ax_current.clear()
        self.ax_current.imshow(np.array(img))
        self.ax_current.set_title(f'Mutated Layout (obstacles: {params.obstacles_left}/{params.obstacles_right})',
                                 fontweight='bold', fontsize=12)
        self.ax_current.axis('off')

    def draw_info_panel(self, ax, params: EnvParameters, seed: int, title: str):
        """Draw parameter info."""
        ax.clear()
        ax.axis('off')
        ax.set_title(title, fontweight='bold', fontsize=11)

        vector = params.to_vector()
        info_text = f"""Raw Parameters:
  obstacles_left:  {params.obstacles_left}
  obstacles_right: {params.obstacles_right}
  resources_left:  {params.resources_left}
  resources_right: {params.resources_right}

Normalized [0,1]:
  [{vector[0]:.3f}, {vector[1]:.3f}, {vector[2]:.3f}, {vector[3]:.3f}]

Vector with Seed:
  ([{vector[0]:.3f}, {vector[1]:.3f}, {vector[2]:.3f}, {vector[3]:.3f}], {seed})

Difficulty:
  Obstacles avg: {(params.obstacles_left + params.obstacles_right) / 2:.1f}/5
  Resources avg: {(params.resources_left + params.resources_right) / 2:.1f}/5
"""
        ax.text(0.05, 0.95, info_text, fontsize=9, family='monospace',
               verticalalignment='top', transform=ax.transAxes)

    def draw_legend(self):
        """Draw visual legend with rendered tiles."""
        self.ax_legend.clear()
        self.ax_legend.axis('off')
        self.ax_legend.set_title('Legend', fontweight='bold', fontsize=11)

        # Extract features from base layout
        features = extract_layout_features(asymm_advantages)

        # Create legend image (6 tiles wide, 5 tiles tall)
        legend_width = 6 * TILE_SIZE
        legend_height = 5 * TILE_SIZE
        legend_img = Image.new("RGB", (legend_width, legend_height), color=COLORS["white"])
        draw = ImageDraw.Draw(legend_img)

        # Draw legend tiles and labels
        tiles = []
        if features['walls']:
            tiles.append(("Wall", lambda d, x, y: OvercookedRenderer.draw_wall_cell(d, x, y)))
        if features['ingredients']:
            tiles.append(("Ingredient", lambda d, x, y: OvercookedRenderer.draw_ingredient_cell(d, x, y)))
        if features['pots']:
            tiles.append(("Pot", lambda d, x, y: OvercookedRenderer.draw_pot_cell(d, x, y)))
        if features['goals']:
            tiles.append(("Goal", lambda d, x, y: OvercookedRenderer.draw_goal_cell(d, x, y)))
        if features['plates']:
            tiles.append(("Plate", lambda d, x, y: OvercookedRenderer.draw_plate_pile_cell(d, x, y)))

        # Add agent rows
        tiles.extend([
            ("Agent 1", lambda d, x, y: OvercookedRenderer.draw_agent_cell(d, x, y, COLORS["red"])),
            ("Agent 2", lambda d, x, y: OvercookedRenderer.draw_agent_cell(d, x, y, COLORS["blue"])),
        ])

        # Render tiles
        for i, (label, draw_fn) in enumerate(tiles):
            col = i % 3
            row = i // 3
            x = col * 2
            y = row

            # Draw tile
            draw_fn(draw, x, y)

            # Draw label text
            label_x = (x + 1) * TILE_SIZE + 5
            label_y = y * TILE_SIZE + TILE_SIZE // 2 - 8
            draw.text((label_x, label_y), label, fill=COLORS["black"], font=None)

        # Add info text at bottom
        info_text = f"Grid: {features['width']}×{features['height']} | Tiles: {TILE_SIZE}×{TILE_SIZE}px"
        draw.text((10, legend_height - 20), info_text, fill=COLORS["black"], font=None)

        self.ax_legend.imshow(np.array(legend_img))
        self.ax_legend.axis('off')

    def draw_history_list(self):
        """Draw history."""
        self.ax_history.clear()
        self.ax_history.axis('off')
        self.ax_history.set_title('Generation History', fontweight='bold', fontsize=12)

        y_pos = 0.95
        for i, entry in enumerate(self.history[-6:]):
            text = f"{i}. {entry['name']} (seed={entry['seed']})"
            self.ax_history.text(0.05, y_pos, text, fontsize=10, family='monospace',
                                transform=self.ax_history.transAxes)
            y_pos -= 0.14

    def on_generate_clicked(self, event):
        """Generate mutation."""
        self.key, subkey = jax.random.split(self.key)
        new_params = self.param_env._randomize_params(subkey)
        self.seed += 1

        self.history.append({
            'name': f'Mutation #{len(self.history)}',
            'params': new_params,
            'vector': new_params.to_vector(),
            'seed': self.seed,
        })

        self.render_current_layout(new_params)
        self.draw_info_panel(self.ax_info_current, new_params, self.seed, "Current Mutation")
        self.draw_history_list()
        self.fig.canvas.draw_idle()


def main():
    """Run the interactive visualizer."""
    print("\nStarting Interactive Parametrized Environment Visualizer...")
    print("Click 'Generate Mutation' button to create new environments\n")

    visualizer = InteractiveEnvVisualizer()
    plt.show()


if __name__ == "__main__":
    main()
