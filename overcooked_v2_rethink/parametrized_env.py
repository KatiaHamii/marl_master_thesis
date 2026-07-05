"""
Parametrized environment for MARL curriculum learning.

Extracts features from base layout and generates parametrized environments.
Can be used both as a training component and for visualization.

Parameters (4-dimensional):
- obstacles_left: Number of walls on left side (1-5, 1=easy, 5=hard)
- obstacles_right: Number of walls on right side (1-5, 1=easy, 5=hard)
- resources_left: Number of ingredient piles on left (1-5, 5=easy, 1=hard)
- resources_right: Number of ingredient piles on right (1-5, 5=easy, 1=hard)
"""

import sys
import numpy as np
import jax
import jax.numpy as jnp
from dataclasses import dataclass
from typing import Dict, Tuple, Optional

# Handle imports
try:
    from layouts import asymm_advantages_recipes_center
except ImportError:
    # Fallback: define layout directly
    asymm_advantages_recipes_center = """
WWWWWWWWW
0 WXR01 X
1   P   W
W A PA  W
WWWBWBWWW
"""

asymm_advantages = """
WWWWWWWWW
O WXWOW X
W   P   W
W A PA  W
WWWBWBWWW
"""


@dataclass
class EnvParameters:
    """Interpretable parameters for environment difficulty (4-dimensional)."""

    obstacles_left: int = 1      # [1, 5] - 1=easy, 5=hard
    obstacles_right: int = 1     # [1, 5] - 1=easy, 5=hard
    resources_left: int = 5      # [1, 5] - 5=easy, 1=hard (inverse)
    resources_right: int = 5     # [1, 5] - 5=easy, 1=hard (inverse)

    @staticmethod
    def get_ranges():
        """Return min/max ranges for each parameter."""
        return {
            'obstacles_left': (1, 5),
            'obstacles_right': (1, 5),
            'resources_left': (1, 5),
            'resources_right': (1, 5),
        }

    def to_vector(self) -> np.ndarray:
        """Convert parameters to normalized vector [0, 1]."""
        ranges = self.get_ranges()
        vector = []

        for param_name, (min_val, max_val) in ranges.items():
            value = getattr(self, param_name)
            normalized = (value - min_val) / (max_val - min_val)
            vector.append(normalized)

        return np.array(vector, dtype=np.float32)

    @classmethod
    def from_vector(cls, vector: np.ndarray) -> 'EnvParameters':
        """Create parameters from normalized vector [0, 1]."""
        ranges = cls.get_ranges()
        params = {}

        for i, (param_name, (min_val, max_val)) in enumerate(ranges.items()):
            normalized = float(vector[i])
            value = int(min_val + normalized * (max_val - min_val))
            params[param_name] = value

        return cls(**params)

    def __str__(self):
        """Human-readable parameter description."""
        ranges = self.get_ranges()
        lines = ["Parameters:"]
        for param_name, (min_val, max_val) in ranges.items():
            value = getattr(self, param_name)
            norm = (value - min_val) / (max_val - min_val)
            lines.append(f"  {param_name:20s}: {value} [{min_val}, {max_val}] → {norm:.3f}")
        return "\n".join(lines)


class ParametrizedOvercookedEnv:
    """
    Parametrized environment wrapper for OvercookedV2.

    Generates environments with controllable difficulty via 4 parameters.
    Can be integrated into training or used standalone.
    """

    def __init__(self, env=None, base_params: EnvParameters = None, randomize_on_reset: bool = True, seed: int = None):
        """
        Args:
            env: OvercookedV2 environment instance (optional for visualization)
            base_params: Base parameters (default: extract from layout)
            randomize_on_reset: If True, randomize params slightly on each reset
            seed: Random seed for reproducibility (optional)
        """
        self.env = env
        self.base_params = base_params or EnvParameters()
        self.current_params = self.base_params
        self.randomize_on_reset = randomize_on_reset
        self.reset_count = 0
        self.seed = seed

        # Initialize PRNG with seed
        if seed is not None:
            self.key = jax.random.PRNGKey(seed)
        else:
            self.key = jax.random.PRNGKey(0)

    def reset(self, key=None):
        """
        Reset environment with new randomized parameters using JAX PRNG.

        Args:
            key: JAX PRNG key (optional, uses internal seed if not provided)

        Returns:
            obs, state: From underlying env (or None if env not set)
            params_vector: Normalized parameter vector [0, 1]
        """
        self.reset_count += 1

        # Use provided key or internal key
        if key is None:
            key = self.key

        if self.randomize_on_reset:
            # Split key for randomization
            key, subkey = jax.random.split(key)

            # Use JAX PRNG to randomize parameters
            self.current_params = self._randomize_params(subkey)

            # Update environment layout if env is set
            if self.env is not None:
                layout_dict = self.params_to_layout_dict(self.current_params)
                self.env.set_layout(layout_dict)

        # Reset underlying environment if it exists
        if self.env is not None:
            key, subkey = jax.random.split(key)
            obs, state = self.env.reset(subkey)
        else:
            obs, state = None, None

        # Update internal key for next reset
        self.key = key

        # Return observation, state, and parameter encoding
        params_vector = self.current_params.to_vector()

        return obs, state, params_vector

    def _randomize_params(self, key) -> EnvParameters:
        """Generate slightly randomized parameters using JAX PRNG."""
        ranges = EnvParameters.get_ranges()
        new_params = {}

        # Generate random deltas for each parameter
        param_count = len(ranges)
        deltas = jax.random.randint(key, (param_count,), -1, 2)  # [-1, 0, 1]

        for i, (param_name, (min_val, max_val)) in enumerate(ranges.items()):
            current_val = getattr(self.base_params, param_name)
            delta = int(deltas[i])
            new_val = np.clip(current_val + delta, min_val, max_val)
            new_params[param_name] = new_val

        return EnvParameters(**new_params)

    @staticmethod
    def params_to_layout_dict(params: EnvParameters) -> Dict:
        """Convert parameters to layout dictionary for env.set_layout()."""
        # This would be implemented to actually generate layout mutations
        # based on the parameters. For now, returns a dict structure.
        return {
            'obstacles_left': params.obstacles_left,
            'obstacles_right': params.obstacles_right,
            'resources_left': params.resources_left,
            'resources_right': params.resources_right,
        }

    def set_params(self, params: EnvParameters):
        """Manually set environment parameters."""
        self.current_params = params
        if self.env is not None:
            layout_dict = self.params_to_layout_dict(params)
            self.env.set_layout(layout_dict)

    def get_params_vector(self) -> np.ndarray:
        """Get current environment encoding as normalized vector."""
        return self.current_params.to_vector()

    def get_params_description(self) -> str:
        """Get human-readable parameter description."""
        return str(self.current_params)

    def step(self, action):
        """Forward step to underlying environment."""
        if self.env is not None:
            return self.env.step(action)
        return None


# ============================================================================
# VISUALIZATION FUNCTIONS (for standalone use)
# ============================================================================

def extract_layout_features(layout_string: str):
    """Extract key features from layout string."""
    lines = [line.strip() for line in layout_string.strip().split('\n') if line.strip()]

    features = {
        'width': max(len(line) for line in lines),
        'height': len(lines),
        'walls': [],
        'ingredients': [],
        'pots': [],
        'goals': [],
        'plates': [],
    }

    for y, line in enumerate(lines):
        for x, char in enumerate(line):
            if char == 'W':
                features['walls'].append((x, y))
            elif char in '01O':  # Ingredient piles (0, 1, or O)
                features['ingredients'].append((x, y))
            elif char == 'P':
                features['pots'].append((x, y))
            elif char == 'X':
                features['goals'].append((x, y))
            elif char == 'B':
                features['plates'].append((x, y))

    return features


def count_obstacles_by_side(features, width):
    """Count obstacles (walls) on left and right sides."""
    mid = width / 2
    left_walls = [w for w in features['walls'] if w[0] < mid]
    right_walls = [w for w in features['walls'] if w[0] >= mid]
    return len(left_walls), len(right_walls)


def count_resources_by_side(features, width):
    """Count resources (ingredient piles) on left and right sides."""
    mid = width / 2
    left_resources = [r for r in features['ingredients'] if r[0] < mid]
    right_resources = [r for r in features['ingredients'] if r[0] >= mid]
    return len(left_resources), len(right_resources)


def generate_random_mutations(num_mutations=5, seed=None):
    """Generate random parameter combinations."""
    if seed is not None:
        rng = np.random.default_rng(seed)
    else:
        rng = np.random.default_rng()

    mutations = []
    for i in range(num_mutations):
        params = EnvParameters(
            obstacles_left=rng.integers(1, 6),
            obstacles_right=rng.integers(1, 6),
            resources_left=rng.integers(1, 6),
            resources_right=rng.integers(1, 6),
        )
        mutations.append(params)

    return mutations


def params_to_vector(params):
    """Convert parameters to normalized vector."""
    return params.to_vector()


def print_base_layout(layout_string: str):
    """Print the base layout and its extracted features."""
    print("\n" + "=" * 80)
    print(f"BASE LAYOUT: {layout_string}")
    print("=" * 80)

    features = extract_layout_features(layout_string)
    walls_left, walls_right = count_obstacles_by_side(features, features['width'])
    res_left, res_right = count_resources_by_side(features, features['width'])

    print("\nLayout String:")
    print(layout_string)

    print("\nExtracted Features:")
    print(f"  Grid dimensions: {features['width']}×{features['height']}")
    print(f"  Walls - Left: {walls_left}, Right: {walls_right}")
    print(f"  Ingredients - Left: {res_left}, Right: {res_right}")
    print(f"  Pots: {features['pots']}")
    print(f"  Goals: {features['goals']}")
    print(f"  Plates: {features['plates']}")

    print("\nParameter Space (based on base layout):")
    print(f"  obstacles_left:  1-5 (base={walls_left}, 1=easy, 5=hard)")
    print(f"  obstacles_right: 1-5 (base={walls_right}, 1=easy, 5=hard)")
    print(f"  resources_left:  1-5 (base={res_left}, 5=easy, 1=hard)")
    print(f"  resources_right: 1-5 (base={res_right}, 5=easy, 1=hard)")


def print_mutations(mutations, seeds=None):
    """Print mutation details."""
    print("\n" + "=" * 80)
    print("GENERATED MUTATIONS")
    print("=" * 80)

    # Create vector representation
    print("\nVector Representation (parameters, seed):")
    print("[")
    for i, params in enumerate(mutations):
        vector = params_to_vector(params)
        seed = seeds[i] if seeds else 42 + i
        print(f"  ([{vector[0]:.3f}, {vector[1]:.3f}, {vector[2]:.3f}, {vector[3]:.3f}], {seed}),")
    print("]")

    # Detailed breakdown
    for i, params in enumerate(mutations, 1):
        vector = params_to_vector(params)

        print(f"\n{'─' * 80}")
        print(f"MUTATION #{i}")
        print(f"{'─' * 80}")

        print("\nRaw Parameters:")
        print(f"  obstacles_left:  {params.obstacles_left} (1=easy, 5=hard)")
        print(f"  obstacles_right: {params.obstacles_right} (1=easy, 5=hard)")
        print(f"  resources_left:  {params.resources_left} (5=easy, 1=hard)")
        print(f"  resources_right: {params.resources_right} (5=easy, 1=hard)")

        print("\nNormalized Parameters [0, 1]:")
        print(f"  obstacles_left:  {vector[0]:.3f}")
        print(f"  obstacles_right: {vector[1]:.3f}")
        print(f"  resources_left:  {vector[2]:.3f}")
        print(f"  resources_right: {vector[3]:.3f}")

        print("\nVector with Seed:")
        seed = seeds[i-1] if seeds else 42 + i - 1
        print(f"  ([{vector[0]:.3f}, {vector[1]:.3f}, {vector[2]:.3f}, {vector[3]:.3f}], {seed})")

        print("\nDifficulty Assessment:")
        avg_obstacles = (params.obstacles_left + params.obstacles_right) / 2
        avg_resources = (params.resources_left + params.resources_right) / 2
        overall_difficulty = (avg_obstacles + (6 - avg_resources)) / 2

        difficulty_level = "Easy" if overall_difficulty < 2.5 else "Medium" if overall_difficulty < 3.5 else "Hard"
        print(f"  Overall difficulty: {overall_difficulty:.2f}/5 ({difficulty_level})")


if __name__ == "__main__":
    """Visualization mode: show base layout and 5 random mutations."""
    print_base_layout("asymm_advantages_recipes_center")

    print("\n" + "=" * 80)
    print("GENERATING 5 RANDOM MUTATIONS (seed=42)")
    print("=" * 80)

    mutations = generate_random_mutations(num_mutations=5, seed=42)
    seeds = [42 + i for i in range(len(mutations))]
    print_mutations(mutations, seeds=seeds)

    print("\n" + "=" * 80)
    print("Vector Format: ([obstacles_left, obstacles_right, resources_left, resources_right], seed)")
    print("\nParameter Ranges:")
    print("  obstacles_left/right: [1, 5]   (1=easy, 5=hard)")
    print("  resources_left/right: [1, 5]   (5=easy, 1=hard)")
    print("  seed: Random seed for reproducibility")
    print("=" * 80 + "\n")
