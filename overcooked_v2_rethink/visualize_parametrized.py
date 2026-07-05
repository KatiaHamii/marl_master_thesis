"""
Visualize parametrized environments extracted from asymm_advantages_recipes_center.

Extracts actual features from the base layout and generates random mutations.
Parameters:
- obstacles (walls): 1-5 per side (1=easy, 5=difficult)
- resources (ingredient piles): 1-5 per side (5=easy, 1=difficult)
"""

import sys
import numpy as np

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


def layout_dict_to_grid(layout_dict):
    """Convert layout dict to 2D grid string for visualization."""
    width = layout_dict.get('width', 9)
    height = layout_dict.get('height', 5)

    # Initialize grid with empty spaces
    grid = [[' ' for _ in range(width)] for _ in range(height)]

    # Add walls
    if 'walls' in layout_dict:
        for pos in layout_dict['walls']:
            if isinstance(pos, (tuple, list)):
                x, y = pos
            else:
                continue
            if 0 <= x < width and 0 <= y < height:
                grid[y][x] = 'W'

    # Add pot
    if 'pot' in layout_dict:
        x, y = layout_dict['pot']
        if 0 <= x < width and 0 <= y < height:
            grid[y][x] = 'P'

    # Add goals
    if 'goal' in layout_dict:
        x, y = layout_dict['goal']
        if 0 <= x < width and 0 <= y < height:
            grid[y][x] = 'X'

    # Add ingredient piles
    if 'onion_pile_pos' in layout_dict:
        for x, y in layout_dict['onion_pile_pos']:
            if 0 <= x < width and 0 <= y < height and grid[y][x] == ' ':
                grid[y][x] = 'O'

    # Add plate piles
    if 'plate_pile_pos' in layout_dict:
        for x, y in layout_dict['plate_pile_pos']:
            if 0 <= x < width and 0 <= y < height and grid[y][x] == ' ':
                grid[y][x] = 'B'

    # Convert grid to string
    grid_str = '\n'.join(''.join(row) for row in grid)
    return grid_str


def print_base_layout():
    """Print the base asymm_advantages_recipes_center layout."""
    print("\n" + "=" * 80)
    print("BASE LAYOUT: asymm_advantages_recipes_center")
    print("=" * 80)

    features = extract_layout_features(asymm_advantages_recipes_center)
    walls_left, walls_right = count_obstacles_by_side(features, features['width'])
    res_left, res_right = count_resources_by_side(features, features['width'])

    print("\nLayout String:")
    print(asymm_advantages_recipes_center)

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


def generate_random_mutations(base_features, num_mutations=5, seed=None):
    """Generate random parameter combinations."""
    if seed is not None:
        rng = np.random.default_rng(seed)
    else:
        rng = np.random.default_rng()

    mutations = []
    for i in range(num_mutations):
        params = {
            'obstacles_left': rng.integers(1, 6),      # 1-5
            'obstacles_right': rng.integers(1, 6),     # 1-5
            'resources_left': rng.integers(1, 6),      # 1-5
            'resources_right': rng.integers(1, 6),     # 1-5
        }
        mutations.append(params)

    return mutations


def params_to_vector(params):
    """Convert parameters dict to normalized vector."""
    obs_left_norm = (params['obstacles_left'] - 1) / 4
    obs_right_norm = (params['obstacles_right'] - 1) / 4
    res_left_norm = (params['resources_left'] - 1) / 4
    res_right_norm = (params['resources_right'] - 1) / 4
    return np.array([obs_left_norm, obs_right_norm, res_left_norm, res_right_norm], dtype=np.float32)


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
        # Normalize to [0, 1]
        obs_left_norm = (params['obstacles_left'] - 1) / 4
        obs_right_norm = (params['obstacles_right'] - 1) / 4
        res_left_norm = (params['resources_left'] - 1) / 4
        res_right_norm = (params['resources_right'] - 1) / 4

        print(f"\n{'─' * 80}")
        print(f"MUTATION #{i}")
        print(f"{'─' * 80}")

        print("\nRaw Parameters:")
        print(f"  obstacles_left:  {params['obstacles_left']} (1=easy, 5=hard)")
        print(f"  obstacles_right: {params['obstacles_right']} (1=easy, 5=hard)")
        print(f"  resources_left:  {params['resources_left']} (5=easy, 1=hard)")
        print(f"  resources_right: {params['resources_right']} (5=easy, 1=hard)")

        print("\nNormalized Parameters [0, 1]:")
        print(f"  obstacles_left:  {obs_left_norm:.3f}")
        print(f"  obstacles_right: {obs_right_norm:.3f}")
        print(f"  resources_left:  {res_left_norm:.3f}")
        print(f"  resources_right: {res_right_norm:.3f}")

        print("\nVector with Seed:")
        seed = seeds[i-1] if seeds else 42 + i - 1
        print(f"  ([{obs_left_norm:.3f}, {obs_right_norm:.3f}, {res_left_norm:.3f}, {res_right_norm:.3f}], {seed})")

        print("\nDifficulty Assessment:")
        avg_obstacles = (params['obstacles_left'] + params['obstacles_right']) / 2
        avg_resources = (params['resources_left'] + params['resources_right']) / 2
        overall_difficulty = (avg_obstacles + (6 - avg_resources)) / 2  # Inverse resources

        difficulty_level = "Easy" if overall_difficulty < 2.5 else "Medium" if overall_difficulty < 3.5 else "Hard"
        print(f"  Overall difficulty: {overall_difficulty:.2f}/5 ({difficulty_level})")


if __name__ == "__main__":
    print_base_layout()

    print("\n" + "=" * 80)
    print("GENERATING 5 RANDOM MUTATIONS (seed=42)")
    print("=" * 80)

    features = extract_layout_features(asymm_advantages_recipes_center)
    mutations = generate_random_mutations(features, num_mutations=5, seed=42)
    seeds = [42 + i for i in range(len(mutations))]
    print_mutations(mutations, seeds=seeds)

    print("\n" + "=" * 80)
    print("Vector Format:")
    print("  (obstacles_left, obstacles_right, resources_left, resources_right, seed)")
    print("\nLegend:")
    print("  obstacles_left/right: Number of walls per side (1-5)")
    print("  resources_left/right: Number of ingredient piles per side (1-5)")
    print("  seed: Random seed for reproducibility")
    print("  Difficulty: Based on obstacle count + inverse resource count")
    print("=" * 80 + "\n")
