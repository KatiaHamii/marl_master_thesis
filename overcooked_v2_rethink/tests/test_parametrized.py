"""
Test script for parametrized environment.

Shows:
1. How parameters encode to vectors
2. How randomization works
3. Layout generation from parameters
"""

import jax
from parametrized_env import EnvParameters, ParametrizedOvercookedEnv


def test_parameter_encoding():
    """Test parameter to vector encoding."""
    print("=" * 60)
    print("TEST 1: Parameter Encoding")
    print("=" * 60)

    # Create base parameters
    params = EnvParameters(
        pot_x=4,
        pot_y=2,
        ingredient_left_x=0,
        ingredient_right_x=7,
        goal_left_x=3,
        goal_right_x=8,
        walls_left=1,
        walls_right=1,
    )

    print("\nBase Parameters (original):")
    print(params)

    # Convert to vector
    vector = params.to_vector()
    print(f"\nEncoded as vector: {vector}")

    # Reconstruct from vector
    reconstructed = EnvParameters.from_vector(vector)
    print("\nReconstructed from vector:")
    print(reconstructed)

    # Verify round-trip
    assert params.pot_x == reconstructed.pot_x
    print("\n✓ Round-trip successful!")


def test_randomization_with_seed():
    """Test randomization with seed for reproducibility."""
    print("\n" + "=" * 60)
    print("TEST 2: Randomization with Seed (Reproducibility)")
    print("=" * 60)

    # Create base parameters
    base_params = EnvParameters()
    print(f"\nBase parameters: {base_params.to_vector()}")

    # Create environment wrapper with seed
    key = jax.random.PRNGKey(42)

    print("\n--- Reset #1 (seed=42) ---")
    # Manually randomize to show mechanism
    from parametrized_env import ParametrizedOvercookedEnv

    class MockEnv:
        def set_layout(self, layout_dict):
            pass

        def reset(self, key):
            return None, None

    mock_env = MockEnv()
    param_env = ParametrizedOvercookedEnv(mock_env, seed=42)

    # Simulate reset
    params1 = param_env._randomize_params(jax.random.PRNGKey(0))
    print(f"Randomized params 1: {params1.to_vector()}")

    print("\n--- Reset #2 (seed=42 again) ---")
    param_env2 = ParametrizedOvercookedEnv(mock_env, seed=42)
    params2 = param_env2._randomize_params(jax.random.PRNGKey(0))
    print(f"Randomized params 2: {params2.to_vector()}")

    # Check reproducibility
    if (params1.to_vector() == params2.to_vector()).all():
        print("\n✓ Reproducibility verified! Same seed → same randomization")
    else:
        print("\n✗ Randomization differs")


def test_layout_generation():
    """Test layout dict generation from parameters."""
    print("\n" + "=" * 60)
    print("TEST 3: Layout Generation")
    print("=" * 60)

    # Test multiple parameter combinations
    test_cases = [
        ("Centered", EnvParameters(pot_x=4, pot_y=2, ingredient_left_x=0, ingredient_right_x=7, goal_left_x=3, goal_right_x=8, walls_left=1, walls_right=1)),
        ("Pot Left", EnvParameters(pot_x=2, pot_y=1, ingredient_left_x=1, ingredient_right_x=6, goal_left_x=2, goal_right_x=7, walls_left=0, walls_right=2)),
        ("Pot Right", EnvParameters(pot_x=6, pot_y=3, ingredient_left_x=0, ingredient_right_x=8, goal_left_x=4, goal_right_x=8, walls_left=2, walls_right=0)),
    ]

    for name, params in test_cases:
        print(f"\n{name}:")
        print(f"  Vector: {params.to_vector()}")

        layout_dict = ParametrizedOvercookedEnv.params_to_layout_dict(params)
        print(f"  Generated layout:")
        print(f"    Grid: {layout_dict['width']}×{layout_dict['height']}")
        print(f"    Walls: {len(layout_dict['walls'])} cells")
        print(f"    Pot: {layout_dict['pot']}")
        print(f"    Goals: left={layout_dict['goal']}")
        print(f"    Ingredients: {layout_dict['onion_pile_pos']}")
        print(f"    Plates: {layout_dict['plate_pile_pos']}")


def test_parameter_ranges():
    """Test parameter ranges."""
    print("\n" + "=" * 60)
    print("TEST 4: Parameter Ranges")
    print("=" * 60)

    ranges = EnvParameters.get_ranges()
    print("\nParameter Ranges:")
    for param_name, (min_val, max_val) in ranges.items():
        print(f"  {param_name}: [{min_val}, {max_val}]")

    print(f"\nTotal parameters: {len(ranges)}")
    print(f"Vector size: {len(EnvParameters().to_vector())}")


if __name__ == "__main__":
    test_parameter_encoding()
    test_randomization_with_seed()
    test_layout_generation()
    test_parameter_ranges()

    print("\n" + "=" * 60)
    print("✓ All tests completed!")
    print("=" * 60)
