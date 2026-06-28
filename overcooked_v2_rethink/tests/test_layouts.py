"""
test_layouts.py — Smoke-test all OvercookedV2 layouts before training.
=======================================================================
Run from MARL/ root:
    JAX_PLATFORMS=cpu uv run --project overcooked_v2_rethink \
        python overcooked_v2_rethink/test_layouts.py
"""

import os, sys

os.environ.setdefault("JAX_PLATFORMS", "cpu")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import jax
import jax.numpy as jnp
import numpy as np

from overcooked_v2_rethink import OvercookedV2, overcooked_v2_layouts


def test_layout(name: str) -> dict:
    """Reset, step once, and return basic stats. Raises on any error."""
    env = OvercookedV2(name, max_steps=400)
    key = jax.random.PRNGKey(0)

    # Reset
    obs, state = env.reset(key)

    # Check obs shapes
    for agent in env.agents:
        assert (
            obs[agent].shape == env.obs_shape
        ), f"{name}: expected obs shape {env.obs_shape}, got {obs[agent].shape}"

    # Step once with a random action
    key, subkey = jax.random.split(key)
    actions = {a: jnp.array(0) for a in env.agents}
    obs2, state2, rewards, dones, info = env.step_env(subkey, state, actions)

    assert state2.time == 1, f"{name}: time should be 1 after one step"
    for agent in env.agents:
        assert obs2[agent].shape == env.obs_shape

    layout = overcooked_v2_layouts[name]
    return {
        "grid": f"{env.height}x{env.width}",
        "obs_shape": str(env.obs_shape),
        "ingredients": layout.num_ingredients,
        "recipes": len(layout.possible_recipes),
        "agents": env.num_agents,
        "trainable": env.num_agents == 2,
    }


def main():
    print(
        f"\n{'Layout':<45} {'Grid':>6}  {'Obs shape':>14}  {'Ing':>4}  {'Recipes':>8}  Status"
    )
    print("-" * 100)

    passed, failed = 0, 0
    for name in overcooked_v2_layouts:
        try:
            info = test_layout(name)
            print(
                f"{name:<45} {info['grid']:>6}  {info['obs_shape']:>14}  "
                f"{info['ingredients']:>4}  {info['recipes']:>8}  OK"
            )
            passed += 1
        except Exception as e:
            print(f"{name:<45} {'':>6}  {'':>14}  {'':>4}  {'':>8}  FAIL — {e}")
            failed += 1

    print()
    print(
        f"Results: {passed} passed, {failed} failed  (total {passed + failed} layouts)"
    )
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
