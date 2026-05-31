import os, sys
os.environ.setdefault("JAX_PLATFORMS", "cpu")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import jax
import jax.numpy as jnp
from overcooked_v2_rethink import OvercookedV2, NUM_OBS_CHANNELS, overcooked_v2_layouts


def test_obs_shape():
    env = OvercookedV2("cramped_room")
    key = jax.random.PRNGKey(0)
    obs, state = env.reset(key)
    H, W = env.height, env.width
    for agent in env.agents:
        assert obs[agent].shape == (H, W, NUM_OBS_CHANNELS), (
            f"Expected ({H},{W},{NUM_OBS_CHANNELS}), got {obs[agent].shape}"
        )
    print(f"  obs shape ({H}x{W}x{NUM_OBS_CHANNELS})   OK")


def test_obs_channels():
    env = OvercookedV2("cramped_room")
    key = jax.random.PRNGKey(0)
    obs, state = env.reset(key)
    o = obs["agent_0"]
    assert o.shape[-1] == 7, f"Expected 7 channels, got {o.shape[-1]}"

    # ch4 = static type: should contain non-zero (walls)
    assert jnp.any(o[..., 4] > 0), "ch4 (static) should not be all zeros"
    print("  channel count == 7             OK")
    print("  ch4 (static) has content       OK")


def test_step():
    env = OvercookedV2("cramped_room")
    key = jax.random.PRNGKey(1)
    obs, state = env.reset(key)

    actions = {agent: jnp.array(0) for agent in env.agents}
    key, subkey = jax.random.split(key)
    obs2, state2, rewards, dones, info = env.step_env(subkey, state, actions)

    assert state2.time == 1
    assert not dones["__all__"]
    for agent in env.agents:
        assert obs2[agent].shape == obs[agent].shape
    print("  step_env runs correctly        OK")


def test_jit_reset_step():
    env = OvercookedV2("cramped_room")
    reset_fn = jax.jit(env.reset)
    step_fn  = jax.jit(env.step_env)

    key = jax.random.PRNGKey(42)
    obs, state = reset_fn(key)
    actions = {agent: jnp.array(5) for agent in env.agents}  # interact
    key, subkey = jax.random.split(key)
    obs2, state2, rewards, dones, _ = step_fn(subkey, state, actions)
    assert obs2["agent_0"].shape[-1] == 7
    print("  jit(reset) + jit(step)         OK")


def test_partial_obs():
    env = OvercookedV2("cramped_room", agent_view_size=2)
    key = jax.random.PRNGKey(7)
    obs, state = env.reset(key)
    view = 2 * 2 + 1
    for agent in env.agents:
        assert obs[agent].shape[-1] == 7
        assert obs[agent].shape[0] <= view
        assert obs[agent].shape[1] <= view
    print(f"  partial obs (view={view})         OK")


def test_all_layouts():
    for name in list(overcooked_v2_layouts.keys())[:5]:
        env = OvercookedV2(name)
        obs, state = env.reset(jax.random.PRNGKey(0))
        assert obs["agent_0"].shape[-1] == 7
    print("  first 5 layouts reset           OK")


if __name__ == "__main__":
    print(f"JAX {jax.__version__} | {jax.devices()}\n")
    test_obs_shape()
    test_obs_channels()
    test_step()
    test_jit_reset_step()
    test_partial_obs()
    test_all_layouts()
    print("\nAll tests passed.")
