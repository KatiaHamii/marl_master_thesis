import os
os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax
import jax.numpy as jnp
import optax
import flax.linen as nn
import numpy as np


def test_jax_basics():
    x = jnp.array([1.0, 2.0, 3.0])
    assert jnp.dot(x, x) == 14.0
    print("  jnp basics         OK")


def test_jit():
    @jax.jit
    def f(x):
        return jnp.sum(x ** 2)

    result = f(jnp.ones(5))
    assert result == 5.0
    print("  jit                OK")


def test_vmap():
    batched = jax.vmap(lambda v: jnp.sum(v ** 2))(jnp.ones((8, 4)))
    assert batched.shape == (8,)
    assert jnp.all(batched == 4.0)
    print("  vmap               OK")


def test_grad():
    f = lambda x: jnp.sum(x ** 2)
    grad = jax.grad(f)(jnp.array([1.0, 2.0, 3.0]))
    assert jnp.allclose(grad, jnp.array([2.0, 4.0, 6.0]))
    print("  grad               OK")


def test_random():
    key = jax.random.PRNGKey(0)
    samples = jax.random.normal(key, shape=(1000,))
    assert samples.shape == (1000,)
    assert abs(float(jnp.mean(samples))) < 0.1
    print("  random             OK")


def test_optax():
    optimizer = optax.adam(1e-3)
    params = {"w": jnp.ones((4, 4))}
    state = optimizer.init(params)
    grads = {"w": jnp.ones((4, 4)) * 0.1}
    updates, new_state = optimizer.update(grads, state)
    assert updates["w"].shape == (4, 4)
    print("  optax (adam)       OK")


def test_flax():
    class MLP(nn.Module):
        @nn.compact
        def __call__(self, x):
            x = nn.Dense(16)(x)
            x = nn.relu(x)
            return nn.Dense(4)(x)

    model = MLP()
    key = jax.random.PRNGKey(1)
    x = jnp.ones((2, 8))
    params = model.init(key, x)
    out = model.apply(params, x)
    assert out.shape == (2, 4)
    print("  flax (MLP)         OK")


if __name__ == "__main__":
    print(f"JAX {jax.__version__} on {jax.devices()}\n")
    test_jax_basics()
    test_jit()
    test_vmap()
    test_grad()
    test_random()
    test_optax()
    test_flax()
    print("\nAll checks passed.")
