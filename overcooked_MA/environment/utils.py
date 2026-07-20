import jax
import jax.numpy as jnp
from .common import Position, Direction


def compute_view_box(x, y, agent_view_size, height, width):
    """Compute the view box for an agent centered at (x, y)."""
    x_low  = jax.lax.clamp(0, x - agent_view_size,     width)
    x_high = jax.lax.clamp(0, x + agent_view_size + 1, width)
    y_low  = jax.lax.clamp(0, y - agent_view_size,     height)
    y_high = jax.lax.clamp(0, y + agent_view_size + 1, height)
    return x_low, x_high, y_low, y_high


def tree_select(predicate, a, b):
    return jax.tree_util.tree_map(lambda x, y: jax.lax.select(predicate, x, y), a, b)


def compute_enclosed_spaces(empty_mask: jnp.ndarray) -> jnp.ndarray:
    """Assign a unique region id to each connected empty cell (used for random agent placement)."""
    height, width = empty_mask.shape
    id_grid = jnp.arange(empty_mask.size, dtype=jnp.int32).reshape(empty_mask.shape)
    id_grid = jnp.where(empty_mask, id_grid, -1)

    def _body_fun(val):
        _, curr = val

        def _next_val(pos):
            neighbors = jax.vmap(pos.move_in_bounds, in_axes=(0, None, None))(
                jnp.array(list(Direction)), width, height
            )
            neighbour_values = curr[neighbors.y, neighbors.x]
            self_value = curr[pos.y, pos.x]
            values = jnp.concatenate([neighbour_values, self_value[jnp.newaxis]], axis=0)
            new_val = jnp.max(values)
            return jax.lax.select(self_value == -1, self_value, new_val)

        pos_y, pos_x = jnp.meshgrid(jnp.arange(height), jnp.arange(width), indexing="ij")
        next_vals = jax.vmap(jax.vmap(_next_val))(Position(x=pos_x, y=pos_y))
        stop = jnp.all(curr == next_vals)
        return stop, next_vals

    initial_val = (False, id_grid)
    _, res = jax.lax.while_loop(lambda val: ~val[0], _body_fun, initial_val)
    return res

