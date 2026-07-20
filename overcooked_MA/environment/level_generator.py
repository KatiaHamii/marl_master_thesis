"""
overcooked_parametrized.py — Parametrized Overcooked environment with PIL rendering

Works with any layout from layouts.py. The base is parsed from a layout string;
mutations generate fully random valid layouts within configurable continuous constraints.

4-parameter continuous encoding (each normalized to [0, 1] in the vector):
    obs_density   base fraction of interior cells filled with obstacle walls   [0, 0.6]
    obs_skew_x    horizontal spatial bias for walls (left vs right)           [-1, 1]
    res_density   base fraction of remaining cells filled with ingredients    [0, 0.4]
    res_skew_x    horizontal spatial bias for ingredients (left vs right)     [-1, 1]

Object counts (agents, goals, pots, plates) are derived from the base layout and
varied randomly across mutations. Spatial gradients allow organic asymmetric generation.

"""

import os
import math
os.environ.setdefault("JAX_PLATFORMS", "cpu")

import numpy as np
import jax
import jax.numpy as jnp
from dataclasses import dataclass
from typing import Tuple

from collections import deque
from PIL import Image, ImageDraw, ImageFont

# ── Layout import ──────────────────────────────────────────────────────────────
import os as _os, re as _re

def _load_layout_strings() -> dict:
    """Parse raw layout string constants from layouts.py without importing it.

    layouts.py uses a relative import (from .common import ...) that fails when
    run standalone.  Reading the file as text sidesteps that entirely.
    """
    path = _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "layouts.py")
    try:
        with open(path) as _f:
            source = _f.read()
        result = {}
        for m in _re.finditer(r'^([a-z][a-z0-9_]+)\s*=\s*"""([\s\S]*?)"""', source, _re.MULTILINE):
            result[m.group(1)] = m.group(2)
        return result
    except Exception:
        return {}

_LAYOUTS = _load_layout_strings()

DEFAULT_LAYOUT = _LAYOUTS.get("asymm_advantages") or """
WWWWWWWWW
O WXWOW X
W   P   W
W A PA  W
WWWBWBWWW
"""

def get_layout(name: str) -> str:
    """Return a layout string by name (from layouts.py).

    Example::

        from overcooked_parametrized import run_interactive_viz, get_layout
        run_interactive_viz(layout_str=get_layout("asymm_advantages_recipes_center"))
    """
    if name not in _LAYOUTS:
        available = ", ".join(sorted(_LAYOUTS))
        raise ValueError(f"Unknown layout '{name}'.  Available: {available}")
    return _LAYOUTS[name]

# ── Object type IDs ────────────────────────────────────────────────────────────

class GridCodes:
    """All grid-cell integer codes in one namespace — import the class, not each constant."""
    EMPTY        = 0
    AGENT_0      = 2   # red  triangle
    AGENT_1      = 3   # blue triangle
    GOAL         = 4   # green filled cell
    POT          = 5   # orange circle
    OBSTACLE     = 8   # wall — dark cell + grey X cross
    PLATE_PILE   = 9   # white circles
    INGREDIENT_0 = 10  # yellow circles
    INGREDIENT_1 = 11  # dark-green circles
    INGREDIENT_2 = 12

    # Density caps (belong here — they constrain how codes are placed)
    MAX_OBS_FRAC = 0.6   # obstacles fill at most 60 % of interior cells per half
    MAX_RES_FRAC = 0.4   # resources fill at most 40 % of remaining interior cells per half

# Module-level aliases — keep existing code working without changes
EMPTY        = GridCodes.EMPTY
AGENT_0      = GridCodes.AGENT_0
AGENT_1      = GridCodes.AGENT_1
GOAL         = GridCodes.GOAL
POT          = GridCodes.POT
OBSTACLE     = GridCodes.OBSTACLE
PLATE_PILE   = GridCodes.PLATE_PILE
INGREDIENT_0 = GridCodes.INGREDIENT_0
INGREDIENT_1 = GridCodes.INGREDIENT_1
INGREDIENT_2 = GridCodes.INGREDIENT_2

# ── Rendering palette ──────────────────────────────────────────────────────────

TILE_PX      = 64

C_BG         = ( 20,  20,  20)
C_GREY       = (130, 130, 130)
C_YELLOW     = (240, 220,  40)
C_DARK_GREEN = ( 40, 130,  40)
C_GREEN      = ( 50, 180,  50)
C_ORANGE     = (210, 130,  30)
C_WHITE      = (255, 255, 255)
C_RED        = (210,  60,  60)
C_BLUE       = ( 60, 100, 210)
C_LEGEND_BG  = (245, 245, 245)

# Module-level aliases for density caps
MAX_OBS_FRAC = GridCodes.MAX_OBS_FRAC
MAX_RES_FRAC = GridCodes.MAX_RES_FRAC

# ── Parameter dataclass ────────────────────────────────────────────────────────

@dataclass
class EnvParams:
    """
    Continuous gradient-based difficulty parameters.
    
    obs_density : base fraction of interior cells to turn into walls [0, 0.6]
    obs_skew_x  : horizontal bias for walls [-1.0 = strict left, 0.0 = uniform, 1.0 = strict right]
    res_density : base fraction of free cells to turn into ingredients [0, 0.4]
    res_skew_x  : horizontal bias for ingredients [-1.0 = strict left, 0.0 = uniform, 1.0 = strict right]
    """
    obs_density: float = 0.0
    obs_skew_x:  float = 0.0
    res_density: float = 0.0
    res_skew_x:  float = 0.0

    def to_vector(self) -> np.ndarray:
        """Projects continuous parameters into a clean normalized [0, 1]^4 vector."""
        return np.array([
            self.obs_density / MAX_OBS_FRAC,
            (self.obs_skew_x + 1.0) / 2.0,
            self.res_density / MAX_RES_FRAC,
            (self.res_skew_x + 1.0) / 2.0,
        ], dtype=np.float32)

    def __str__(self) -> str:
        v = self.to_vector()
        return (
            f"obs_den={self.obs_density:.2f}({v[0]:.2f})  "
            f"obs_skew={self.obs_skew_x:.2f}({v[1]:.2f})  \n"
            f"res_den={self.res_density:.2f}({v[2]:.2f})  "
            f"res_skew={self.res_skew_x:.2f}({v[3]:.2f})"
        )

# ── Random layout generator ────────────────────────────────────────────────────

def generate_random_layout(
    H: int,
    W: int,
    obs_density,  # float or JAX float scalar
    obs_skew_x,   # float or JAX float scalar [-1.0, 1.0]
    res_density,  # float or JAX float scalar
    res_skew_x,   # float or JAX float scalar [-1.0, 1.0]
    key,
    num_agents: int = 2,
    num_goals = 2,
    num_pots  = 2,
    num_plates: int = 1,
    ing0: int = INGREDIENT_0,
    ing1: int = INGREDIENT_0,
) -> jnp.ndarray:
    """
    JAX-compatible gradient-biased layout generator. Vmappable and JIT-compilable.
    Uses continuous spatial priorities to organically guide obstacle and resource densities.
    """
    N = H * W
    keys = jax.random.split(key, 12)

    grid = jnp.full(N, EMPTY, dtype=jnp.int32)
    occ  = jnp.zeros(N, dtype=jnp.bool_)

    # Generate flat cell indices
    r_idx = jnp.repeat(jnp.arange(H), W)
    c_idx = jnp.tile(jnp.arange(W), H)
    
    # Static cell masks
    interior_flat = (r_idx > 0) & (r_idx < H - 1) & (c_idx > 0) & (c_idx < W - 1)
    all_flat      = jnp.ones(N, dtype=jnp.bool_)

    # Continuous horizontal coordinates mapped to [-1.0, 1.0]
    x_coords = (c_idx.astype(jnp.float32) - (W - 1) / 2.0) / jnp.maximum((W - 1) / 2.0, 1.0)

    def place_biased(grid, occ, eligible_flat, n, obj, skew_x, rng):
        """Places 'n' objects by sorting cells using spatial weight combined with noise."""
        spatial_weight = skew_x * x_coords
        noise = jax.random.uniform(rng, (N,))
        priority = spatial_weight + noise
        
        # Suppress non-eligible cells to prevent them from being selected
        priority = jnp.where(eligible_flat, priority, -99999.0)
        perm = jnp.argsort(-priority)

        def _step(carry, idx):
            g, o, cnt = carry
            can = eligible_flat[idx] & ~o[idx] & (cnt < n)
            g = g.at[idx].set(jnp.where(can, jnp.int32(obj), g[idx]))
            o = o.at[idx].set(o[idx] | can)
            cnt = cnt + can.astype(jnp.int32)
            return (g, o, cnt), None

        (grid, occ, _), _ = jax.lax.scan(_step, (grid, occ, jnp.int32(0)), perm)
        return grid, occ

    def place_uniform(grid, occ, eligible_flat, n, obj, rng):
        """Standard uniform placement helper for agents and stations."""
        perm = jax.random.permutation(rng, N)

        def _step(carry, idx):
            g, o, cnt = carry
            can = eligible_flat[idx] & ~o[idx] & (cnt < n)
            g = g.at[idx].set(jnp.where(can, jnp.int32(obj), g[idx]))
            o = o.at[idx].set(o[idx] | can)
            cnt = cnt + can.astype(jnp.int32)
            return (g, o, cnt), None

        (grid, occ, _), _ = jax.lax.scan(_step, (grid, occ, jnp.int32(0)), perm)
        return grid, occ

    # 1. Spawn Agents uniformly across the interior space
    k_a0, k_a1 = jax.random.split(keys[8])
    grid, occ = place_uniform(grid, occ, interior_flat, jnp.int32(1), AGENT_0, k_a0)
    grid, occ = place_uniform(grid, occ, interior_flat, jnp.int32(1), AGENT_1, k_a1)

    # 2. Spawn Obstacle Walls using horizontal spatial gradients
    n_interior = jnp.sum(interior_flat)
    n_walls    = (obs_density * n_interior.astype(jnp.float32)).astype(jnp.int32)
    grid, occ  = place_biased(grid, occ, interior_flat, n_walls, OBSTACLE, obs_skew_x, keys[0])

    # 3. Spawn Ingredient Piles across remaining free space based on resource gradients
    n_free        = jnp.sum(all_flat & ~occ)
    n_ingredients = (res_density * n_free.astype(jnp.float32)).astype(jnp.int32)
    n_ing0        = n_ingredients // 2
    n_ing1        = n_ingredients - n_ing0
    
    k_res0, k_res1 = jax.random.split(keys[2])
    grid, occ = place_biased(grid, occ, all_flat, n_ing0, ing0, res_skew_x, k_res0)
    grid, occ = place_biased(grid, occ, all_flat, n_ing1, ing1, res_skew_x, k_res1)

    # 4. Place remaining structural targets uniformly
    grid, occ = place_uniform(grid, occ, all_flat, jnp.asarray(num_goals, jnp.int32), GOAL,       keys[4])
    grid, occ = place_uniform(grid, occ, all_flat, jnp.asarray(num_pots,  jnp.int32), POT,        keys[5])
    grid, occ = place_uniform(grid, occ, all_flat, jnp.int32(num_plates),             PLATE_PILE, keys[6])

    return grid.reshape(H, W)


# ── Reachability check (BFS) ───────────────────────────────────────────────────
def compute_min_cycle(grid: np.ndarray) -> int:
    """
    Returns the minimum delivery cycle length for a single agent.
    Enforces that both agents are within the same walkable space.
    Returns 9999 if the layout is blocked, split, or structurally invalid.

    This is the unified/improved version — kept in sync with the identical
    implementation in environment/generator.py (the lightweight-NumPy stack).
    Over the earlier 4-leg version it adds: a cooperative-connectivity check
    (both agents must share one walkable region), interaction zones defined as
    agent-reachable cells adjacent to each object (not the object cell itself),
    and the full 5-leg recipe cycle including the plate->pot->goal dish pickup.
    """
    grid = np.asarray(grid)
    H, W = grid.shape
    walkable = {EMPTY, AGENT_0, AGENT_1}
    dirs = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    # Gather positions of both agents
    agent_starts = [(r, c) for r in range(H) for c in range(W) if grid[r, c] in (AGENT_0, AGENT_1)]

    # Validation: ensure exactly two agents are spawned on the map
    if len(agent_starts) < 2:
        return 9999

    # Check path connectivity between Agent 1 and Agent 2 (cooperative reachability)
    start_agent = agent_starts[0]
    target_agent = agent_starts[1]

    visited_coop = {start_agent}
    queue_coop = deque([start_agent])
    agents_connected = False

    while queue_coop:
        r, c = queue_coop.popleft()
        if (r, c) == target_agent:
            agents_connected = True
            break

        for dr, dc in dirs:
            nr, nc = r + dr, c + dc
            if 0 <= nr < H and 0 <= nc < W and (nr, nc) not in visited_coop:
                if grid[nr, nc] in walkable:
                    visited_coop.add((nr, nc))
                    queue_coop.append((nr, nc))

    # Reject layout if agents are split into separate spaces or trapped
    if not agents_connected:
        return 9999

    def _get_adjacent_walkable(target_codes: set) -> set:
        adj_cells = set()
        for r in range(H):
            for c in range(W):
                if grid[r, c] in target_codes:
                    for dr, dc in dirs:
                        nr, nc = r + dr, c + dc
                        # The adjacent cell MUST be reachable by agents
                        if 0 <= nr < H and 0 <= nc < W and (nr, nc) in visited_coop:
                            adj_cells.add((nr, nc))
        return adj_cells

    # Gather interaction zones (walkable boundaries) for all critical tools
    onion_zones = _get_adjacent_walkable({INGREDIENT_0, INGREDIENT_1})
    pot_zones   = _get_adjacent_walkable({POT})
    plate_zones = _get_adjacent_walkable({PLATE_PILE})
    goal_zones  = _get_adjacent_walkable({GOAL})

    # Reject layout if any critical object has zero accessible adjacent cells
    if not (agent_starts and onion_zones and pot_zones and plate_zones and goal_zones):
        return 9999

    # Multi-source BFS for the exact path distance between item zones
    def _bfs_distance(starts: set, targets: set) -> int:
        if not starts or not targets:
            return 9999
        # Immediate success if starting zones already overlap with target zones
        if not starts.isdisjoint(targets):
            return 0
        visited = set(starts)
        queue = deque([(r, c, 0) for r, c in starts])
        while queue:
            r, c, dist = queue.popleft()
            if (r, c) in targets:
                return dist
            for dr, dc in dirs:
                nr, nc = r + dr, c + dc
                if 0 <= nr < H and 0 <= nc < W and grid[nr, nc] in walkable:
                    if (nr, nc) in targets:
                        return dist + 1
                    if (nr, nc) not in visited:
                        visited.add((nr, nc))
                        queue.append((nr, nc, dist + 1))
        return 9999

    # Full sequential recipe cycle loop for Overcooked
    d1 = _bfs_distance(set(agent_starts), onion_zones)   # spawns -> ingredient crate
    d2 = _bfs_distance(onion_zones, pot_zones)            # ingredient -> pot
    d3 = _bfs_distance(pot_zones, plate_zones)            # pot -> plate pile
    d4 = _bfs_distance(plate_zones, pot_zones)            # plate -> pot (pick up dish)
    d5 = _bfs_distance(pot_zones, goal_zones)             # pot -> delivery station
    total = d1 + d2 + d3 + d4 + d5
    return total if total < 9999 else 9999

def is_valid_layout(grid) -> bool:
    """Return True if the generated layout can successfully complete a full cooking task loop."""
    grid = np.asarray(grid)
    H, W = grid.shape
    walkable = {EMPTY, AGENT_0, AGENT_1}

    task_cats: dict = {
        "ingredient": frozenset({INGREDIENT_0, INGREDIENT_1}),
        "pot":        frozenset({POT}),
        "plate":      frozenset({PLATE_PILE}),
        "goal":       frozenset({GOAL}),
    }

    agent_pos = [(r, c) for r in range(H) for c in range(W) if grid[r, c] in (AGENT_0, AGENT_1)]
    if not agent_pos:
        return False

    for cat_types in task_cats.values():
        if not any(grid[r, c] in cat_types for r in range(H) for c in range(W)):
            return False

    all_task_types = frozenset(t for ts in task_cats.values() for t in ts)
    dirs = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    def reachable_cats(start: tuple) -> set:
        visited = {start}
        queue   = deque([start])
        found: set = set()
        while queue:
            r, c = queue.popleft()
            for dr, dc in dirs:
                nr, nc = r + dr, c + dc
                if not (0 <= nr < H and 0 <= nc < W):
                    continue
                cell = grid[nr, nc]
                if cell in all_task_types:
                    for name, ts in task_cats.items():
                        if cell in ts:
                            found.add(name)
                elif (nr, nc) not in visited and cell in walkable:
                    visited.add((nr, nc))
                    queue.append((nr, nc))
        return found

    required = frozenset(task_cats)
    per_agent = [reachable_cats(pos) for pos in agent_pos]

    if not all("goal" in cats for cats in per_agent):
        return False

    return any(cats >= required for cats in per_agent)


# ── Tile renderer (PIL, 64×64 px) ─────────────────────────────────────────────

_PILE_POS = [(0.50, 0.15), (0.28, 0.42), (0.78, 0.38), (0.38, 0.78), (0.72, 0.74)]

def _circles_on_grey(draw: ImageDraw.Draw, positions, color, r_frac=0.14, s=TILE_PX):
    draw.rectangle([(0, 0), (s - 1, s - 1)], fill=C_GREY)
    r = int(r_frac * s)
    for fx, fy in positions:
        cx, cy = int(fx * s), int(fy * s)
        draw.ellipse([(cx - r, cy - r), (cx + r, cy + r)], fill=color)

def _add_grid_lines(draw: ImageDraw.Draw, s: int):
    lw = max(1, s // 32)
    draw.rectangle([(0, 0), (lw - 1, s - 1)], fill=C_GREY)
    draw.rectangle([(0, 0), (s - 1, lw - 1)], fill=C_GREY)

def render_tile(obj: int, size: int = TILE_PX) -> Image.Image:
    img = Image.new("RGB", (size, size), C_BG)
    draw = ImageDraw.Draw(img)
    s = size

    if obj == OBSTACLE:
        lw  = max(3, s // 12)
        pad = max(4, s // 10)
        draw.line([(pad, pad), (s - 1 - pad, s - 1 - pad)], fill=C_GREY, width=lw)
        draw.line([(s - 1 - pad, pad), (pad, s - 1 - pad)], fill=C_GREY, width=lw)
    elif obj == GOAL:
        draw.rectangle([(0, 0), (s - 1, s - 1)], fill=C_GREY)
        pad = max(2, int(0.10 * s))
        draw.rectangle([(pad, pad), (s - 1 - pad, s - 1 - pad)], fill=C_GREEN)
    elif obj == POT:
        C_POT = (25, 25, 25)
        draw.rectangle([(0, 0), (s - 1, s - 1)], fill=C_GREY)
        pad = max(3, int(0.10 * s))
        body_top = int(0.30 * s)
        draw.rectangle([(pad, body_top), (s - 1 - pad, int(0.90 * s))], fill=C_POT)
        hw = max(3, int(0.09 * s))
        cx = s // 2
        draw.rectangle([(cx - hw, int(0.12 * s)), (cx + hw, body_top)], fill=C_POT)
    elif obj == PLATE_PILE:
        _circles_on_grey(draw, [(0.30, 0.30), (0.72, 0.40), (0.40, 0.72)], C_WHITE, r_frac=0.18, s=size)
    elif obj == INGREDIENT_0:
        _circles_on_grey(draw, _PILE_POS, C_YELLOW, s=size)
    elif obj == INGREDIENT_1:
        _circles_on_grey(draw, _PILE_POS, C_DARK_GREEN, s=size)
    elif obj in (AGENT_0, AGENT_1):
        color = C_RED if obj == AGENT_0 else C_BLUE
        pts = [
            (int(s * 0.50), int(s * 0.10)),
            (int(s * 0.88), int(s * 0.86)),
            (int(s * 0.12), int(s * 0.86)),
        ]
        draw.polygon(pts, fill=color)

    _add_grid_lines(draw, s)
    return img

# ── Legend helpers ─────────────────────────────────────────────────────────────

_LABEL = {
    OBSTACLE:     "Wall",
    GOAL:         "Delivery\nStation",
    POT:          "Pot",
    PLATE_PILE:   "Plate",
    INGREDIENT_0: "Ingredient",
    INGREDIENT_1: "Ingredient",
    AGENT_0:      "Agent 1",
    AGENT_1:      "Agent 2",
}

def _get_font(size: int = 13) -> ImageFont.ImageFont:
    for fp in (
        "/System/Library/Fonts/Helvetica.ttc",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ):
        try:
            return ImageFont.truetype(fp, size)
        except OSError:
            continue
    return ImageFont.load_default()

def render_legend_2col(grid: np.ndarray, tile_size: int = TILE_PX, extra_types: tuple = ()) -> Image.Image:
    present = sorted(set(grid.flatten()) | set(extra_types))
    items   = [(o, _LABEL[o]) for o in present if o in _LABEL]
    font    = _get_font(13)
    ncols   = 2
    nrows   = math.ceil(len(items) / ncols)
    pad     = 6
    label_w = 90
    item_w  = tile_size + label_w + pad
    item_h  = tile_size + pad
    legend  = Image.new("RGB", (item_w * ncols + pad, item_h * nrows + pad), C_LEGEND_BG)
    draw    = ImageDraw.Draw(legend)
    for i, (obj, label) in enumerate(items):
        col = i % ncols
        row = i // ncols
        x = pad + col * item_w
        y = pad + row * item_h
        legend.paste(render_tile(obj, size=tile_size), (x, y))
        bbox = draw.textbbox((0, 0), label, font=font)
        th = bbox[3] - bbox[1]
        draw.text((x + tile_size + 5, y + (tile_size - th) // 2), label, fill=(30, 30, 30), font=font)
    return legend

def render_env(grid: np.ndarray, tile_size: int = TILE_PX) -> Image.Image:
    H, W = grid.shape
    img = Image.new("RGB", (W * tile_size, H * tile_size), C_BG)
    for r in range(H):
        for c in range(W):
            img.paste(render_tile(grid[r, c], size=tile_size), (c * tile_size, r * tile_size))
    return img

def parse_layout_string(layout_str: str) -> np.ndarray:
    rows = [r for r in layout_str.split('\n') if r]
    H    = len(rows)
    W    = max(len(r) for r in rows)
    grid = np.full((H, W), EMPTY, dtype=np.int32)
    agent_count = 0
    for r, row in enumerate(rows):
        for c, ch in enumerate(row):
            if ch == 'W':
                grid[r, c] = OBSTACLE
            elif ch == 'X':
                grid[r, c] = GOAL
            elif ch == 'B':
                grid[r, c] = PLATE_PILE
            elif ch == 'P':
                grid[r, c] = POT
            elif ch == 'A':
                grid[r, c] = AGENT_0 if agent_count == 0 else AGENT_1
                agent_count += 1
            elif ch == 'O':
                grid[r, c] = INGREDIENT_0
            elif ch.isdigit():
                grid[r, c] = INGREDIENT_0 if int(ch) % 2 == 0 else INGREDIENT_1
    return grid

# ── Parametrized environment ───────────────────────────────────────────────────

class ParametrizedOvercooked:
    """Parametrized Overcooked wrapper handling discrete mutations via continuous spatial scaling."""
    codes = GridCodes

    @staticmethod
    def make_layout(H, W, params: "EnvParams", key, **kwargs) -> np.ndarray:
        ingredient_types = kwargs.pop("ingredient_types", (INGREDIENT_0,))
        return generate_random_layout(
            H, W,
            params.obs_density, params.obs_skew_x,
            params.res_density, params.res_skew_x,
            key,
            ing0=int(ingredient_types[0]),
            ing1=int(ingredient_types[-1]),
            **kwargs,
        )

    @staticmethod
    def validate(grid: np.ndarray) -> bool:
        return is_valid_layout(grid)

    @staticmethod
    def from_string(layout_str: str) -> np.ndarray:
        return parse_layout_string(layout_str)

    def __init__(self, base_grid: np.ndarray, seed: int = 42, curriculum: bool = False, min_cycle: int = 0, max_cycle: int = 999):
        self.base_grid = base_grid
        self.H, self.W = base_grid.shape
        self.seed = seed
        self.key  = jax.random.PRNGKey(seed)

        self.num_agents = int((base_grid == AGENT_0).sum() + (base_grid == AGENT_1).sum())
        self.num_goals  = int((base_grid == GOAL).sum())
        self.num_pots   = int((base_grid == POT).sum())
        self.num_plates = int((base_grid == PLATE_PILE).sum())
        self.max_goals  = self.num_agents + 2
        self._max_pots    = self.num_pots
        self._max_plates  = self.num_plates
        self._dims_only   = False

        self.ingredient_types = tuple(
            t for t in (INGREDIENT_0, INGREDIENT_1) if (base_grid == t).any()
        ) or (INGREDIENT_0,)

        self.current_params = EnvParams()
        self.current_seed   = 0
        self._counter       = 0
        self.grid = base_grid.copy()

        self.curriculum   = curriculum
        self.cycle_range  = (min_cycle, max_cycle)
        self.target_pots  = None

    @classmethod
    def from_dims(cls, H: int, W: int, seed: int = 42, max_pots: int = 3, max_goals: int = 3, max_plates: int = 2):
        obj = cls.__new__(cls)
        obj.base_grid     = None
        obj.H, obj.W      = H, W
        obj.seed          = seed
        obj.key           = jax.random.PRNGKey(seed)
        obj.num_agents    = 2
        obj.num_goals     = 1
        obj.num_pots      = 1
        obj.num_plates    = 1
        obj.max_goals     = max_goals
        obj._max_pots     = max_pots
        obj._max_plates   = max_plates
        obj._dims_only    = True
        obj.ingredient_types = (INGREDIENT_0, INGREDIENT_1)
        obj.current_params = EnvParams()
        obj.current_seed  = 0
        obj._counter      = 0
        obj.grid          = None
        obj.curriculum    = False
        obj.cycle_range   = (0, 999)
        
        obj.obs_range = [0.0, MAX_OBS_FRAC]
        obj.res_range = [0.0, MAX_RES_FRAC]
        obj.target_pots = None

        return obj

    def _generate_from_key(self, candidate_key) -> Tuple[np.ndarray, "EnvParams"]:
        """Python path generating discrete numpy grids from continuous distributions."""
        k_params, k_layout = jax.random.split(candidate_key)
        k_obs, k_res, k_ng, k_np = jax.random.split(k_params, 4)
        

        obs_density = float(jax.random.uniform(k_obs, minval=self.obs_range[0], maxval=self.obs_range[1]))
        obs_skew_x  = float(jax.random.uniform(k_res, minval=-1.0, maxval=1.0))
        res_density = float(jax.random.uniform(k_ng,  minval=self.res_range[0], maxval=self.res_range[1]))
        res_skew_x  = float(jax.random.uniform(k_np,  minval=-1.0, maxval=1.0))
        
        params = EnvParams(obs_density, obs_skew_x, res_density, res_skew_x)

        k_ng2, k_np2, k_npl = jax.random.split(k_layout, 3)
        num_goals  = int(jax.random.randint(k_ng2, (), 1, self.max_goals + 1))
        num_pots   = int(np.clip(self.target_pots, 1, self._max_pots)) if self.target_pots is not None \
            else int(jax.random.randint(k_np2, (), 1, self._max_pots + 1))
        num_plates = int(jax.random.randint(k_npl, (), 1, self._max_plates + 1)) if self._dims_only else self.num_plates

        grid = generate_random_layout(
            self.H, self.W,
            obs_density, obs_skew_x, res_density, res_skew_x,
            k_layout,
            num_agents=self.num_agents,
            num_goals=num_goals,
            num_pots=num_pots,
            num_plates=num_plates,
            ing0=int(self.ingredient_types[0]),
            ing1=int(self.ingredient_types[-1]),
        )
        return np.asarray(grid), params

    def _generate_from_key_jax(self, candidate_key) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Pure JAX pathway keeping all vectors completely compiled within XLA."""
        k_params, k_layout = jax.random.split(candidate_key)
        k_obs, k_res, k_ng, k_np = jax.random.split(k_params, 4)

        obs_density = jax.random.uniform(k_obs, minval=float(self.obs_range[0]), maxval=float(self.obs_range[1]))
        obs_skew_x  = jax.random.uniform(k_res, minval=-1.0, maxval=1.0)
        res_density = jax.random.uniform(k_ng,  minval=0.0, maxval=float(MAX_RES_FRAC))
        res_skew_x  = jax.random.uniform(k_np,  minval=-1.0, maxval=1.0)

        k_ng2, k_np2 = jax.random.split(k_layout)
        num_goals = jax.random.randint(k_ng2, (), 1, self.max_goals + 1)
        num_pots = jnp.asarray(int(np.clip(self.target_pots, 1, self._max_pots)), dtype=jnp.int32) \
            if self.target_pots is not None else jax.random.randint(k_np2, (), 1, self._max_pots + 1)

        grid = generate_random_layout(
            self.H, self.W,
            obs_density, obs_skew_x, res_density, res_skew_x,
            k_layout,
            num_agents=self.num_agents,
            num_goals=num_goals,
            num_pots=num_pots,
            num_plates=self.num_plates,
            ing0=int(self.ingredient_types[0]),
            ing1=int(self.ingredient_types[-1]),
        )
        return grid, jnp.stack([obs_density, obs_skew_x, res_density, res_skew_x])

    def reset(self, max_retries: int = 50) -> Tuple[np.ndarray, EnvParams]:
        for _ in range(max_retries):
            self._counter += 1
            key = jax.random.fold_in(jax.random.PRNGKey(self.seed), self._counter)
            candidate, params = self._generate_from_key(key)
            if is_valid_layout(candidate):
                if self.curriculum:
                    c = compute_min_cycle(candidate)
                    if not (self.cycle_range[0] <= c <= self.cycle_range[1]):
                        continue
                self.current_params = params
                self.current_seed   = self._counter
                self.grid = candidate
                return params.to_vector(), params

        self.current_params = EnvParams()
        self.current_seed   = 0
        self.grid = self.base_grid.copy() if self.base_grid is not None else np.zeros((self.H, self.W), dtype=np.int32)
        return EnvParams().to_vector(), EnvParams()
    
    def set_cycle_range(self, min_cycle: int, max_cycle: int):
        self.cycle_range = (min_cycle, max_cycle)

    def encoding(self) -> Tuple[np.ndarray, int]:
        return self.current_params.to_vector(), self.current_seed
    
    def _mutate_grid_jax(self, grid: jnp.ndarray, key: jnp.ndarray, edit_step: int = 1) -> jnp.ndarray:
        """
        Universal Grid Mutation for ACCEL and Evolutionary SFL.
        Randomly flips 'edit_step' number of tiles (Empty <-> Wall).
        """
        N = self.H * self.W
        
        def _single_mutation(carry, _):
            current_grid, current_key = carry
            k1, k2, next_key = jax.random.split(current_key, 3)
            
            # Identify valid cells to mutate (do not touch agents, pots, goals, ingredients)
            is_empty = (current_grid == EMPTY)
            is_wall = (current_grid == OBSTACLE)
            
            # Safely pick one empty cell and one wall cell
            # Adding a tiny epsilon to probabilities to avoid NaNs if there are no walls/empty cells
            p_empty = is_empty.astype(jnp.float32) + 1e-8
            p_wall = is_wall.astype(jnp.float32) + 1e-8
            
            empty_idx = jax.random.choice(k1, jnp.arange(N), p=p_empty)
            wall_idx = jax.random.choice(k2, jnp.arange(N), p=p_wall)
            
            # 50% chance to add a wall, 50% chance to remove a wall
            add_wall = jax.random.bernoulli(k1, 0.5)
            
            # Apply the mutation
            new_grid = jax.lax.cond(
                add_wall,
                lambda g: g.at[empty_idx].set(jnp.int32(OBSTACLE)),
                lambda g: g.at[wall_idx].set(jnp.int32(EMPTY)),
                current_grid
            )
            return (new_grid, next_key), None

        # Flatten the grid for easier 1D indexing
        flat_grid = grid.flatten()
        
        # Apply the mutation 'edit_step' times using a JAX scan loop
        (mutated_flat, _), _ = jax.lax.scan(_single_mutation, (flat_grid, key), None, length=edit_step)
        
        return mutated_flat.reshape(self.H, self.W)

# ── Layout stats helper ────────────────────────────────────────────────────────

def _grid_stats(grid: np.ndarray) -> dict:
    """Computes left vs right balances to let you evaluate the gradient skew performance."""
    H, W = grid.shape
    mid = W / 2
    interior = [(r, c) for r in range(1, H - 1) for c in range(1, W - 1)]
    all_cells = [(r, c) for r in range(H) for c in range(W)]

    int_l = [(r, c) for r, c in interior if c <  mid]
    int_r = [(r, c) for r, c in interior if c >= mid]
    all_l = [(r, c) for r, c in all_cells if c <  mid]
    all_r = [(r, c) for r, c in all_cells if c >= mid]

    obs_l = sum(1 for r, c in int_l if grid[r, c] == OBSTACLE)
    obs_r = sum(1 for r, c in int_r if grid[r, c] == OBSTACLE)

    skip = {AGENT_0, AGENT_1, OBSTACLE}
    ing_l = sum(1 for r, c in all_l if grid[r, c] in (INGREDIENT_0, INGREDIENT_1))
    ing_r = sum(1 for r, c in all_r if grid[r, c] in (INGREDIENT_0, INGREDIENT_1))
    ing_den_l = sum(1 for r, c in all_l if grid[r, c] not in skip)
    ing_den_r = sum(1 for r, c in all_r if grid[r, c] not in skip)

    return dict(
        obs_l=obs_l, obs_dl=len(int_l),
        obs_r=obs_r, obs_dr=len(int_r),
        ing_l=ing_l, ing_dl=ing_den_l,
        ing_r=ing_r, ing_dr=ing_den_r,
    )

# ── Interactive visualization window ──────────────────────────────────────────

def run_interactive_viz(
    layout_str: str = None,
    seed: int = 42,
    curriculum: bool = False,
    min_cycle: int = 0,
    max_cycle: int = 999,
    grid_size: tuple = None,
):
    import matplotlib
    for backend in ("MacOSX", "Qt5Agg", "GTK3Agg", "WXAgg"):
        try:
            matplotlib.use(backend)
            break
        except Exception:
            continue
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
    from matplotlib.widgets import Button

    dims_mode = grid_size is not None

    if dims_mode:
        H, W  = grid_size
        env   = ParametrizedOvercooked.from_dims(H, W, seed=seed)
        # generate one sample to display in the "first layout" panel
        env.reset()
        first_grid = env.grid.copy()
        legend_grid = first_grid
    else:
        layout_str = layout_str or DEFAULT_LAYOUT
        base_grid  = parse_layout_string(layout_str)
        H, W       = base_grid.shape
        env        = ParametrizedOvercooked(
            base_grid=base_grid, seed=seed,
            curriculum=curriculum, min_cycle=min_cycle, max_cycle=max_cycle,
        )
        first_grid  = base_grid
        legend_grid = base_grid

    fig = plt.figure(figsize=(14, 8))
    title = f"Random Dims {W}×{H}" if dims_mode else "Parametrized Gradient Environment Visualization"
    fig.suptitle(title, fontweight="bold", fontsize=12, y=0.98)

    gs = GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.2, left=0.04, right=0.97, top=0.93, bottom=0.12)

    # ── First layout panel ────────────────────────────────────────────────
    ax_base_img = fig.add_subplot(gs[0, 0])
    ax_base_img.imshow(np.array(render_env(first_grid)))
    ax_base_img.axis("off")
    ax_base_img.set_title("Sample Layout" if dims_mode else "Base Layout", fontweight="bold", fontsize=9)

    if dims_mode:
        base_info = (
            f"Grid: {W}×{H}   seed={seed}\n"
            f"Mode: random dims (like training)\n"
            f"max_pots={env._max_pots}  max_goals={env.max_goals}\n"
            f"max_plates={env._max_plates}\n\n"
            "Vector encoding:\n"
            "  [obs_den, obs_skew_x, res_den, res_skew_x]\n"
            "  All randomized per reset"
        )
    else:
        base_info = (
            f"Grid: {W}×{H}   seed={seed}\n"
            f"Agents: {env.num_agents}  Goals: {env.num_goals}  "
            f"Pots: {env.num_pots}  Plates: {env.num_plates}\n\n"
            "Vector encoding details:\n"
            "  [obs_den, obs_skew_x, res_den, res_skew_x]\n"
            "  Densities mapped [0, 1] relative to Caps\n"
            "  Skews mapped from [-1, 1] range to [0, 1]"
        )
    ax_base_info = fig.add_subplot(gs[1, 0])
    ax_base_info.axis("off")
    ax_base_info.set_title("Layout Info", fontweight="bold", fontsize=9)
    ax_base_info.text(0.05, 0.97, base_info, transform=ax_base_info.transAxes, fontsize=8, va="top", fontfamily="monospace")

    # ── Mutation slots ────────────────────────────────────────────────────
    SLOTS = [(0, 1), (0, 2), (1, 1)]
    slot_axes: list[tuple] = []
    for r, c in SLOTS:
        inner = GridSpecFromSubplotSpec(2, 1, subplot_spec=gs[r, c], height_ratios=[3, 1], hspace=0.08)
        ax_img = fig.add_subplot(inner[0])
        ax_txt = fig.add_subplot(inner[1])
        ax_txt.axis("off")
        slot_axes.append((ax_img, ax_txt))

    # ── Legend ────────────────────────────────────────────────────────────
    ax_leg = fig.add_subplot(gs[1, 2])
    ax_leg.imshow(np.array(render_legend_2col(legend_grid, extra_types=(OBSTACLE,))))
    ax_leg.axis("off")
    ax_leg.set_title("Legend", fontweight="bold", fontsize=9)

    # ── Button ────────────────────────────────────────────────────────────
    ax_btn = fig.add_axes([0.40, 0.03, 0.20, 0.05])
    btn_label = "Generate Layout" if dims_mode else "Generate Mutation"
    btn    = Button(ax_btn, btn_label)
    slot_idx = [0]

    def _on_click(_event):
        if slot_idx[0] >= len(slot_axes):
            for ax_i, ax_t in slot_axes:
                ax_i.cla(); ax_t.cla(); ax_t.axis("off")
            slot_idx[0] = 0

        vec, params = env.reset()
        ax_i, ax_t = slot_axes[slot_idx[0]]

        ax_i.imshow(np.array(render_env(env.grid)))
        ax_i.axis("off")
        slot_label = "Layout" if dims_mode else "Mutation"
        ax_i.set_title(f"{slot_label} {slot_idx[0] + 1}", fontweight="bold", fontsize=9)

        n_goals = int((env.grid == GOAL).sum())
        n_pots  = int((env.grid == POT).sum())
        st    = _grid_stats(env.grid)
        cycle = compute_min_cycle(env.grid)

        enc = (
            f"Vector: [{vec[0]:.2f}, {vec[1]:.2f}, {vec[2]:.2f}, {vec[3]:.2f}]\n"
            f"Params: den_O={params.obs_density:.2f} skw_O={params.obs_skew_x:.2f}\n"
            f"        den_R={params.res_density:.2f} skw_R={params.res_skew_x:.2f}\n"
            f"Counts: goals={n_goals}  pots={n_pots}  "
            f"walls={st['obs_l']}/{st['obs_r']}  res={st['ing_l']}/{st['ing_r']}\n"
            f"Metrics: path_cycle={cycle}  seed={env.current_seed}"
        )
        ax_t.text(0.5, 0.5, enc, transform=ax_t.transAxes, fontsize=7.5, ha="center", va="center", fontfamily="monospace")

        slot_idx[0] += 1
        fig.canvas.draw_idle()

    btn.on_clicked(_on_click)
    plt.show(block=True)

if __name__ == "__main__":
    import argparse as _ap
    _p = _ap.ArgumentParser(description="Parametrized Overcooked Gradient Visualizer")
    _p.add_argument("layout",       nargs="?", default=None, help="Named layout from layouts.py (ignored if --dims is set)")
    _p.add_argument("--dims",       default=None, metavar="HxW", help="Random dims mode, e.g. 5x9 — same process as ACCEL/SFL training")
    _p.add_argument("--seed",       type=int, default=42)
    _p.add_argument("--curriculum", action="store_true")
    _p.add_argument("--min-cycle",  type=int, default=0)
    _p.add_argument("--max-cycle",  type=int, default=999)
    _args = _p.parse_args()

    if _args.dims is not None:
        _h, _w = (int(x) for x in _args.dims.lower().split("x"))
        run_interactive_viz(grid_size=(_h, _w), seed=_args.seed)
    else:
        _layout_str = get_layout(_args.layout) if _args.layout else DEFAULT_LAYOUT
        run_interactive_viz(
            layout_str=_layout_str,
            seed=_args.seed,
            curriculum=_args.curriculum,
            min_cycle=_args.min_cycle,
            max_cycle=_args.max_cycle,
        )
        
# python level_generator.py --dims 7x7 --seed 42
