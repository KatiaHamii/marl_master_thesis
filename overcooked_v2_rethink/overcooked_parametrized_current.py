"""
overcooked_parametrized.py — Parametrized Overcooked environment with PIL rendering

Works with any layout from layouts.py.  The base is parsed from a layout string;
mutations generate fully random valid layouts within configurable constraints.

4-parameter encoding (each normalised to [0, 1]):
    obstacles_left   fraction of left-half interior cells → obstacle walls   [0, 0.8]
    obstacles_right  same for right half
    resources_left   fraction of remaining left cells → ingredient piles      [0, 0.8]
    resources_right  same for right half

Object counts (agents, goals, pots, plates) are derived from the base layout and
varied randomly across mutations; only density and position of walls/resources
are captured in the 4-vector.
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
    Fraction-based difficulty parameters.

    obstacles_left/right : fraction of interior cells per half → walls    [0, 0.8]
    resources_left/right : fraction of remaining cells per half → ingredients [0, 0.8]
    """
    obstacles_left:  float = 0.0
    obstacles_right: float = 0.0
    resources_left:  float = 0.0
    resources_right: float = 0.0

    def to_vector(self) -> np.ndarray:
        return np.array([
            self.obstacles_left  / MAX_OBS_FRAC,
            self.obstacles_right / MAX_OBS_FRAC,
            self.resources_left  / MAX_RES_FRAC,
            self.resources_right / MAX_RES_FRAC,
        ], dtype=np.float32)

    def __str__(self) -> str:
        v = self.to_vector()
        return (
            f"obs_L={self.obstacles_left:.2f}({v[0]:.2f})  "
            f"obs_R={self.obstacles_right:.2f}({v[1]:.2f})  "
            f"res_L={self.resources_left:.2f}({v[2]:.2f})  "
            f"res_R={self.resources_right:.2f}({v[3]:.2f})"
        )

# ── Random layout generator ────────────────────────────────────────────────────

def generate_random_layout(
    H: int,
    W: int,
    params: EnvParams,
    key,
    num_agents: int = 2,
    num_goals:  int = 2,
    num_pots:   int = 2,
    num_plates: int = 1,
    ingredient_types: tuple = (INGREDIENT_0, INGREDIENT_1),
) -> np.ndarray:
    """
    Build a fully random layout within the density constraints.

    Only H × W is fixed.  No border is forced — every cell is up for grabs.
    Placement order:
      1. Agents       → interior only (they must walk)
      2. Obstacles    → interior only, left/right halves by density param
      3. Ingredients  → any remaining cell, using only the types from the base
      4. Goals / pots / plates → any remaining cell

    ingredient_types controls which ingredient IDs appear (derived from the
    base layout so e.g. cramped_room never spawns INGREDIENT_1).
    """
    grid = np.full((H, W), EMPTY, dtype=np.int32)
    occupied: set = set()
    mid = W / 2

    all_cells      = [(r, c) for r in range(H) for c in range(W)]
    interior_cells = [(r, c) for r in range(H) for c in range(W)
                      if 0 < r < H - 1 and 0 < c < W - 1]

    keys = jax.random.split(key, 10)

    def _place(pool, n, obj, rng_key):
        if n <= 0 or not pool:
            return
        perm = np.array(jax.random.permutation(rng_key, len(pool)))
        placed = 0
        for i in perm:
            if placed >= n:
                break
            r, c = pool[int(i)]
            if (r, c) not in occupied:
                grid[r, c] = obj
                occupied.add((r, c))
                placed += 1

    # 1. Agents — interior only
    k_agents = jax.random.split(keys[8], max(num_agents, 2))
    for i, agent_obj in enumerate([AGENT_0, AGENT_1][:num_agents]):
        _place([p for p in interior_cells if p not in occupied], 1, agent_obj, k_agents[i])

    # 2. Walls (obstacles) — interior only, per half
    int_left  = [p for p in interior_cells if p[1] <  mid and p not in occupied]
    int_right = [p for p in interior_cells if p[1] >= mid and p not in occupied]
    _place(int_left,  int(params.obstacles_left  * len(int_left)),  OBSTACLE, keys[0])
    _place(int_right, int(params.obstacles_right * len(int_right)), OBSTACLE, keys[1])

    # 3. Ingredients — any cell, left→type[0], right→type[-1] (same if only one type)
    ing_left  = ingredient_types[0]
    ing_right = ingredient_types[-1]
    rem_left  = [p for p in all_cells if p[1] <  mid and p not in occupied]
    rem_right = [p for p in all_cells if p[1] >= mid and p not in occupied]
    _place(rem_left,  int(params.resources_left  * len(rem_left)),  ing_left,  keys[2])
    _place(rem_right, int(params.resources_right * len(rem_right)), ing_right, keys[3])

    # 4. Goals, pots, plate piles — any remaining cell
    def _free():
        return [p for p in all_cells if p not in occupied]

    _place(_free(), num_goals,  GOAL,       keys[4])
    _place(_free(), num_pots,   POT,        keys[5])
    _place(_free(), num_plates, PLATE_PILE, keys[6])

    return grid



# ── Reachability check (BFS) ───────────────────────────────────────────────────
def compute_min_cycle(grid: np.ndarray) -> int:
    """Return the minimum delivery cycle length for the given layout.

    Approximates the shortest path for one complete delivery:
      agents → ingredient → pot → plate_pile → goal

    Each segment is the minimum BFS distance from any cell of the source
    type to any adjacent cell of the target type.  Shorter total = easier.
    """
    H, W = grid.shape
    walkable = {EMPTY, AGENT_0, AGENT_1}
    dirs = [(-1, 0), (1, 0), (0, -1), (0, 1)] # one row up, one row down, one col left, one col right

    def _bfs_min(starts: list[tuple], targets: frozenset) -> int:
        """Min BFS steps walking from any start until adjacent to any target."""
        visited: set = set(starts)
        queue: deque = deque(starts)
        steps = 0
        while queue:
            for _ in range(len(queue)):
                r, c = queue.popleft() #row, col
                for dr, dc in dirs: # delta row, delta col
                    nr, nc = r + dr, c + dc # compute neighbour cells
                    if not (0 <= nr < H and 0 <= nc < W):
                        continue
                    if grid[nr, nc] in targets:
                        return steps + 1
                    if (nr, nc) not in visited and grid[nr, nc] in walkable:
                        visited.add((nr, nc))
                        queue.append((nr, nc))
            steps += 1
        return 9999  # unreachable

    agents      = [(r, c) for r in range(H) for c in range(W)
                   if grid[r, c] in (AGENT_0, AGENT_1)]
    ingredients = [(r, c) for r in range(H) for c in range(W)
                   if grid[r, c] in (INGREDIENT_0, INGREDIENT_1)]
    pots        = [(r, c) for r in range(H) for c in range(W)
                   if grid[r, c] == POT]
    plates      = [(r, c) for r in range(H) for c in range(W)
                   if grid[r, c] == PLATE_PILE]

    if not (agents and ingredients and pots and plates):
        return 9999

    d1 = _bfs_min(agents,      frozenset({INGREDIENT_0, INGREDIENT_1}))
    d2 = _bfs_min(ingredients, frozenset({POT}))
    d3 = _bfs_min(pots,        frozenset({PLATE_PILE}))
    d4 = _bfs_min(plates,      frozenset({GOAL}))
    return d1 + d2 + d3 + d4

def is_valid_layout(grid: np.ndarray) -> bool:
    """
    Return True iff the layout is task-completable:

      1. Every agent can adjacently reach at least one GOAL.
      2. All four task categories exist in the grid:
         ingredient (INGREDIENT_0/1), POT, PLATE_PILE, GOAL.
      3. At least one agent can adjacently reach ALL four categories.

    "Adjacently reachable" means BFS over walkable cells (EMPTY/AGENT) visits
    a cell that is a direct neighbor of the target object.
    """
    H, W = grid.shape
    walkable = {EMPTY, AGENT_0, AGENT_1}

    task_cats: dict = {
        "ingredient": frozenset({INGREDIENT_0, INGREDIENT_1}),
        "pot":        frozenset({POT}),
        "plate":      frozenset({PLATE_PILE}),
        "goal":       frozenset({GOAL}),
    }

    agent_pos = [(r, c) for r in range(H) for c in range(W)
                 if grid[r, c] in (AGENT_0, AGENT_1)]
    if not agent_pos:
        return False

    # All four task-critical object types must be present
    for cat_types in task_cats.values():
        if not any(grid[r, c] in cat_types for r in range(H) for c in range(W)):
            return False

    all_task_types = frozenset(t for ts in task_cats.values() for t in ts)
    dirs = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    def reachable_cats(start: tuple) -> set:
        """BFS from *start*; return set of task category names adjacently touched."""
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

    # Rule 1: every agent reaches a goal, for further steps
    if not all("goal" in cats for cats in per_agent):
        return False

    # Rule 2: at least one agent reaches everything needed to cook
    return any(cats >= required for cats in per_agent)


# ── Tile renderer (PIL, 64×64 px) ─────────────────────────────────────────────

_PILE_POS = [(0.50, 0.15), (0.28, 0.42), (0.78, 0.38), (0.38, 0.78), (0.72, 0.74)] #fixed pixel positions of the circles drawn inside a single tile for pile objects 


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
        # body — wide, fills most of the cell
        pad = max(3, int(0.10 * s))
        body_top = int(0.30 * s)
        draw.rectangle([(pad, body_top), (s - 1 - pad, int(0.90 * s))], fill=C_POT)
        # handle — narrow knob centred on top of the body
        hw = max(3, int(0.09 * s))
        cx = s // 2
        draw.rectangle([(cx - hw, int(0.12 * s)), (cx + hw, body_top)], fill=C_POT)

    elif obj == PLATE_PILE:
        _circles_on_grey(draw, [(0.30, 0.30), (0.72, 0.40), (0.40, 0.72)],
                         C_WHITE, r_frac=0.18, s=size)

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


def render_legend_2col(grid: np.ndarray, tile_size: int = TILE_PX,
                       extra_types: tuple = ()) -> Image.Image:
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
        draw.text((x + tile_size + 5, y + (tile_size - th) // 2),
                  label, fill=(30, 30, 30), font=font)
    return legend

# ── Full environment render ────────────────────────────────────────────────────

def render_env(grid: np.ndarray, tile_size: int = TILE_PX) -> Image.Image:
    H, W = grid.shape
    img = Image.new("RGB", (W * tile_size, H * tile_size), C_BG)
    for r in range(H):
        for c in range(W):
            img.paste(render_tile(grid[r, c], size=tile_size), (c * tile_size, r * tile_size))
    return img

# ── Layout string parser ───────────────────────────────────────────────────────

def parse_layout_string(layout_str: str) -> np.ndarray:
    """
    Convert an Overcooked layout string to a rendering grid.

    W → BORDER_WALL (edge) or OBSTACLE (interior)
    X → GOAL   B → PLATE_PILE   P → POT
    A → AGENT_0 (first), AGENT_1 (second)
    O → INGREDIENT_0 (onion, always type 0)
    digit → even digit = INGREDIENT_0, odd digit = INGREDIENT_1
    """
    rows = [r for r in layout_str.split('\n') if r]
    H    = len(rows)
    W    = max(len(r) for r in rows)
    grid = np.full((H, W), EMPTY, dtype=np.int32)
    agent_count = 0
    for r, row in enumerate(rows):
        for c, ch in enumerate(row):
            border = (r == 0 or r == H - 1 or c == 0 or c == W - 1)
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
                grid[r, c] = INGREDIENT_0          # O = onion = always type 0
            elif ch.isdigit():
                # digit value selects ingredient type: even→0, odd→1
                grid[r, c] = INGREDIENT_0 if int(ch) % 2 == 0 else INGREDIENT_1
    return grid

# ── Parametrized environment ───────────────────────────────────────────────────

class ParametrizedOvercooked:
    """
    Parametrized Overcooked environment — works with any parsed layout.

    The base grid defines grid size and object counts (agents, goals, pots,
    plates).  Each reset() generates a fully random layout by sampling density
    params uniformly in [0, MAX_*_FRAC] and calling generate_random_layout().

    Encoding: (vector ∈ [0,1]^4, seed)

    Single-import usage (no need to import individual constants or helpers)::

        from overcooked_v2_rethink.overcooked_parametrized_current import (
            ParametrizedOvercooked, EnvParams
        )
        G = ParametrizedOvercooked.codes        # GridCodes namespace
        grid = ParametrizedOvercooked.make_layout(...)
        ok   = ParametrizedOvercooked.validate(grid)
        g    = ParametrizedOvercooked.from_string(layout_str)
    """

    # ── Class-level access to codes and helpers ────────────────────────────────
    codes = GridCodes

    @staticmethod
    def make_layout(H, W, params, key, **kwargs) -> np.ndarray:
        """Alias for generate_random_layout — available without a separate import."""
        return generate_random_layout(H, W, params, key, **kwargs)

    @staticmethod
    def validate(grid: np.ndarray) -> bool:
        """Alias for is_valid_layout — available without a separate import."""
        return is_valid_layout(grid)

    @staticmethod
    def from_string(layout_str: str) -> np.ndarray:
        """Alias for parse_layout_string — available without a separate import."""
        return parse_layout_string(layout_str)

    # ── Instance ───────────────────────────────────────────────────────────────

    def __init__(self, base_grid: np.ndarray, seed: int = 42, curriculum: bool = False, min_cycle: int = 0, max_cycle: int = 999):
        self.base_grid = base_grid
        self.H, self.W = base_grid.shape
        self.seed = seed
        self.key  = jax.random.PRNGKey(seed)

        # Counts and object types derived from base layout (stay fixed across mutations)
        self.num_agents = int((base_grid == AGENT_0).sum() + (base_grid == AGENT_1).sum())
        self.num_goals  = int((base_grid == GOAL).sum())
        self.num_pots   = int((base_grid == POT).sum())
        self.num_plates = int((base_grid == PLATE_PILE).sum())
        self.max_goals  = self.num_agents + 2

        # Only use ingredient types present in the base (e.g. cramped_room has only INGREDIENT_0)
        self.ingredient_types = tuple(
            t for t in (INGREDIENT_0, INGREDIENT_1) if (base_grid == t).any()
        ) or (INGREDIENT_0,)

        self.current_params = EnvParams()
        self.current_seed   = 0
        self._counter       = 0          # increments on every attempt
        self.grid = base_grid.copy()
        
        # Curriculum learning parameters (allocation radius from CACTUS https://arxiv.org/abs/2401.05860)
        self.curriculum   = curriculum 
        self.cycle_range  = (min_cycle, max_cycle)

    # ── internal: generate from a single JAX key ──────────────────────────────

    def _generate_from_key(self, candidate_key) -> Tuple[np.ndarray, "EnvParams"]:
        """Deterministically produce a layout+params from one JAX key."""
        k_params, k_layout = jax.random.split(candidate_key)
        k_obs, k_res, k_ng, k_np = jax.random.split(k_params, 4)

        params = EnvParams(
            obstacles_left=float(jax.random.uniform(k_obs, minval=0.0, maxval=MAX_OBS_FRAC)),
            obstacles_right=float(jax.random.uniform(k_res, minval=0.0, maxval=MAX_OBS_FRAC)),
            resources_left=float(jax.random.uniform(k_ng,  minval=0.0, maxval=MAX_RES_FRAC)),
            resources_right=float(jax.random.uniform(k_np,  minval=0.0, maxval=MAX_RES_FRAC)),
        )

        k_ng2, k_np2 = jax.random.split(k_layout)
        num_goals = int(jax.random.randint(k_ng2, (), 1, self.max_goals + 1))
        num_pots  = int(jax.random.randint(k_np2, (), 1, max(self.num_pots, 1) + 1))

        grid = generate_random_layout(
            self.H, self.W, params, k_layout,
            num_agents=self.num_agents,
            num_goals=num_goals,
            num_pots=num_pots,
            num_plates=self.num_plates,
            ingredient_types=self.ingredient_types,
        )
        return grid, params

    # ── public API ────────────────────────────────────────────────────────────

    def reset(self, max_retries: int = 50) -> Tuple[np.ndarray, EnvParams]:
        """
        Try up to max_retries layouts.  Each attempt uses a simple counter as
        the seed (fold_in with the env seed), so current_seed stays a small
        readable integer and any mutation can be reproduced via reset_with_seed.
        """
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
        self.grid = self.base_grid.copy()
        return EnvParams().to_vector(), EnvParams()
    
    def set_cycle_range(self, min_cycle: int, max_cycle: int):
        """Set the cycle length range for curriculum learning."""
        self.cycle_range = (min_cycle, max_cycle)

    # def reset_with_seed(self, mutation_seed: int) -> Tuple[np.ndarray, EnvParams]:
    #     """Reproduce a mutation exactly by its seed number."""
    #     key = jax.random.fold_in(jax.random.PRNGKey(self.seed), mutation_seed)
    #     candidate, params = self._generate_from_key(key)
    #     self.current_params = params
    #     self.current_seed   = mutation_seed
    #     self.grid = candidate
    #     return params.to_vector(), params

    def encoding(self) -> Tuple[np.ndarray, int]:
        return self.current_params.to_vector(), self.current_seed

# ── Layout stats helper ────────────────────────────────────────────────────────

def _grid_stats(grid: np.ndarray) -> dict:
    """
    Return human-readable counts for the mutation label:
      obs_l, obs_denom_l  — walls placed / interior-left cells available
      obs_r, obs_denom_r
      ing_l, ing_denom_l  — ingredients placed / cells available before ing. placement
      ing_r, ing_denom_r
    """
    H, W = grid.shape
    mid = W / 2
    interior = [(r, c) for r in range(1, H - 1) for c in range(1, W - 1)]
    all_cells = [(r, c) for r in range(H) for c in range(W)]

    int_l = [(r, c) for r, c in interior if c <  mid]
    int_r = [(r, c) for r, c in interior if c >= mid]
    all_l = [(r, c) for r, c in all_cells if c <  mid]
    all_r = [(r, c) for r, c in all_cells if c >= mid]

    # obstacles live in interior cells only
    obs_l = sum(1 for r, c in int_l if grid[r, c] == OBSTACLE)
    obs_r = sum(1 for r, c in int_r if grid[r, c] == OBSTACLE)

    # ingredient denominator = all cells that were free after agents+obstacles
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

def run_interactive_viz(layout_str: str = None, seed: int = 42,
                        curriculum: bool = False, min_cycle: int = 0, max_cycle: int = 999):
    """
    Open an interactive matplotlib window.

    Layout (2 rows × 3 cols):
      [base image]  [mutation 1]  [mutation 2]
      [base info ]  [mutation 3]  [legend    ]

    Click "Generate Mutation" to fill the next slot.
    After all 3 slots are filled the next click resets them.
    """
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

    layout_str = layout_str or DEFAULT_LAYOUT
    base_grid  = parse_layout_string(layout_str)
    H, W       = base_grid.shape
    env        = ParametrizedOvercooked(base_grid=base_grid, seed=seed,
                                        curriculum=curriculum,
                                        min_cycle=min_cycle, max_cycle=max_cycle)

    fig = plt.figure(figsize=(14, 8))
    fig.suptitle("Parametrized Environment Visualization (Overcooked Style)",
                 fontweight="bold", fontsize=12, y=0.98)

    gs = GridSpec(2, 3, figure=fig,
                  hspace=0.45, wspace=0.2,
                  left=0.04, right=0.97, top=0.93, bottom=0.12)

    # ── Base layout ───────────────────────────────────────────────────────
    ax_base_img = fig.add_subplot(gs[0, 0])
    ax_base_img.imshow(np.array(render_env(base_grid)))
    ax_base_img.axis("off")
    ax_base_img.set_title("Base Layout", fontweight="bold", fontsize=9)

    base_info = (
        f"Grid: {W}×{H}   seed={seed}\n"
        f"Agents: {env.num_agents}  Goals: {env.num_goals}  "
        f"Pots: {env.num_pots}  Plates: {env.num_plates}\n"
        f"Max goals per mutation: {env.max_goals}\n"
        f"Obstacle / resource cap: {MAX_OBS_FRAC*100:.0f}% of free cells\n\n"
        "Vector encoding:\n"
        "  [obs_left, obs_right, res_left, res_right]\n"
        "  each = density / 0.8,  normalised to [0, 1]"
    )
    ax_base_info = fig.add_subplot(gs[1, 0])
    ax_base_info.axis("off")
    ax_base_info.set_title("Layout Info", fontweight="bold", fontsize=9)
    ax_base_info.text(0.05, 0.97, base_info,
                      transform=ax_base_info.transAxes,
                      fontsize=8, va="top", fontfamily="monospace")

    # ── Mutation slots ────────────────────────────────────────────────────
    SLOTS = [(0, 1), (0, 2), (1, 1)]
    slot_axes: list[tuple] = []
    for r, c in SLOTS:
        inner = GridSpecFromSubplotSpec(
            2, 1, subplot_spec=gs[r, c], height_ratios=[3, 1], hspace=0.08
        )
        ax_img = fig.add_subplot(inner[0])
        ax_txt = fig.add_subplot(inner[1])
        ax_txt.axis("off")
        slot_axes.append((ax_img, ax_txt))

    # ── Legend ────────────────────────────────────────────────────────────
    ax_leg = fig.add_subplot(gs[1, 2])
    ax_leg.imshow(np.array(render_legend_2col(base_grid, extra_types=(OBSTACLE,))))
    ax_leg.axis("off")
    ax_leg.set_title("Legend", fontweight="bold", fontsize=9)
    ax_leg.text(0.98, 0.02,
                f"Grid: {W}×{H} | Tiles: {TILE_PX}×{TILE_PX}px",
                transform=ax_leg.transAxes, fontsize=6,
                ha="right", va="bottom", color="grey")

    # ── Button ────────────────────────────────────────────────────────────
    ax_btn = fig.add_axes([0.40, 0.03, 0.20, 0.05])
    btn    = Button(ax_btn, "Generate Mutation")
    slot_idx = [0]

    def _on_click(_event):
        if slot_idx[0] >= len(slot_axes):
            for ax_i, ax_t in slot_axes:
                ax_i.cla(); ax_t.cla(); ax_t.axis("off")
            slot_idx[0] = 0

        vec, _ = env.reset()
        ax_i, ax_t = slot_axes[slot_idx[0]]

        ax_i.imshow(np.array(render_env(env.grid)))
        ax_i.axis("off")
        ax_i.set_title(f"Mutation {slot_idx[0] + 1}", fontweight="bold", fontsize=9)

        n_goals = int((env.grid == GOAL).sum())
        n_pots  = int((env.grid == POT).sum())
        st    = _grid_stats(env.grid)
        cycle = compute_min_cycle(env.grid)  # theoretical minimum walking distance if the agent already knows the perfect path
        enc = (
            f"[{vec[0]:.2f}, {vec[1]:.2f}, {vec[2]:.2f}, {vec[3]:.2f}]\n"
            f"obs  L={st['obs_l']}/{st['obs_dl']}  R={st['obs_r']}/{st['obs_dr']}   "
            f"res  L={st['ing_l']}/{st['ing_dl']}  R={st['ing_r']}/{st['ing_dr']}\n"
            f"goals={n_goals}  pots={n_pots}  cycle={cycle}  seed={env.current_seed}"
        )
        ax_t.text(0.5, 0.5, enc, transform=ax_t.transAxes, fontsize=7.5,
                  ha="center", va="center", fontfamily="monospace")

        slot_idx[0] += 1
        fig.canvas.draw_idle()

    btn.on_clicked(_on_click)
    plt.show(block=True)

# ── Standalone entry point ─────────────────────────────────────────────────────

# random mode (unchanged)
# python overcooked_parametrized.py cramped_room

# curriculum mode — only show layouts with cycle between 6 and 12
# python overcooked_parametrized.py cramped_room --curriculum --min-cycle 6 --max-cycle 12

if __name__ == "__main__":
    import argparse as _ap
    _p = _ap.ArgumentParser(description="Parametrized Overcooked visualizer")
    _p.add_argument("layout",      nargs="?", default=None,
                    help="Layout name from layouts.py (default: asymm_advantages)")
    _p.add_argument("--seed",      type=int, default=42)
    _p.add_argument("--curriculum", action="store_true",
                    help="Only accept layouts whose min delivery cycle is in [min-cycle, max-cycle]")
    _p.add_argument("--min-cycle", type=int, default=0,
                    help="Minimum delivery cycle length (curriculum mode)")
    _p.add_argument("--max-cycle", type=int, default=999,
                    help="Maximum delivery cycle length (curriculum mode)")
    _args = _p.parse_args()

    _layout_str = get_layout(_args.layout) if _args.layout else DEFAULT_LAYOUT
    run_interactive_viz(
        layout_str=_layout_str,
        seed=_args.seed,
        curriculum=_args.curriculum,
        min_cycle=_args.min_cycle,
        max_cycle=_args.max_cycle,
    )
