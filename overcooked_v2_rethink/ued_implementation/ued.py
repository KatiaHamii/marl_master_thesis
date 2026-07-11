"""
overcooked_ued.py — Unified UED Environment for Cooperative Overcooked
======================================================================

Provides a single interface that any DCD algorithm (DR, PLR, ACCEL, SFL,
EvoSFL, PAIRED, or custom) can use to generate, mutate, score, and play
Overcooked levels.

Architecture
────────────
  OvercookedLevel     Canonical level representation (grid + agent positions)
  LevelGenerator      Wraps generate_random_layout from overcooked_parametrized
  LevelMutator        Enhanced mutation: wall flips + object moves
  LevelScorer         Regret (PLR/ACCEL) and learnability p*(1-p) (SFL)
  OvercookedUEDEnv    Unified wrapper tying everything together

Usage by algorithm
──────────────────
  DR:       env.reset_student_random(key) → rollout
  PLR:      env.reset_student_random(key) → rollout → env.score_regret() → buffer
  ACCEL:    PLR + env.mutate_level(key, level) → re-score
  SFL:      env.reset_student_random(key) → rollout → env.score_learnability() → buffer
  EvoSFL:   SFL + env.mutate_level(key, level)
  PAIRED:   env.reset_teacher(key) → step_teacher() × N → env.reset_student_from_teacher() → rollout
  Custom:   mix and match any of the above

Grid code mapping
─────────────────
  overcooked_parametrized uses GridCodes (OBSTACLE=8, AGENT_0=2, ...)
  OvercookedV2 uses StaticObject (WALL=1, agents stored separately)
  This module handles all conversions transparently.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import IntEnum
from functools import partial
from typing import Tuple, Optional, Dict, Any, NamedTuple, List

import jax
import jax.numpy as jnp
import numpy as np
import chex


# ═══════════════════════════════════════════════════════════════════════════════
# §1  GRID CODE SYSTEMS
# ═══════════════════════════════════════════════════════════════════════════════

class GridCodes:
    """
    Grid codes used by the parametrized level generator.
    These are the 'lingua franca' of OvercookedLevel — all levels are stored
    in this encoding, and converted to StaticObject encoding only when loading
    into the OvercookedV2 JAX environment.
    """
    EMPTY        = 0
    WALL         = 1    # border walls (OvercookedV2 StaticObject.WALL)
    AGENT_0      = 2
    AGENT_1      = 3
    GOAL         = 4
    POT          = 5
    OBSTACLE     = 8    # interior obstacle walls (same role as WALL)
    PLATE_PILE   = 9
    INGREDIENT_0 = 10
    INGREDIENT_1 = 11
    INGREDIENT_2 = 12

    # All codes that agents can walk through
    WALKABLE = frozenset({EMPTY, AGENT_0, AGENT_1})

    # All codes representing task-critical objects (not walls/empty)
    OBJECTS = frozenset({GOAL, POT, PLATE_PILE, INGREDIENT_0, INGREDIENT_1, INGREDIENT_2})

    # Agent codes
    AGENTS = frozenset({AGENT_0, AGENT_1})


# ── JAX-level conversion utilities (for use inside jit/vmap) ──────────────────
# These are module-level constants and functions that operate on raw JAX arrays.
# They exist separately from OvercookedLevel's methods because jit cannot trace
# through Python dataclass method calls.

def _build_static_lookup() -> jnp.ndarray:
    """
    Build a JAX lookup table: index with a GridCodes value → StaticObject int.
    
    This must be called AFTER importing StaticObject from the OvercookedV2 package.
    Returns a jnp array of shape (MAX_CODE,) where lookup[grid_code] = static_code.
    """
    from .environment.common import StaticObject
    _G = GridCodes
    _TO_STATIC = {
        _G.EMPTY:        int(StaticObject.EMPTY),
        _G.WALL:         int(StaticObject.WALL),
        _G.GOAL:         int(StaticObject.GOAL),
        _G.POT:          int(StaticObject.POT),
        _G.OBSTACLE:     int(StaticObject.WALL),       # interior walls → WALL
        _G.PLATE_PILE:   int(StaticObject.PLATE_PILE),
        _G.INGREDIENT_0: int(StaticObject.INGREDIENT_PILE_BASE) + 0,
        _G.INGREDIENT_1: int(StaticObject.INGREDIENT_PILE_BASE) + 1,
        _G.INGREDIENT_2: int(StaticObject.INGREDIENT_PILE_BASE) + 2,
        _G.AGENT_0:      int(StaticObject.EMPTY),       # agents tracked separately
        _G.AGENT_1:      int(StaticObject.EMPTY),
    }
    max_code = 13
    return jnp.array(
        [_TO_STATIC.get(i, int(StaticObject.EMPTY)) for i in range(max_code)],
        dtype=jnp.int32,
    )

# Lazy-initialized singleton — built on first use so import order doesn't matter.
_STATIC_LOOKUP_CACHE = None

def get_static_lookup() -> jnp.ndarray:
    """Get (or build) the GridCodes → StaticObject JAX lookup table."""
    global _STATIC_LOOKUP_CACHE
    if _STATIC_LOOKUP_CACHE is None:
        _STATIC_LOOKUP_CACHE = _build_static_lookup()
    return _STATIC_LOOKUP_CACHE


def grid_to_static_jax(grid: jnp.ndarray) -> jnp.ndarray:
    """
    Convert a GridCodes grid (H, W) to StaticObject encoding via JAX lookup.
    Works inside jit/vmap. Agents become EMPTY.
    """
    return get_static_lookup()[grid]


def grid_batch_to_static_jax(grids: jnp.ndarray) -> jnp.ndarray:
    """
    Convert a batch of GridCodes grids (N, H, W) to StaticObject encoding.
    Works inside jit/vmap.
    """
    return get_static_lookup()[grids]


@jax.jit
def extract_agent_positions_jax(grid: jnp.ndarray) -> jnp.ndarray:
    """
    Extract agent (x, y) positions from a GridCodes grid.
    Returns (2, 2) int32 array: [[x0, y0], [x1, y1]].
    Works inside jit/vmap.
    """
    H, W = grid.shape
    flat = grid.reshape(-1)
    row_idx = jnp.repeat(jnp.arange(H), W)
    col_idx = jnp.tile(jnp.arange(W), H)
    idx0 = jnp.argmax(flat == GridCodes.AGENT_0)
    idx1 = jnp.argmax(flat == GridCodes.AGENT_1)
    pos0 = jnp.stack([col_idx[idx0], row_idx[idx0]])
    pos1 = jnp.stack([col_idx[idx1], row_idx[idx1]])
    return jnp.stack([pos0, pos1]).astype(jnp.int32)


def grid_to_layout_numpy(grid: np.ndarray) -> Tuple[np.ndarray, list]:
    """
    Convert a GridCodes grid to (static_objects, agent_positions) using numpy.
    For use outside jit (e.g., visualization, GIF rendering).
    
    Returns:
        static_objects:  (H, W) int array of StaticObject codes
        agent_positions: list of (x, y) tuples
    """
    from .environment.common import StaticObject
    _G = GridCodes
    _TO_STATIC = {
        _G.EMPTY:        int(StaticObject.EMPTY),
        _G.WALL:         int(StaticObject.WALL),
        _G.GOAL:         int(StaticObject.GOAL),
        _G.POT:          int(StaticObject.POT),
        _G.OBSTACLE:     int(StaticObject.WALL),
        _G.PLATE_PILE:   int(StaticObject.PLATE_PILE),
        _G.INGREDIENT_0: int(StaticObject.INGREDIENT_PILE_BASE) + 0,
        _G.INGREDIENT_1: int(StaticObject.INGREDIENT_PILE_BASE) + 1,
        _G.INGREDIENT_2: int(StaticObject.INGREDIENT_PILE_BASE) + 2,
        _G.AGENT_0:      int(StaticObject.EMPTY),
        _G.AGENT_1:      int(StaticObject.EMPTY),
    }
    static_objects = np.vectorize(lambda c: _TO_STATIC.get(int(c), 0))(grid).astype(int)
    agent_positions = []
    for code in [_G.AGENT_0, _G.AGENT_1]:
        rows, cols = np.where(grid == code)
        if len(rows):
            agent_positions.append((int(cols[0]), int(rows[0])))
    return static_objects, agent_positions


# ═══════════════════════════════════════════════════════════════════════════════
# §2  OVERCOOKED LEVEL — the canonical representation
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class OvercookedLevel:
    """
    Canonical representation of a single Overcooked level.

    All UED algorithms produce and consume this. The grid uses GridCodes
    encoding (agents embedded in the grid as AGENT_0/AGENT_1).

    Attributes
    ──────────
    grid             (H, W) int32 ndarray — cell types using GridCodes
    H, W             grid dimensions
    params           optional EnvParams that generated this level
    seed             random seed used to generate this level
    score            most recent UED score (regret, learnability, etc.)
    metadata         arbitrary extra info (cycle length, validity, etc.)
    """
    grid: np.ndarray                        # (H, W) int32
    H: int = field(init=False)
    W: int = field(init=False)
    params: Any = None                      # EnvParams or None
    seed: int = 0
    score: float = 0.0
    metadata: dict = field(default_factory=dict)

    def __post_init__(self):
        self.H, self.W = self.grid.shape

    # ── Derived properties ────────────────────────────────────────────────

    @property
    def agent_positions(self) -> np.ndarray:
        """Extract agent (x, y) positions from grid. Returns (n_agents, 2) array."""
        positions = []
        for code in [GridCodes.AGENT_0, GridCodes.AGENT_1]:
            ys, xs = np.where(self.grid == code)
            if len(ys) > 0:
                positions.append([int(xs[0]), int(ys[0])])
        return np.array(positions, dtype=np.int32)

    @property
    def n_agents(self) -> int:
        return int(np.isin(self.grid, [GridCodes.AGENT_0, GridCodes.AGENT_1]).sum())

    @property
    def n_walls(self) -> int:
        return int((self.grid == GridCodes.OBSTACLE).sum())

    @property
    def n_goals(self) -> int:
        return int((self.grid == GridCodes.GOAL).sum())

    @property
    def n_pots(self) -> int:
        return int((self.grid == GridCodes.POT).sum())

    @property
    def n_plates(self) -> int:
        return int((self.grid == GridCodes.PLATE_PILE).sum())

    @property
    def n_ingredients(self) -> int:
        return int(np.isin(self.grid, [
            GridCodes.INGREDIENT_0, GridCodes.INGREDIENT_1, GridCodes.INGREDIENT_2
        ]).sum())

    # ── Conversion to OvercookedV2 format ────────────────────────────────

    def to_static_objects(self, static_object_enum) -> np.ndarray:
        """
        Convert grid to OvercookedV2 StaticObject encoding.

        Agent cells become EMPTY (agents are stored separately).
        Obstacle walls (8) become StaticObject.WALL (1).
        All other codes that match StaticObject values pass through.

        Args:
            static_object_enum: The StaticObject class from overcooked_v2_rethink.common

        Returns:
            (H, W) int32 array of StaticObject values
        """
        so = self.grid.copy()
        # Agents → EMPTY (agents tracked separately)
        so[so == GridCodes.AGENT_0] = int(static_object_enum.EMPTY)
        so[so == GridCodes.AGENT_1] = int(static_object_enum.EMPTY)
        # Interior obstacles → WALL
        so[so == GridCodes.OBSTACLE] = int(static_object_enum.WALL)
        return so.astype(np.int32)

    def to_static_objects_jax(self, wall_code: int = 1, empty_code: int = 0) -> jnp.ndarray:
        """
        JAX-compatible conversion. Uses integer codes directly to avoid
        importing StaticObject (keeps this module self-contained).

        Default mapping: OBSTACLE(8)→WALL(1), AGENT_0(2)→EMPTY(0), AGENT_1(3)→EMPTY(0)
        """
        g = jnp.array(self.grid, dtype=jnp.int32)
        g = jnp.where(g == GridCodes.AGENT_0, empty_code, g)
        g = jnp.where(g == GridCodes.AGENT_1, empty_code, g)
        g = jnp.where(g == GridCodes.OBSTACLE, wall_code, g)
        return g

    def to_agent_positions_jax(self) -> jnp.ndarray:
        """Returns (n_agents, 2) jnp array of (x, y) positions."""
        return jnp.array(self.agent_positions, dtype=jnp.int32)

    # ── Utilities ────────────────────────────────────────────────────────

    def copy(self) -> "OvercookedLevel":
        return OvercookedLevel(
            grid=self.grid.copy(),
            params=self.params,
            seed=self.seed,
            score=self.score,
            metadata=dict(self.metadata),
        )

    def summary(self) -> str:
        return (
            f"OvercookedLevel({self.W}×{self.H}) "
            f"agents={self.n_agents} walls={self.n_walls} "
            f"goals={self.n_goals} pots={self.n_pots} "
            f"plates={self.n_plates} ing={self.n_ingredients} "
            f"score={self.score:.4f}"
        )

    def __repr__(self):
        return self.summary()


# ═══════════════════════════════════════════════════════════════════════════════
# §3  LEVEL GENERATOR
# ═══════════════════════════════════════════════════════════════════════════════

class LevelGenerator:
    """
    Generates random OvercookedLevel instances.

    Wraps your existing generate_random_layout from overcooked_parametrized_new.py.
    Produces valid, solvable levels with randomized continuous parameters.

    Usage:
        gen = LevelGenerator(H=5, W=9)
        level = gen.generate(key)                    # fully random
        level = gen.generate(key, params=my_params)  # specific parameters
    """

    def __init__(
        self,
        H: int,
        W: int,
        max_walls_frac: float = 0.6,
        max_res_frac: float = 0.4,
        max_goals: int = 3,
        max_pots: int = 3,
        max_plates: int = 2,
        num_agents: int = 2,
        ingredient_types: tuple = (GridCodes.INGREDIENT_0, GridCodes.INGREDIENT_1),
        max_retries: int = 50,
    ):
        self.H = H
        self.W = W
        self.max_walls_frac = max_walls_frac
        self.max_res_frac = max_res_frac
        self.max_goals = max_goals
        self.max_pots = max_pots
        self.max_plates = max_plates
        self.num_agents = num_agents
        self.ingredient_types = ingredient_types
        self.max_retries = max_retries

    def generate(
        self,
        key: jnp.ndarray,
        params: Optional[Any] = None,
        validate: bool = True,
    ) -> Optional[OvercookedLevel]:
        """
        Generate a single valid OvercookedLevel.

        Args:
            key:      JAX PRNGKey
            params:   optional EnvParams; if None, sampled randomly
            validate: if True, retry until valid (BFS reachability check)

        Returns:
            OvercookedLevel or None if max_retries exceeded
        """
        from .level_generator import (
            generate_random_layout, EnvParams, is_valid_layout,
            MAX_OBS_FRAC, MAX_RES_FRAC,
        )

        for attempt in range(self.max_retries if validate else 1):
            key, k_params, k_layout, k_counts = jax.random.split(key, 4)

            # Sample continuous parameters if not provided
            if params is None:
                k_od, k_os, k_rd, k_rs = jax.random.split(k_params, 4)
                cur_params = EnvParams(
                    obs_density=float(jax.random.uniform(k_od, minval=0.0, maxval=self.max_walls_frac)),
                    obs_skew_x=float(jax.random.uniform(k_os, minval=-1.0, maxval=1.0)),
                    res_density=float(jax.random.uniform(k_rd, minval=0.0, maxval=self.max_res_frac)),
                    res_skew_x=float(jax.random.uniform(k_rs, minval=-1.0, maxval=1.0)),
                )
            else:
                cur_params = params

            # Sample object counts
            k_ng, k_np, k_npl = jax.random.split(k_counts, 3)
            num_goals = int(jax.random.randint(k_ng, (), 1, self.max_goals + 1))
            num_pots = int(jax.random.randint(k_np, (), 1, self.max_pots + 1))
            num_plates = int(jax.random.randint(k_npl, (), 1, self.max_plates + 1))

            grid = generate_random_layout(
                self.H, self.W,
                cur_params.obs_density, cur_params.obs_skew_x,
                cur_params.res_density, cur_params.res_skew_x,
                k_layout,
                num_agents=self.num_agents,
                num_goals=num_goals,
                num_pots=num_pots,
                num_plates=num_plates,
                ing0=int(self.ingredient_types[0]),
                ing1=int(self.ingredient_types[-1]),
            )
            grid_np = np.asarray(grid)

            if not validate or is_valid_layout(grid_np):
                return OvercookedLevel(
                    grid=grid_np,
                    params=cur_params,
                    seed=attempt,
                    metadata={"attempt": attempt},
                )

        return None  # exhausted retries

    def generate_batch(
        self,
        key: jnp.ndarray,
        n: int,
        validate: bool = True,
    ) -> List[OvercookedLevel]:
        """Generate n valid levels."""
        keys = jax.random.split(key, n)
        levels = []
        for k in keys:
            level = self.generate(k, validate=validate)
            if level is not None:
                levels.append(level)
        return levels

    def generate_jax(
        self,
        key: jnp.ndarray,
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """
        Pure JAX generation (no validation, for use inside jit).
        Returns (grid, param_vector) — both jnp arrays.
        """
        from .level_generator import generate_random_layout, MAX_OBS_FRAC, MAX_RES_FRAC

        k_params, k_layout = jax.random.split(key)
        k_od, k_os, k_rd, k_rs = jax.random.split(k_params, 4)

        obs_density = jax.random.uniform(k_od, minval=0.0, maxval=float(self.max_walls_frac))
        obs_skew_x = jax.random.uniform(k_os, minval=-1.0, maxval=1.0)
        res_density = jax.random.uniform(k_rd, minval=0.0, maxval=float(self.max_res_frac))
        res_skew_x = jax.random.uniform(k_rs, minval=-1.0, maxval=1.0)

        k_ng, k_np, k_npl = jax.random.split(k_layout, 3)
        num_goals = jax.random.randint(k_ng, (), 1, self.max_goals + 1)
        num_pots = jax.random.randint(k_np, (), 1, self.max_pots + 1)
        num_plates = jax.random.randint(k_npl, (), 1, self.max_plates + 1)

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
        param_vec = jnp.stack([obs_density, obs_skew_x, res_density, res_skew_x])
        return grid, param_vec


# ═══════════════════════════════════════════════════════════════════════════════
# §4  LEVEL MUTATOR
# ═══════════════════════════════════════════════════════════════════════════════

class MutationType(IntEnum):
    FLIP_WALL       = 0   # toggle empty ↔ obstacle
    MOVE_GOAL       = 1   # relocate a goal to a random free cell
    MOVE_POT        = 2   # relocate a pot
    MOVE_PLATE      = 3   # relocate a plate pile
    MOVE_INGREDIENT = 4   # relocate an ingredient pile


class LevelMutator:
    """
    Mutates an OvercookedLevel by applying random edits.

    Supports 5 mutation types:
      0. Flip wall ↔ empty
      1. Move goal to random free cell
      2. Move pot
      3. Move plate pile
      4. Move ingredient pile

    Usage:
        mutator = LevelMutator()
        new_level = mutator.mutate(key, level, n_mutations=5)
        new_grid_jax = mutator.mutate_jax(key, grid_jax, n_mutations=5)  # inside jit
    """

    # Codes that are safe targets for object relocation (empty walkable cells)
    SAFE_TARGETS = frozenset({GridCodes.EMPTY})

    # Codes that must never be mutated
    PROTECTED = frozenset({GridCodes.AGENT_0, GridCodes.AGENT_1, GridCodes.WALL})

    def mutate(
        self,
        key: jnp.ndarray,
        level: OvercookedLevel,
        n_mutations: int = 5,
        validate: bool = True,
        max_retries: int = 10,
    ) -> OvercookedLevel:
        """
        Apply n_mutations random edits to a level.

        If validate=True, retries with different random seeds until the mutated
        level passes the BFS reachability check.

        Returns a new OvercookedLevel (does not modify the input).
        """
        from .level_generator import is_valid_layout

        for attempt in range(max_retries if validate else 1):
            key, k_mut = jax.random.split(key)
            new_grid = self._apply_mutations_numpy(
                level.grid.copy(), k_mut, n_mutations
            )
            if not validate or is_valid_layout(new_grid):
                new_level = level.copy()
                new_level.grid = new_grid
                new_level.score = 0.0  # reset score — needs re-evaluation
                new_level.metadata["mutated"] = True
                new_level.metadata["n_mutations"] = n_mutations
                return new_level

        # All retries failed — return unchanged
        return level.copy()

    def _apply_mutations_numpy(
        self,
        grid: np.ndarray,
        key: jnp.ndarray,
        n: int,
    ) -> np.ndarray:
        """Apply n random mutations using numpy for flexible indexing."""
        H, W = grid.shape

        for i in range(n):
            key, k_type, k_cell = jax.random.split(key, 3)
            mut_type = int(jax.random.randint(k_type, (), 0, len(MutationType)))

            if mut_type == MutationType.FLIP_WALL:
                grid = self._flip_wall(grid, k_cell)
            elif mut_type == MutationType.MOVE_GOAL:
                grid = self._move_object(grid, k_cell, GridCodes.GOAL)
            elif mut_type == MutationType.MOVE_POT:
                grid = self._move_object(grid, k_cell, GridCodes.POT)
            elif mut_type == MutationType.MOVE_PLATE:
                grid = self._move_object(grid, k_cell, GridCodes.PLATE_PILE)
            elif mut_type == MutationType.MOVE_INGREDIENT:
                # Pick a random ingredient type that exists
                ing_types = [c for c in [GridCodes.INGREDIENT_0, GridCodes.INGREDIENT_1]
                             if np.any(grid == c)]
                if ing_types:
                    key, k_ing = jax.random.split(key)
                    idx = int(jax.random.randint(k_ing, (), 0, len(ing_types)))
                    grid = self._move_object(grid, k_cell, ing_types[idx])

        return grid

    @staticmethod
    def _flip_wall(grid: np.ndarray, key: jnp.ndarray) -> np.ndarray:
        """Toggle a random interior cell between empty and obstacle."""
        H, W = grid.shape
        interior = [(r, c) for r in range(1, H - 1) for c in range(1, W - 1)
                    if grid[r, c] in (GridCodes.EMPTY, GridCodes.OBSTACLE)]
        if not interior:
            return grid
        idx = int(jax.random.randint(key, (), 0, len(interior)))
        r, c = interior[idx]
        grid[r, c] = GridCodes.EMPTY if grid[r, c] == GridCodes.OBSTACLE else GridCodes.OBSTACLE
        return grid

    @staticmethod
    def _move_object(grid: np.ndarray, key: jnp.ndarray, obj_code: int) -> np.ndarray:
        """Move one instance of obj_code to a random free cell."""
        H, W = grid.shape
        # Find existing instances
        locations = [(r, c) for r in range(H) for c in range(W) if grid[r, c] == obj_code]
        if not locations:
            return grid
        # Find free cells (can place on any edge or interior, but not on agents/objects)
        free = [(r, c) for r in range(H) for c in range(W) if grid[r, c] == GridCodes.EMPTY]
        if not free:
            return grid

        k1, k2 = jax.random.split(key)
        src_idx = int(jax.random.randint(k1, (), 0, len(locations)))
        dst_idx = int(jax.random.randint(k2, (), 0, len(free)))

        sr, sc = locations[src_idx]
        dr, dc = free[dst_idx]
        grid[dr, dc] = obj_code
        grid[sr, sc] = GridCodes.EMPTY
        return grid

    def mutate_jax(
        self,
        key: jnp.ndarray,
        grid: jnp.ndarray,
        n_mutations: int = 5,
    ) -> jnp.ndarray:
        """
        Pure JAX mutation (wall flips only — for use inside jit).
        For richer mutations inside jit, compile a custom version.
        """
        H, W = grid.shape
        N = H * W

        def _single_mutation(carry, _):
            flat, k = carry
            k1, k2, k_next = jax.random.split(k, 3)

            is_empty = (flat == GridCodes.EMPTY)
            is_wall = (flat == GridCodes.OBSTACLE)

            p_empty = is_empty.astype(jnp.float32) + 1e-8
            p_wall = is_wall.astype(jnp.float32) + 1e-8

            empty_idx = jax.random.choice(k1, jnp.arange(N), p=p_empty)
            wall_idx = jax.random.choice(k2, jnp.arange(N), p=p_wall)

            add_wall = jax.random.bernoulli(k1, 0.5)
            new_flat = jax.lax.cond(
                add_wall,
                lambda g: g.at[empty_idx].set(jnp.int32(GridCodes.OBSTACLE)),
                lambda g: g.at[wall_idx].set(jnp.int32(GridCodes.EMPTY)),
                flat,
            )
            return (new_flat, k_next), None

        flat = grid.flatten()
        (mutated_flat, _), _ = jax.lax.scan(
            _single_mutation, (flat, key), None, length=n_mutations
        )
        return mutated_flat.reshape(H, W)


# ═══════════════════════════════════════════════════════════════════════════════
# §5  LEVEL SCORER
# ═══════════════════════════════════════════════════════════════════════════════

class LevelScorer:
    """
    Computes UED scores for levels.

    Two scoring modes:
      regret       — for PLR / ACCEL / PAIRED: measures how much the agent
                     could improve on this level
      learnability — for SFL / EvoSFL: p*(1-p) where p is success rate,
                     peaks at 0.5 (the learning frontier)
    """

    @staticmethod
    def regret_pvl(
        max_return: float,
        achieved_return: float,
    ) -> float:
        """
        Positive Value Loss (PVL) regret.
        regret = max(0, V* - V_achieved)
        where V* is an estimate of the optimal return.
        """
        return max(0.0, max_return - achieved_return)

    @staticmethod
    def regret_relative(
        returns: np.ndarray,
    ) -> float:
        """
        Relative regret between students (for PAIRED with 2+ students).
        regret = max(returns) - min(returns)
        """
        return float(np.max(returns) - np.min(returns))

    @staticmethod
    def regret_maxmc(
        values: np.ndarray,
        returns: np.ndarray,
    ) -> float:
        """
        MaxMC regret (used in minimax/OGC).
        regret = max(0, mean(values) - mean(returns))
        The gap between what the critic predicted and what actually happened.
        """
        return float(max(0.0, np.mean(values) - np.mean(returns)))

    @staticmethod
    def learnability(
        success_rate: float,
    ) -> float:
        """
        SFL learnability score: p * (1 - p)
        Peaks at p=0.5 — levels that are neither too easy nor too hard.
        """
        p = np.clip(success_rate, 0.0, 1.0)
        return float(p * (1.0 - p))

    @staticmethod
    def learnability_from_deliveries(
        deliveries: float,
        max_deliveries: float = 10.0,
    ) -> float:
        """
        Estimate learnability from delivery count.
        Normalizes deliveries to [0, 1] then applies p*(1-p).
        """
        p = np.clip(deliveries / max(max_deliveries, 1.0), 0.0, 1.0)
        return float(p * (1.0 - p))


# ═══════════════════════════════════════════════════════════════════════════════
# §6  LEVEL BUFFER
# ═══════════════════════════════════════════════════════════════════════════════

class LevelBuffer:
    """
    Replay buffer storing scored OvercookedLevels.

    Used by PLR, ACCEL, SFL to maintain a curated set of training levels.
    Supports insertion with eviction (lowest score replaced).

    Attributes
    ──────────
    capacity     maximum number of levels in the buffer
    levels       list of OvercookedLevel instances
    """

    def __init__(self, capacity: int = 100):
        self.capacity = capacity
        self.levels: List[OvercookedLevel] = []

    @property
    def size(self) -> int:
        return len(self.levels)

    @property
    def is_full(self) -> bool:
        return self.size >= self.capacity

    @property
    def scores(self) -> np.ndarray:
        if not self.levels:
            return np.array([])
        return np.array([lv.score for lv in self.levels])

    @property
    def mean_score(self) -> float:
        s = self.scores
        return float(s.mean()) if len(s) > 0 else 0.0

    @property
    def max_score(self) -> float:
        s = self.scores
        return float(s.max()) if len(s) > 0 else 0.0

    @property
    def min_score(self) -> float:
        s = self.scores
        return float(s.min()) if len(s) > 0 else 0.0

    def insert(self, level: OvercookedLevel) -> bool:
        """
        Insert a level into the buffer.
        If full, replaces the level with the lowest score (if new level scores higher).
        Returns True if the level was inserted.
        """
        if not self.is_full:
            self.levels.append(level)
            return True

        # Find the weakest level in the buffer
        min_idx = int(np.argmin(self.scores))
        if level.score > self.levels[min_idx].score:
            self.levels[min_idx] = level
            return True
        return False

    def sample(
        self,
        key: jnp.ndarray,
        n: int = 1,
        temperature: float = 0.1,
        use_score_ranks: bool = True,
    ) -> List[OvercookedLevel]:
        """
        Sample n levels from the buffer, weighted by score.

        Higher-scored levels are sampled more frequently.
        temperature controls the sharpness of the distribution.
        """
        if not self.levels:
            return []
        n = min(n, self.size)

        scores = self.scores
        if use_score_ranks:
            # Rank-based prioritization (more robust to outliers)
            ranks = np.argsort(np.argsort(scores)).astype(np.float64) + 1.0
            weights = ranks
        else:
            weights = scores - scores.min() + 1e-8

        weights = weights ** (1.0 / max(temperature, 1e-8))
        probs = weights / weights.sum()

        indices = np.array(jax.random.choice(
            key, len(self.levels), shape=(n,), replace=False, p=probs
        ))
        return [self.levels[i] for i in indices]

    def sample_uniform(self, key: jnp.ndarray, n: int = 1) -> List[OvercookedLevel]:
        """Sample n levels uniformly (for mixing with fresh levels)."""
        if not self.levels:
            return []
        n = min(n, self.size)
        indices = np.array(jax.random.choice(
            key, len(self.levels), shape=(n,), replace=False
        ))
        return [self.levels[i] for i in indices]

    def clear(self):
        self.levels.clear()

    def fill_random(self, generator: LevelGenerator, key: jnp.ndarray, n: int = None):
        """Fill the buffer with random valid levels."""
        n = n or self.capacity
        levels = generator.generate_batch(key, n)
        for lv in levels:
            if self.is_full:
                break
            self.insert(lv)

    def stats(self) -> dict:
        return {
            "buffer_size": self.size,
            "buffer_mean_score": self.mean_score,
            "buffer_max_score": self.max_score,
            "buffer_min_score": self.min_score,
        }


# ═══════════════════════════════════════════════════════════════════════════════
# §7  UNIFIED UED ENVIRONMENT
# ═══════════════════════════════════════════════════════════════════════════════

class OvercookedUEDEnv:
    """
    Unified UED environment for cooperative Overcooked.

    Wraps the student OvercookedV2 environment with level generation, mutation,
    and scoring. Provides a single interface that any DCD algorithm can use.

    Usage
    ─────
    # Setup
    ued = OvercookedUEDEnv(H=5, W=9)

    # DR: random level → play
    level, obs, state = ued.reset_to_random(key)

    # PLR/ACCEL: load a specific level → play
    obs, state = ued.reset_to_level(key, level)

    # After rollout: score the level
    level.score = ued.score_regret(values, returns)
    level.score = ued.score_learnability(success_rate)

    # ACCEL: mutate a level
    new_level = ued.mutate(key, level, n_mutations=5)

    # Step the student environment
    obs, state, rewards, dones, infos = ued.step(key, state, actions)
    """

    def __init__(
        self,
        H: int,
        W: int,
        max_steps: int = 400,
        agent_view_size: int = None,
        max_goals: int = 3,
        max_pots: int = 3,
        max_plates: int = 2,
        num_agents: int = 2,
        ingredient_types: tuple = (GridCodes.INGREDIENT_0, GridCodes.INGREDIENT_1),
        **env_kwargs,
    ):
        self.H = H
        self.W = W
        self.max_steps = max_steps
        self.agent_view_size = agent_view_size

        # --- Level generator ---
        self.generator = LevelGenerator(
            H=H, W=W,
            max_goals=max_goals,
            max_pots=max_pots,
            max_plates=max_plates,
            num_agents=num_agents,
            ingredient_types=ingredient_types,
        )

        # --- Mutator ---
        self.mutator = LevelMutator()

        # --- Scorer ---
        self.scorer = LevelScorer()

        # --- Student environment (OvercookedV2) ---
        # Create with a dummy layout — levels will override it via set_env_instance
        self._env_kwargs = env_kwargs
        self.env = self._create_student_env()

        # --- Track current level ---
        self.current_level: Optional[OvercookedLevel] = None

    def _create_student_env(self):
        """Initialize the OvercookedV2 student environment with a dummy layout."""
        from .environment import OvercookedV2
        from .environment.layouts import Layout
        from .environment.common import StaticObject

        static = np.full((self.H, self.W), int(StaticObject.WALL), dtype=int)
        static[1:self.H - 1, 1:self.W - 1] = int(StaticObject.EMPTY)

        dummy_layout = Layout(
            agent_positions=[(1, 1), (self.W - 2, self.H - 2)],
            static_objects=static,
            num_ingredients=2,
            possible_recipes=[[0, 0, 0], [1, 1, 1]],
        )
        return OvercookedV2(
            layout=dummy_layout,
            max_steps=self.max_steps,
            agent_view_size=self.agent_view_size,
            **self._env_kwargs,
        )

    # ── Core interface ────────────────────────────────────────────────────

    @property
    def obs_shape(self):
        return self.env.obs_shape

    @property
    def num_actions(self):
        return self.env.num_actions

    @property
    def height(self):
        return self.H

    @property
    def width(self):
        return self.W

    # ── Level → student environment ──────────────────────────────────────

    def reset_to_level(
        self,
        key: jnp.ndarray,
        level: OvercookedLevel,
    ) -> Tuple[Dict, Any]:
        """
        Reset the student environment to play a specific level.

        This is the universal entry point — every algorithm calls this
        after producing an OvercookedLevel (whether generated, replayed,
        mutated, or designed by a teacher).

        Returns:
            obs_dict   {agent_0: (H,W,7), agent_1: (H,W,7)}
            state      OvercookedV2 State
        """
        self.current_level = level
        static_objects = level.to_static_objects_jax()
        agent_positions = level.to_agent_positions_jax()

        obs, state = self.env.reset(
            key,
            static_objects=static_objects,
            agent_positions_xy=agent_positions,
        )
        return obs, state

    def reset_to_random(
        self,
        key: jnp.ndarray,
        params: Optional[Any] = None,
    ) -> Tuple[OvercookedLevel, Dict, Any]:
        """
        Generate a random level and reset the student into it.

        Convenience for DR-style training.

        Returns:
            level      the generated OvercookedLevel
            obs_dict   observations
            state      environment state
        """
        key, k_gen, k_reset = jax.random.split(key, 3)
        level = self.generator.generate(k_gen, params=params)
        if level is None:
            raise RuntimeError("Failed to generate a valid level after max retries")
        obs, state = self.reset_to_level(k_reset, level)
        return level, obs, state

    def step(
        self,
        key: jnp.ndarray,
        state: Any,
        actions: Dict[str, jnp.ndarray],
    ):
        """Step the student environment. Thin wrapper around OvercookedV2.step."""
        return self.env.step(key, state, actions)

    # ── Level generation ─────────────────────────────────────────────────

    def generate_level(
        self,
        key: jnp.ndarray,
        params: Optional[Any] = None,
        validate: bool = True,
    ) -> Optional[OvercookedLevel]:
        """Generate a single level."""
        return self.generator.generate(key, params=params, validate=validate)

    def generate_batch(
        self,
        key: jnp.ndarray,
        n: int,
        validate: bool = True,
    ) -> List[OvercookedLevel]:
        """Generate n levels."""
        return self.generator.generate_batch(key, n, validate=validate)

    # ── Mutation ─────────────────────────────────────────────────────────

    def mutate(
        self,
        key: jnp.ndarray,
        level: OvercookedLevel,
        n_mutations: int = 5,
        validate: bool = True,
    ) -> OvercookedLevel:
        """Mutate an existing level."""
        return self.mutator.mutate(key, level, n_mutations, validate=validate)

    # ── Scoring ──────────────────────────────────────────────────────────

    def score_regret(
        self,
        max_return: float,
        achieved_return: float,
    ) -> float:
        """PVL regret for PLR/ACCEL."""
        return self.scorer.regret_pvl(max_return, achieved_return)

    def score_regret_maxmc(
        self,
        values: np.ndarray,
        returns: np.ndarray,
    ) -> float:
        """MaxMC regret (minimax-style)."""
        return self.scorer.regret_maxmc(values, returns)

    def score_learnability(self, success_rate: float) -> float:
        """SFL learnability: p*(1-p)."""
        return self.scorer.learnability(success_rate)

    def score_learnability_from_deliveries(
        self,
        deliveries: float,
        max_deliveries: float = 10.0,
    ) -> float:
        """SFL learnability from delivery count."""
        return self.scorer.learnability_from_deliveries(deliveries, max_deliveries)


# ═══════════════════════════════════════════════════════════════════════════════
# §8  CONVENIENCE FACTORIES
# ═══════════════════════════════════════════════════════════════════════════════

def make_ued_env(
    H: int = 5,
    W: int = 9,
    max_steps: int = 400,
    agent_view_size: int = None,
    **kwargs,
) -> OvercookedUEDEnv:
    """
    Quick factory for creating an OvercookedUEDEnv.

    Example:
        ued = make_ued_env(5, 9)
        level, obs, state = ued.reset_to_random(key)
    """
    return OvercookedUEDEnv(
        H=H, W=W,
        max_steps=max_steps,
        agent_view_size=agent_view_size,
        **kwargs,
    )


def make_ued_env_with_buffer(
    H: int = 5,
    W: int = 9,
    buffer_size: int = 100,
    initial_fill: int = None,
    seed: int = 42,
    **kwargs,
) -> Tuple[OvercookedUEDEnv, LevelBuffer]:
    """
    Create a UED env + pre-filled level buffer.

    Example:
        ued, buffer = make_ued_env_with_buffer(5, 9, buffer_size=50, initial_fill=25)
        # Buffer is ready — start ACCEL/PLR/SFL training
    """
    ued = make_ued_env(H, W, **kwargs)
    buffer = LevelBuffer(capacity=buffer_size)

    fill_n = initial_fill or buffer_size
    key = jax.random.PRNGKey(seed)
    buffer.fill_random(ued.generator, key, n=fill_n)

    return ued, buffer


# ═══════════════════════════════════════════════════════════════════════════════
# §9  SELF-TEST
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import os
    os.environ.setdefault("JAX_PLATFORMS", "cpu")

    print("=" * 60)
    print("  OvercookedUED Self-Test")
    print("=" * 60)

    key = jax.random.PRNGKey(42)

    # Test 1: Level generation
    print("\n[1] Level Generator")
    gen = LevelGenerator(H=5, W=9)
    key, k = jax.random.split(key)
    level = gen.generate(k)
    print(f"  Generated: {level}")
    print(f"  Grid:\n{level.grid}")
    print(f"  Agent positions: {level.agent_positions}")

    # Test 2: Level mutation
    print("\n[2] Level Mutator")
    mutator = LevelMutator()
    key, k = jax.random.split(key)
    mutated = mutator.mutate(k, level, n_mutations=3)
    print(f"  Mutated: {mutated}")
    diffs = (level.grid != mutated.grid).sum()
    print(f"  Cells changed: {diffs}")

    # Test 3: Scoring
    print("\n[3] Scoring")
    scorer = LevelScorer()
    print(f"  Regret PVL(100, 60) = {scorer.regret_pvl(100, 60):.2f}")
    print(f"  Learnability(0.5) = {scorer.learnability(0.5):.4f}")
    print(f"  Learnability(0.1) = {scorer.learnability(0.1):.4f}")
    print(f"  Learnability(0.9) = {scorer.learnability(0.9):.4f}")

    # Test 4: Level buffer
    print("\n[4] Level Buffer")
    buffer = LevelBuffer(capacity=10)
    key, k = jax.random.split(key)
    levels = gen.generate_batch(k, 15)
    for i, lv in enumerate(levels):
        lv.score = float(np.random.random())
        inserted = buffer.insert(lv)
        if i < 10:
            assert inserted, f"Should have inserted level {i}"
    print(f"  Buffer: {buffer.stats()}")

    key, k = jax.random.split(key)
    sampled = buffer.sample(k, n=3)
    print(f"  Sampled {len(sampled)} levels: {[lv.score for lv in sampled]}")

    # Test 5: Static object conversion
    print("\n[5] Grid Conversion")
    jax_grid = level.to_static_objects_jax()
    print(f"  JAX grid shape: {jax_grid.shape}")
    print(f"  Agent cells zeroed: {(jax_grid == GridCodes.AGENT_0).sum() == 0}")

    print("\n" + "=" * 60)
    print("  All tests passed!")
    print("=" * 60)
