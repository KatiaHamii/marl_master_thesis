import random
from collections import deque
import numpy as np
from config import GridCodes

def compute_min_cycle(grid: np.ndarray) -> int:
    """
    Returns the minimum delivery cycle length for a single agent.
    Enforces that both agents are within the same walkable space.
    Returns 9999 if the layout is blocked, split, or structurally invalid.
    """
    grid = np.asarray(grid)
    H, W = grid.shape
    walkable = {GridCodes.EMPTY, GridCodes.AGENT_0, GridCodes.AGENT_1}
    dirs = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    
    #  Gather positions of both agents
    agent_starts = [(r, c) for r in range(H) for c in range(W) if grid[r, c] in (GridCodes.AGENT_0, GridCodes.AGENT_1)]
    
    # Validation: Ensure exactly two agents are spawned on the map
    if len(agent_starts) < 2:
        return 9999
    
    #  Check path connectivity between Agent 1 and Agent 2 (Cooperative Reachability)
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

    # def _get_adjacent_walkable(target_codes: set) -> set:
    #     adj_cells = set()
    #     for r in range(H):
    #         for c in range(W):
    #             if grid[r, c] in target_codes:
    #                 for dr, dc in dirs:
    #                     nr, nc = r + dr, c + dc
    #                     if 0 <= nr < H and 0 <= nc < W and grid[nr, nc] in walkable:
    #                         adj_cells.add((nr, nc))
    #     return adj_cells
    
    def _get_adjacent_walkable(target_codes: set) -> set:
        adj_cells = set()
        for r in range(H):
            for c in range(W):
                if grid[r, c] in target_codes:
                    for dr, dc in dirs:
                        nr, nc = r + dr, c + dc
                        # CRITICAL CHANGE: The adjacent cell MUST be reachable by agents
                        if 0 <= nr < H and 0 <= nc < W and (nr, nc) in visited_coop:
                            adj_cells.add((nr, nc))
        return adj_cells

    # Gather interaction zones (walkable boundaries) for all critical tools
    onion_zones  = _get_adjacent_walkable({GridCodes.INGREDIENT_0, GridCodes.INGREDIENT_1})
    pot_zones    = _get_adjacent_walkable({GridCodes.POT})
    plate_zones  = _get_adjacent_walkable({GridCodes.PLATE_PILE})
    goal_zones   = _get_adjacent_walkable({GridCodes.GOAL})

    # Reject layout if any critical object has zero accessible adjacent cells
    if not (agent_starts and onion_zones and pot_zones and plate_zones and goal_zones):
        return 9999

    #  Multi-source BFS to calculate exact path distance between item zones
    def _bfs_distance(starts: set, targets: set) -> int:
        
        if not starts or not targets: return 9999
        
        # Immediate success if starting zones already overlap with target zones
        if not starts.isdisjoint(targets): return 0
        visited = set(starts)
        # Initialize queue correctly using explicit coordinate unpacking
        queue = deque([(r, c, 0) for r, c in starts])
        while queue:
            r, c, dist = queue.popleft()
            
            # Target zone reached successfully
            if (r, c) in targets:
                return dist
            
            for dr, dc in dirs:
                nr, nc = r + dr, c + dc
                if 0 <= nr < H and 0 <= nc < W and grid[nr, nc] in walkable:
                    if (nr, nc) in targets: return dist + 1
                    if (nr, nc) not in visited:
                        visited.add((nr, nc))
                        queue.append((nr, nc, dist + 1))
        return 9999

    # 5. Evaluate full sequential recipe cycle loop for Overcooked
    # Step A: From agent spawns to onion crate
    d1 = _bfs_distance(set(agent_starts), onion_zones)
    # Step B: From onion crate to cooking pot
    d2 = _bfs_distance(onion_zones, pot_zones)
    # Step C: From cooking pot to plate pile
    d3 = _bfs_distance(pot_zones, plate_zones)
    # Step D: From plate pile back to cooking pot (to pick up dish)
    d4 = _bfs_distance(plate_zones, pot_zones)
    # Step E: From cooking pot to delivery station
    d5 = _bfs_distance(pot_zones, goal_zones)
    total = d1 + d2 + d3 + d4 + d5
    return total if total < 9999 else 9999

# def is_valid_layout(grid: np.ndarray) -> bool:
#     return compute_min_cycle(grid) < 9999
def is_valid_layout(grid: np.ndarray) -> bool:
    """
    Validates map integrity by leveraging the unified compute_min_cycle metric.
    Filters out blocked paths, isolated agents, and corrupt short-cycle anomalies.
    """
    try:
        cycle_cost = compute_min_cycle(grid)
        # Any cycle under 5 steps means structural overlap/suffocation in Overcooked.
        # 9999 indicates completely broken topology or isolated components.
        return 5 <= cycle_cost < 9999
    except Exception:
        return False


class LevelGenerator:
    """Generates Overcooked levels based on a 4D UED vector (obs_density, obs_skew_x, res_density, res_skew_x)."""
    
    def __init__(self, height=9, width=11):
        # default dimensions for the Overcooked environment; can be overridden in create_level_elements
        self.height = height
        self.width = width
        
    def create_level_elements(self, obs_density, obs_skew_x, res_density, res_skew_x, seed=None):
        print(f"Creating level elements with obs_density={obs_density}, obs_skew_x={obs_skew_x}, res_density={res_density}, res_skew_x={res_skew_x}, seed={seed}")
        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)
            
        interior_h = self.height - 2
        interior_w = self.width - 2
        total_interior = interior_h * interior_w
        
        num_obstacles = int((obs_density * GridCodes.MAX_OBS_FRAC) * total_interior)
        num_resources = int((res_density * GridCodes.MAX_RES_FRAC) * (total_interior - num_obstacles))
        num_extra_resources = max(0, num_resources - 4)

        return {
            "num_obstacles": num_obstacles,
            "obs_skew_x": (obs_skew_x * 2.0) - 1.0,
            "num_extra_resources": num_extra_resources,
            "res_skew_x": (res_skew_x * 2.0) - 1.0,
            "height": self.height,
            "width": self.width
        }

    def calculate_element_coords(self, level_elems, max_trials=1000):
        """
        Generates physical coordinates based on level element quotas.
        Dynamically relaxes wall density if layout connectivity fails.
        All internal comments inside the code are strictly in English.
        """
        h, w = level_elems["height"], level_elems["width"]
        interior_w = w - 2
        
        # Extract the base target numbers from UED vector
        target_obstacles = level_elems["num_obstacles"]
        requested_extra = level_elems["num_extra_resources"]

        for trial in range(max_trials):
            # Dynamic relaxation: every 100 failed trials, reduce walls by 5% to open paths
            if trial < 50:
                relaxation_factor = 1.0
            else:
                relaxation_factor = max(0.0, 1.0 - (trial - 50) / 850.0)
            current_obstacles = max(0, int(target_obstacles * relaxation_factor))
            
            grid = np.full((h, w), GridCodes.EMPTY, dtype=int)
            
            # Setup outer static borders
            grid[0, :], grid[h-1, :], grid[:, 0], grid[:, w-1] = (
                GridCodes.OBSTACLE, GridCodes.OBSTACLE, GridCodes.OBSTACLE, GridCodes.OBSTACLE
            )
            
            interior_cells = [(r, c) for r in range(1, h-1) for c in range(1, w-1)]

            def get_skew_weights(cells, skew):
                weights = [max(0.01, 1.0 + (2.0 * (c - 1) / (interior_w - 1) - 1.0) * skew) for r, c in cells]
                total = sum(weights)
                return [w / total for w in weights]

            # 1. Spawn relaxed internal walls
            if current_obstacles > 0 and len(interior_cells) >= current_obstacles:
                p = get_skew_weights(interior_cells, level_elems["obs_skew_x"])
                idx = np.random.choice(len(interior_cells), size=current_obstacles, replace=False, p=p)
                for i in sorted(idx, reverse=True):
                    r, c = interior_cells.pop(i)
                    grid[r, c] = GridCodes.OBSTACLE

            # 2. Place Mandatory Minimal Viable Inventory (MVI)
            mandatory = [
                GridCodes.POT, GridCodes.GOAL, GridCodes.PLATE_PILE, 
                GridCodes.INGREDIENT_0, GridCodes.AGENT_0, GridCodes.AGENT_1
            ]
            
            if len(interior_cells) < len(mandatory):
                continue
                
            random.shuffle(interior_cells)
            for obj in mandatory:
                r, c = interior_cells.pop()
                grid[r, c] = obj

            # 3. Handle extra resources dynamically based on remaining grid slots
            available_space = len(interior_cells)
            actual_extra = min(requested_extra, available_space)

            if actual_extra > 0:
                p = get_skew_weights(interior_cells, level_elems["res_skew_x"])
                idx = np.random.choice(len(interior_cells), size=actual_extra, replace=False, p=p)
                for i in sorted(idx, reverse=True):
                    r, c = interior_cells.pop(i)
                    grid[r, c] = random.choice([GridCodes.INGREDIENT_0, GridCodes.INGREDIENT_1])

            # 4. Run full path connectivity and cycle validation via BFS
            if is_valid_layout(grid):
                # Safe density calculation protecting against division by zero
                interior_area = (h - 2) * (w - 2)
                rec_obs_frac = (current_obstacles / interior_area) if interior_area > 0 else 0.0
                
                # Available interior space left for resources after wall placement
                remaining_space = available_space + actual_extra  
                rec_res_frac = (actual_extra / remaining_space) if remaining_space > 0 else 0.0

                # Print the final successful summary ONCE per layout generation
                print(
                    f"[Success] Valid layout found on trial {trial + 1}/{max_trials}. "
                    f"Obstacles: {current_obstacles} (relaxed from {target_obstacles}) | "
                    f"Final Density: {rec_obs_frac:.3f}. "
                    f"Recommended MAX_OBS_FRAC: {rec_obs_frac:.3f}, MAX_RES_FRAC: {rec_res_frac:.3f}"
                )
                # Update the elements dictionary to reflect actual placed values for dashboard rendering
                level_elems["num_obstacles"] = current_obstacles
                level_elems["num_extra_resources"] = actual_extra
                return grid

        # If everything fails, fall back to a clean MVI-only open layout instead of crashing
        print(f"[Warning] Map generation forced fallback open layout on size {w}x{h}")
        grid = np.full((h, w), GridCodes.EMPTY, dtype=int)
        grid[0, :], grid[h-1, :], grid[:, 0], grid[:, w-1] = (
            GridCodes.OBSTACLE, GridCodes.OBSTACLE, GridCodes.OBSTACLE, GridCodes.OBSTACLE
        )
        interior_cells = [(r, c) for r in range(1, h-1) for c in range(1, w-1)]
        random.shuffle(interior_cells)
        for obj in [GridCodes.POT, GridCodes.GOAL, GridCodes.PLATE_PILE, GridCodes.INGREDIENT_0, GridCodes.AGENT_0, GridCodes.AGENT_1]:
            r, c = interior_cells.pop()
            grid[r, c] = obj
        level_elems["num_obstacles"] = 0
        level_elems["num_extra_resources"] = 0
        return grid