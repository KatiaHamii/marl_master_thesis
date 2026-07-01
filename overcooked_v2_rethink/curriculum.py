"""
curriculum.py — Lightweight auto-curriculum for OvercookedV2 IPPO
=================================================================

Manages a sequence of layouts with increasing difficulty. Agents auto-promote
when reaching performance thresholds on the current level.

Curriculum state is persisted to allow resumable training.
"""

import json
from pathlib import Path
from typing import List, Dict, Any
from dataclasses import dataclass, asdict


@dataclass
class CurriculumLevel:
    """A difficulty level in the curriculum."""
    level_id: int
    layout_name: str
    threshold_deliveries: float  # min avg deliveries needed to promote
    eval_window: int = 50  # episodes to average over for threshold check


def make_default_curriculum(obs_size: int = None) -> List[CurriculumLevel]:
    """
    Create a default easy→hard curriculum.

    Args:
        obs_size: grid size hint (e.g., 5x5 or 6x9) — unused for now,
                  but here for future parameterization.

    Returns:
        List of CurriculumLevel objects.
    """
    return [
        CurriculumLevel(0, "cramped_room_v2", threshold_deliveries=5.0),
        CurriculumLevel(1, "asymm_advantages_recipes_center", threshold_deliveries=8.0),
        CurriculumLevel(2, "coord_ring_synth_6x9_b", threshold_deliveries=6.0),
    ]


class CurriculumManager:
    """Tracks curriculum progress and decides when to promote to harder layouts."""

    def __init__(
        self,
        levels: List[CurriculumLevel],
        eval_window: int = 50,
    ):
        """
        Args:
            levels: list of CurriculumLevel defining the curriculum sequence
            eval_window: number of episodes to average for threshold checks
        """
        self.levels = levels
        self.eval_window = eval_window
        self.current_level = 0
        self.level_update_history = [(0, 0)]  # list of (update_num, level_id)

        # Per-layout performance tracking: layout_name → [delivery counts]
        self.layout_deliveries: Dict[str, List[float]] = {
            lvl.layout_name: [] for lvl in levels
        }

    @property
    def current_layout(self) -> str:
        """Get the current layout name."""
        return self.levels[self.current_level].layout_name

    @property
    def current_threshold(self) -> float:
        """Get the promotion threshold for current level."""
        return self.levels[self.current_level].threshold_deliveries

    def record_episode(self, layout_name: str, deliveries: int):
        """Record deliveries achieved in an episode."""
        if layout_name not in self.layout_deliveries:
            self.layout_deliveries[layout_name] = []
        self.layout_deliveries[layout_name].append(float(deliveries))

    def check_promotion(self, update_num: int) -> bool:
        """
        Check if we should promote to the next level.

        Returns True if promoted, False otherwise.
        """
        if self.current_level >= len(self.levels) - 1:
            return False  # Already at hardest level

        current_layout = self.current_layout
        deliveries = self.layout_deliveries.get(current_layout, [])

        if len(deliveries) < self.eval_window:
            return False  # Not enough data yet

        # Average over last eval_window episodes
        recent_avg = sum(deliveries[-self.eval_window:]) / self.eval_window
        threshold = self.current_threshold

        if recent_avg >= threshold:
            self.current_level += 1
            self.level_update_history.append((update_num, self.current_level))
            return True

        return False

    def get_status(self) -> Dict[str, Any]:
        """Return current curriculum status."""
        current_layout = self.current_layout
        deliveries = self.layout_deliveries.get(current_layout, [])
        recent_avg = (
            sum(deliveries[-self.eval_window:]) / self.eval_window
            if len(deliveries) >= self.eval_window
            else 0.0
        )

        return {
            "current_level": self.current_level,
            "current_layout": current_layout,
            "threshold": self.current_threshold,
            "recent_avg_deliveries": recent_avg,
            "episodes_on_current_level": len(deliveries),
        }

    def save(self, path: Path):
        """Persist curriculum state to JSON."""
        state = {
            "current_level": self.current_level,
            "level_update_history": self.level_update_history,
            "layout_deliveries": self.layout_deliveries,
        }
        with open(path, "w") as f:
            json.dump(state, f, indent=2)

    @staticmethod
    def load(path: Path, levels: List[CurriculumLevel]) -> "CurriculumManager":
        """Load curriculum state from JSON."""
        with open(path, "r") as f:
            state = json.load(f)

        mgr = CurriculumManager(levels)
        mgr.current_level = state["current_level"]
        mgr.level_update_history = state["level_update_history"]
        mgr.layout_deliveries = state["layout_deliveries"]
        return mgr
