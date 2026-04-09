"""
utils.py — minimal utilities for Minesweeper MAPPO scripts
===========================================================
Provides load_config and get, mirroring the interface of
rl_experiments/utils.py so training and curriculum scripts
can import from utils without path gymnastics.
"""

import yaml


def load_config(path: str = "config.yaml") -> dict:
    """Load a YAML config file and return it as a plain dict."""
    with open(path, "r") as f:
        return yaml.safe_load(f)


def get(cfg: dict, *keys, default=None):
    """
    Safe nested key access.
    Example: get(cfg, "mappo", "learning_rate", default=3e-4)
    """
    val = cfg
    for k in keys:
        if not isinstance(val, dict) or k not in val:
            return default
        val = val[k]
    return val
