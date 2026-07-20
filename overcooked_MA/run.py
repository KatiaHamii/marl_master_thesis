"""
run.py — CLI entry point for the overcooked_MA training pipeline.

    python3 run.py --config config.yaml
    python3 run.py --config config.yaml --steps 100000 --seed 1

Thin by design: argument parsing only — all real logic lives in pipeline.py.
"""

import argparse
from pipeline import run_pipeline

def main():
    ap = argparse.ArgumentParser(description="Run overcooked_MA SFL training from a config.yaml")
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--algo", default=None, choices=["sfl", "sfl_jax"],
                     help="sfl = lightweight NumPy path; sfl_jax = real JAX PPO+SFL stack")
    ap.add_argument("--grid-size", dest="grid_size", default=None, metavar="HxW")
    ap.add_argument("--steps", type=int, default=None)
    ap.add_argument("--envs", type=int, default=None)
    ap.add_argument("--episode-len", dest="episode_len", type=int, default=None)
    ap.add_argument("--view-size", dest="view_size", type=int, default=None,
                     help="Partial-observability crop radius. Supported by sfl_jax; "
                          "sfl (NumPy) raises if this is set.")
    ap.add_argument("--reward-mode", dest="reward_mode", default=None,
                     choices=["shaped", "sparse", "delivery"],
                     help="Supported by sfl_jax; sfl (NumPy) only supports 'shaped'.")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--checkpoint-every", dest="checkpoint_every", type=int, default=None)
    ap.add_argument("--out-dir", dest="out_dir", default=None)
    args = ap.parse_args()

    overrides = {k: v for k, v in vars(args).items() if k != "config"}
    run_pipeline(args.config, overrides)

if __name__ == "__main__":
    main()
