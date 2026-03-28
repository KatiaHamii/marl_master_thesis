"""
run_experiments.py — train all agents and compare them
=======================================================
Adding a new algorithm?
  1. Drop a new file in agents/
  2. Add it to the AGENTS dict below
  3. Run this script — nothing else changes

Usage:
    python run_experiments.py
    python run_experiments.py --config config.yaml   # explicit path
    python run_experiments.py --agents qlearning     # single agent
"""

import argparse
import gymnasium as gym
import numpy as np

from utils import load_config, get, set_seed, evaluate_agent, Plotter

# ── register agents here ────────────────────────────────────
from agents.qlearning import QLearningAgent
from agents.dqn import DQNAgent

AGENTS = {
    "Q-Learning": QLearningAgent,
    "DQN": DQNAgent,
}


# ─────────────────────────────────────────────
#  GENERIC TRAINING LOOP
#  Works for any agent that implements:
#    .select_action(obs) -> int
#    .update(...)        -> loss or None
#    .store(...) [optional, for DQN]
#    .end_episode()
# ─────────────────────────────────────────────
def train(agent, env, cfg: dict) -> list[float]:
    episodes = get(cfg, "training", "episodes")
    rewards = []

    for ep in range(episodes):
        obs, _ = env.reset()
        total = 0.0

        while True:
            action = agent.select_action(obs)
            next_obs, r, term, trunc, _ = env.step(action)
            done = term or trunc

            # Q-Learning: direct update
            # DQN: store then update from replay
            if hasattr(agent, "store"):
                agent.store(obs, action, r, next_obs, done)
                agent.update()
            else:
                agent.update(obs, action, r, next_obs, done)

            obs = next_obs
            total += r
            if done:
                break

        rewards.append(total)
        agent.end_episode()

        if (ep + 1) % 100 == 0:
            avg = np.mean(rewards[-50:])
            eps = float(agent.epsilon)
            print(f"  [{ep+1:4d}/{episodes}]  last-50 avg: {avg:6.1f}" f"  ε={eps:.3f}")

    return rewards


# ─────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument(
        "--agents",
        nargs="+",
        default=list(AGENTS.keys()),
        help="Which agents to run (default: all)",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_seed(get(cfg, "training", "seed"))

    env_name = get(cfg, "environment", "name")
    plotter = Plotter(cfg)
    eval_results = {}

    for name in args.agents:
        if name not in AGENTS:
            print(f"Unknown agent '{name}'. Available: {list(AGENTS.keys())}")
            continue

        print(f"\n{'─'*50}")
        print(f"  Training: {name}")
        print(f"{'─'*50}")

        env = gym.make(env_name)
        state_dim = env.observation_space.shape[0]
        action_dim = env.action_space.n

        agent = AGENTS[name](cfg, state_dim, action_dim)
        rewards = train(agent, env, cfg)
        env.close()

        plotter.add(name, rewards)

        # Evaluate
        print(f"  Evaluating {name}...")
        agent.epsilon.value = 0.0  # pure greedy during eval
        result = evaluate_agent(
            env_factory=lambda: gym.make(env_name),
            select_action=agent.select_action,
            cfg=cfg,
        )
        eval_results[name] = result

    plotter.print_summary(eval_results)
    plotter.save_and_show("comparison")


if __name__ == "__main__":
    main()
