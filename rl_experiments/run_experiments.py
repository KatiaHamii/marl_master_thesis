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
    python run_experiments.py --env FrozenLake-v1
    python run_experiments.py --env FrozenLake-v1 --agents DQN PPO
    python run_experiments.py --config config.yaml --agents Q-Learning
"""

import argparse
import gymnasium as gym
import numpy as np

from utils import (
    load_config,
    get,
    set_seed,
    evaluate_agent,
    Plotter,
    make_env_factory,
    onehot,
    RewardShaper,
)

# ── register agents here ────────────────────────────────────
from agents.qlearning import QLearningAgent
from agents.dqn import DQNAgent
from agents.ppo import PPOAgent

AGENTS = {"Q-Learning": QLearningAgent, "DQN": DQNAgent, "PPO": PPOAgent}


# ─────────────────────────────────────────────
#  OBSERVATION PREPROCESSING
#  CartPole  → raw float vector (pass through)
#  FrozenLake → integer tile → one-hot vector
#               (for DQN/PPO which need vectors)
#               (Q-Learning uses the int directly)
# ─────────────────────────────────────────────
def make_obs_fn(env_name: str, n_states: int, agent_name: str):
    """Returns a function that preprocesses raw observations."""
    if env_name == "FrozenLake-v1" and agent_name != "Q-Learning":
        return lambda obs: onehot(obs, n_states)
    return lambda obs: obs  # pass through


# ─────────────────────────────────────────────
#  CURRICULUM — grow map size as agent improves
# ─────────────────────────────────────────────
def curriculum_size(rewards: list, cfg: dict) -> int:
    """Return the map size based on recent performance."""
    if not get(cfg, "frozenlake", "curriculum", default=False):
        return get(cfg, "frozenlake", "size", default=4)
    thresholds = get(cfg, "frozenlake", "curriculum_thresholds", default=[])
    avg = np.mean(rewards[-100:]) if len(rewards) >= 100 else 0.0
    size = get(cfg, "frozenlake", "size", default=4)
    for threshold, next_size in thresholds:
        if avg >= threshold:
            size = next_size
    return size


# ─────────────────────────────────────────────
#  GENERIC TRAINING LOOP
#  Works for any agent that implements:
#    .select_action(obs) -> int
#    .update(...)        -> loss or None
#    .store(...) [optional, for DQN]
#    .end_episode()
# ─────────────────────────────────────────────
# def train(agent, env, cfg: dict) -> list[float]:
#     episodes = get(cfg, "training", "episodes")
#     rewards = []

#     for ep in range(episodes):
#         obs, _ = env.reset()
#         total = 0.0

#         while True:
#             action = agent.select_action(obs)
#             next_obs, r, term, trunc, _ = env.step(action)
#             done = term or trunc

#             # Q-Learning: direct update
#             # DQN: store then update from replay
#             if hasattr(agent, "store"):
#                 agent.store(obs, action, r, next_obs, done)
#                 agent.update()
#             else:
#                 agent.update(obs, action, r, next_obs, done)

#             obs = next_obs
#             total += r
#             if done:
#                 break

#         rewards.append(total)
#         agent.end_episode()

#         if (ep + 1) % 100 == 0:
#             avg = np.mean(rewards[-50:])
#             eps = float(agent.epsilon)
#             print(f"  [{ep+1:4d}/{episodes}]  last-50 avg: {avg:6.1f}" f"  ε={eps:.3f}")

#     return rewards


# ─────────────────────────────────────────────
#  GENERIC TRAINING LOOP
# ─────────────────────────────────────────────
def train(
    agent, env_factory, cfg: dict, obs_fn=None, agent_name: str = ""
) -> list[float]:
    env_name = get(cfg, "environment", "name")
    is_fl = env_name == "FrozenLake-v1"
    randomise = is_fl and get(cfg, "frozenlake", "domain_randomisation", default=False)
    curriculum = is_fl and get(cfg, "frozenlake", "curriculum", default=False)
    shaping = is_fl and get(cfg, "frozenlake", "reward_shaping", default=False)

    # Use frozenlake-specific episode count if available
    if is_fl:
        episodes = get(
            cfg, "frozenlake", "episodes", default=get(cfg, "training", "episodes")
        )
    else:
        episodes = get(cfg, "training", "episodes")

    obs_fn = obs_fn or (lambda o: o)
    rewards = []

    # For fixed envs, create once. Randomised envs recreate each episode.
    env, map_desc = env_factory() if not (randomise or curriculum) else (None, None)

    for ep in range(episodes):
        if randomise or curriculum:
            if env is not None:
                env.close()
            size = curriculum_size(rewards, cfg)
            env, map_desc = make_env_factory(cfg, size=size)()

        # Fresh shaper each episode — map changes each time
        shaper = (
            RewardShaper(cfg, map_desc) if shaping and map_desc is not None else None
        )

        raw_obs, _ = env.reset()
        obs = obs_fn(raw_obs)
        total = 0.0

        while True:
            action = agent.select_action(obs)
            raw_next, r, term, trunc, _ = env.step(action)
            done = term or trunc
            next_obs = obs_fn(raw_next)

            # Apply reward shaping during training only
            shaped_r = (
                shaper.shape(int(raw_obs), int(raw_next), r, done) if shaper else r
            )

            if hasattr(agent, "store"):
                agent.store(obs, action, shaped_r, next_obs, done)
                agent.update()
            else:
                agent.update(obs, action, shaped_r, next_obs, done)

            obs = next_obs
            raw_obs = raw_next
            total += r  # always track ORIGINAL reward for plotting
            if done:
                break

        rewards.append(total)
        agent.end_episode()

        if (ep + 1) % 500 == 0:
            avg = np.mean(rewards[-100:])
            eps = float(agent.epsilon)
            size_str = ""
            if curriculum:
                s = curriculum_size(rewards, cfg)
                size_str = f"  map={s}x{s}"
            print(
                f"  [{ep+1:5d}/{episodes}]  last-100 avg: {avg:.3f}"
                f"  ε={eps:.3f}{size_str}"
            )

    env.close()
    return rewards


# ─────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--agents", nargs="+", default=list(AGENTS.keys()))
    parser.add_argument("--env", default=None, help="Override env, e.g. FrozenLake-v1")
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_seed(get(cfg, "training", "seed"))

    if args.env:
        cfg["environment"]["name"] = args.env
        if args.env == "FrozenLake-v1":
            cfg["environment"]["solved_threshold"] = 0.78

    env_name = get(cfg, "environment", "name")
    plotter = Plotter(cfg)
    eval_results = {}

    for name in args.agents:
        if name not in AGENTS:
            print(f"Unknown agent '{name}'. Available: {list(AGENTS.keys())}")
            continue

        print(f"\n{'─'*52}")
        print(f"  Training: {name}  on  {env_name}")
        if env_name == "FrozenLake-v1":
            print(f"  Reward shaping : {get(cfg, 'frozenlake', 'reward_shaping')}")
            print(
                f"  Randomisation  : {get(cfg, 'frozenlake', 'domain_randomisation')}"
            )
            print(f"  Slippery       : {get(cfg, 'frozenlake', 'is_slippery')}")
            ep_count = get(
                cfg, "frozenlake", "episodes", default=get(cfg, "training", "episodes")
            )
            print(f"  Episodes       : {ep_count}")
        print(f"{'─'*52}")

        # Probe env for dims
        probe_env, _ = make_env_factory(cfg)()
        action_dim = probe_env.action_space.n

        if env_name == "FrozenLake-v1":
            n_states = probe_env.observation_space.n
            state_dim = 1 if name == "Q-Learning" else n_states
        else:
            state_dim = probe_env.observation_space.shape[0]
            n_states = None
        probe_env.close()

        obs_fn = make_obs_fn(env_name, n_states, name)
        factory = make_env_factory(cfg)
        agent = AGENTS[name](cfg, state_dim, action_dim)
        rewards = train(agent, factory, cfg, obs_fn=obs_fn, agent_name=name)

        plotter.add(name, rewards)

        # Evaluate on fixed map (no reward shaping) for fair comparison
        print(f"  Evaluating {name} (original reward, fixed map)...")
        agent.epsilon.value = 0.0
        # temporarily disable randomisation for eval
        cfg_eval = dict(cfg)
        if env_name == "FrozenLake-v1":
            cfg_eval["frozenlake"] = dict(cfg["frozenlake"])
            cfg_eval["frozenlake"]["domain_randomisation"] = False
            cfg_eval["frozenlake"]["reward_shaping"] = False

        result = evaluate_agent(
            env_factory=lambda: make_env_factory(cfg_eval)()[0],
            select_action=lambda o, fn=obs_fn: agent.select_action(fn(o)),
            cfg=cfg,
        )
        eval_results[name] = result

    plotter.print_summary(eval_results)
    tag = env_name.lower().replace("-", "_").replace("v1", "").rstrip("_")
    plotter.save_and_show(f"comparison_{tag}")


if __name__ == "__main__":
    main()
