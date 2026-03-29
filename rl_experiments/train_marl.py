"""
train_marl.py — train 2 cooperative PPO agents on FrozenLake
=============================================================
Architecture: Independent PPO (IPPO)
  - two separate SB3 PPO agents
  - each sees own pos + partner pos + goal pos
  - both receive the same joint reward
  - no communication during execution

Uses:
  PettingZoo  — multi-agent env API
  SuperSuit   — converts PettingZoo env → SB3-compatible vec env
  SB3 PPO     — battle-tested policy gradient implementation

Usage:
    python train_marl.py
    python train_marl.py --timesteps 200000
    python train_marl.py --render              # watch trained agents
"""

import argparse
import os
import time
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecMonitor
import supersuit as ss

from utils import load_config, get
from envs.frozen_lake_marl import env_creator, FrozenLakeMARLEnv


# ─────────────────────────────────────────────
#  ENV FACTORY
#  SuperSuit converts PettingZoo → SB3 VecEnv
#  Each agent gets its own parallel env copy
# ─────────────────────────────────────────────
def make_vec_env(cfg: dict, n_envs: int = 4):
    """
    Wraps our PettingZoo env into a vectorised SB3-compatible env.
    SuperSuit requires a ParallelEnv, so we convert from AEC first.
    concat_vec_envs_v1 expects a *callable* that returns a VecEnv.
    """
    from pettingzoo.utils.conversions import aec_to_parallel

    def _make():
        return aec_to_parallel(env_creator(cfg=cfg))

    env = ss.pettingzoo_env_to_vec_env_v1(_make())
    env = ss.concat_vec_envs_v1(env, n_envs, num_cpus=0, base_class="stable_baselines3")
    env = VecMonitor(env)
    return env


# ─────────────────────────────────────────────
#  IPPO TRAINER
# ─────────────────────────────────────────────
class IPPOTrainer:
    """
    Independent PPO — two SB3 PPO agents sharing the same env.

    Both agents are trained simultaneously. After every `train_freq`
    steps each agent updates its own network using experience it
    collected from the shared environment.
    """

    def __init__(self, cfg: dict, total_timesteps: int, save_dir: str = "results/marl"):
        self.cfg = cfg
        self.total_timesteps = total_timesteps
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)

        marl = cfg.get("marl", {})
        self.n_envs = marl.get("n_envs", 4)
        self.lr = marl.get("learning_rate", 3e-4)
        self.n_steps = marl.get("n_steps", 256)
        self.batch_size = marl.get("batch_size", 64)
        self.n_epochs = marl.get("n_epochs", 4)
        self.clip_range = marl.get("clip_range", 0.2)
        self.ent_coef = marl.get("ent_coef", 0.01)

        print(f"\n{'─'*52}")
        print(f"  IPPO — 3-agent cooperative FrozenLake")
        print(f"  Timesteps : {total_timesteps:,}")
        print(f"  Parallel envs : {self.n_envs} per agent")
        print(f"  Device    : auto (SB3 picks best available)")
        print(f"{'─'*52}\n")

        # Each agent gets its own independent vec_env → true IPPO
        # This prevents agents from sharing experience and converging to one policy
        agent_names = ["agent_0", "agent_1", "agent_2"]
        self.agents = {
            name: self._make_ppo(name, make_vec_env(cfg, n_envs=self.n_envs))
            for name in agent_names
        }

    def _make_ppo(self, name: str, env) -> PPO:
        """Create one SB3 PPO agent with its own env."""
        return PPO(
            policy="MlpPolicy",
            env=env,
            learning_rate=self.lr,
            n_steps=self.n_steps,
            batch_size=self.batch_size,
            n_epochs=self.n_epochs,
            clip_range=self.clip_range,
            ent_coef=self.ent_coef,
            gamma=get(self.cfg, "training", "discount", default=0.99),
            gae_lambda=0.95,
            verbose=0,
            policy_kwargs=dict(net_arch=[64, 64]),  # same as your custom PPO
        )

    # ─────────────────────────────────────────────
    #  TRAINING
    # ─────────────────────────────────────────────
    def train(self):
        """
        Alternate training between agent_0 and agent_1.
        Each agent collects n_steps * n_envs transitions then updates.
        """
        steps_per_agent = self.total_timesteps // 2
        rewards_log = {"agent_0": [], "agent_1": []}
        t_start = time.time()

        print("Training agent_0 and agent_1 alternately...\n")

        chunk = steps_per_agent // 20  # log 20 times total
        logged_at = 0

        for agent_name, agent in self.agents.items():
            print(f"  Training {agent_name}...")
            agent.learn(
                total_timesteps=steps_per_agent,
                reset_num_timesteps=True,
                progress_bar=True,
            )

        elapsed = time.time() - t_start
        print(f"\nTraining done in {elapsed:.1f}s")
        return rewards_log

    # ─────────────────────────────────────────────
    #  EVALUATION
    # ─────────────────────────────────────────────
    def evaluate(self, n_episodes: int = 50) -> dict:
        """
        Run both agents greedily on a fixed map.
        Returns success rate and mean episode reward.
        """
        # fixed map for fair eval — no randomisation
        raw_env = env_creator(cfg=self.cfg)

        successes = 0
        ep_rewards = []

        for ep in range(n_episodes):
            raw_env.reset()
            ep_reward = 0.0

            # AEC API: agent_iter() + last() is the standard loop
            for agent in raw_env.agent_iter():
                obs, reward, termination, truncation, _ = raw_env.last()
                ep_reward += reward

                if termination or truncation:
                    raw_env.step(None)
                else:
                    action, _ = self.agents[agent].predict(
                        obs.reshape(1, -1), deterministic=True
                    )
                    raw_env.step(int(action[0]))

            ep_rewards.append(ep_reward)
            if ep_reward >= 1.0:
                successes += 1

        raw_env.close()

        result = {
            "success_rate": successes / n_episodes,
            "mean_reward": float(np.mean(ep_rewards)),
            "std_reward": float(np.std(ep_rewards)),
        }
        print(f"\n── Evaluation ({n_episodes} episodes) ──")
        print(f"  Success rate : {result['success_rate']*100:.1f}%")
        print(
            f"  Mean reward  : {result['mean_reward']:.3f} ± {result['std_reward']:.3f}"
        )
        return result

    # ─────────────────────────────────────────────
    #  SAVE / LOAD
    # ─────────────────────────────────────────────
    def save(self):
        for name, agent in self.agents.items():
            path = self.save_dir / f"{name}.zip"
            agent.save(str(path))
            print(f"  Saved {name} → {path}")

    def load(self):
        for name in self.agents:
            path = self.save_dir / f"{name}.zip"
            if path.exists():
                self.agents[name] = PPO.load(str(path), env=self.agents[name].get_env())
                print(f"  Loaded {name} ← {path}")

    # ─────────────────────────────────────────────
    #  RENDER  — watch the trained agents play
    # ─────────────────────────────────────────────
    def render(self, n_episodes: int = 5):
        env = FrozenLakeMARLEnv(cfg=self.cfg, render_mode="human")

        for ep in range(n_episodes):
            env.reset()
            ep_reward = 0.0
            step = 0

            for agent in env.agent_iter():
                obs, reward, termination, truncation, _ = env.last()
                ep_reward += reward
                env.render()
                step += 1

                if termination or truncation:
                    env.step(None)
                else:
                    action, _ = self.agents[agent].predict(
                        obs.reshape(1, -1), deterministic=False  # stochastic → varied paths
                    )
                    env.step(int(action[0]))

            result = "SUCCESS" if ep_reward >= 1.0 else "failed"
            print(f"  Episode {ep+1}: {result}  (reward={ep_reward:.2f}, steps={step})")

        env.close()


# ─────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument(
        "--timesteps", type=int, default=None, help="Override total training timesteps"
    )
    parser.add_argument(
        "--render", action="store_true", help="Render trained agents after training"
    )
    parser.add_argument(
        "--load", action="store_true", help="Load saved agents instead of training"
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    marl = cfg.get("marl", {})
    steps = args.timesteps or marl.get("total_timesteps", 200_000)

    trainer = IPPOTrainer(cfg=cfg, total_timesteps=steps)

    if args.load:
        trainer.load()
    else:
        trainer.train()
        trainer.save()

    trainer.evaluate(n_episodes=get(cfg, "environment", "eval_episodes", default=50))

    if args.render:
        n_eval = get(cfg, "environment", "eval_episodes", default=50)
        trainer.render(n_episodes=n_eval)


if __name__ == "__main__":
    main()
