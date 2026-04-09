"""
train_minesweeper.py — MAPPO on 2-agent cooperative Minesweeper
================================================================
Architecture: MAPPO (Multi-Agent PPO)
  Shared actor:       π(a | o_i)       one network for both agents
  Centralized critic: V(o_0, o_1)      global state value estimate
  CTDE: centralized training, decentralized execution

AEC training loop:
  Agents take turns (agent_0 → agent_1 → ...).  For each step the trainer
  receives (local_obs, global_obs, action, log_prob, value, reward, done).
  global_obs = local_obs of acting agent + observed obs of partner agent.
  Reward is read from env.last_step_reward — the immediate shared reward
  set by MinesweeperMARLEnv.step() before cumulative accumulation.

Usage:
    python train_minesweeper.py
    python train_minesweeper.py --timesteps 300000
    python train_minesweeper.py --render
    python train_minesweeper.py --load --render


    # render-only (no training) at default speed
python train_minesweeper.py --render-only

# render-only, slow (1 move/sec — easy to follow)
python train_minesweeper.py --render-only --fps 1

# render-only, fast
python train_minesweeper.py --render-only --fps 15 --episodes 10

# after training, render at custom speed
python train_minesweeper.py --render --fps 2

"""

import argparse
import csv
import datetime
import time
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from envs.minesweeper_marl import env_creator, MinesweeperMARLEnv
from agents.mappo import MAPPOTrainer
from utils import load_config

SAVE_DIR = Path("results/minesweeper")


# ─────────────────────────────────────────────
#  MAPPO RUNNER
# ─────────────────────────────────────────────


class MAPPORunner:
    """
    Wraps the MAPPO training loop for Minesweeper.

    Handles the PettingZoo AEC env interaction, reward tracking, and
    periodic logging.  Keeps the same structure as IPPOTrainer in
    train_marl.py so the two are easy to compare.
    """

    def __init__(self, cfg: dict, total_timesteps: int, save_dir: Path = SAVE_DIR):
        self.cfg = cfg
        self.total_timesteps = total_timesteps
        self.save_dir = save_dir
        self.save_dir.mkdir(parents=True, exist_ok=True)

        ms = cfg.get("minesweeper", {})

        # probe env to read obs/action dimensions
        _probe = MinesweeperMARLEnv(cfg=cfg)
        obs_dim = _probe.observation_space("agent_0").shape[0]
        action_dim = _probe.action_space("agent_0").n
        _probe.close()

        self.trainer = MAPPOTrainer(cfg, obs_dim, action_dim, n_agents=2)
        self._reward_log: list[float] = []  # one entry per 100 episodes

        print(f"\n{'─'*52}")
        print(f"  MAPPO — 2-agent cooperative Minesweeper")
        print(f"  Board   : {ms.get('rows', 5)}×{ms.get('cols', 5)}")
        print(f"  Mines   : {ms.get('n_mines', 3)}")
        print(f"  Steps   : {total_timesteps:,}")
        print(f"  Device  : auto (honours system.device in config)")
        print(f"{'─'*52}\n")

    # ── training ─────────────────────────────────────────────────

    def train(self) -> list[float]:
        """
        Main training loop.

        For each env step:
          1. Get local obs from env.last() for the acting agent.
          2. Get partner obs via env.observe(partner) — always accessible in AEC.
          3. Form global_obs = concat(local_obs, partner_obs).
          4. Sample action + log_prob from shared actor.
          5. Get V from centralized critic (global_obs).
          6. Step env, read immediate reward from env.last_step_reward.
          7. Store transition, call maybe_update().

        Returns the mean-reward log (one value per 100 episodes).
        """
        env = env_creator(cfg=self.cfg)
        t_start = time.time()
        ep_count = 0
        env_steps = 0
        ep_rewards_buf: list[float] = []

        # ── CSV logger ───────────────────────────────────────────
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        csv_path = self.save_dir / f"training_log_{ts}.csv"
        _CSV_COLS = [
            "episode",
            "env_steps",
            "ep_reward",
            "win",
            "safe_revealed",
            "total_safe",
            "ep_actions",
            "mine_hit",
        ]
        csv_file = open(csv_path, "w", newline="")
        csv_writer = csv.DictWriter(csv_file, fieldnames=_CSV_COLS)
        csv_writer.writeheader()
        print(f"  Logging episodes → {csv_path}\n")

        while env_steps < self.total_timesteps:
            # ── single episode ───────────────────────────────────
            env.reset()
            ep_reward = 0.0
            ep_actions = 0

            for agent in env.agent_iter():
                obs, _, term, trunc, _ = env.last()
                done = term or trunc

                if done:
                    env.step(None)
                    continue

                # partner obs is always accessible in AEC (no side effects)
                partner = "agent_1" if agent == "agent_0" else "agent_0"
                global_obs = np.concatenate([obs, env.observe(partner)])

                value = self.trainer.get_value(global_obs)
                action, log_prob = self.trainer.get_action(agent, obs, mask=env.action_mask())

                env.step(action)
                env_steps += 1
                ep_actions += 1

                # immediate reward — delegated through OrderEnforcingWrapper
                reward = env.last_step_reward
                ep_reward += reward

                is_done = env.terminations.get(agent, False) or env.truncations.get(
                    agent, False
                )
                self.trainer.store(
                    agent,
                    obs,
                    global_obs,
                    action,
                    log_prob,
                    value,
                    reward,
                    is_done,
                )
                self.trainer.maybe_update()

                if env_steps >= self.total_timesteps:
                    break

            ep_count += 1
            ep_rewards_buf.append(ep_reward)

            # safe_revealed and mine_hit are public attrs on env (via __getattr__)
            csv_writer.writerow(
                {
                    "episode": ep_count,
                    "env_steps": env_steps,
                    "ep_reward": round(ep_reward, 4),
                    "win": int(not env.mine_hit and not any(env.truncations.values())),
                    "safe_revealed": env.safe_revealed,
                    "total_safe": env.safe_revealed
                    + (self.cfg.get("minesweeper", {}).get("n_mines", 3)),
                    "ep_actions": ep_actions,
                    "mine_hit": int(env.mine_hit),
                }
            )
            csv_file.flush()  # write immediately so file is readable during training

            if ep_count % 100 == 0:
                mean_r = float(np.mean(ep_rewards_buf[-100:]))
                self._reward_log.append(mean_r)
                elapsed = time.time() - t_start
                print(
                    f"  Ep {ep_count:5d}  |  steps {env_steps:7d}  |"
                    f"  mean reward (100ep): {mean_r:+.3f}  |  {elapsed:.0f}s"
                )

        csv_file.close()
        env.close()
        self.trainer.save(str(self.save_dir / "mappo"))
        print(f"\nTraining done — {ep_count} episodes, {env_steps} steps")
        print(f"Episode log → {csv_path}")
        return self._reward_log

    # ── evaluation ───────────────────────────────────────────────

    def evaluate(self, n_episodes: int = 50) -> dict:
        """
        Run trained agents greedily.
        Win = episode ends with net positive total reward (mine never hit).
        """
        env = env_creator(cfg=self.cfg)
        wins = 0
        ep_rewards: list[float] = []

        for _ in range(n_episodes):
            env.reset()
            ep_reward = 0.0

            for agent in env.agent_iter():
                obs, _, term, trunc, _ = env.last()
                if term or trunc:
                    env.step(None)
                else:
                    action, _ = self.trainer.get_action(agent, obs, mask=env.action_mask())
                    env.step(action)
                    ep_reward += env.last_step_reward

            ep_rewards.append(ep_reward)
            # true win: all safe cells cleared, no mine hit, no timeout
            if not env.mine_hit and not any(env.truncations.values()):
                wins += 1

        env.close()
        result = {
            "win_rate": wins / n_episodes,
            "mean_reward": float(np.mean(ep_rewards)),
            "std_reward": float(np.std(ep_rewards)),
        }
        print(f"\n── Evaluation ({n_episodes} episodes) ──")
        print(f"  Win rate    : {result['win_rate'] * 100:.1f}%")
        print(
            f"  Mean reward : {result['mean_reward']:.3f} ± {result['std_reward']:.3f}"
        )
        return result

    # ── plot ─────────────────────────────────────────────────────

    def plot(self):
        if not self._reward_log:
            print("No training data to plot — run train() first.")
            return

        rewards = self._reward_log
        window = 5

        fig, ax = plt.subplots(figsize=(10, 5))
        fig.suptitle(
            "MAPPO — Minesweeper training progress", fontsize=13, fontweight="bold"
        )

        ax.plot(rewards, color="#3b8be0", linewidth=0.5, alpha=0.3)
        if len(rewards) >= window:
            smoothed = np.convolve(rewards, np.ones(window) / window, mode="valid")
            ax.plot(
                range(window - 1, len(rewards)),
                smoothed,
                color="#3b8be0",
                linewidth=2,
                label="MAPPO (mean 100ep)",
            )

        ax.axhline(0, color="#aaaaaa", linewidth=0.8, linestyle="--")
        ax.set_xlabel("Checkpoint (×100 episodes)")
        ax.set_ylabel("Mean episode reward")
        ax.legend(fontsize=10)
        ax.spines[["top", "right"]].set_visible(False)

        out = self.save_dir / "training_curve.png"
        plt.tight_layout()
        plt.savefig(out, dpi=150)
        print(f"  Saved plot → {out}")
        plt.show()
        plt.close()

    # ── render ───────────────────────────────────────────────────

    def render(self, n_episodes: int = 5, fps: int = 4):
        """
        Watch trained agents play in a pygame window.

        fps controls playback speed:
          1  — very slow (easy to follow each move)
          4  — default (comfortable watching speed)
          10 — fast
          30 — very fast (almost instant)
        """
        import pygame

        env = MinesweeperMARLEnv(cfg=self.cfg, render_mode="human")
        env.render_fps = fps  # env reads this in _render_pygame()
        env.episode_total = n_episodes  # shown in info bar
        pause_ms = max(400, 2000 // fps)
        print(f"\nRendering {n_episodes} episodes  (fps={fps})...\n")

        for ep in range(n_episodes):
            env.reset()
            env.episode_num = ep + 1  # shown in info bar as "Ep X/Y"
            ep_reward, steps = 0.0, 0

            for agent in env.agent_iter():
                obs, _, term, trunc, _ = env.last()

                if term or trunc:
                    env.render()  # show final board state (mines revealed)
                    env.step(None)
                else:
                    action, _ = self.trainer.get_action(agent, obs, mask=env.action_mask())
                    env.step(action)
                    env.render()
                    ep_reward += env.last_step_reward
                    steps += 1

            won = not env.mine_hit and not any(env.truncations.values())
            result = (
                "WIN"
                if won
                else ("LOST — mine hit" if env.mine_hit else "LOST — timeout")
            )
            print(
                f"  Episode {ep + 1}: {result}  "
                f"(reward={ep_reward:.2f}, steps={steps})"
            )
            pygame.time.wait(pause_ms)  # pause on final board before next episode

        env.close()


# ─────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(
        description="Train or render MAPPO agents on cooperative Minesweeper.",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument(
        "--timesteps", type=int, default=None, help="Override total training timesteps"
    )
    parser.add_argument(
        "--render",
        action="store_true",
        help="Open pygame window after training/loading",
    )
    parser.add_argument(
        "--render-only",
        action="store_true",
        help="Skip training — load saved model and render immediately\n"
        "(shortcut for --load --render)",
    )
    parser.add_argument(
        "--load", action="store_true", help="Load saved model instead of training"
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=15,
        help="Number of episodes to render (default: 5)",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=4,
        help="Rendering speed in frames per second\n"
        "  1  = very slow   4  = default\n"
        "  10 = fast        30 = very fast  (default: 4)",
    )
    args = parser.parse_args()

    # --render-only is a shortcut for --load --render
    if args.render_only:
        args.load = True
        args.render = True

    cfg = load_config(args.config)
    ms = cfg.get("minesweeper", {})
    steps = args.timesteps or ms.get("total_timesteps", 200_000)

    runner = MAPPORunner(cfg=cfg, total_timesteps=steps)

    if args.load:
        runner.trainer.load(str(SAVE_DIR / "mappo"))
    else:
        runner.train()
        runner.plot()

    if not args.render_only:
        runner.evaluate(n_episodes=ms.get("eval_episodes", 50))

    if args.render:
        runner.render(n_episodes=args.episodes, fps=args.fps)


if __name__ == "__main__":
    main()
