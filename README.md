# MARL Master Thesis — Independent PPO on Cooperative FrozenLake

Multi-Agent Reinforcement Learning experiments using **Independent PPO (IPPO)** with 3 cooperative agents on a custom FrozenLake environment.

---

## What was built

### Environment — `rl_experiments/envs/frozen_lake_marl.py`
A custom 2/3-agent cooperative FrozenLake following the **PettingZoo AEC API**:
- 3 agents navigate a grid independently (no communication)
- Each agent starts at a different safe tile spread across the map
- Each agent sees: own position + all partners' positions + goal position + agent identity (all one-hot encoded)
- Reward: `+1.0` for reaching the goal, `-0.3` for falling in a hole, `-0.005` per step, `+0.02` for moving closer to goal
- Episode ends when any agent reaches the goal or falls in a hole
- Supports domain randomisation (random map per episode), configurable grid size, and slippery movement

### Training — `rl_experiments/train_marl.py`
**Independent PPO (IPPO)** using Stable-Baselines3:
- Each agent has its own separate vectorised environment → truly independent learning
- Agents are trained sequentially (agent_0 → agent_1 → agent_2)
- Reward logging via SB3 callbacks → training curves saved automatically
- Saves/loads model weights as `.zip` files

### Benchmark — `rl_experiments/benchmark_marl.py`
Trains and evaluates agents across three map configurations in one run:

| Scenario | Success rate |
|---|---|
| 4×4 non-slippery | 94% |
| 6×6 non-slippery | 90% |
| 8×8 slippery | 80% |

### Ablation study — `rl_experiments/ablation_marl.py`
Tests how many agents are needed for good performance:

| Condition | Success rate |
|---|---|
| Random baseline | 26% |
| 1 trained agent | 16% |
| 2 trained agents | 60% |
| 3 trained agents | 82% |

Key finding: 1 trained agent performs *worse* than random because untrained teammates hit holes and end the episode early.

---

## Project structure

```
rl_experiments/
├── envs/
│   └── frozen_lake_marl.py   # custom 3-agent PettingZoo environment
├── results/
│   ├── marl/                 # saved agent weights from train_marl.py
│   ├── benchmark/            # results from benchmark_marl.py
│   └── ablation/             # results from ablation_marl.py
├── config.yaml               # all hyperparameters and env settings
├── train_marl.py             # main training script
├── benchmark_marl.py         # multi-scenario benchmark
├── ablation_marl.py          # ablation study (1/2/3 agents vs random)
└── utils.py                  # shared utilities
```

---

## Setup

```bash
# install dependencies
uv sync

# or with pip
pip install stable-baselines3 pettingzoo supersuit pygame tensorboard tqdm rich
```

---

## How to run

### Train agents (uses config.yaml settings)
```bash
cd rl_experiments
uv run python train_marl.py
```

### Train with custom timesteps
```bash
uv run python train_marl.py --timesteps 1000000
```

### Evaluate saved agents (no retraining)
```bash
uv run python train_marl.py --load
```

### Watch agents play (pygame window)
```bash
uv run python train_marl.py --load --render
```

### Train + render in one command
```bash
uv run python train_marl.py --render
```

---

### Run benchmark (4×4 / 6×6 / 8×8)
```bash
uv run python benchmark_marl.py
```
Saves results and comparison plot to `results/benchmark/`.

---

### Run ablation study (1 / 2 / 3 agents vs random)
```bash
# train all conditions + plot
uv run python ablation_marl.py

# render all conditions (must train first)
uv run python ablation_marl.py --render-only --episodes 3

# train + render in one command
uv run python ablation_marl.py --render --episodes 3
```
Saves bar chart to `results/ablation/ablation.png`.

---

## Key config options — `config.yaml`

```yaml
frozenlake:
  size: 6              # grid size (4, 6, or 8)
  is_slippery: false   # true adds stochastic movement
  domain_randomisation: true  # random map each episode

marl:
  total_timesteps: 500000   # training budget (split across agents)
  n_envs: 4                 # parallel environments per agent
  learning_rate: 0.0003
  n_steps: 256
  batch_size: 64

environment:
  eval_episodes: 50    # episodes used for evaluation
```

---

## Results location

| File | Description |
|---|---|
| `results/marl/agent_*.zip` | Latest trained agent weights |
| `results/marl/training_curves.png` | Training progress per agent |
| `results/benchmark/comparison.png` | Training curves across map sizes |
| `results/ablation/ablation.png` | Ablation bar chart |
