# RL Experiments

Modular RL framework for comparing algorithms on CartPole-v1.
All agents share the same config, training loop, evaluation, and plotting.

## Structure

```
rl_experiments/
├── config.yaml          ← all hyperparameters live here
├── utils.py             ← shared: config, seeding, replay buffer, plotting, eval
├── run_experiments.py   ← single entry point to train & compare agents
├── agents/
│   ├── qlearning.py     ← tabular Q-Learning
│   ├── dqn.py           ← Deep Q-Network
│   └── <your_agent>.py  ← add new algorithms here
└── results/             ← saved plots appear here
```

## Setup

```bash
pip install gymnasium torch matplotlib numpy pyyaml
```

## Run all agents

```bash
python run_experiments.py
```

## Run a single agent

```bash
python run_experiments.py --agents DQN
python run_experiments.py --agents Q-Learning
```

## Adding a new algorithm

1. Create `agents/my_agent.py` with a class that implements:
   - `select_action(obs) -> int`
   - `update(obs, action, reward, next_obs, done)` (or `store` + `update()` for replay-based)
   - `end_episode()`
   - `self.epsilon` — an `EpsilonSchedule` instance

2. Register it in `run_experiments.py`:
   ```python
   from agents.my_agent import MyAgent
   AGENTS = {
       "Q-Learning": QLearningAgent,
       "DQN":        DQNAgent,
       "My Agent":   MyAgent,      # ← add here
   }
   ```

3. Optionally add a section to `config.yaml` for its specific hyperparameters.

4. Run `python run_experiments.py` — the comparison plot updates automatically.

## Tweaking hyperparameters

Edit `config.yaml` — all agents pick up the change on the next run.
The `training` section (episodes, discount, epsilon schedule) is shared by all agents.
Agent-specific sections (`qlearning`, `dqn`) only affect that agent.