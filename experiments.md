### Learning map

suggested learning order based on that map:

**Step 1** — Nail single-agent RL. Implement Q-learning and then PPO on a simple environment (CartPole, then LunarLander). Don't skip this — MARL is much harder to debug if you don't already have intuition for rewards, discount factors, and training stability.
- Buildng simple RL to refresh basics in RL 
  - basic Q-Learning with 4-dimensional continuous state space (solved with BINS) and BELLMAN UPDATE & EPSILON-GREEDY POLICY
  - DQN: the network learns to approximate the Q-function throught training, and generalises it. The network takes 4 raw floats and outputs 3 Q-values.
  - PPO: the network learns a policy directly, and is more stable than DQN. The network takes 4 raw floats and outputs a probability distribution over actions.
    - Components:
      - **Replay Buffer:** `state, action, reward, next_state, done` are stored there. During training we sample random batch from it, because the each step is correlated with the prev one.
      - **Target network:** frozen copy of the main network, updated only every `n` episodes to compute Q-target in the Bellman update -> makes target stable amd long enogh for the online network to learn from it.

- 
  
**Step 2** — Understand Dec-POMDPs. Read the formal definition and get comfortable with the idea that each robot has its own observation function. The joint reward is what makes it cooperative rather than competitive.

**Step 3** — Implement CTDE with MADDPG or MAPPO. Start with PettingZoo's simple cooperative environments (e.g. simple_spread — agents must cover landmarks). MADDPG is conceptually clean to start with; MAPPO is more stable and scales better.

**Step 4** — Tackle reward and credit assignment. This is where cooperative tasks get subtle. A shared team reward is simple but the robots struggle to learn who contributed. Techniques like QMIX and QPLEX mix individual value estimates into a joint value in a structured way.

**Step 5** — Graduate to physical simulation. Once you have the learning loop working, try Isaac Gym or MuJoCo for proper robot kinematics. This is where you'll face sim-to-real gaps too.


