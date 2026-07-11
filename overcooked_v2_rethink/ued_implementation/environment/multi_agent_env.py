"""Minimal MultiAgentEnv base class for OvercookedV2."""

class MultiAgentEnv:
    """Base class for multi-agent JAX environments."""
    def __init__(self, num_agents: int = 2):
        self.num_agents = num_agents
