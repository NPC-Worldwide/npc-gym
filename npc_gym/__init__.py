"""
npc-gym: A Gymnasium-style framework for training hybrid LLM+ML agents

Core concepts:
- Environments simulate games with partial information
- Agents combine fast pattern recognition (System 1) with LLM reasoning (System 2)
- Training evolves model ensembles through gameplay
- Streaming NLP processing for real-time text games
- PID (Partial Information Decomposition) for multi-agent inference

Inspired by OpenAI Gym but designed for cognitive agents.
"""

from npc_gym.core.env import Environment, GameState, Observation, Action
from npc_gym.core.spaces import (
    Space, DiscreteSpace, ContinuousSpace, TextSpace,
    DeckSpace, CardSpace, CompositeSpace
)
from npc_gym.core.info import InformationStructure, InfoPartition
from npc_gym.core.agent import Agent, HybridAgent

__version__ = "0.1.0"
__author__ = "Christopher Agostino"

__all__ = [
    # Core
    "Environment",
    "GameState",
    "Observation",
    "Action",
    # Spaces
    "Space",
    "DiscreteSpace",
    "ContinuousSpace",
    "TextSpace",
    "DeckSpace",
    "CardSpace",
    "CompositeSpace",
    # Information
    "InformationStructure",
    "InfoPartition",
    # Agents
    "Agent",
    "HybridAgent",
]


def make(env_id: str, **kwargs) -> "Environment":
    """
    Create an environment by ID.

    Args:
        env_id: Environment identifier (e.g., "CardGame-v1", "PartialInfo-v1")
        **kwargs: Environment-specific configuration

    Returns:
        Configured Environment instance

    Example:
        >>> env = npc_gym.make("InfoPoker-v1", num_players=4)
        >>> obs, info = env.reset()
        >>> action = agent.act(obs)
        >>> obs, reward, terminated, truncated, info = env.step(action)
    """
    from npc_gym.envs import REGISTRY

    if env_id not in REGISTRY:
        available = ", ".join(REGISTRY.keys())
        raise ValueError(f"Unknown environment: {env_id}. Available: {available}")

    env_cls = REGISTRY[env_id]
    return env_cls(**kwargs)


def list_envs() -> list:
    """List all registered environments."""
    from npc_gym.envs import REGISTRY
    return list(REGISTRY.keys())
