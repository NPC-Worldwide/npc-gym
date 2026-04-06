"""
npc-gym: A gymnasium-compatible framework for training hybrid LLM+ML agents.

Drop-in replacement for gymnasium:

    import npc_gym as gym
    env = gym.make("SlimeVolley-v1")       # npc-gym env
    env = gym.make("CartPole-v1")          # proxies to gymnasium

Extends gymnasium with multi-agent games, partial information,
text/LLM observations, and evolutionary training.
"""

# Core environment
from npc_gym.core.env import Environment, GameState, Observation, Action

# Spaces — gymnasium-compatible names
from npc_gym.core.spaces import (
    Space, DiscreteSpace, ContinuousSpace, TextSpace,
    DeckSpace, CardSpace, CompositeSpace,
)
from npc_gym.core.compat import (
    Box, Discrete, Text, Dict, Tuple,
    MultiBinary, MultiBinarySpace,
    MultiDiscrete, MultiDiscreteSpace,
    TupleSpace,
)

# Wrappers
from npc_gym.core.compat import (
    Wrapper, ObservationWrapper, ActionWrapper, RewardWrapper,
    TimeLimit, ClipReward, FlattenObservation,
)

# Information structures
from npc_gym.core.info import InformationStructure, InfoPartition

# Agents
from npc_gym.core.agent import Agent, HybridAgent

# Registration and env creation
from npc_gym.core.compat import make, register, list_envs

# Namespace for spaces (like gymnasium.spaces)
from npc_gym.core import spaces

__version__ = "0.1.1"
__author__ = "Christopher Agostino"

__all__ = [
    # Core
    "Environment", "GameState", "Observation", "Action",
    # Spaces (gymnasium names)
    "Space", "Box", "Discrete", "MultiBinary", "MultiDiscrete",
    "Dict", "Tuple", "Text",
    # Spaces (npc-gym specific)
    "DiscreteSpace", "ContinuousSpace", "TextSpace",
    "DeckSpace", "CardSpace", "CompositeSpace",
    "MultiBinarySpace", "MultiDiscreteSpace", "TupleSpace",
    # Wrappers
    "Wrapper", "ObservationWrapper", "ActionWrapper", "RewardWrapper",
    "TimeLimit", "ClipReward", "FlattenObservation",
    # Information
    "InformationStructure", "InfoPartition",
    # Agents
    "Agent", "HybridAgent",
    # Registration
    "make", "register", "list_envs",
    # Submodules
    "spaces",
]
