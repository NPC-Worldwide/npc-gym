"""
Gymnasium compatibility layer.

Provides the types and functions needed so that:

    import npc_gym as gym
    env = gym.make("SlimeVolley-v1")
    obs = env.reset()
    obs, reward, terminated, truncated, info = env.step(action)

Also proxies unknown env IDs to gymnasium if installed, so existing
gymnasium scripts work unchanged:

    import npc_gym as gym
    env = gym.make("CartPole-v1")  # falls through to gymnasium
"""

from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np


# ---------------------------------------------------------------------------
# Additional space types for gymnasium parity
# ---------------------------------------------------------------------------

from npc_gym.core.spaces import (
    Space, DiscreteSpace, ContinuousSpace, TextSpace,
    CompositeSpace, CardSpace, DeckSpace,
)

# Gymnasium-compatible aliases
Box = ContinuousSpace
Discrete = DiscreteSpace
Text = TextSpace
Dict = CompositeSpace


class MultiBinarySpace(Space):
    """Space of binary vectors: {0, 1}^n."""

    def __init__(self, n: int, seed: int = None):
        super().__init__(dtype=np.int8, seed=seed)
        self.n = n

    def sample(self) -> np.ndarray:
        return self._np_random.integers(0, 2, size=(self.n,), dtype=np.int8)

    def contains(self, x: Any) -> bool:
        x = np.asarray(x)
        return x.shape == (self.n,) and np.all((x == 0) | (x == 1))

    @property
    def shape(self) -> Tuple:
        return (self.n,)

    def __repr__(self):
        return f"MultiBinary({self.n})"


class MultiDiscreteSpace(Space):
    """Space of multi-dimensional discrete values."""

    def __init__(self, nvec, seed: int = None):
        super().__init__(dtype=np.int64, seed=seed)
        self.nvec = np.asarray(nvec, dtype=np.int64)

    def sample(self) -> np.ndarray:
        return np.array([
            self._np_random.integers(0, n) for n in self.nvec
        ], dtype=np.int64)

    def contains(self, x: Any) -> bool:
        x = np.asarray(x)
        return (
            x.shape == self.nvec.shape
            and np.all(x >= 0)
            and np.all(x < self.nvec)
        )

    @property
    def shape(self) -> Tuple:
        return tuple(self.nvec.shape)

    def __repr__(self):
        return f"MultiDiscrete({self.nvec})"


class TupleSpace(Space):
    """Ordered tuple of sub-spaces."""

    def __init__(self, spaces: tuple, seed: int = None):
        super().__init__(seed=seed)
        self.spaces = spaces

    def sample(self):
        return tuple(s.sample() for s in self.spaces)

    def contains(self, x: Any) -> bool:
        if not isinstance(x, (tuple, list)) or len(x) != len(self.spaces):
            return False
        return all(s.contains(v) for s, v in zip(self.spaces, x))

    @property
    def shape(self):
        return tuple(s.shape for s in self.spaces)

    def __repr__(self):
        return f"Tuple({', '.join(repr(s) for s in self.spaces)})"


MultiBinary = MultiBinarySpace
MultiDiscrete = MultiDiscreteSpace
Tuple = TupleSpace


# ---------------------------------------------------------------------------
# Wrapper base classes (gymnasium-compatible)
# ---------------------------------------------------------------------------

class Wrapper:
    """
    Base wrapper class. Wraps an environment and delegates all calls
    to the wrapped env by default.

    Subclass and override specific methods to modify behavior.
    """

    def __init__(self, env):
        self.env = env

    def __getattr__(self, name):
        return getattr(self.env, name)

    def reset(self, **kwargs):
        return self.env.reset(**kwargs)

    def step(self, action):
        return self.env.step(action)

    def render(self, **kwargs):
        return self.env.render(**kwargs)

    def close(self):
        return self.env.close()

    @property
    def unwrapped(self):
        return self.env.unwrapped if hasattr(self.env, 'unwrapped') else self.env


class ObservationWrapper(Wrapper):
    """Wrapper that modifies observations."""

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        return self.observation(obs), info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        return self.observation(obs), reward, terminated, truncated, info

    def observation(self, observation):
        raise NotImplementedError


class ActionWrapper(Wrapper):
    """Wrapper that modifies actions."""

    def step(self, action):
        return self.env.step(self.action(action))

    def action(self, action):
        raise NotImplementedError

    def reverse_action(self, action):
        raise NotImplementedError


class RewardWrapper(Wrapper):
    """Wrapper that modifies rewards."""

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        return obs, self.reward(reward), terminated, truncated, info

    def reward(self, reward):
        raise NotImplementedError


# ---------------------------------------------------------------------------
# Common wrappers
# ---------------------------------------------------------------------------

class TimeLimit(Wrapper):
    """Truncate episodes after max_steps."""

    def __init__(self, env, max_episode_steps: int):
        super().__init__(env)
        self.max_episode_steps = max_episode_steps
        self._elapsed_steps = 0

    def reset(self, **kwargs):
        self._elapsed_steps = 0
        return self.env.reset(**kwargs)

    def step(self, action):
        result = self.env.step(action)
        self._elapsed_steps += 1

        if len(result) == 5:
            obs, reward, terminated, truncated, info = result
        else:
            obs, reward, done, info = result
            terminated = done
            truncated = False

        if self._elapsed_steps >= self.max_episode_steps:
            truncated = True

        return obs, reward, terminated, truncated, info


class ClipReward(RewardWrapper):
    """Clip rewards to [min_r, max_r]."""

    def __init__(self, env, min_r: float = -1.0, max_r: float = 1.0):
        super().__init__(env)
        self.min_r = min_r
        self.max_r = max_r

    def reward(self, reward):
        return max(self.min_r, min(self.max_r, reward))


class FlattenObservation(ObservationWrapper):
    """Flatten dict/structured observations to a flat numpy array."""

    def observation(self, observation):
        if isinstance(observation, dict):
            parts = []
            for v in observation.values():
                parts.append(np.asarray(v).flatten())
            return np.concatenate(parts)
        return np.asarray(observation).flatten()


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------

_CUSTOM_REGISTRY: Dict[str, Any] = {}


def register(id: str, entry_point=None, **kwargs):
    """
    Register a custom environment.

    Args:
        id: Environment ID (e.g., "MyEnv-v0")
        entry_point: Class or "module:Class" string
        **kwargs: Default kwargs passed to the constructor
    """
    _CUSTOM_REGISTRY[id] = {
        "entry_point": entry_point,
        "kwargs": kwargs,
    }


def make(env_id: str, **kwargs):
    """
    Create an environment by ID.

    Resolution order:
    1. npc-gym's built-in REGISTRY
    2. Custom registered environments
    3. Gymnasium (if installed) — for CartPole, MountainCar, etc.
    """
    from npc_gym.envs import REGISTRY

    # 1. npc-gym built-in
    if env_id in REGISTRY:
        env_cls = REGISTRY[env_id]
        return env_cls(**kwargs)

    # 2. Custom registered
    if env_id in _CUSTOM_REGISTRY:
        spec = _CUSTOM_REGISTRY[env_id]
        ep = spec["entry_point"]
        merged = {**spec["kwargs"], **kwargs}
        if isinstance(ep, str):
            module_name, cls_name = ep.rsplit(":", 1)
            import importlib
            module = importlib.import_module(module_name)
            cls = getattr(module, cls_name)
            return cls(**merged)
        return ep(**merged)

    # 3. Fall through to gymnasium
    try:
        import gymnasium
        return gymnasium.make(env_id, **kwargs)
    except ImportError:
        pass
    except gymnasium.error.NameNotFound:
        pass

    # 4. Try old gym
    try:
        import gym
        return gym.make(env_id, **kwargs)
    except ImportError:
        pass

    available = list(REGISTRY.keys()) + list(_CUSTOM_REGISTRY.keys())
    raise ValueError(
        f"Unknown environment: {env_id}. "
        f"npc-gym envs: {', '.join(available)}. "
        f"Install gymnasium for classic envs (CartPole, etc.)."
    )


def list_envs() -> list:
    """List all available environments."""
    from npc_gym.envs import REGISTRY
    envs = list(REGISTRY.keys()) + list(_CUSTOM_REGISTRY.keys())
    try:
        import gymnasium
        envs += list(gymnasium.registry.keys())
    except ImportError:
        pass
    return sorted(set(envs))
