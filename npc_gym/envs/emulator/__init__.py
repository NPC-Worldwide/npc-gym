"""
Emulator-based game environments.

Supports game emulation (e.g., Game Boy via PyBoy) with
vision model integration for screen understanding.
"""

from npc_gym.envs.emulator.base import EmulatorEnv, EmulatorConfig
from npc_gym.envs.emulator.vision import VisionProcessor, VisionConfig

__all__ = [
    "EmulatorEnv",
    "EmulatorConfig",
    "VisionProcessor",
    "VisionConfig",
]

# Optional imports for specific emulators
try:
    from npc_gym.envs.emulator.pokemon import PokemonEnv, PokemonConfig
    __all__.extend(["PokemonEnv", "PokemonConfig"])
except ImportError:
    pass  # PyBoy not installed
