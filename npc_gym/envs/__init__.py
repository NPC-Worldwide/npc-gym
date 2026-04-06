"""
Environment implementations for npc-gym.

Built-in environments:
- CardGame: Base class for card-based games
- InfoPoker: Poker-style partial information game with text/cards
- HypothesisBJ: Blackjack-style hypothesis formation game
- SynthesisTournament: Multi-round debate/synthesis tournament
- GridWorld: Spatial navigation with partial observability
- TicTacToe: Classic two-player competitive game
- EmulatorEnv: Base class for game emulator environments (PyBoy, etc.)
- PokemonEnv: Pokemon game environment with vision processing
"""

from npc_gym.envs.card_game import CardGame, CardGameConfig
from npc_gym.envs.info_poker import InfoPoker
from npc_gym.envs.hypothesis_bj import HypothesisBlackjack
from npc_gym.envs.synthesis import SynthesisTournament
from npc_gym.envs.grid_world import GridWorld, GridWorldConfig, Maze, ItemCollector
from npc_gym.envs.tictactoe import TicTacToe, ConnectFour
from npc_gym.envs.slime_volleyball import SlimeVolleyEnv

# Emulator environments (optional - may not have dependencies)
from npc_gym.envs.emulator import EmulatorEnv, EmulatorConfig, VisionProcessor, VisionConfig

try:
    from npc_gym.envs.emulator.pokemon import PokemonEnv, PokemonConfig
    _HAS_POKEMON = True
except ImportError:
    _HAS_POKEMON = False

# Registry of available environments
REGISTRY = {
    # Card games
    "CardGame-v1": CardGame,
    "InfoPoker-v1": InfoPoker,
    "HypothesisBJ-v1": HypothesisBlackjack,
    "Synthesis-v1": SynthesisTournament,
    # Spatial/navigation
    "GridWorld-v1": GridWorld,
    "Maze-v1": Maze,
    "ItemCollector-v1": ItemCollector,
    # Competitive board games
    "TicTacToe-v1": TicTacToe,
    "ConnectFour-v1": ConnectFour,
    # Physics / RL
    "SlimeVolley-v1": SlimeVolleyEnv,
    # Emulator environments
    "Emulator-v0": EmulatorEnv,
}

# Add Pokemon if available
if _HAS_POKEMON:
    REGISTRY["Pokemon-v1"] = PokemonEnv

__all__ = [
    # Card games
    "CardGame",
    "CardGameConfig",
    "InfoPoker",
    "HypothesisBlackjack",
    "SynthesisTournament",
    # Spatial
    "GridWorld",
    "GridWorldConfig",
    "Maze",
    "ItemCollector",
    # Competitive
    "TicTacToe",
    "ConnectFour",
    # Physics / RL
    "SlimeVolleyEnv",
    # Emulator
    "EmulatorEnv",
    "EmulatorConfig",
    "VisionProcessor",
    "VisionConfig",
    # Registry
    "REGISTRY",
]

# Conditionally add Pokemon to __all__
if _HAS_POKEMON:
    __all__.extend(["PokemonEnv", "PokemonConfig"])
