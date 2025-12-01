"""
Pokemon environment for training LLM+ML agents.

Wraps Pokemon Game Boy games (Red/Blue/Yellow/Crystal/etc.)
for hybrid agent training with:
- Vision-based observation
- Memory-reading for game state
- Reward shaping for progression
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
from enum import Enum

from npc_gym.envs.emulator.base import EmulatorEnv, EmulatorConfig


class PokemonGame(Enum):
    """Supported Pokemon games."""
    RED = "red"
    BLUE = "blue"
    YELLOW = "yellow"
    GOLD = "gold"
    SILVER = "silver"
    CRYSTAL = "crystal"


@dataclass
class PokemonConfig(EmulatorConfig):
    """Configuration for Pokemon environments."""
    game_version: PokemonGame = PokemonGame.CRYSTAL

    # Reward shaping
    badge_reward: float = 1000.0
    pokemon_caught_reward: float = 50.0
    level_up_reward: float = 10.0
    battle_win_reward: float = 20.0
    new_location_reward: float = 5.0
    hp_loss_penalty: float = -1.0
    faint_penalty: float = -50.0

    # Exploration bonuses
    exploration_bonus: float = 1.0
    item_pickup_reward: float = 5.0


class PokemonEnv(EmulatorEnv):
    """
    Pokemon game environment for RL training.

    Provides:
    - Screen capture and vision processing
    - Game state reading from memory
    - Reward shaping for game progression
    - Battle/exploration/menu handling

    Memory Addresses (Crystal version - others differ):
    - Player position: 0xDCB4 (Y), 0xDCB5 (X)
    - Map ID: 0xDA00
    - Badge count: 0xD857
    - Party size: 0xDCD7
    - Money: 0xD84E-0xD850

    Usage:
        config = PokemonConfig(
            rom_path="Pokemon-CrystalVersion.gbc",
            game_version=PokemonGame.CRYSTAL,
        )
        env = PokemonEnv(config=config)
        obs, info = env.reset()

        while not done:
            action = agent.act(obs)
            obs, reward, done, truncated, info = env.step(action)
    """

    env_id = "Pokemon-v1"

    # Extended button set for menu navigation
    BUTTONS = ["a", "b", "start", "select", "up", "down", "left", "right"]

    # Memory addresses by game version
    MEMORY_MAPS = {
        PokemonGame.CRYSTAL: {
            "player_x": 0xDCB5,
            "player_y": 0xDCB4,
            "map_id": 0xDA00,
            "map_bank": 0xDA01,
            "badge_flags": 0xD857,
            "party_size": 0xDCD7,
            "money_1": 0xD84E,
            "money_2": 0xD84F,
            "money_3": 0xD850,
            "player_hp_1": 0xDCDF,
            "player_hp_2": 0xDCE0,
            "player_max_hp_1": 0xDCE1,
            "player_max_hp_2": 0xDCE2,
            "in_battle": 0xD22D,
            "battle_type": 0xD230,
        },
        PokemonGame.RED: {
            "player_x": 0xD362,
            "player_y": 0xD361,
            "map_id": 0xD35E,
            "badge_flags": 0xD356,
            "party_size": 0xD163,
            "money_1": 0xD347,
            "money_2": 0xD348,
            "money_3": 0xD349,
            "player_hp_1": 0xD16C,
            "player_hp_2": 0xD16D,
            "player_max_hp_1": 0xD18D,
            "player_max_hp_2": 0xD18E,
            "in_battle": 0xD057,
        },
        PokemonGame.YELLOW: {
            # Similar to Red with some differences
            "player_x": 0xD362,
            "player_y": 0xD361,
            "map_id": 0xD35E,
            "badge_flags": 0xD356,
            "party_size": 0xD163,
        },
    }

    def __init__(self, config: PokemonConfig = None, **kwargs):
        self.pokemon_config = config or PokemonConfig()
        super().__init__(config=self.pokemon_config, **kwargs)

        # Get memory map for this game version
        self.memory_map = self.MEMORY_MAPS.get(
            self.pokemon_config.game_version,
            self.MEMORY_MAPS[PokemonGame.CRYSTAL]
        )

        # Tracking for rewards
        self.visited_locations = set()
        self.pokemon_caught = 0
        self.badges_earned = 0
        self.total_levels = 0
        self.battles_won = 0

        # Previous state for delta rewards
        self.prev_state: Dict[str, Any] = {}

    def _read_game_state(self) -> Dict[str, Any]:
        """Read Pokemon game state from memory."""
        state = {
            "frame": self.frame_count,
            "running": self._is_emulator_running(),
        }

        # Read position
        if "player_x" in self.memory_map:
            state["player_x"] = self._read_memory(self.memory_map["player_x"])
            state["player_y"] = self._read_memory(self.memory_map["player_y"])

        # Read map
        if "map_id" in self.memory_map:
            state["map_id"] = self._read_memory(self.memory_map["map_id"])

        # Read badges
        if "badge_flags" in self.memory_map:
            badge_byte = self._read_memory(self.memory_map["badge_flags"])
            state["badges"] = bin(badge_byte).count("1")

        # Read party
        if "party_size" in self.memory_map:
            state["party_size"] = self._read_memory(self.memory_map["party_size"])

        # Read money (BCD format)
        if "money_1" in self.memory_map:
            m1 = self._read_memory(self.memory_map["money_1"])
            m2 = self._read_memory(self.memory_map["money_2"])
            m3 = self._read_memory(self.memory_map["money_3"])
            state["money"] = self._bcd_to_int(m1, m2, m3)

        # Read HP
        if "player_hp_1" in self.memory_map:
            hp_high = self._read_memory(self.memory_map["player_hp_1"])
            hp_low = self._read_memory(self.memory_map["player_hp_2"])
            state["hp"] = (hp_high << 8) + hp_low

            if "player_max_hp_1" in self.memory_map:
                max_hp_high = self._read_memory(self.memory_map["player_max_hp_1"])
                max_hp_low = self._read_memory(self.memory_map["player_max_hp_2"])
                state["max_hp"] = (max_hp_high << 8) + max_hp_low
                state["hp_percent"] = state["hp"] / max(state["max_hp"], 1)

        # Check battle status
        if "in_battle" in self.memory_map:
            state["in_battle"] = self._read_memory(self.memory_map["in_battle"]) != 0

        # Location tracking
        location_key = (state.get("map_id", 0), state.get("player_x", 0), state.get("player_y", 0))
        state["new_location"] = location_key not in self.visited_locations
        if state["new_location"]:
            self.visited_locations.add(location_key)
        state["locations_visited"] = len(self.visited_locations)

        self.game_data = state
        return state

    def _bcd_to_int(self, *bytes_bcd) -> int:
        """Convert BCD encoded bytes to integer."""
        result = 0
        for b in bytes_bcd:
            result = result * 100 + ((b >> 4) * 10) + (b & 0x0F)
        return result

    def _compute_rewards(self) -> Dict[str, float]:
        """Compute rewards based on game progress."""
        reward = 0.0
        cfg = self.pokemon_config

        current = self.game_data
        prev = self.prev_state

        # Badge rewards
        current_badges = current.get("badges", 0)
        prev_badges = prev.get("badges", 0)
        if current_badges > prev_badges:
            reward += cfg.badge_reward * (current_badges - prev_badges)
            self.badges_earned = current_badges

        # New location bonus
        if current.get("new_location", False):
            reward += cfg.new_location_reward

        # HP changes
        current_hp_pct = current.get("hp_percent", 1.0)
        prev_hp_pct = prev.get("hp_percent", 1.0)
        if current_hp_pct < prev_hp_pct:
            hp_loss = prev_hp_pct - current_hp_pct
            reward += cfg.hp_loss_penalty * hp_loss * 100

        # Faint penalty
        if current.get("hp", 1) == 0 and prev.get("hp", 1) > 0:
            reward += cfg.faint_penalty

        # Party size increase (caught Pokemon)
        current_party = current.get("party_size", 0)
        prev_party = prev.get("party_size", 0)
        if current_party > prev_party:
            reward += cfg.pokemon_caught_reward * (current_party - prev_party)
            self.pokemon_caught += (current_party - prev_party)

        # Exploration bonus for frames spent moving to new areas
        if current.get("player_x") != prev.get("player_x") or \
           current.get("player_y") != prev.get("player_y"):
            reward += cfg.exploration_bonus * 0.1

        # Save current state for next comparison
        self.prev_state = current.copy()

        # Scale and return
        scaled_reward = reward * cfg.reward_scale

        return {self.player_ids[0]: scaled_reward}

    def _is_terminal(self) -> bool:
        """Check if game has ended."""
        # Pokemon games don't really end, so use step limit
        if self.state and self.state.step >= self.config.max_steps:
            return True

        # Could add specific endings (e.g., all badges, Elite Four beaten)
        if self.badges_earned >= 8:
            # All badges collected - optional termination
            pass

        return False

    def _get_valid_actions(self, player_id: str) -> List[str]:
        """Get valid actions based on game context."""
        # All buttons always valid for Pokemon
        return self.BUTTONS + ["none"]

    def _format_game_state(self) -> str:
        """Format Pokemon-specific game state."""
        parts = []

        if "player_x" in self.game_data:
            parts.append(f"Position: ({self.game_data['player_x']}, {self.game_data['player_y']})")

        if "map_id" in self.game_data:
            parts.append(f"Map: {self.game_data['map_id']}")

        if "badges" in self.game_data:
            parts.append(f"Badges: {self.game_data['badges']}/8")

        if "party_size" in self.game_data:
            parts.append(f"Party: {self.game_data['party_size']} Pokemon")

        if "hp_percent" in self.game_data:
            parts.append(f"HP: {self.game_data['hp_percent']:.0%}")

        if "in_battle" in self.game_data and self.game_data["in_battle"]:
            parts.append("IN BATTLE")

        parts.append(f"Explored: {len(self.visited_locations)} tiles")

        return ", ".join(parts)

    def _render_text(self) -> str:
        """Render Pokemon game state."""
        lines = [
            f"=== Pokemon ({self.pokemon_config.game_version.value}) ===",
            f"Frame: {self.frame_count}",
            f"Step: {self.state.step if self.state else 0}",
            "",
        ]

        if "player_x" in self.game_data:
            lines.append(f"Position: ({self.game_data['player_x']}, {self.game_data['player_y']})")

        if "badges" in self.game_data:
            lines.append(f"Badges: {''.join(['⭐' if i < self.game_data['badges'] else '○' for i in range(8)])}")

        if "party_size" in self.game_data:
            lines.append(f"Pokemon in party: {self.game_data['party_size']}")

        if "money" in self.game_data:
            lines.append(f"Money: ₽{self.game_data['money']}")

        if self.game_data.get("in_battle"):
            lines.append("")
            lines.append(">>> IN BATTLE <<<")

        lines.append("")
        lines.append(f"Locations explored: {len(self.visited_locations)}")
        lines.append(f"Total caught: {self.pokemon_caught}")

        if self.scene_description:
            lines.append("")
            lines.append("Screen: " + self.scene_description.summary[:100])

        return "\n".join(lines)


def make_pokemon_env(
    rom_path: str,
    game_version: str = "crystal",
    **kwargs
) -> PokemonEnv:
    """
    Create a Pokemon environment.

    Args:
        rom_path: Path to the Pokemon ROM file
        game_version: Game version (red, blue, yellow, gold, silver, crystal)
        **kwargs: Additional config options

    Returns:
        Configured PokemonEnv
    """
    version_map = {
        "red": PokemonGame.RED,
        "blue": PokemonGame.BLUE,
        "yellow": PokemonGame.YELLOW,
        "gold": PokemonGame.GOLD,
        "silver": PokemonGame.SILVER,
        "crystal": PokemonGame.CRYSTAL,
    }

    game = version_map.get(game_version.lower(), PokemonGame.CRYSTAL)

    config = PokemonConfig(
        rom_path=rom_path,
        game_version=game,
        **kwargs
    )

    return PokemonEnv(config=config)
