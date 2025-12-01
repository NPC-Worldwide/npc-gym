"""
Base emulator environment for game-based RL.

Provides integration with game emulators (PyBoy, etc.) for
training agents on classic games using hybrid LLM+ML approaches.
"""

from __future__ import annotations
from abc import abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Callable
from enum import Enum
import time

from PIL import Image

from npc_gym.core.env import (
    Environment, GameState, Observation, Action, Phase, Trace
)
from npc_gym.core.spaces import DiscreteSpace
from npc_gym.core.info import InfoPartition
from npc_gym.envs.emulator.vision import VisionProcessor, VisionConfig, SceneDescription


class EmulatorType(Enum):
    """Supported emulator types."""
    PYBOY = "pyboy"  # Game Boy / Game Boy Color
    MGBA = "mgba"    # Game Boy Advance
    MOCK = "mock"    # For testing without emulator


@dataclass
class EmulatorConfig:
    """Configuration for emulator environments."""
    # ROM settings
    rom_path: str = ""
    save_state_path: str = ""

    # Display settings
    window_type: str = "null"  # null, SDL2, headless
    emulation_speed: int = 0   # 0 = max speed, 1 = normal

    # Frame settings
    frames_per_step: int = 1   # Frames to run per env step
    frame_skip: int = 0        # Additional frames to skip
    max_steps: int = 10000

    # Vision settings
    vision_config: VisionConfig = field(default_factory=VisionConfig)
    capture_every_frame: bool = False

    # Action mapping
    action_repeat: int = 1     # Repeat each action N frames

    # Rewards
    reward_scale: float = 1.0


class EmulatorEnv(Environment):
    """
    Base class for emulator-based game environments.

    Wraps game emulators (PyBoy, mGBA, etc.) to provide
    Gymnasium-compatible interfaces with:
    - Screen capture and vision processing
    - Button/input mapping
    - Save state management
    - Frame-level control

    Subclasses implement game-specific:
    - Memory reading for rewards/game state
    - Action validation
    - Victory/loss conditions

    Example:
        class MyGameEnv(EmulatorEnv):
            def _read_game_state(self) -> Dict:
                # Read memory addresses
                return {"score": self._read_memory(0x1234)}

            def _compute_rewards(self) -> Dict[str, float]:
                return {"player_0": self.game_data.get("score", 0)}
    """

    env_id = "Emulator-v0"
    min_players = 1
    max_players = 1

    # Standard button mappings (subclasses override)
    BUTTONS = ["a", "b", "start", "select", "up", "down", "left", "right"]

    def __init__(
        self,
        config: EmulatorConfig = None,
        **kwargs
    ):
        # Default to single player
        kwargs.setdefault("num_players", 1)
        super().__init__(**kwargs)

        self.config = config or EmulatorConfig()
        self.emulator = None
        self.vision = VisionProcessor(self.config.vision_config)

        # Game state tracking
        self.game_data: Dict[str, Any] = {}
        self.last_screen: Optional[Image.Image] = None
        self.scene_description: Optional[SceneDescription] = None
        self.frame_count: int = 0

        # Action space
        self.action_space = DiscreteSpace(choices=self.BUTTONS + ["none"])

        # History for learning
        self.screen_history: List[Image.Image] = []
        self.action_history_detailed: List[Tuple[str, Dict]] = []

    def _setup_emulator(self) -> Any:
        """
        Initialize the emulator. Subclasses can override.

        Returns:
            Emulator instance (PyBoy, etc.)
        """
        if not self.config.rom_path:
            # Return mock emulator for testing
            return MockEmulator()

        try:
            from pyboy import PyBoy
            emu = PyBoy(
                self.config.rom_path,
                window_type=self.config.window_type
            )
            emu.set_emulation_speed(self.config.emulation_speed)

            # Load save state if provided
            if self.config.save_state_path:
                with open(self.config.save_state_path, "rb") as f:
                    emu.load_state(f)

            return emu

        except ImportError:
            print("PyBoy not installed. Using mock emulator.")
            return MockEmulator()
        except Exception as e:
            print(f"Failed to load ROM: {e}. Using mock emulator.")
            return MockEmulator()

    def _setup_game(self) -> GameState:
        """Initialize emulator and game state."""
        # Initialize emulator
        self.emulator = self._setup_emulator()
        self.frame_count = 0
        self.game_data = {}
        self.screen_history = []

        # Capture initial screen
        self._capture_screen()

        # Read initial game state
        self._read_game_state()

        state = GameState(
            phase=Phase.PLAYING,
            step=0,
            current_player=self.player_ids[0],
            player_order=self.player_ids.copy(),
            player_states={
                pid: {"score": 0, "actions_taken": 0}
                for pid in self.player_ids
            },
            metadata={
                "max_steps": self.config.max_steps,
                "frame_count": 0,
                "rom": self.config.rom_path,
            }
        )

        return state

    def _capture_screen(self) -> Image.Image:
        """Capture current emulator screen."""
        if hasattr(self.emulator, 'screen_image'):
            self.last_screen = self.emulator.screen_image()
        elif hasattr(self.emulator, 'screen'):
            # PyBoy specific
            screen = self.emulator.screen
            if hasattr(screen, 'image'):
                self.last_screen = screen.image
            else:
                self.last_screen = Image.new('RGB', (160, 144), color='black')
        else:
            # Mock
            self.last_screen = Image.new('RGB', (160, 144), color='gray')

        # Process with vision
        self.scene_description = self.vision.process(self.last_screen)

        # Keep history
        if self.config.capture_every_frame:
            self.screen_history.append(self.last_screen.copy())
            if len(self.screen_history) > 100:
                self.screen_history.pop(0)

        return self.last_screen

    def _read_game_state(self) -> Dict[str, Any]:
        """
        Read game-specific state from emulator memory.

        Subclasses should override to read relevant addresses.

        Returns:
            Dict of game state values
        """
        # Base implementation - subclasses override
        self.game_data = {
            "frame": self.frame_count,
            "running": self._is_emulator_running(),
        }
        return self.game_data

    def _is_emulator_running(self) -> bool:
        """Check if emulator is still running."""
        if hasattr(self.emulator, 'tick'):
            return True
        return False

    def _read_memory(self, address: int) -> int:
        """Read a byte from emulator memory."""
        if hasattr(self.emulator, 'memory'):
            return self.emulator.memory[address]
        elif hasattr(self.emulator, 'get_memory_value'):
            return self.emulator.get_memory_value(address)
        return 0

    def _write_memory(self, address: int, value: int) -> None:
        """Write a byte to emulator memory."""
        if hasattr(self.emulator, 'memory'):
            self.emulator.memory[address] = value

    def _press_button(self, button: str) -> None:
        """Press a button on the emulator."""
        if hasattr(self.emulator, 'button_press'):
            self.emulator.button_press(button)
        elif hasattr(self.emulator, 'button'):
            self.emulator.button(button)
        elif hasattr(self.emulator, 'press_button'):
            self.emulator.press_button(button)

    def _release_button(self, button: str) -> None:
        """Release a button on the emulator."""
        if hasattr(self.emulator, 'button_release'):
            self.emulator.button_release(button)
        elif hasattr(self.emulator, 'release_button'):
            self.emulator.release_button(button)

    def _tick(self, frames: int = 1) -> None:
        """Advance emulator by N frames."""
        for _ in range(frames):
            if hasattr(self.emulator, 'tick'):
                self.emulator.tick()
            self.frame_count += 1

    def _get_observation(self, player_id: str) -> Observation:
        """Get observation including screen description."""
        # Visual observation from scene
        visual_desc = ""
        if self.scene_description:
            visual_desc = self.scene_description.as_text()

        # Game state observation
        state_desc = self._format_game_state()

        info_partition = InfoPartition(
            player_id=player_id,
            private=[
                f"Screen: {visual_desc}",
                f"Game state: {state_desc}",
            ],
            public=[
                f"Frame: {self.frame_count}",
                f"Step: {self.state.step}",
            ],
        )

        valid_actions = self._get_valid_actions(player_id)

        game_state = {
            "frame": self.frame_count,
            "screen_description": visual_desc,
            **self.game_data,
        }

        return Observation(
            player_id=player_id,
            info_partition=info_partition,
            game_state=game_state,
            valid_actions=valid_actions,
            action_history=self.state.action_history[-10:] if self.state else [],
            step=self.state.step if self.state else 0,
        )

    def _format_game_state(self) -> str:
        """Format game data as string."""
        parts = []
        for key, value in self.game_data.items():
            parts.append(f"{key}={value}")
        return ", ".join(parts)

    def _get_valid_actions(self, player_id: str) -> List[str]:
        """Get valid button actions."""
        # Most emulator games allow all buttons
        return self.BUTTONS + ["none"]

    def _apply_action(self, action: Action) -> None:
        """Apply button press to emulator."""
        button = action.action_type.lower()

        if button in self.BUTTONS:
            # Press button
            self._press_button(button)

            # Hold for action_repeat frames
            self._tick(self.config.action_repeat)

            # Release button
            self._release_button(button)

            # Additional frame skip
            if self.config.frame_skip > 0:
                self._tick(self.config.frame_skip)
        elif button == "none":
            # Just advance frames
            self._tick(self.config.frames_per_step)
        else:
            # Unknown action
            self._tick(1)

        # Capture screen after action
        self._capture_screen()

        # Update game state
        self._read_game_state()

        # Track action
        self.action_history_detailed.append((button, self.game_data.copy()))

        # Update player state
        if self.state and action.player_id in self.state.player_states:
            self.state.player_states[action.player_id]["actions_taken"] += 1

    @abstractmethod
    def _compute_rewards(self) -> Dict[str, float]:
        """
        Compute rewards based on game state.

        Subclasses must implement game-specific reward logic.
        """
        pass

    @abstractmethod
    def _is_terminal(self) -> bool:
        """
        Check if game has ended.

        Subclasses must implement game-specific termination.
        """
        pass

    def save_state(self, path: str) -> None:
        """Save emulator state."""
        if hasattr(self.emulator, 'save_state'):
            with open(path, "wb") as f:
                self.emulator.save_state(f)

    def load_state(self, path: str) -> None:
        """Load emulator state."""
        if hasattr(self.emulator, 'load_state'):
            with open(path, "rb") as f:
                self.emulator.load_state(f)
            self._capture_screen()
            self._read_game_state()

    def get_screen(self) -> Optional[Image.Image]:
        """Get the current screen as PIL Image."""
        return self.last_screen

    def save_screenshot(self, path: str) -> None:
        """Save current screen to file."""
        if self.last_screen:
            self.last_screen.save(path)

    def close(self) -> None:
        """Clean up emulator."""
        if hasattr(self.emulator, 'stop'):
            self.emulator.stop()
        elif hasattr(self.emulator, 'close'):
            self.emulator.close()

    def _render_text(self) -> str:
        """Render current state as text."""
        lines = [
            f"=== {self.env_id} (Frame {self.frame_count}) ===",
            f"Step: {self.state.step if self.state else 0}",
            "",
            "Game State:",
        ]

        for key, value in self.game_data.items():
            lines.append(f"  {key}: {value}")

        if self.scene_description:
            lines.append("")
            lines.append("Screen:")
            lines.append(f"  {self.scene_description.summary}")

        return "\n".join(lines)


class MockEmulator:
    """Mock emulator for testing without actual ROM."""

    def __init__(self):
        self.frame = 0
        self.buttons_pressed = set()
        self._memory = [0] * 0x10000

    def tick(self):
        self.frame += 1
        return False

    def screen_image(self) -> Image.Image:
        """Generate a mock screen."""
        # Create gradient based on frame
        img = Image.new('RGB', (160, 144))
        pixels = img.load()
        for y in range(144):
            for x in range(160):
                r = (self.frame + x) % 256
                g = (self.frame + y) % 256
                b = (x + y) % 256
                pixels[x, y] = (r, g, b)
        return img

    def button_press(self, button: str):
        self.buttons_pressed.add(button)

    def button_release(self, button: str):
        self.buttons_pressed.discard(button)

    @property
    def memory(self):
        return self._memory

    def stop(self):
        pass
