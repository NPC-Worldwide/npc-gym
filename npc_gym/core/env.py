"""
Core Environment class for npc-gym.

Follows Gymnasium API but extended for:
- Multi-agent games
- Partial information
- Text/LLM observations and actions
- Trace collection for training
"""

from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union, Callable
from enum import Enum
import copy
import time

from npc_gym.core.spaces import Space, DiscreteSpace, CompositeSpace
from npc_gym.core.info import InformationStructure, InfoPartition


class Phase(Enum):
    """Game phases."""
    SETUP = "setup"
    PLAYING = "playing"
    SHOWDOWN = "showdown"
    TERMINAL = "terminal"


@dataclass
class Action:
    """
    A player action with optional metadata.

    Can be simple (fold/call) or structured (bet amount + reasoning).
    """
    player_id: str
    action_type: str
    value: Any = None
    reasoning: Optional[str] = None
    confidence: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "player_id": self.player_id,
            "action_type": self.action_type,
            "value": self.value,
            "reasoning": self.reasoning,
            "confidence": self.confidence,
            "metadata": self.metadata,
        }


@dataclass
class Observation:
    """
    What a player observes at a given state.

    Combines:
    - Information partition (private/public info)
    - Game state (phase, pot, positions, etc.)
    - Valid actions
    - History of actions
    """
    player_id: str
    info_partition: InfoPartition
    game_state: Dict[str, Any]
    valid_actions: List[str]
    action_history: List[Action] = field(default_factory=list)
    step: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "player_id": self.player_id,
            "private_info": self.info_partition.private,
            "public_info": self.info_partition.public,
            "game_state": self.game_state,
            "valid_actions": self.valid_actions,
            "action_history": [a.to_dict() for a in self.action_history[-10:]],  # Last 10
            "step": self.step,
        }

    def as_text(self) -> str:
        """Render observation as text for LLM consumption."""
        parts = [
            f"=== Observation for {self.player_id} (Step {self.step}) ===",
            "",
            self.info_partition.as_text(),
            "",
            "Game State:",
        ]

        for key, val in self.game_state.items():
            parts.append(f"  {key}: {val}")

        parts.append("")
        parts.append(f"Valid Actions: {', '.join(self.valid_actions)}")

        if self.action_history:
            parts.append("")
            parts.append("Recent Actions:")
            for action in self.action_history[-5:]:
                parts.append(f"  {action.player_id}: {action.action_type} {action.value or ''}")

        return "\n".join(parts)


@dataclass
class GameState:
    """
    Complete game state (may not be fully observable).

    This is the "god view" of the game - used for:
    - Environment logic
    - Evaluation
    - Replay/visualization
    """
    phase: Phase = Phase.SETUP
    step: int = 0
    current_player: Optional[str] = None
    player_order: List[str] = field(default_factory=list)
    player_states: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    info_structure: Optional[InformationStructure] = None
    action_history: List[Action] = field(default_factory=list)
    rewards: Dict[str, float] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def next_player(self) -> Optional[str]:
        """Get the next player in order."""
        if not self.player_order or not self.current_player:
            return None
        idx = self.player_order.index(self.current_player)
        next_idx = (idx + 1) % len(self.player_order)
        return self.player_order[next_idx]

    def clone(self) -> "GameState":
        """Deep copy the game state."""
        new_state = GameState(
            phase=self.phase,
            step=self.step,
            current_player=self.current_player,
            player_order=self.player_order.copy(),
            player_states=copy.deepcopy(self.player_states),
            info_structure=self.info_structure.clone() if self.info_structure else None,
            action_history=self.action_history.copy(),
            rewards=self.rewards.copy(),
            metadata=copy.deepcopy(self.metadata),
        )
        return new_state


@dataclass
class StepResult:
    """Result of an environment step."""
    observations: Dict[str, Observation]  # Per-player observations
    rewards: Dict[str, float]             # Per-player rewards
    terminated: bool                       # Game ended naturally
    truncated: bool                        # Game ended by limit
    info: Dict[str, Any]                  # Additional info


@dataclass
class Trace:
    """
    A complete game trace for training.

    Contains:
    - Full action/observation history
    - Final rewards
    - Ground truth (for evaluation)
    """
    env_id: str
    player_ids: List[str]
    steps: List[Dict[str, Any]] = field(default_factory=list)
    final_rewards: Dict[str, float] = field(default_factory=dict)
    winner: Optional[str] = None
    ground_truth: Any = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def add_step(
        self,
        observations: Dict[str, Observation],
        actions: Dict[str, Action],
        rewards: Dict[str, float]
    ) -> None:
        """Record a step in the trace."""
        self.steps.append({
            "observations": {k: v.to_dict() for k, v in observations.items()},
            "actions": {k: v.to_dict() for k, v in actions.items()},
            "rewards": rewards,
        })

    def to_preference_pairs(
        self,
        min_reward_gap: float = 0.2
    ) -> List[Dict[str, Any]]:
        """
        Convert trace to DPO preference pairs.

        Compares actions at each step, preferring those that led to higher rewards.
        """
        pairs = []

        # Simple version: compare final winner vs losers
        if self.winner and len(self.player_ids) > 1:
            winner_reward = self.final_rewards.get(self.winner, 0)
            for player_id in self.player_ids:
                if player_id != self.winner:
                    loser_reward = self.final_rewards.get(player_id, 0)
                    if winner_reward - loser_reward >= min_reward_gap:
                        # Find representative actions
                        for step in self.steps:
                            winner_action = step["actions"].get(self.winner)
                            loser_action = step["actions"].get(player_id)
                            if winner_action and loser_action:
                                pairs.append({
                                    "prompt": str(step["observations"].get(self.winner, {})),
                                    "chosen": str(winner_action),
                                    "rejected": str(loser_action),
                                    "reward_gap": winner_reward - loser_reward,
                                })

        return pairs


class Environment(ABC):
    """
    Base class for all npc-gym environments.

    Follows Gymnasium API:
    - reset() -> observation, info
    - step(action) -> observation, reward, terminated, truncated, info

    Extended for multi-agent:
    - Observations and rewards are dicts keyed by player_id
    - Actions can be single (current player) or dict (simultaneous)
    """

    # Subclasses should override these
    env_id: str = "BaseEnv-v0"
    max_players: int = 10
    min_players: int = 2

    def __init__(
        self,
        player_ids: List[str] = None,
        num_players: int = 2,
        seed: int = None,
        collect_traces: bool = True,
        **kwargs
    ):
        if player_ids:
            self.player_ids = player_ids
        else:
            self.player_ids = [f"player_{i}" for i in range(num_players)]

        if not (self.min_players <= len(self.player_ids) <= self.max_players):
            raise ValueError(
                f"Player count {len(self.player_ids)} not in range "
                f"[{self.min_players}, {self.max_players}]"
            )

        self.seed = seed
        self.collect_traces = collect_traces
        self._rng = self._create_rng(seed)

        # State
        self.state: Optional[GameState] = None
        self.current_trace: Optional[Trace] = None

        # Spaces (subclasses should define)
        self.observation_space: Optional[Space] = None
        self.action_space: Optional[Space] = None

    def _create_rng(self, seed: int = None):
        """Create random number generator."""
        import numpy as np
        return np.random.default_rng(seed)

    @abstractmethod
    def _setup_game(self) -> GameState:
        """Initialize the game state. Subclasses implement this."""
        pass

    @abstractmethod
    def _get_observation(self, player_id: str) -> Observation:
        """Get observation for a specific player."""
        pass

    @abstractmethod
    def _get_valid_actions(self, player_id: str) -> List[str]:
        """Get valid actions for a player."""
        pass

    @abstractmethod
    def _apply_action(self, action: Action) -> None:
        """Apply an action to the game state."""
        pass

    @abstractmethod
    def _compute_rewards(self) -> Dict[str, float]:
        """Compute rewards for all players."""
        pass

    @abstractmethod
    def _is_terminal(self) -> bool:
        """Check if game has ended."""
        pass

    def reset(
        self,
        seed: int = None,
        options: Dict = None
    ) -> Tuple[Dict[str, Observation], Dict[str, Any]]:
        """
        Reset the environment to initial state.

        Returns:
            observations: Dict mapping player_id to their observation
            info: Additional information
        """
        if seed is not None:
            self._rng = self._create_rng(seed)

        # Initialize game
        self.state = self._setup_game()

        # Initialize trace
        if self.collect_traces:
            self.current_trace = Trace(
                env_id=self.env_id,
                player_ids=self.player_ids.copy(),
            )

        # Get initial observations
        observations = {
            player_id: self._get_observation(player_id)
            for player_id in self.player_ids
        }

        info = {
            "phase": self.state.phase.value,
            "current_player": self.state.current_player,
        }

        return observations, info

    def step(
        self,
        action: Union[Action, Dict[str, Action]]
    ) -> Tuple[Dict[str, Observation], Dict[str, float], bool, bool, Dict[str, Any]]:
        """
        Take a step in the environment.

        Args:
            action: Either a single Action (for current player)
                   or a dict of Actions (for simultaneous moves)

        Returns:
            observations: Dict of player observations
            rewards: Dict of player rewards
            terminated: Whether game ended naturally
            truncated: Whether game ended by limit
            info: Additional information
        """
        if self.state is None:
            raise RuntimeError("Must call reset() before step()")

        # Normalize action to dict
        if isinstance(action, Action):
            actions = {action.player_id: action}
        else:
            actions = action

        # Record pre-step observations
        pre_observations = {
            player_id: self._get_observation(player_id)
            for player_id in self.player_ids
        }

        # Apply actions
        for player_id, act in actions.items():
            self._apply_action(act)
            self.state.action_history.append(act)

        # Advance state
        self.state.step += 1
        if self.state.info_structure:
            self.state.info_structure.step()

        # Check terminal
        terminated = self._is_terminal()
        truncated = self.state.step >= self.state.metadata.get("max_steps", 1000)

        # Compute rewards
        rewards = self._compute_rewards()
        self.state.rewards = rewards

        # Get new observations
        observations = {
            player_id: self._get_observation(player_id)
            for player_id in self.player_ids
        }

        # Record trace
        if self.collect_traces and self.current_trace:
            self.current_trace.add_step(pre_observations, actions, rewards)
            if terminated or truncated:
                self.current_trace.final_rewards = rewards

        info = {
            "phase": self.state.phase.value,
            "current_player": self.state.current_player,
            "step": self.state.step,
        }

        return observations, rewards, terminated, truncated, info

    def get_trace(self) -> Optional[Trace]:
        """Get the current game trace."""
        return self.current_trace

    def render(self, mode: str = "text") -> Optional[str]:
        """
        Render the current state.

        Modes:
        - "text": Return text representation
        - "dict": Return dict representation
        - "html": Return HTML for web visualization
        """
        if self.state is None:
            return None

        if mode == "text":
            return self._render_text()
        elif mode == "dict":
            return self._render_dict()
        elif mode == "html":
            return self._render_html()
        else:
            raise ValueError(f"Unknown render mode: {mode}")

    def _render_text(self) -> str:
        """Default text rendering."""
        lines = [
            f"=== {self.env_id} ===",
            f"Phase: {self.state.phase.value}",
            f"Step: {self.state.step}",
            f"Current Player: {self.state.current_player}",
            "",
            "Player States:"
        ]
        for player_id, pstate in self.state.player_states.items():
            lines.append(f"  {player_id}: {pstate}")
        return "\n".join(lines)

    def _render_dict(self) -> Dict:
        """Dict representation for serialization."""
        return {
            "env_id": self.env_id,
            "phase": self.state.phase.value,
            "step": self.state.step,
            "current_player": self.state.current_player,
            "player_states": self.state.player_states,
            "rewards": self.state.rewards,
        }

    def _render_html(self) -> str:
        """HTML rendering for web visualization."""
        # Default: wrap text in pre tag
        return f"<pre>{self._render_text()}</pre>"

    def close(self) -> None:
        """Clean up resources."""
        pass

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(players={self.player_ids})"


class MultiPhaseEnvironment(Environment):
    """
    Base class for games with distinct phases.

    Examples:
    - Poker: Preflop, Flop, Turn, River, Showdown
    - Blackjack: Deal, Play, Showdown
    """

    phases: List[Phase] = [Phase.SETUP, Phase.PLAYING, Phase.TERMINAL]

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._phase_handlers: Dict[Phase, Callable] = {}

    def register_phase_handler(
        self,
        phase: Phase,
        handler: Callable[[GameState], GameState]
    ) -> None:
        """Register a handler for a specific phase."""
        self._phase_handlers[phase] = handler

    def advance_phase(self) -> None:
        """Move to the next phase."""
        if self.state is None:
            return

        current_idx = self.phases.index(self.state.phase)
        if current_idx < len(self.phases) - 1:
            next_phase = self.phases[current_idx + 1]
            self.state.phase = next_phase

            # Run phase handler if registered
            handler = self._phase_handlers.get(next_phase)
            if handler:
                self.state = handler(self.state)
