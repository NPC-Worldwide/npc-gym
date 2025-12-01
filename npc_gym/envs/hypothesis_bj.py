"""
HypothesisBlackjack: Blackjack-style hypothesis formation game.

Based on bj.py/poker.py research - agents form hypotheses from partial
information, trying to be concise (word count = card value) while accurate.

Key mechanics:
- "Bust" if hypothesis exceeds word limit (21 words)
- Score = Correctness / Word Count
- System 1/2 hybrid: fast gut feeling + deliberative override
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
from enum import Enum

from npc_gym.core.env import Environment, GameState, Observation, Action, Phase, Trace
from npc_gym.core.spaces import DiscreteSpace, TextSpace, CompositeSpace, Card
from npc_gym.core.info import TextPIDInfo, InfoPartition


class BJAction(Enum):
    """Blackjack actions."""
    HIT = "hit"      # Request more information
    STAND = "stand"  # Keep current hypothesis
    DOUBLE = "double"  # Double confidence, one more card


class BJPhase(Enum):
    """Game phases."""
    DEAL = "deal"
    PLAY = "play"
    SHOWDOWN = "showdown"
    TERMINAL = "terminal"


@dataclass
class HypothesisBJConfig:
    """Configuration for Hypothesis Blackjack."""
    # Source text
    source_text: str = ""
    chunk_by: str = "sentence"  # Larger chunks than poker

    # Limits
    bust_threshold: int = 21  # Max words in hypothesis
    max_hits: int = 5  # Max times can hit
    ante: int = 50

    # Scoring
    correctness_weight: float = 1.0
    brevity_weight: float = 1.0  # Score = correctness * brevity_weight / word_count

    # System 1 simulation
    use_gut_feeling: bool = True
    gut_hit_bias: float = 0.5  # Probability of gut suggesting "hit"


@dataclass
class BJPlayerState:
    """Player state in Hypothesis Blackjack."""
    player_id: str
    stack: int
    hand: List[str] = field(default_factory=list)  # Info fragments
    hypothesis: str = ""
    word_count: int = 0
    confidence: float = 0.5
    busted: bool = False
    standing: bool = False
    hits_taken: int = 0

    def can_act(self) -> bool:
        return not self.busted and not self.standing


class HypothesisBlackjack(Environment):
    """
    Hypothesis Blackjack - concise inference game.

    Rules:
    - Each player gets initial fragments (like hole cards)
    - Players can "hit" to get more info or "stand" to keep hypothesis
    - If hypothesis exceeds word limit, player "busts"
    - Score = (Correctness / Word Count) * multiplier
    - Best score wins the pot

    This game specifically trains for:
    - Knowing when you have enough information
    - Balancing accuracy vs. brevity
    - System 1/2 decision making (gut vs. deliberation)
    """

    env_id = "HypothesisBJ-v1"
    min_players = 1
    max_players = 8

    def __init__(
        self,
        source_text: str = None,
        config: HypothesisBJConfig = None,
        **kwargs
    ):
        super().__init__(**kwargs)

        self.config = config or HypothesisBJConfig()
        if source_text:
            self.config.source_text = source_text

        self.ground_truth = self.config.source_text

        # Chunk the text into fragments
        self.fragments = self._chunk_text(self.config.source_text)

        # Player states
        self.players: Dict[str, BJPlayerState] = {}
        self.pot: int = 0
        self.deck: List[str] = []  # Remaining fragments

        # Action space
        self.action_space = DiscreteSpace(choices=[a.value for a in BJAction])
        self.observation_space = CompositeSpace({
            "hand": TextSpace(max_length=1000),
            "hypothesis": TextSpace(max_length=500),
            "word_count": DiscreteSpace(n=100),
        })

    def _chunk_text(self, text: str) -> List[str]:
        """Chunk text into fragments."""
        import re

        if self.config.chunk_by == "sentence":
            chunks = [s.strip() for s in re.split(r'[.!?]+', text) if s.strip()]
        elif self.config.chunk_by == "word":
            chunks = text.split()
        elif self.config.chunk_by == "paragraph":
            chunks = [p.strip() for p in text.split('\n\n') if p.strip()]
        else:
            chunks = [text]

        return chunks

    def _setup_game(self) -> GameState:
        """Initialize the game."""
        import random

        # Reset deck
        self.deck = self.fragments.copy()
        random.shuffle(self.deck)

        # Initialize players
        self.players = {}
        self.pot = 0

        for pid in self.player_ids:
            self.players[pid] = BJPlayerState(
                player_id=pid,
                stack=1000,  # Default starting stack
            )
            # Ante up
            self.players[pid].stack -= self.config.ante
            self.pot += self.config.ante

        # Deal initial fragments (2 per player)
        for pid in self.player_ids:
            if len(self.deck) >= 2:
                self.players[pid].hand = [self.deck.pop(), self.deck.pop()]
            elif self.deck:
                self.players[pid].hand = [self.deck.pop()]

        # Create game state
        state = GameState(
            phase=BJPhase.PLAY,
            step=0,
            current_player=self.player_ids[0],
            player_order=self.player_ids.copy(),
            player_states={pid: {"stack": p.stack, "busted": p.busted}
                          for pid, p in self.players.items()},
            metadata={"pot": self.pot, "max_steps": 50}
        )

        return state

    def _get_observation(self, player_id: str) -> Observation:
        """Get observation for player."""
        player = self.players[player_id]

        # Generate gut feeling suggestion
        gut_suggestion = self._get_gut_suggestion(player)

        info_partition = InfoPartition(
            player_id=player_id,
            private=player.hand,
            public=[],  # No public cards in blackjack style
        )

        game_state = {
            "pot": self.pot,
            "your_stack": player.stack,
            "your_hand": player.hand,
            "hand_size": len(player.hand),
            "current_hypothesis": player.hypothesis,
            "word_count": player.word_count,
            "bust_threshold": self.config.bust_threshold,
            "hits_taken": player.hits_taken,
            "max_hits": self.config.max_hits,
            "gut_suggestion": gut_suggestion,
            "busted": player.busted,
            "standing": player.standing,
            "other_players": {
                pid: {"standing": p.standing, "busted": p.busted}
                for pid, p in self.players.items()
                if pid != player_id
            }
        }

        return Observation(
            player_id=player_id,
            info_partition=info_partition,
            game_state=game_state,
            valid_actions=self._get_valid_actions(player_id),
            step=self.state.step if self.state else 0,
        )

    def _get_gut_suggestion(self, player: BJPlayerState) -> str:
        """
        Simulate System 1 "gut feeling" suggestion.

        Based on word count vs bust threshold.
        """
        if not self.config.use_gut_feeling:
            return "none"

        # Simple heuristic: hit probability decreases as word count approaches limit
        remaining_room = self.config.bust_threshold - player.word_count
        hit_probability = max(0, remaining_room / self.config.bust_threshold)

        # Add bias
        hit_probability = (hit_probability + self.config.gut_hit_bias) / 2

        import random
        if random.random() < hit_probability:
            return "hit"
        return "stand"

    def _get_valid_actions(self, player_id: str) -> List[str]:
        """Get valid actions for player."""
        player = self.players[player_id]

        if not player.can_act():
            return []

        actions = [BJAction.STAND.value]

        if player.hits_taken < self.config.max_hits and self.deck:
            actions.append(BJAction.HIT.value)

        return actions

    def _apply_action(self, action: Action) -> None:
        """Apply player action."""
        player_id = action.player_id
        player = self.players[player_id]
        action_type = action.action_type.lower()

        if action_type == BJAction.HIT.value:
            if self.deck:
                new_fragment = self.deck.pop()
                player.hand.append(new_fragment)
                player.hits_taken += 1

        elif action_type == BJAction.STAND.value:
            player.standing = True

        # Update hypothesis from action
        if action.reasoning:
            player.hypothesis = action.reasoning
            player.word_count = len(action.reasoning.split())

            # Check for bust
            if player.word_count > self.config.bust_threshold:
                player.busted = True
                player.standing = True  # Can't act anymore

        if action.confidence is not None:
            player.confidence = action.confidence

        # Advance to next player or phase
        self._advance_game()

    def _advance_game(self) -> None:
        """Advance to next player or showdown."""
        # Find next active player
        current_idx = self.player_ids.index(self.state.current_player)

        for i in range(1, len(self.player_ids) + 1):
            next_idx = (current_idx + i) % len(self.player_ids)
            next_player = self.player_ids[next_idx]

            if self.players[next_player].can_act():
                self.state.current_player = next_player
                return

        # No one can act - go to showdown
        self.state.phase = BJPhase.SHOWDOWN

    def _compute_rewards(self) -> Dict[str, float]:
        """Compute rewards based on hypothesis quality."""
        rewards = {pid: 0.0 for pid in self.player_ids}

        if self.state.phase != BJPhase.SHOWDOWN:
            return rewards

        # Score each non-busted player
        scores = {}
        for pid, player in self.players.items():
            if player.busted:
                scores[pid] = 0.0
                continue

            if not player.hypothesis:
                scores[pid] = 0.0
                continue

            # Evaluate correctness
            correctness = self._evaluate_hypothesis(player.hypothesis)

            # Brevity bonus: fewer words = better
            if player.word_count > 0:
                brevity = self.config.brevity_weight / player.word_count
            else:
                brevity = 0

            scores[pid] = correctness * brevity * 10  # Scale up

        # Winner takes pot
        if scores:
            winner = max(scores, key=scores.get)
            rewards[winner] = self.pot - self.config.ante  # Net profit

            for pid in self.player_ids:
                if pid != winner:
                    rewards[pid] = -self.config.ante  # Lost ante

            # Store in trace
            if self.current_trace:
                self.current_trace.winner = winner
                self.current_trace.ground_truth = self.ground_truth
                self.current_trace.metadata["scores"] = scores
                self.current_trace.metadata["hypotheses"] = {
                    pid: self.players[pid].hypothesis
                    for pid in self.player_ids
                }

        self.state.phase = BJPhase.TERMINAL
        return rewards

    def _evaluate_hypothesis(self, hypothesis: str) -> float:
        """Evaluate hypothesis correctness (0-1)."""
        # Simple word overlap
        hyp_words = set(hypothesis.lower().split())
        truth_words = set(self.ground_truth.lower().split())

        if not truth_words or not hyp_words:
            return 0.0

        intersection = len(hyp_words & truth_words)
        precision = intersection / len(hyp_words)
        recall = intersection / len(truth_words)

        if precision + recall > 0:
            f1 = 2 * precision * recall / (precision + recall)
            return f1

        return 0.0

    def _is_terminal(self) -> bool:
        """Check if game is over."""
        return self.state.phase == BJPhase.TERMINAL

    def _render_text(self) -> str:
        """Render game state."""
        lines = [
            f"=== Hypothesis Blackjack ===",
            f"Pot: ${self.pot}",
            f"Phase: {self.state.phase.value}",
            f"Remaining cards: {len(self.deck)}",
            "",
            "Players:"
        ]

        for pid, player in self.players.items():
            status = "BUST" if player.busted else ("STAND" if player.standing else "ACTIVE")
            marker = "<--" if pid == self.state.current_player else ""
            lines.append(f"  {pid}: {player.word_count} words [{status}] {marker}")
            if player.hypothesis:
                lines.append(f"    Hypothesis: {player.hypothesis[:50]}...")

        return "\n".join(lines)
