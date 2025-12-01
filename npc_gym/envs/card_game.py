"""
CardGame: Base class for card-based game environments.

Abstracts common mechanics:
- Deck management (shuffle, deal, draw)
- Hand management per player
- Betting rounds
- Showdown evaluation
- Information revelation schedules

Can be instantiated directly or subclassed for specific games.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Callable, Tuple
from enum import Enum
import random

from npc_gym.core.env import (
    Environment, MultiPhaseEnvironment, GameState, Observation, Action,
    Phase, StepResult, Trace
)
from npc_gym.core.spaces import (
    Space, DiscreteSpace, ContinuousSpace, CompositeSpace,
    Card, DeckSpace, CardSpace
)
from npc_gym.core.info import InformationStructure, InfoPartition, Visibility


class BettingAction(Enum):
    """Standard betting actions."""
    FOLD = "fold"
    CHECK = "check"
    CALL = "call"
    BET = "bet"
    RAISE = "raise"
    ALL_IN = "all_in"


@dataclass
class CardGameConfig:
    """Configuration for a card game environment."""
    # Deck configuration
    deck_type: str = "standard"  # "standard", "text", "custom"
    source_text: Optional[str] = None  # For text-based decks
    chunk_by: str = "word"  # How to chunk text: "word", "sentence", "char:N"
    custom_cards: Optional[List[Card]] = None

    # Deal configuration
    hole_cards: int = 2  # Cards dealt to each player privately
    community_schedule: List[int] = field(default_factory=lambda: [3, 1, 1])  # Flop, turn, river

    # Betting configuration
    small_blind: int = 25
    big_blind: int = 50
    starting_stack: int = 1000
    min_bet: int = 50
    max_rounds: int = 4  # Preflop, Flop, Turn, River

    # Evaluation
    hand_evaluator: Optional[Callable] = None  # Custom hand evaluation function
    use_hypothesis_scoring: bool = False  # Score based on hypothesis quality

    # Game rules
    allow_check: bool = True
    allow_all_in: bool = True
    max_raises_per_round: int = 4


@dataclass
class PlayerState:
    """State for a single player in the card game."""
    player_id: str
    stack: int
    hand: List[Card] = field(default_factory=list)
    bet_this_round: int = 0
    total_bet: int = 0
    folded: bool = False
    all_in: bool = False
    is_dealer: bool = False
    is_small_blind: bool = False
    is_big_blind: bool = False

    # For hypothesis games
    hypothesis: Optional[str] = None
    confidence: float = 0.5
    initial_fragments: List[str] = field(default_factory=list)

    def can_act(self) -> bool:
        """Check if player can take action."""
        return not self.folded and not self.all_in

    def to_dict(self) -> Dict[str, Any]:
        return {
            "player_id": self.player_id,
            "stack": self.stack,
            "hand_size": len(self.hand),
            "bet_this_round": self.bet_this_round,
            "total_bet": self.total_bet,
            "folded": self.folded,
            "all_in": self.all_in,
        }


class CardGamePhase(Enum):
    """Phases in a card game."""
    SETUP = "setup"
    BLINDS = "blinds"
    DEAL = "deal"
    PREFLOP = "preflop"
    FLOP = "flop"
    TURN = "turn"
    RIVER = "river"
    SHOWDOWN = "showdown"
    TERMINAL = "terminal"


class CardGame(MultiPhaseEnvironment):
    """
    General card game environment.

    Supports:
    - Standard playing cards or text-chunk cards
    - Configurable betting rounds
    - Partial information (hole cards private, community public)
    - Multiple evaluation methods (hand ranking, hypothesis scoring)
    """

    env_id = "CardGame-v1"
    phases = [
        CardGamePhase.SETUP,
        CardGamePhase.BLINDS,
        CardGamePhase.DEAL,
        CardGamePhase.PREFLOP,
        CardGamePhase.FLOP,
        CardGamePhase.TURN,
        CardGamePhase.RIVER,
        CardGamePhase.SHOWDOWN,
        CardGamePhase.TERMINAL,
    ]

    def __init__(
        self,
        config: CardGameConfig = None,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.config = config or CardGameConfig()

        # Create deck
        self.deck = self._create_deck()

        # Player states (initialized on reset)
        self.players: Dict[str, PlayerState] = {}

        # Game state
        self.pot: int = 0
        self.community_cards: List[Card] = []
        self.current_bet: int = 0
        self.raises_this_round: int = 0
        self.dealer_position: int = 0
        self.action_position: int = 0

        # Define spaces
        self.action_space = DiscreteSpace(
            choices=[a.value for a in BettingAction]
        )
        self.observation_space = CompositeSpace({
            "hand": CardSpace(cards=self.deck._initial_cards),
            "stack": ContinuousSpace(0, self.config.starting_stack * 10),
            "pot": ContinuousSpace(0, self.config.starting_stack * len(self.player_ids)),
            "current_bet": ContinuousSpace(0, self.config.starting_stack),
        })

    def _create_deck(self) -> DeckSpace:
        """Create the deck based on configuration."""
        if self.config.deck_type == "standard":
            return DeckSpace(card_type="standard", seed=self.seed)
        elif self.config.deck_type == "text" and self.config.source_text:
            return DeckSpace.from_text(
                self.config.source_text,
                chunk_by=self.config.chunk_by,
                seed=self.seed
            )
        elif self.config.deck_type == "custom" and self.config.custom_cards:
            return DeckSpace(cards=self.config.custom_cards, seed=self.seed)
        else:
            return DeckSpace(card_type="standard", seed=self.seed)

    def _setup_game(self) -> GameState:
        """Initialize game state."""
        # Reset deck
        self.deck.reset()
        self.deck.shuffle()

        # Initialize player states
        self.players = {}
        for i, player_id in enumerate(self.player_ids):
            self.players[player_id] = PlayerState(
                player_id=player_id,
                stack=self.config.starting_stack,
            )

        # Set dealer and blinds
        self.dealer_position = 0
        dealer_id = self.player_ids[self.dealer_position]
        sb_id = self.player_ids[(self.dealer_position + 1) % len(self.player_ids)]
        bb_id = self.player_ids[(self.dealer_position + 2) % len(self.player_ids)]

        self.players[dealer_id].is_dealer = True
        self.players[sb_id].is_small_blind = True
        self.players[bb_id].is_big_blind = True

        # Reset game state
        self.pot = 0
        self.community_cards = []
        self.current_bet = 0
        self.raises_this_round = 0

        # Create information structure
        info_structure = InformationStructure(self.player_ids)

        # Build game state
        state = GameState(
            phase=CardGamePhase.SETUP,
            step=0,
            current_player=None,
            player_order=self.player_ids.copy(),
            player_states={pid: self.players[pid].to_dict() for pid in self.player_ids},
            info_structure=info_structure,
            metadata={
                "max_steps": 100,
                "pot": 0,
                "community_cards": [],
            }
        )

        # Advance through setup phases
        self._post_blinds()
        self._deal_hole_cards()

        state.phase = CardGamePhase.PREFLOP
        state.current_player = self._get_first_actor()

        return state

    def _post_blinds(self) -> None:
        """Post small and big blinds."""
        for player_id, player in self.players.items():
            if player.is_small_blind:
                blind = min(self.config.small_blind, player.stack)
                player.stack -= blind
                player.bet_this_round = blind
                player.total_bet = blind
                self.pot += blind
            elif player.is_big_blind:
                blind = min(self.config.big_blind, player.stack)
                player.stack -= blind
                player.bet_this_round = blind
                player.total_bet = blind
                self.pot += blind
                self.current_bet = blind

    def _deal_hole_cards(self) -> None:
        """Deal hole cards to all players."""
        for player_id in self.player_ids:
            cards = self.deck.deal(self.config.hole_cards)
            self.players[player_id].hand = cards

            # Store initial fragments for hypothesis games
            if self.config.use_hypothesis_scoring:
                self.players[player_id].initial_fragments = [
                    c.value for c in cards if isinstance(c.value, str)
                ]

            # Add to information structure as private
            if self.state and self.state.info_structure:
                for card in cards:
                    self.state.info_structure.add_item(
                        card, Visibility.PRIVATE, owner=player_id
                    )

    def _deal_community(self, count: int) -> List[Card]:
        """Deal community cards."""
        cards = self.deck.deal(count)
        self.community_cards.extend(cards)

        # Add to information structure as public
        if self.state and self.state.info_structure:
            for card in cards:
                self.state.info_structure.add_item(card, Visibility.PUBLIC)

        return cards

    def _get_first_actor(self) -> str:
        """Get the first player to act in current round."""
        # UTG (after big blind) for preflop, dealer+1 for postflop
        if self.state.phase == CardGamePhase.PREFLOP:
            start_pos = (self.dealer_position + 3) % len(self.player_ids)
        else:
            start_pos = (self.dealer_position + 1) % len(self.player_ids)

        # Find first active player
        for i in range(len(self.player_ids)):
            pos = (start_pos + i) % len(self.player_ids)
            player_id = self.player_ids[pos]
            if self.players[player_id].can_act():
                return player_id

        return self.player_ids[0]

    def _get_observation(self, player_id: str) -> Observation:
        """Get observation for a specific player."""
        player = self.players[player_id]

        # Get info partition
        info_partition = InfoPartition(
            player_id=player_id,
            private=[str(c) for c in player.hand],
            public=[str(c) for c in self.community_cards],
        )

        # Build game state view
        game_state = {
            "pot": self.pot,
            "current_bet": self.current_bet,
            "your_stack": player.stack,
            "your_bet": player.bet_this_round,
            "to_call": self.current_bet - player.bet_this_round,
            "phase": self.state.phase.value if self.state else "unknown",
            "players": {
                pid: {
                    "stack": p.stack,
                    "bet": p.bet_this_round,
                    "folded": p.folded,
                    "all_in": p.all_in,
                }
                for pid, p in self.players.items()
                if pid != player_id
            }
        }

        # Add hypothesis if in hypothesis mode
        if self.config.use_hypothesis_scoring and player.hypothesis:
            game_state["your_hypothesis"] = player.hypothesis
            game_state["your_confidence"] = player.confidence

        return Observation(
            player_id=player_id,
            info_partition=info_partition,
            game_state=game_state,
            valid_actions=self._get_valid_actions(player_id),
            action_history=self.state.action_history if self.state else [],
            step=self.state.step if self.state else 0,
        )

    def _get_valid_actions(self, player_id: str) -> List[str]:
        """Get valid actions for a player."""
        player = self.players[player_id]

        if not player.can_act():
            return []

        actions = [BettingAction.FOLD.value]

        to_call = self.current_bet - player.bet_this_round

        if to_call == 0:
            if self.config.allow_check:
                actions.append(BettingAction.CHECK.value)
            actions.append(BettingAction.BET.value)
        else:
            if to_call <= player.stack:
                actions.append(BettingAction.CALL.value)
            if self.raises_this_round < self.config.max_raises_per_round:
                if player.stack > to_call:
                    actions.append(BettingAction.RAISE.value)

        if self.config.allow_all_in and player.stack > 0:
            actions.append(BettingAction.ALL_IN.value)

        return actions

    def _apply_action(self, action: Action) -> None:
        """Apply a player action."""
        player_id = action.player_id
        player = self.players[player_id]
        action_type = action.action_type.lower()

        if action_type == BettingAction.FOLD.value:
            player.folded = True

        elif action_type == BettingAction.CHECK.value:
            pass  # No change

        elif action_type == BettingAction.CALL.value:
            to_call = min(self.current_bet - player.bet_this_round, player.stack)
            player.stack -= to_call
            player.bet_this_round += to_call
            player.total_bet += to_call
            self.pot += to_call
            if player.stack == 0:
                player.all_in = True

        elif action_type == BettingAction.BET.value:
            amount = action.value or self.config.min_bet
            amount = min(max(amount, self.config.min_bet), player.stack)
            player.stack -= amount
            player.bet_this_round += amount
            player.total_bet += amount
            self.pot += amount
            self.current_bet = player.bet_this_round
            self.raises_this_round += 1
            if player.stack == 0:
                player.all_in = True

        elif action_type == BettingAction.RAISE.value:
            to_call = self.current_bet - player.bet_this_round
            raise_amount = action.value or self.config.min_bet
            total = min(to_call + raise_amount, player.stack)
            player.stack -= total
            player.bet_this_round += total
            player.total_bet += total
            self.pot += total
            self.current_bet = player.bet_this_round
            self.raises_this_round += 1
            if player.stack == 0:
                player.all_in = True

        elif action_type == BettingAction.ALL_IN.value:
            amount = player.stack
            player.stack = 0
            player.bet_this_round += amount
            player.total_bet += amount
            self.pot += amount
            player.all_in = True
            if player.bet_this_round > self.current_bet:
                self.current_bet = player.bet_this_round
                self.raises_this_round += 1

        # Store hypothesis/confidence if provided
        if action.reasoning:
            player.hypothesis = action.reasoning
        if action.confidence is not None:
            player.confidence = action.confidence

        # Advance to next player or phase
        self._advance_action()

    def _advance_action(self) -> None:
        """Advance to next player or next phase."""
        active_players = [p for p in self.players.values() if p.can_act()]

        if len(active_players) <= 1:
            # Only one player left (others folded)
            self._go_to_showdown()
            return

        # Check if betting round is complete
        all_matched = all(
            p.bet_this_round == self.current_bet or p.all_in
            for p in self.players.values()
            if not p.folded
        )

        if all_matched and self.state.step > 0:
            self._advance_phase()
        else:
            # Next player
            self._next_player()

    def _next_player(self) -> None:
        """Move to next active player."""
        current_idx = self.player_ids.index(self.state.current_player)

        for i in range(1, len(self.player_ids) + 1):
            next_idx = (current_idx + i) % len(self.player_ids)
            next_player = self.player_ids[next_idx]
            if self.players[next_player].can_act():
                self.state.current_player = next_player
                return

    def _advance_phase(self) -> None:
        """Move to next game phase."""
        # Reset for new betting round
        for player in self.players.values():
            player.bet_this_round = 0
        self.current_bet = 0
        self.raises_this_round = 0

        current_phase = self.state.phase

        if current_phase == CardGamePhase.PREFLOP:
            self.state.phase = CardGamePhase.FLOP
            if len(self.config.community_schedule) > 0:
                self._deal_community(self.config.community_schedule[0])
        elif current_phase == CardGamePhase.FLOP:
            self.state.phase = CardGamePhase.TURN
            if len(self.config.community_schedule) > 1:
                self._deal_community(self.config.community_schedule[1])
        elif current_phase == CardGamePhase.TURN:
            self.state.phase = CardGamePhase.RIVER
            if len(self.config.community_schedule) > 2:
                self._deal_community(self.config.community_schedule[2])
        elif current_phase == CardGamePhase.RIVER:
            self._go_to_showdown()
            return

        # Set first actor for new round
        self.state.current_player = self._get_first_actor()

    def _go_to_showdown(self) -> None:
        """Move to showdown phase."""
        self.state.phase = CardGamePhase.SHOWDOWN

    def _compute_rewards(self) -> Dict[str, float]:
        """Compute rewards for all players."""
        rewards = {pid: 0.0 for pid in self.player_ids}

        if self.state.phase != CardGamePhase.SHOWDOWN:
            return rewards

        # Get active players
        active = [pid for pid, p in self.players.items() if not p.folded]

        if len(active) == 1:
            # Last player standing wins
            winner = active[0]
            rewards[winner] = self.pot - self.players[winner].total_bet
            for pid in self.player_ids:
                if pid != winner:
                    rewards[pid] = -self.players[pid].total_bet
        else:
            # Evaluate hands and determine winner
            if self.config.use_hypothesis_scoring:
                winner, scores = self._evaluate_hypotheses(active)
            else:
                winner, scores = self._evaluate_hands(active)

            if winner:
                rewards[winner] = self.pot - self.players[winner].total_bet
                for pid in self.player_ids:
                    if pid != winner:
                        rewards[pid] = -self.players[pid].total_bet

            # Store scores in trace
            if self.current_trace:
                self.current_trace.metadata["hand_scores"] = scores

        # Mark terminal
        self.state.phase = CardGamePhase.TERMINAL

        return rewards

    def _evaluate_hands(self, player_ids: List[str]) -> Tuple[Optional[str], Dict[str, float]]:
        """Evaluate hands using standard poker ranking or custom evaluator."""
        scores = {}

        for pid in player_ids:
            player = self.players[pid]
            hand = player.hand + self.community_cards

            if self.config.hand_evaluator:
                scores[pid] = self.config.hand_evaluator(hand)
            else:
                # Simple evaluation: sum of card values
                score = sum(self._card_value(c) for c in hand)
                scores[pid] = score

        if scores:
            winner = max(scores, key=scores.get)
            return winner, scores

        return None, scores

    def _card_value(self, card: Card) -> int:
        """Get numeric value of a card."""
        if card.suit:  # Standard playing card
            rank_values = {
                "2": 2, "3": 3, "4": 4, "5": 5, "6": 6, "7": 7, "8": 8,
                "9": 9, "10": 10, "J": 11, "Q": 12, "K": 13, "A": 14
            }
            return rank_values.get(card.value, 0)
        else:
            # Text card - value by length or other metric
            return len(str(card.value))

    def _evaluate_hypotheses(
        self,
        player_ids: List[str]
    ) -> Tuple[Optional[str], Dict[str, float]]:
        """
        Evaluate hypotheses for hypothesis-based games.

        This is where PID comes in - we evaluate how well players
        inferred the underlying truth from partial information.
        """
        scores = {}

        # Get ground truth (full text or correct answer)
        ground_truth = self.config.source_text or ""

        for pid in player_ids:
            player = self.players[pid]

            if player.hypothesis:
                # Score based on hypothesis quality
                # This could use LLM-based evaluation like in holdem.py
                score = self._score_hypothesis(
                    player.hypothesis,
                    ground_truth,
                    player.initial_fragments
                )
                scores[pid] = score * player.confidence
            else:
                scores[pid] = 0.0

        if scores:
            winner = max(scores, key=scores.get)
            return winner, scores

        return None, scores

    def _score_hypothesis(
        self,
        hypothesis: str,
        ground_truth: str,
        initial_fragments: List[str]
    ) -> float:
        """
        Score a hypothesis against ground truth.

        Override this for custom evaluation logic.
        """
        # Simple word overlap scoring
        hyp_words = set(hypothesis.lower().split())
        truth_words = set(ground_truth.lower().split())

        if not truth_words:
            return 0.0

        overlap = len(hyp_words & truth_words)
        precision = overlap / len(hyp_words) if hyp_words else 0
        recall = overlap / len(truth_words)

        # F1 score
        if precision + recall > 0:
            return 2 * precision * recall / (precision + recall)
        return 0.0

    def _is_terminal(self) -> bool:
        """Check if game is over."""
        return self.state.phase == CardGamePhase.TERMINAL

    def _render_text(self) -> str:
        """Render game state as text."""
        lines = [
            f"=== {self.env_id} ===",
            f"Phase: {self.state.phase.value}",
            f"Pot: ${self.pot}",
            f"Current Bet: ${self.current_bet}",
            f"Community: {' '.join(str(c) for c in self.community_cards)}",
            "",
            "Players:"
        ]

        for pid, player in self.players.items():
            status = "FOLD" if player.folded else ("ALL-IN" if player.all_in else "ACTIVE")
            marker = "<--" if pid == self.state.current_player else ""
            lines.append(
                f"  {pid}: ${player.stack} (bet: ${player.bet_this_round}) [{status}] {marker}"
            )

        return "\n".join(lines)
