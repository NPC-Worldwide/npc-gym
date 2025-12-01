"""
Streaming PID Environment for real-time NLP games.

Wraps any PID-style environment to support streaming text input
where information is revealed progressively over time.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple
from enum import Enum
import time

from npc_gym.core.env import Environment, GameState, Observation, Action, Phase, Trace
from npc_gym.core.spaces import DiscreteSpace
from npc_gym.core.info import InfoPartition, InformationStructure
from npc_gym.streaming.processor import (
    TextStream,
    StreamChunk,
    StreamDeck,
    ChunkStrategy,
    DealConfig,
)


class StreamPhase(Enum):
    """Phases of a streaming game."""
    SETUP = "setup"
    STREAMING = "streaming"  # Text is being revealed
    HYPOTHESIS = "hypothesis"  # Players form hypotheses
    VOTING = "voting"  # Players vote on hypotheses
    REVEAL = "reveal"  # Full text revealed
    SCORING = "scoring"
    DONE = "done"


@dataclass
class StreamingConfig:
    """Configuration for streaming PID games."""
    # Streaming settings
    chunk_strategy: ChunkStrategy = ChunkStrategy.SENTENCE
    deal_rate: float = 1.0  # Chunks per second in real-time mode
    real_time: bool = False  # If False, deal immediately on step

    # Game structure
    chunks_per_round: int = 1  # How many chunks dealt per game step
    rounds_before_hypothesis: int = 5  # Deal rounds before hypothesis phase
    hypothesis_rounds: int = 3  # Rounds for hypothesis refinement
    voting_enabled: bool = True

    # Rewards
    correct_hypothesis_reward: float = 100.0
    partial_match_reward: float = 10.0
    voting_accuracy_reward: float = 20.0
    information_bonus: float = 5.0  # Bonus for using less information

    # Deck settings
    max_hand_size: int = 10
    public_reveal_per_round: int = 0  # Chunks revealed to all per round


class StreamingPIDEnv(Environment):
    """
    Streaming Partial Information Decomposition Environment.

    Text is streamed to players as chunks (sentences, words, etc.).
    Each player receives different chunks, like being dealt cards.
    Players must form hypotheses about the complete text or its meaning.

    Supports two modes:
    - Turn-based: Chunks dealt on each step
    - Real-time: Chunks dealt based on wall-clock time

    This environment can work with any text source and any
    ground truth evaluation function.

    Usage:
        env = StreamingPIDEnv(
            text="The quick brown fox jumps over the lazy dog.",
            ground_truth="A fox jumping over a dog",
            config=StreamingConfig(chunk_strategy=ChunkStrategy.WORD)
        )

        obs = env.reset(player_ids=["p1", "p2", "p3"])
        done = False

        while not done:
            actions = {pid: agent.act(obs[pid]) for pid, agent in agents.items()}
            obs, rewards, done, info = env.step(actions)
    """

    env_id = "StreamingPID-v1"
    min_players = 2
    max_players = 8

    def __init__(
        self,
        text: str = None,
        ground_truth: Any = None,
        evaluator: Callable[[str, Any], float] = None,
        config: StreamingConfig = None,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.config = config or StreamingConfig()

        # Source text and truth
        self.source_text = text or ""
        self.ground_truth = ground_truth
        self.evaluator = evaluator or self._default_evaluator

        # Text processing
        self.text_stream = TextStream(strategy=self.config.chunk_strategy)
        self.deck: Optional[StreamDeck] = None

        # Game state
        self.current_phase = StreamPhase.SETUP
        self.hypotheses: Dict[str, List[str]] = {}  # player -> hypothesis history
        self.votes: Dict[str, str] = {}  # player -> voted_hypothesis_owner
        self.scores: Dict[str, float] = {}
        self.round_number: int = 0

        # Real-time state
        self.stream_start_time: float = 0
        self.last_deal_time: float = 0

        # Action spaces depend on phase
        self.action_space = DiscreteSpace(
            choices=["wait", "hypothesize", "vote", "pass"]
        )

    def set_text(self, text: str, ground_truth: Any = None) -> None:
        """Set source text and optional ground truth."""
        self.source_text = text
        if ground_truth is not None:
            self.ground_truth = ground_truth

    def set_evaluator(self, evaluator: Callable[[str, Any], float]) -> None:
        """Set custom hypothesis evaluator function."""
        self.evaluator = evaluator

    def _setup_game(self) -> GameState:
        """Initialize the streaming game."""
        # Create deck from text
        self.deck = StreamDeck(
            player_ids=self.player_ids,
            config=DealConfig(
                deal_rate=self.config.deal_rate,
                max_hand_size=self.config.max_hand_size,
            )
        )

        # Process text into chunks
        num_chunks = self.deck.from_text(
            self.source_text,
            strategy=self.config.chunk_strategy
        )

        # Shuffle for fair distribution
        self.deck.shuffle()

        # Initialize game state
        self.hypotheses = {pid: [] for pid in self.player_ids}
        self.votes = {}
        self.scores = {pid: 0.0 for pid in self.player_ids}
        self.round_number = 0
        self.current_phase = StreamPhase.STREAMING

        self.stream_start_time = time.time()
        self.last_deal_time = 0

        # Initial deal
        self._deal_round()

        state = GameState(
            phase=Phase.PLAYING,
            step=0,
            current_player=self.player_ids[0],
            player_order=self.player_ids.copy(),
            player_states={
                pid: {
                    "hand_size": len(self.deck.get_hand(pid)),
                    "hypothesis_count": 0,
                    "has_voted": False,
                }
                for pid in self.player_ids
            },
            metadata={
                "total_chunks": num_chunks,
                "deck_remaining": self.deck.deck_size(),
                "phase": self.current_phase.value,
            }
        )

        return state

    def _deal_round(self) -> None:
        """Deal chunks for this round."""
        for _ in range(self.config.chunks_per_round):
            self.deck.deal_round()

        # Public reveal
        if self.config.public_reveal_per_round > 0:
            self.deck.deal_to_public(self.config.public_reveal_per_round)

        self.last_deal_time = time.time()

    def _get_observation(self, player_id: str) -> Observation:
        """Get observation for a player."""
        # Private info: their hand chunks
        hand_text = self.deck.get_hand_text(player_id)
        hand_chunks = self.deck.get_hand(player_id)

        # Public info: revealed chunks, other player hand sizes
        public_text = self.deck.get_public_text()
        hand_sizes = self.deck.hand_sizes()

        # Other players' hypotheses (if any shared)
        other_hypotheses = {}
        if self.current_phase in [StreamPhase.VOTING, StreamPhase.REVEAL]:
            for pid, hyp_list in self.hypotheses.items():
                if pid != player_id and hyp_list:
                    other_hypotheses[pid] = hyp_list[-1]  # Latest hypothesis

        info_partition = InfoPartition(
            player_id=player_id,
            private=[
                f"Your information fragments: {hand_text}",
                f"Number of fragments: {len(hand_chunks)}",
            ],
            public=[
                f"Common knowledge: {public_text}" if public_text else "No common info yet",
                f"Round: {self.round_number}",
                f"Phase: {self.current_phase.value}",
            ],
        )

        # Determine valid actions based on phase
        valid_actions = self._get_valid_actions_for_phase(player_id)

        game_state = {
            "phase": self.current_phase.value,
            "round": self.round_number,
            "hand_text": hand_text,
            "hand_chunks": [c.content for c in hand_chunks],
            "public_text": public_text,
            "hand_sizes": hand_sizes,
            "my_hypotheses": self.hypotheses.get(player_id, []),
            "other_hypotheses": other_hypotheses,
            "deck_remaining": self.deck.deck_size(),
            "has_voted": player_id in self.votes,
        }

        return Observation(
            player_id=player_id,
            info_partition=info_partition,
            game_state=game_state,
            valid_actions=valid_actions,
            step=self.state.step,
        )

    def _get_valid_actions(self, player_id: str) -> List[str]:
        """Get valid actions for a player (required by base class)."""
        return self._get_valid_actions_for_phase(player_id)

    def _get_valid_actions_for_phase(self, player_id: str) -> List[str]:
        """Get valid actions for current phase."""
        actions = []

        if self.current_phase == StreamPhase.STREAMING:
            actions = ["wait", "hypothesize"]

        elif self.current_phase == StreamPhase.HYPOTHESIS:
            actions = ["hypothesize", "pass"]

        elif self.current_phase == StreamPhase.VOTING:
            if player_id not in self.votes:
                # Can vote for any other player's hypothesis
                for pid in self.player_ids:
                    if pid != player_id and self.hypotheses.get(pid):
                        actions.append(f"vote:{pid}")
            actions.append("pass")

        elif self.current_phase == StreamPhase.REVEAL:
            actions = ["pass"]

        return actions if actions else ["pass"]

    def _apply_action(self, action: Action) -> None:
        """Process a player action."""
        player_id = action.player_id
        action_type = action.action_type

        if action_type == "wait":
            pass  # Just waiting for more info

        elif action_type == "hypothesize":
            # Player submits hypothesis (from action.value or reasoning)
            hypothesis = action.value or action.reasoning or ""
            if hypothesis:
                self.hypotheses[player_id].append(hypothesis)
                self.state.player_states[player_id]["hypothesis_count"] += 1

        elif action_type.startswith("vote:"):
            # Vote for another player's hypothesis
            voted_for = action_type.split(":", 1)[1]
            if voted_for in self.player_ids and voted_for != player_id:
                self.votes[player_id] = voted_for
                self.state.player_states[player_id]["has_voted"] = True

        elif action_type == "pass":
            pass

        # Update deck remaining in metadata
        self.state.metadata["deck_remaining"] = self.deck.deck_size()

    def step(self, actions: Dict[str, Action]) -> Tuple[Dict[str, Observation], Dict[str, float], bool, Dict[str, Any]]:
        """Execute one step of the game."""
        # Process all actions
        for player_id, action in actions.items():
            self._apply_action(action)

        # Advance round
        self.round_number += 1
        self.state.step = self.round_number

        # Phase transitions
        self._check_phase_transition()

        # Deal more chunks if still streaming
        if self.current_phase == StreamPhase.STREAMING and self.deck.deck_size() > 0:
            self._deal_round()

        # Update state
        self.state.metadata["phase"] = self.current_phase.value

        # Compute rewards
        rewards = self._compute_rewards()

        # Check termination
        done = self._is_terminal()
        if done:
            self.state.phase = Phase.TERMINAL

        # Get observations
        observations = {pid: self._get_observation(pid) for pid in self.player_ids}

        info = {
            "phase": self.current_phase.value,
            "round": self.round_number,
            "deck_remaining": self.deck.deck_size(),
        }

        return observations, rewards, done, info

    def _check_phase_transition(self) -> None:
        """Check and handle phase transitions."""
        if self.current_phase == StreamPhase.STREAMING:
            # Move to hypothesis phase when deck empty or enough rounds
            if self.deck.deck_size() == 0 or self.round_number >= self.config.rounds_before_hypothesis:
                self.current_phase = StreamPhase.HYPOTHESIS

        elif self.current_phase == StreamPhase.HYPOTHESIS:
            # Move to voting after hypothesis rounds
            hypothesis_rounds_done = self.round_number - self.config.rounds_before_hypothesis
            if hypothesis_rounds_done >= self.config.hypothesis_rounds:
                if self.config.voting_enabled:
                    self.current_phase = StreamPhase.VOTING
                else:
                    self.current_phase = StreamPhase.REVEAL

        elif self.current_phase == StreamPhase.VOTING:
            # Move to reveal when all have voted or timeout
            all_voted = all(pid in self.votes for pid in self.player_ids)
            if all_voted:
                self.current_phase = StreamPhase.REVEAL

        elif self.current_phase == StreamPhase.REVEAL:
            # Move to scoring
            self.current_phase = StreamPhase.SCORING

        elif self.current_phase == StreamPhase.SCORING:
            self.current_phase = StreamPhase.DONE

    def _compute_rewards(self) -> Dict[str, float]:
        """Compute rewards for all players."""
        rewards = {pid: 0.0 for pid in self.player_ids}

        if self.current_phase == StreamPhase.SCORING:
            # Evaluate hypotheses
            for pid in self.player_ids:
                if self.hypotheses.get(pid):
                    latest = self.hypotheses[pid][-1]
                    score = self.evaluator(latest, self.ground_truth)
                    rewards[pid] += score * self.config.correct_hypothesis_reward

                    # Information efficiency bonus
                    info_used = len(self.deck.get_hand(pid))
                    total_chunks = len(list(self.text_stream.process(self.source_text)))
                    if total_chunks > 0:
                        efficiency = 1 - (info_used / total_chunks)
                        rewards[pid] += efficiency * self.config.information_bonus

            # Voting accuracy rewards
            if self.config.voting_enabled and self.votes:
                # Find best hypothesis
                best_score = 0
                best_player = None
                for pid in self.player_ids:
                    if self.hypotheses.get(pid):
                        score = self.evaluator(self.hypotheses[pid][-1], self.ground_truth)
                        if score > best_score:
                            best_score = score
                            best_player = pid

                # Reward correct votes
                for voter, voted_for in self.votes.items():
                    if voted_for == best_player:
                        rewards[voter] += self.config.voting_accuracy_reward

            self.scores = rewards.copy()

        return rewards

    def _is_terminal(self) -> bool:
        """Check if game is over."""
        return self.current_phase == StreamPhase.DONE

    def _default_evaluator(self, hypothesis: str, ground_truth: Any) -> float:
        """Default evaluator: simple text similarity."""
        if not hypothesis or not ground_truth:
            return 0.0

        hyp_lower = hypothesis.lower()
        truth_lower = str(ground_truth).lower()

        # Word overlap score
        hyp_words = set(hyp_lower.split())
        truth_words = set(truth_lower.split())

        if not truth_words:
            return 0.0

        overlap = len(hyp_words & truth_words)
        precision = overlap / len(hyp_words) if hyp_words else 0
        recall = overlap / len(truth_words)

        if precision + recall == 0:
            return 0.0

        f1 = 2 * (precision * recall) / (precision + recall)
        return f1

    def _render_text(self) -> str:
        """Render current state."""
        lines = [f"=== Streaming PID (Round {self.round_number}) ==="]
        lines.append(f"Phase: {self.current_phase.value}")
        lines.append(f"Deck remaining: {self.deck.deck_size()}")
        lines.append("")

        lines.append("Public info:")
        lines.append(f"  {self.deck.get_public_text() or '(none)'}")
        lines.append("")

        for pid in self.player_ids:
            lines.append(f"{pid}:")
            lines.append(f"  Hand: {self.deck.get_hand_text(pid)[:100]}...")
            if self.hypotheses.get(pid):
                lines.append(f"  Hypothesis: {self.hypotheses[pid][-1][:50]}...")
            if pid in self.votes:
                lines.append(f"  Voted for: {self.votes[pid]}")

        if self.current_phase == StreamPhase.SCORING:
            lines.append("")
            lines.append("Scores:")
            for pid, score in self.scores.items():
                lines.append(f"  {pid}: {score:.2f}")

        return "\n".join(lines)


def make_streaming_env(
    text: str,
    ground_truth: Any = None,
    strategy: ChunkStrategy = ChunkStrategy.SENTENCE,
    **kwargs
) -> StreamingPIDEnv:
    """Helper to create a streaming PID environment."""
    config = StreamingConfig(chunk_strategy=strategy, **kwargs)
    return StreamingPIDEnv(
        text=text,
        ground_truth=ground_truth,
        config=config,
    )
