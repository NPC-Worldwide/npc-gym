"""
SynthesisTournament: Multi-round debate and synthesis game.

Based on debate.py research - agents debate, judges vote, and ideas
are synthesized weighted by vote margin.

Key mechanics:
- Tournament bracket structure
- Debate rounds between pairs
- Judge panel voting
- Weighted synthesis of winning + losing arguments
- Evolutionary pressure toward better ideas
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Callable
from enum import Enum
import random

from npc_gym.core.env import Environment, GameState, Observation, Action, Phase, Trace
from npc_gym.core.spaces import DiscreteSpace, TextSpace, CompositeSpace
from npc_gym.core.info import InfoPartition


class SynthesisAction(Enum):
    """Actions in synthesis tournament."""
    ARGUE = "argue"      # Make an argument
    REBUT = "rebut"      # Counter opponent's argument
    CONCEDE = "concede"  # Accept opponent's point
    SYNTHESIZE = "synthesize"  # Propose merged position


class SynthesisPhase(Enum):
    """Tournament phases."""
    MATCHMAKING = "matchmaking"
    OPENING = "opening"
    DEBATE = "debate"
    VOTING = "voting"
    SYNTHESIS = "synthesis"
    NEXT_ROUND = "next_round"
    TERMINAL = "terminal"


@dataclass
class SynthesisTournamentConfig:
    """Configuration for synthesis tournament."""
    # Topic
    initial_topic: str = ""

    # Tournament structure
    debate_turns: int = 2  # Turns per player in debate
    num_judges: int = 3

    # Synthesis weights
    winner_weight: float = 0.7  # How much winning argument dominates synthesis
    loser_integration: float = 0.3  # How much losing argument is integrated

    # LLM settings
    judge_model: str = "llama3.2"
    judge_provider: str = "ollama"
    synthesizer_model: str = "llama3.2"
    synthesizer_provider: str = "ollama"


@dataclass
class Debater:
    """A debater in the tournament."""
    player_id: str
    current_argument: str = ""
    argument_history: List[str] = field(default_factory=list)
    wins: int = 0
    eliminated: bool = False


@dataclass
class Match:
    """A single debate match."""
    debater1_id: str
    debater2_id: str
    argument1: str = ""
    argument2: str = ""
    transcript: List[Dict[str, str]] = field(default_factory=list)
    votes: Dict[str, str] = field(default_factory=dict)  # judge_id -> winner_id
    winner_id: Optional[str] = None
    synthesis: str = ""


class SynthesisTournament(Environment):
    """
    Synthesis Tournament - evolutionary idea refinement.

    Structure:
    1. N debaters start with the same topic
    2. Paired into matches (tournament bracket)
    3. Each match: opening statements, debate turns, judge voting
    4. Winner's argument survives, enriched by loser's best points
    5. Continue until one idea remains

    This trains:
    - Argumentation and persuasion
    - Finding common ground
    - Synthesis of opposing views
    - Evaluating argument quality
    """

    env_id = "Synthesis-v1"
    min_players = 2
    max_players = 16

    def __init__(
        self,
        initial_topic: str = None,
        config: SynthesisTournamentConfig = None,
        **kwargs
    ):
        super().__init__(**kwargs)

        self.config = config or SynthesisTournamentConfig()
        if initial_topic:
            self.config.initial_topic = initial_topic

        # Ensure power of 2 players for bracket
        n = len(self.player_ids)
        if n & (n - 1) != 0:
            # Round down to nearest power of 2
            import math
            n = 2 ** int(math.log2(n))
            self.player_ids = self.player_ids[:n]

        # Tournament state
        self.debaters: Dict[str, Debater] = {}
        self.current_match: Optional[Match] = None
        self.matches_this_round: List[Match] = []
        self.completed_matches: List[Match] = []
        self.round_number: int = 1
        self.current_turn: int = 0

        # Judges (generated per match)
        self.judges: List[str] = []

        # Action space
        self.action_space = DiscreteSpace(
            choices=[a.value for a in SynthesisAction]
        )
        self.observation_space = CompositeSpace({
            "topic": TextSpace(max_length=500),
            "your_argument": TextSpace(max_length=1000),
            "opponent_argument": TextSpace(max_length=1000),
        })

    def _setup_game(self) -> GameState:
        """Initialize tournament."""
        # Initialize debaters
        self.debaters = {
            pid: Debater(
                player_id=pid,
                current_argument=self.config.initial_topic,
            )
            for pid in self.player_ids
        }

        # Create first round matches
        self._create_round_matches()

        state = GameState(
            phase=SynthesisPhase.OPENING,
            step=0,
            current_player=self.current_match.debater1_id if self.current_match else None,
            player_order=self.player_ids.copy(),
            player_states={pid: {"eliminated": False, "wins": 0} for pid in self.player_ids},
            metadata={"round": 1, "max_steps": 200}
        )

        return state

    def _create_round_matches(self) -> None:
        """Create matches for current round."""
        active = [d for d in self.debaters.values() if not d.eliminated]
        random.shuffle(active)

        self.matches_this_round = []
        for i in range(0, len(active), 2):
            if i + 1 < len(active):
                match = Match(
                    debater1_id=active[i].player_id,
                    debater2_id=active[i + 1].player_id,
                    argument1=active[i].current_argument,
                    argument2=active[i + 1].current_argument,
                )
                self.matches_this_round.append(match)

        if self.matches_this_round:
            self.current_match = self.matches_this_round[0]
        else:
            self.current_match = None

    def _get_observation(self, player_id: str) -> Observation:
        """Get observation for player."""
        debater = self.debaters.get(player_id)

        if not debater or debater.eliminated:
            return Observation(
                player_id=player_id,
                info_partition=InfoPartition(player_id=player_id),
                game_state={"eliminated": True},
                valid_actions=[],
                step=self.state.step if self.state else 0,
            )

        # Get current match context
        opponent_id = None
        opponent_argument = ""
        if self.current_match:
            if self.current_match.debater1_id == player_id:
                opponent_id = self.current_match.debater2_id
                opponent_argument = self.current_match.argument2
            elif self.current_match.debater2_id == player_id:
                opponent_id = self.current_match.debater1_id
                opponent_argument = self.current_match.argument1

        info_partition = InfoPartition(
            player_id=player_id,
            private=[debater.current_argument],
            public=[self.config.initial_topic] + [m.synthesis for m in self.completed_matches],
        )

        game_state = {
            "topic": self.config.initial_topic,
            "round": self.round_number,
            "your_argument": debater.current_argument,
            "opponent_id": opponent_id,
            "opponent_argument": opponent_argument,
            "transcript": self.current_match.transcript if self.current_match else [],
            "turn": self.current_turn,
            "phase": self.state.phase.value if self.state else "unknown",
        }

        return Observation(
            player_id=player_id,
            info_partition=info_partition,
            game_state=game_state,
            valid_actions=self._get_valid_actions(player_id),
            step=self.state.step if self.state else 0,
        )

    def _get_valid_actions(self, player_id: str) -> List[str]:
        """Get valid actions."""
        debater = self.debaters.get(player_id)

        if not debater or debater.eliminated:
            return []

        if not self.current_match:
            return []

        # Check if it's this player's turn in current match
        is_in_match = (
            player_id == self.current_match.debater1_id or
            player_id == self.current_match.debater2_id
        )

        if not is_in_match:
            return []

        if self.state.phase == SynthesisPhase.OPENING:
            return [SynthesisAction.ARGUE.value]
        elif self.state.phase == SynthesisPhase.DEBATE:
            return [
                SynthesisAction.ARGUE.value,
                SynthesisAction.REBUT.value,
                SynthesisAction.CONCEDE.value,
            ]
        elif self.state.phase == SynthesisPhase.SYNTHESIS:
            return [SynthesisAction.SYNTHESIZE.value]

        return []

    def _apply_action(self, action: Action) -> None:
        """Apply player action."""
        player_id = action.player_id
        debater = self.debaters.get(player_id)

        if not debater or not self.current_match:
            return

        action_type = action.action_type.lower()

        # Record in transcript
        self.current_match.transcript.append({
            "player_id": player_id,
            "action": action_type,
            "content": action.reasoning or "",
            "turn": self.current_turn,
        })

        # Update argument
        if action.reasoning:
            debater.current_argument = action.reasoning
            debater.argument_history.append(action.reasoning)

            # Update match arguments
            if player_id == self.current_match.debater1_id:
                self.current_match.argument1 = action.reasoning
            else:
                self.current_match.argument2 = action.reasoning

        # Advance game
        self._advance_match()

    def _advance_match(self) -> None:
        """Advance the current match."""
        if self.state.phase == SynthesisPhase.OPENING:
            # Switch to other debater or move to debate
            if self.state.current_player == self.current_match.debater1_id:
                self.state.current_player = self.current_match.debater2_id
            else:
                self.state.phase = SynthesisPhase.DEBATE
                self.state.current_player = self.current_match.debater1_id
                self.current_turn = 0

        elif self.state.phase == SynthesisPhase.DEBATE:
            self.current_turn += 1

            # Alternate between debaters
            if self.state.current_player == self.current_match.debater1_id:
                self.state.current_player = self.current_match.debater2_id
            else:
                self.state.current_player = self.current_match.debater1_id

            # Check if debate is over
            if self.current_turn >= self.config.debate_turns * 2:
                self.state.phase = SynthesisPhase.VOTING
                self._run_voting()

    def _run_voting(self) -> None:
        """Run judge voting for current match."""
        if not self.current_match:
            return

        votes = self._get_judge_votes(
            self.current_match.argument1,
            self.current_match.argument2,
            self.current_match.debater1_id,
            self.current_match.debater2_id,
            self.current_match.transcript,
        )

        self.current_match.votes = votes

        # Count votes
        vote_counts = {}
        for winner_id in votes.values():
            vote_counts[winner_id] = vote_counts.get(winner_id, 0) + 1

        # Determine winner
        if vote_counts:
            winner_id = max(vote_counts, key=vote_counts.get)
            loser_id = (
                self.current_match.debater2_id
                if winner_id == self.current_match.debater1_id
                else self.current_match.debater1_id
            )

            self.current_match.winner_id = winner_id
            self.debaters[winner_id].wins += 1
            self.debaters[loser_id].eliminated = True

            # Run synthesis
            self._run_synthesis(winner_id, loser_id, vote_counts)

    def _get_judge_votes(
        self,
        argument1: str,
        argument2: str,
        debater1_id: str,
        debater2_id: str,
        transcript: List[Dict],
    ) -> Dict[str, str]:
        """Get votes from judge panel."""
        votes = {}

        try:
            from npcpy.llm_funcs import get_llm_response

            transcript_text = "\n".join(
                f"{t['player_id']}: {t['content']}"
                for t in transcript
            )

            for i in range(self.config.num_judges):
                prompt = f"""You are Judge #{i+1} evaluating a debate.

TOPIC: {self.config.initial_topic}

{debater1_id}'s ARGUMENT:
{argument1}

{debater2_id}'s ARGUMENT:
{argument2}

DEBATE TRANSCRIPT:
{transcript_text}

Which debater's argument was more persuasive and well-founded?
Respond with ONLY the name of the winner: {debater1_id} or {debater2_id}
"""
                response = get_llm_response(
                    prompt,
                    model=self.config.judge_model,
                    provider=self.config.judge_provider,
                )

                result = response.get('response', '').strip()
                if debater1_id in result:
                    votes[f"judge_{i}"] = debater1_id
                else:
                    votes[f"judge_{i}"] = debater2_id

        except Exception as e:
            print(f"Voting failed: {e}")
            # Random fallback
            for i in range(self.config.num_judges):
                votes[f"judge_{i}"] = random.choice([debater1_id, debater2_id])

        return votes

    def _run_synthesis(
        self,
        winner_id: str,
        loser_id: str,
        vote_counts: Dict[str, int]
    ) -> None:
        """Synthesize winning and losing arguments."""
        if not self.current_match:
            return

        winner = self.debaters[winner_id]
        loser = self.debaters[loser_id]

        winner_votes = vote_counts.get(winner_id, 0)
        loser_votes = vote_counts.get(loser_id, 0)
        total_votes = winner_votes + loser_votes

        try:
            from npcpy.llm_funcs import get_llm_response

            prompt = f"""You are a Master Synthesizer creating a superior argument by merging two competing ideas.

The jury voted {winner_votes} to {loser_votes}.

WINNING ARGUMENT ({winner_id}):
{winner.current_argument}

LOSING ARGUMENT ({loser_id}):
{loser.current_argument}

Create a single, synthesized argument that:
1. Is PRIMARILY based on the winning argument (it won for a reason)
2. INCORPORATES the most valuable insights from the losing argument
3. Is stronger than either original argument alone

The synthesis should be weighted: {winner_votes}/{total_votes} winning, {loser_votes}/{total_votes} losing.

Output ONLY the synthesized argument paragraph:
"""
            response = get_llm_response(
                prompt,
                model=self.config.synthesizer_model,
                provider=self.config.synthesizer_provider,
            )

            synthesis = response.get('response', winner.current_argument).strip()

        except Exception as e:
            print(f"Synthesis failed: {e}")
            synthesis = winner.current_argument

        self.current_match.synthesis = synthesis
        winner.current_argument = synthesis

        # Move to next match or round
        self._advance_tournament()

    def _advance_tournament(self) -> None:
        """Advance to next match or round."""
        if self.current_match:
            self.completed_matches.append(self.current_match)
            self.matches_this_round.remove(self.current_match)

        if self.matches_this_round:
            self.current_match = self.matches_this_round[0]
            self.state.phase = SynthesisPhase.OPENING
            self.state.current_player = self.current_match.debater1_id
            self.current_turn = 0
        else:
            # Check if tournament is over
            active = [d for d in self.debaters.values() if not d.eliminated]
            if len(active) <= 1:
                self.state.phase = SynthesisPhase.TERMINAL
            else:
                # Start next round
                self.round_number += 1
                self._create_round_matches()
                if self.current_match:
                    self.state.phase = SynthesisPhase.OPENING
                    self.state.current_player = self.current_match.debater1_id
                else:
                    self.state.phase = SynthesisPhase.TERMINAL

    def _compute_rewards(self) -> Dict[str, float]:
        """Compute rewards based on tournament performance."""
        rewards = {pid: 0.0 for pid in self.player_ids}

        if self.state.phase != SynthesisPhase.TERMINAL:
            return rewards

        # Reward based on how far each player got
        for pid, debater in self.debaters.items():
            if not debater.eliminated:
                # Winner
                rewards[pid] = 100.0 + debater.wins * 10
            else:
                # Reward based on wins before elimination
                rewards[pid] = debater.wins * 10

        # Store final idea in trace
        if self.current_trace:
            active = [d for d in self.debaters.values() if not d.eliminated]
            if active:
                self.current_trace.winner = active[0].player_id
                self.current_trace.ground_truth = active[0].current_argument
                self.current_trace.metadata["final_synthesis"] = active[0].current_argument
                self.current_trace.metadata["all_syntheses"] = [
                    m.synthesis for m in self.completed_matches
                ]

        return rewards

    def _is_terminal(self) -> bool:
        """Check if tournament is over."""
        return self.state.phase == SynthesisPhase.TERMINAL

    def _render_text(self) -> str:
        """Render tournament state."""
        lines = [
            f"=== Synthesis Tournament Round {self.round_number} ===",
            f"Topic: {self.config.initial_topic[:50]}...",
            f"Phase: {self.state.phase.value}",
            "",
            "Debaters:"
        ]

        for pid, debater in self.debaters.items():
            status = "ELIMINATED" if debater.eliminated else f"ACTIVE (wins: {debater.wins})"
            lines.append(f"  {pid}: {status}")

        if self.current_match:
            lines.append("")
            lines.append(f"Current Match: {self.current_match.debater1_id} vs {self.current_match.debater2_id}")

        return "\n".join(lines)

    def get_final_synthesis(self) -> Optional[str]:
        """Get the final synthesized idea from the tournament."""
        active = [d for d in self.debaters.values() if not d.eliminated]
        if active:
            return active[0].current_argument
        return None
