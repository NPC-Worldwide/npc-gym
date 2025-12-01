"""
InfoPoker: Poker-style partial information game.

Based on the holdem.py research - text is chunked into "cards" and dealt
to players who must form hypotheses about the underlying information.

Key concepts:
- Text fragments as cards
- Hypothesis formation from partial info
- Betting reflects confidence in inference
- Judge panel evaluation at showdown
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Callable
import random

from npc_gym.envs.card_game import CardGame, CardGameConfig, PlayerState, CardGamePhase
from npc_gym.core.spaces import Card, DeckSpace
from npc_gym.core.env import Action, Observation, GameState
from npc_gym.core.info import TextPIDInfo


@dataclass
class InfoPokerConfig(CardGameConfig):
    """Configuration for InfoPoker environment."""
    # Text source
    source_text: str = ""
    chunk_by: str = "word"  # "word", "sentence", "paragraph"

    # Hypothesis settings
    require_hypothesis: bool = True
    hypothesis_max_words: int = 50

    # Evaluation
    num_judges: int = 5
    use_llm_judges: bool = True
    judge_model: str = "llama3.2"
    judge_provider: str = "ollama"

    # Scoring weights
    accuracy_weight: float = 0.6
    brevity_weight: float = 0.2
    confidence_calibration_weight: float = 0.2

    def __post_init__(self):
        self.deck_type = "text"
        self.use_hypothesis_scoring = True


class InfoPoker(CardGame):
    """
    Information Poker - Partial Information Decomposition game.

    Players receive text fragments as "hole cards" and community cards.
    They must infer the underlying question/answer from partial information.
    Betting reflects confidence in their hypothesis.
    """

    env_id = "InfoPoker-v1"

    def __init__(
        self,
        source_text: str = None,
        config: InfoPokerConfig = None,
        **kwargs
    ):
        if config is None:
            config = InfoPokerConfig(source_text=source_text or "")
        elif source_text:
            config.source_text = source_text

        super().__init__(config=config, **kwargs)
        self.info_config = config

        # Store ground truth
        self.ground_truth = config.source_text

        # Track hypotheses
        self.hypotheses: Dict[str, Dict[str, Any]] = {}

    def _setup_game(self) -> GameState:
        """Initialize the game with text-based cards."""
        state = super()._setup_game()

        # Initialize hypothesis tracking
        self.hypotheses = {
            pid: {
                "initial": None,
                "current": None,
                "history": [],
                "confidence": 0.5,
            }
            for pid in self.player_ids
        }

        return state

    def _get_observation(self, player_id: str) -> Observation:
        """Get observation with hypothesis prompt."""
        obs = super()._get_observation(player_id)

        # Add hypothesis context
        player = self.players[player_id]
        obs.game_state["hypothesis_prompt"] = self._get_hypothesis_prompt(player_id)
        obs.game_state["your_fragments"] = player.initial_fragments
        obs.game_state["current_hypothesis"] = self.hypotheses[player_id]["current"]
        obs.game_state["hypothesis_history"] = self.hypotheses[player_id]["history"]

        return obs

    def _get_hypothesis_prompt(self, player_id: str) -> str:
        """Generate prompt for hypothesis formation."""
        player = self.players[player_id]

        prompt = f"""You are playing Information Poker.

YOUR PRIVATE FRAGMENTS (hole cards):
{' | '.join(player.initial_fragments)}

PUBLIC FRAGMENTS (community cards):
{' | '.join(str(c) for c in self.community_cards)}

YOUR TASK:
1. INFER the underlying question or topic from these fragments
2. Formulate a HYPOTHESIS answering that question
3. BET based on your confidence in this hypothesis

Your hypothesis should be:
- Specific and falsifiable
- Concise (max {self.info_config.hypothesis_max_words} words)
- Based on logical inference from the fragments

Current hypothesis: {self.hypotheses[player_id]['current'] or 'None yet'}
"""
        return prompt

    def _apply_action(self, action: Action) -> None:
        """Apply action and track hypothesis."""
        player_id = action.player_id

        # Extract hypothesis from reasoning
        if action.reasoning:
            self.hypotheses[player_id]["current"] = action.reasoning
            self.hypotheses[player_id]["history"].append({
                "phase": self.state.phase.value,
                "hypothesis": action.reasoning,
                "confidence": action.confidence or 0.5,
            })

            if self.hypotheses[player_id]["initial"] is None:
                self.hypotheses[player_id]["initial"] = action.reasoning

        if action.confidence is not None:
            self.hypotheses[player_id]["confidence"] = action.confidence

        # Apply betting action
        super()._apply_action(action)

    def _evaluate_hypotheses(
        self,
        player_ids: List[str]
    ) -> tuple[Optional[str], Dict[str, float]]:
        """
        Evaluate hypotheses using judge panel or simple scoring.
        """
        scores = {}

        for pid in player_ids:
            hypothesis = self.hypotheses[pid]["current"]
            confidence = self.hypotheses[pid]["confidence"]

            if not hypothesis:
                scores[pid] = 0.0
                continue

            if self.info_config.use_llm_judges:
                score = self._llm_judge_evaluation(
                    hypothesis,
                    self.ground_truth,
                    self.players[pid].initial_fragments
                )
            else:
                score = self._simple_evaluation(
                    hypothesis,
                    self.ground_truth
                )

            # Apply confidence calibration
            # Reward accurate confidence, penalize overconfidence on wrong answers
            base_score = score
            if score > 0.5:  # Good hypothesis
                calibration = 1.0 + (confidence - 0.5) * 0.2  # Bonus for high confidence
            else:  # Bad hypothesis
                calibration = 1.0 - (confidence - 0.5) * 0.3  # Penalty for overconfidence

            final_score = base_score * calibration

            # Brevity bonus
            word_count = len(hypothesis.split())
            brevity_bonus = max(0, 1 - word_count / 100) * self.info_config.brevity_weight

            scores[pid] = final_score * self.info_config.accuracy_weight + brevity_bonus

        if scores:
            winner = max(scores, key=scores.get)

            # Store in trace
            if self.current_trace:
                self.current_trace.winner = winner
                self.current_trace.ground_truth = self.ground_truth
                self.current_trace.metadata["hypothesis_scores"] = scores
                self.current_trace.metadata["hypotheses"] = {
                    pid: self.hypotheses[pid]["current"]
                    for pid in player_ids
                }

            return winner, scores

        return None, scores

    def _llm_judge_evaluation(
        self,
        hypothesis: str,
        ground_truth: str,
        initial_fragments: List[str]
    ) -> float:
        """
        Use LLM judges to evaluate hypothesis quality.

        Evaluates:
        1. Inferential soundness - Was the logic reasonable?
        2. Accuracy - Does it match the ground truth?
        3. Insight - Did they understand the core meaning?
        """
        try:
            from npcpy.llm_funcs import get_llm_response

            scores = []

            for i in range(self.info_config.num_judges):
                prompt = f"""You are Judge #{i+1} evaluating an inference task.

FULL SOURCE TEXT (ground truth):
"{ground_truth}"

AGENT'S INITIAL FRAGMENTS (what they saw):
"{' | '.join(initial_fragments)}"

AGENT'S HYPOTHESIS:
"{hypothesis}"

EVALUATION CRITERIA (score 1.0 to 10.0):
1. INFERENTIAL SOUNDNESS: Was the reasoning from fragments to hypothesis logical?
2. SEMANTIC ACCURACY: Does the hypothesis capture the essence of the source?
3. INSIGHT QUALITY: Did they identify the key information despite limited data?

Respond with ONLY a JSON object:
{{"score": 7.5, "reasoning": "brief explanation"}}
"""
                response = get_llm_response(
                    prompt,
                    model=self.info_config.judge_model,
                    provider=self.info_config.judge_provider,
                    format='json'
                )

                result = response.get('response', {})
                if isinstance(result, dict):
                    score = float(result.get('score', 5.0))
                    # Add noise for diversity
                    noise = random.gauss(0, 0.3)
                    score = max(1.0, min(10.0, score + noise))
                    scores.append(score)

            if scores:
                return sum(scores) / len(scores) / 10.0  # Normalize to 0-1

        except Exception as e:
            print(f"LLM judge evaluation failed: {e}")

        # Fallback to simple evaluation
        return self._simple_evaluation(hypothesis, ground_truth)

    def _simple_evaluation(self, hypothesis: str, ground_truth: str) -> float:
        """Simple word overlap evaluation."""
        hyp_words = set(hypothesis.lower().split())
        truth_words = set(ground_truth.lower().split())

        if not truth_words or not hyp_words:
            return 0.0

        # Jaccard similarity
        intersection = len(hyp_words & truth_words)
        union = len(hyp_words | truth_words)

        return intersection / union if union > 0 else 0.0

    def get_preference_pairs(self, min_score_gap: float = 0.2) -> List[Dict]:
        """
        Generate DPO preference pairs from this game.

        Returns pairs of (winner_hypothesis, loser_hypothesis) for training.
        """
        if not self.current_trace:
            return []

        scores = self.current_trace.metadata.get("hypothesis_scores", {})
        hypotheses = self.current_trace.metadata.get("hypotheses", {})

        if len(scores) < 2:
            return []

        pairs = []
        sorted_players = sorted(scores.keys(), key=lambda p: scores[p], reverse=True)

        # Compare each pair
        for i, winner in enumerate(sorted_players[:-1]):
            for loser in sorted_players[i+1:]:
                gap = scores[winner] - scores[loser]
                if gap >= min_score_gap:
                    pairs.append({
                        "prompt": self.ground_truth[:500],  # Truncated source
                        "chosen": hypotheses.get(winner, ""),
                        "rejected": hypotheses.get(loser, ""),
                        "chosen_score": scores[winner],
                        "rejected_score": scores[loser],
                        "reward_gap": gap,
                    })

        return pairs


def load_benchmark_text(
    dataset_name: str,
    sample_idx: int = 0,
    config_name: str = None
) -> str:
    """
    Load text from a HuggingFace benchmark dataset.

    Supports: ai2_arc, hellaswag, etc.
    """
    try:
        from datasets import load_dataset

        ds = load_dataset(dataset_name, name=config_name, split='train')

        if sample_idx >= len(ds):
            sample_idx = 0

        entry = ds[sample_idx]

        if dataset_name == "ai2_arc":
            choices_text = " ".join(entry['choices']['text'])
            return entry['question'] + " " + choices_text
        elif dataset_name == "hellaswag":
            endings_text = " ".join(entry['endings'])
            return entry['ctx'] + " " + endings_text
        else:
            # Generic: concatenate all string fields
            text_parts = []
            for key, value in entry.items():
                if isinstance(value, str):
                    text_parts.append(value)
            return " ".join(text_parts)

    except Exception as e:
        print(f"Failed to load benchmark: {e}")
        return ""
