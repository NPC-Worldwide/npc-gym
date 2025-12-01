"""
Voter models for PID proposal evaluation.

Voters evaluate and rank proposals from Proposers. They can use
various strategies from simple voting to learned preference models.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple
from enum import Enum
import random


class VotingStrategy(Enum):
    """Voting strategies for proposal selection."""
    MAJORITY = "majority"  # Most votes wins
    WEIGHTED = "weighted"  # Votes weighted by voter confidence
    RANKED_CHOICE = "ranked_choice"  # Ranked choice voting
    APPROVAL = "approval"  # Approve multiple proposals
    CONFIDENCE = "confidence"  # Use proposal confidence directly
    LEARNED = "learned"  # ML model for ranking


@dataclass
class Vote:
    """A vote on proposals."""
    voter_id: str
    rankings: List[int]  # Proposal indices in preference order
    scores: List[float]  # Score for each proposal
    confidence: float  # Voter's confidence in their vote
    reasoning: str = ""


@dataclass
class VoterConfig:
    """Configuration for a Voter model."""
    name: str = "voter"
    model: str = "llama3.2"
    provider: str = "ollama"

    strategy: VotingStrategy = VotingStrategy.WEIGHTED
    num_top_k: int = 3  # How many proposals to consider
    use_reasoning: bool = True  # Include reasoning in votes


class Voter:
    """
    Evaluates and ranks proposals from Proposers.

    Can operate independently or as part of a voting ensemble.
    Voters are trained to identify good proposals and calibrate
    their confidence scores.

    Usage:
        voter = Voter(config=VoterConfig())

        # Vote on proposals
        vote = voter.vote(proposals, context)

        # Get ranked proposals
        ranked = voter.rank(proposals, context)
    """

    def __init__(
        self,
        config: VoterConfig = None,
        npc: Any = None,
    ):
        self.config = config or VoterConfig()
        self.npc = npc
        self.id = self.config.name

        # Track performance
        self.vote_history: List[Tuple[Vote, int]] = []  # (vote, actual_best_idx)
        self.accuracy: float = 0.0

    @classmethod
    def from_npc(cls, npc: Any) -> "Voter":
        """Create Voter from npcpy NPC."""
        config = VoterConfig(
            name=npc.name,
            model=getattr(npc, 'model', 'llama3.2'),
            provider=getattr(npc, 'provider', 'ollama'),
        )
        return cls(config=config, npc=npc)

    def vote(
        self,
        proposals: List[Any],  # List of Proposal objects
        context: str = "",
        info_fragments: List[str] = None,
    ) -> Vote:
        """
        Vote on a list of proposals.

        Args:
            proposals: List of Proposal objects to evaluate
            context: Optional context for evaluation
            info_fragments: Optional info fragments for reference

        Returns:
            Vote object with rankings and scores
        """
        if not proposals:
            return Vote(
                voter_id=self.id,
                rankings=[],
                scores=[],
                confidence=0.0,
            )

        # Build evaluation prompt
        prompt = self._build_vote_prompt(proposals, context, info_fragments)

        # Get vote from LLM or heuristic
        if self.npc:
            vote = self._vote_with_llm(prompt, proposals)
        else:
            vote = self._vote_heuristic(proposals)

        return vote

    def rank(
        self,
        proposals: List[Any],
        context: str = "",
    ) -> List[Tuple[Any, float]]:
        """
        Rank proposals by quality.

        Returns:
            List of (proposal, score) tuples, sorted by score descending
        """
        vote = self.vote(proposals, context)

        ranked = []
        for idx, score in enumerate(vote.scores):
            if idx < len(proposals):
                ranked.append((proposals[idx], score))

        return sorted(ranked, key=lambda x: x[1], reverse=True)

    def _build_vote_prompt(
        self,
        proposals: List[Any],
        context: str,
        info_fragments: List[str],
    ) -> str:
        """Build prompt for voting."""
        proposals_text = ""
        for i, p in enumerate(proposals):
            content = getattr(p, 'content', str(p))
            conf = getattr(p, 'confidence', 0.5)
            proposals_text += "\n{0}. [{1:.2f}] {2}".format(i + 1, conf, content)

        fragments_text = ""
        if info_fragments:
            fragments_text = "\n".join("- " + f for f in info_fragments)

        info_section = ""
        if fragments_text:
            info_section = "AVAILABLE INFORMATION:\n" + fragments_text + "\n"

        context_section = ""
        if context:
            context_section = "CONTEXT: " + context + "\n"

        prompt = """Evaluate these hypothesis proposals and vote on which is most likely correct.

PROPOSALS:{proposals}

{info}{context}
For each proposal, assess:
1. How well does it fit the available information?
2. How specific and testable is it?
3. How confident should we be?

Output JSON:
{{
    "rankings": [list of proposal numbers in preference order, e.g., [2, 1, 3]],
    "scores": [score 0-1 for each proposal in order they were presented],
    "best": proposal_number,
    "reasoning": "brief explanation of your evaluation"
}}
""".format(proposals=proposals_text, info=info_section, context=context_section)
        return prompt

    def _vote_with_llm(self, prompt: str, proposals: List[Any]) -> Vote:
        """Use LLM to vote."""
        try:
            response = self.npc.get_llm_response(prompt, format='json')
            result = response.get('response', {})

            rankings = result.get('rankings', list(range(1, len(proposals) + 1)))
            # Convert to 0-indexed
            rankings = [r - 1 for r in rankings if isinstance(r, int)]

            scores = result.get('scores', [0.5] * len(proposals))
            scores = [float(s) for s in scores[:len(proposals)]]

            # Pad if needed
            while len(scores) < len(proposals):
                scores.append(0.5)

            return Vote(
                voter_id=self.id,
                rankings=rankings,
                scores=scores,
                confidence=max(scores) if scores else 0.5,
                reasoning=result.get('reasoning', ''),
            )

        except Exception as e:
            return self._vote_heuristic(proposals)

    def _vote_heuristic(self, proposals: List[Any]) -> Vote:
        """Heuristic voting based on proposal confidence."""
        scores = []
        for p in proposals:
            # Use proposal's own confidence as starting point
            conf = getattr(p, 'confidence', 0.5)
            scores.append(conf)

        # Rank by score
        indexed_scores = list(enumerate(scores))
        indexed_scores.sort(key=lambda x: x[1], reverse=True)
        rankings = [idx for idx, _ in indexed_scores]

        return Vote(
            voter_id=self.id,
            rankings=rankings,
            scores=scores,
            confidence=max(scores) if scores else 0.5,
            reasoning="Heuristic vote based on proposal confidence",
        )

    def record_outcome(self, vote: Vote, actual_best_idx: int) -> None:
        """Record actual outcome for training."""
        self.vote_history.append((vote, actual_best_idx))

        # Update accuracy
        correct = 0
        for v, actual in self.vote_history[-100:]:
            if v.rankings and v.rankings[0] == actual:
                correct += 1
        self.accuracy = correct / len(self.vote_history[-100:])


class VotingEnsemble:
    """
    Ensemble of voters for aggregated decision making.

    Combines votes from multiple voters using various aggregation
    strategies to produce more robust rankings.
    """

    def __init__(self, strategy: VotingStrategy = VotingStrategy.WEIGHTED):
        self.voters: Dict[str, Voter] = {}
        self.strategy = strategy
        self.voter_weights: Dict[str, float] = {}

    def add_voter(self, voter: Voter, weight: float = 1.0) -> None:
        """Add a voter to the ensemble."""
        self.voters[voter.id] = voter
        self.voter_weights[voter.id] = weight

    def vote(
        self,
        proposals: List[Any],
        context: str = "",
    ) -> Dict[str, Any]:
        """
        Aggregate votes from all voters.

        Returns:
            Dict with aggregated results
        """
        # Collect votes
        votes = {}
        for voter_id, voter in self.voters.items():
            votes[voter_id] = voter.vote(proposals, context)

        # Aggregate based on strategy
        if self.strategy == VotingStrategy.MAJORITY:
            result = self._aggregate_majority(votes, proposals)
        elif self.strategy == VotingStrategy.WEIGHTED:
            result = self._aggregate_weighted(votes, proposals)
        elif self.strategy == VotingStrategy.RANKED_CHOICE:
            result = self._aggregate_ranked_choice(votes, proposals)
        else:
            result = self._aggregate_weighted(votes, proposals)

        result["individual_votes"] = votes
        return result

    def _aggregate_majority(
        self,
        votes: Dict[str, Vote],
        proposals: List[Any],
    ) -> Dict[str, Any]:
        """Majority voting."""
        first_choice_counts = [0] * len(proposals)

        for vote in votes.values():
            if vote.rankings:
                first_choice_counts[vote.rankings[0]] += 1

        winner_idx = max(range(len(proposals)), key=lambda i: first_choice_counts[i])

        return {
            "winner_idx": winner_idx,
            "winner": proposals[winner_idx] if winner_idx < len(proposals) else None,
            "vote_counts": first_choice_counts,
            "final_scores": [c / len(votes) for c in first_choice_counts],
        }

    def _aggregate_weighted(
        self,
        votes: Dict[str, Vote],
        proposals: List[Any],
    ) -> Dict[str, Any]:
        """Weighted voting by voter confidence and weight."""
        weighted_scores = [0.0] * len(proposals)
        total_weight = 0

        for voter_id, vote in votes.items():
            weight = self.voter_weights.get(voter_id, 1.0) * vote.confidence
            total_weight += weight

            for i, score in enumerate(vote.scores):
                if i < len(weighted_scores):
                    weighted_scores[i] += score * weight

        # Normalize
        if total_weight > 0:
            weighted_scores = [s / total_weight for s in weighted_scores]

        winner_idx = max(range(len(proposals)), key=lambda i: weighted_scores[i])

        return {
            "winner_idx": winner_idx,
            "winner": proposals[winner_idx] if winner_idx < len(proposals) else None,
            "final_scores": weighted_scores,
            "total_weight": total_weight,
        }

    def _aggregate_ranked_choice(
        self,
        votes: Dict[str, Vote],
        proposals: List[Any],
    ) -> Dict[str, Any]:
        """Ranked choice / instant runoff voting."""
        n = len(proposals)
        if n == 0:
            return {"winner_idx": -1, "winner": None, "final_scores": []}

        # Copy rankings for elimination process
        active_votes = {vid: list(v.rankings) for vid, v in votes.items()}
        eliminated = set()
        round_results = []

        while len(eliminated) < n - 1:
            # Count first choices
            counts = [0] * n
            for rankings in active_votes.values():
                for idx in rankings:
                    if idx not in eliminated:
                        counts[idx] += 1
                        break

            round_results.append(counts.copy())

            # Check for majority
            total = sum(counts)
            for i, c in enumerate(counts):
                if c > total / 2:
                    return {
                        "winner_idx": i,
                        "winner": proposals[i],
                        "final_scores": [c / total if total > 0 else 0 for c in counts],
                        "rounds": round_results,
                    }

            # Eliminate lowest
            active_counts = [(i, c) for i, c in enumerate(counts) if i not in eliminated]
            if not active_counts:
                break
            min_idx = min(active_counts, key=lambda x: x[1])[0]
            eliminated.add(min_idx)

        # Return remaining candidate
        remaining = [i for i in range(n) if i not in eliminated]
        winner_idx = remaining[0] if remaining else 0

        return {
            "winner_idx": winner_idx,
            "winner": proposals[winner_idx] if winner_idx < n else None,
            "final_scores": [1.0 if i == winner_idx else 0.0 for i in range(n)],
            "rounds": round_results,
        }

    def update_weights(self, outcomes: Dict[str, float]) -> None:
        """Update voter weights based on accuracy outcomes."""
        for voter_id, accuracy in outcomes.items():
            if voter_id in self.voter_weights:
                current = self.voter_weights[voter_id]
                self.voter_weights[voter_id] = 0.9 * current + 0.1 * accuracy
