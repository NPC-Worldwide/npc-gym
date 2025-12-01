"""
Proposer models for PID hypothesis generation.

Proposers are small, specialized models that generate candidate
hypotheses from partial information fragments. They are designed
to be fast and efficient, trained to maximize hypothesis quality
while minimizing information requirements.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple
from enum import Enum
import random
import time


@dataclass
class ProposerConfig:
    """Configuration for a Proposer model."""
    name: str = "proposer"
    model: str = "llama3.2"  # Small model by default
    provider: str = "ollama"

    # Specialization
    domain: str = "general"  # Can specialize: "math", "code", "reasoning", etc.
    max_info_tokens: int = 100  # Max input tokens to use

    # Generation settings
    temperature: float = 0.7
    max_output_tokens: int = 200
    num_proposals: int = 3  # How many proposals to generate

    # Training
    learning_rate: float = 1e-4
    use_confidence: bool = True  # Output confidence scores


@dataclass
class Proposal:
    """A hypothesis proposal from a Proposer."""
    content: str
    confidence: float  # 0-1 confidence score
    proposer_id: str
    info_used: List[str]  # Which fragments were used
    timestamp: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __str__(self) -> str:
        return f"[{self.confidence:.2f}] {self.content}"


class Proposer:
    """
    A small model that generates hypotheses from partial information.

    Designed to be fast and efficient, proposers are trained to:
    1. Make reasonable guesses with limited information
    2. Output calibrated confidence scores
    3. Identify what additional information would help

    Usage:
        proposer = Proposer(config=ProposerConfig(domain="math"))

        # From npcpy NPC
        proposer = Proposer.from_npc(my_npc)

        # Generate proposals
        proposals = proposer.propose(info_fragments)
    """

    def __init__(
        self,
        config: ProposerConfig = None,
        npc: Any = None,  # Optional npcpy NPC
    ):
        self.config = config or ProposerConfig()
        self.npc = npc
        self.id = self.config.name

        # Track performance
        self.proposal_history: List[Tuple[Proposal, float]] = []  # (proposal, actual_score)
        self.calibration_error: float = 0.0

    @classmethod
    def from_npc(cls, npc: Any, domain: str = "general") -> "Proposer":
        """Create Proposer from an npcpy NPC."""
        config = ProposerConfig(
            name=npc.name,
            model=getattr(npc, 'model', 'llama3.2'),
            provider=getattr(npc, 'provider', 'ollama'),
            domain=domain,
        )
        return cls(config=config, npc=npc)

    def propose(
        self,
        info_fragments: List[str],
        context: str = "",
        num_proposals: int = None,
    ) -> List[Proposal]:
        """
        Generate hypothesis proposals from information fragments.

        Args:
            info_fragments: List of partial information pieces
            context: Optional additional context
            num_proposals: Override default number of proposals

        Returns:
            List of Proposal objects
        """
        n = num_proposals or self.config.num_proposals

        # Build prompt
        prompt = self._build_prompt(info_fragments, context)

        # Generate proposals
        proposals = []
        for i in range(n):
            proposal = self._generate_one(prompt, i)
            proposal.info_used = info_fragments.copy()
            proposals.append(proposal)

        return proposals

    def _build_prompt(self, info_fragments: List[str], context: str) -> str:
        """Build prompt for proposal generation."""
        fragments_text = "\n".join(f"- {f}" for f in info_fragments)

        prompt = f"""You have access to partial information about something.
Based ONLY on these fragments, generate a hypothesis about what the complete information might be.

INFORMATION FRAGMENTS:
{fragments_text}

{f"CONTEXT: {context}" if context else ""}

Generate a hypothesis. Be specific but acknowledge uncertainty.
Output JSON:
{{
    "hypothesis": "your best guess based on the fragments",
    "confidence": 0.0 to 1.0 (how confident are you?),
    "reasoning": "brief explanation of your reasoning",
    "missing_info": "what additional info would help?"
}}
"""
        return prompt

    def _generate_one(self, prompt: str, variation: int = 0) -> Proposal:
        """Generate a single proposal."""
        # Add variation
        if variation > 0:
            prompt += f"\n\nThis is alternative hypothesis #{variation + 1}. Consider different interpretations."

        # If we have an NPC, use it
        if self.npc:
            try:
                response = self.npc.get_llm_response(prompt, format='json')
                result = response.get('response', {})

                return Proposal(
                    content=result.get('hypothesis', ''),
                    confidence=float(result.get('confidence', 0.5)),
                    proposer_id=self.id,
                    info_used=[],
                    metadata={
                        'reasoning': result.get('reasoning', ''),
                        'missing_info': result.get('missing_info', ''),
                    }
                )
            except Exception as e:
                # Fallback
                return Proposal(
                    content=f"Error generating proposal: {e}",
                    confidence=0.1,
                    proposer_id=self.id,
                    info_used=[],
                )

        # Without NPC, return placeholder
        return Proposal(
            content="[Placeholder - no LLM available]",
            confidence=0.5,
            proposer_id=self.id,
            info_used=[],
        )

    def record_outcome(self, proposal: Proposal, actual_score: float) -> None:
        """Record the actual outcome for a proposal (for training)."""
        self.proposal_history.append((proposal, actual_score))

        # Update calibration error
        errors = []
        for p, score in self.proposal_history[-100:]:  # Last 100
            errors.append(abs(p.confidence - score))
        self.calibration_error = sum(errors) / len(errors) if errors else 0

    def get_calibration_stats(self) -> Dict[str, float]:
        """Get calibration statistics."""
        if not self.proposal_history:
            return {"calibration_error": 0, "samples": 0}

        confidences = [p.confidence for p, _ in self.proposal_history]
        actuals = [s for _, s in self.proposal_history]

        return {
            "calibration_error": self.calibration_error,
            "avg_confidence": sum(confidences) / len(confidences),
            "avg_actual": sum(actuals) / len(actuals),
            "samples": len(self.proposal_history),
        }


class ProposerEnsemble:
    """
    An ensemble of Proposer models with different specializations.

    Coordinates multiple proposers to generate diverse hypotheses
    and routes information efficiently.

    Usage:
        ensemble = ProposerEnsemble()
        ensemble.add_proposer(math_proposer)
        ensemble.add_proposer(reasoning_proposer)

        # Generate proposals from all
        proposals = ensemble.propose_all(fragments)

        # Or route to best proposer
        proposals = ensemble.propose_routed(fragments)
    """

    def __init__(self):
        self.proposers: Dict[str, Proposer] = {}
        self.routing_scores: Dict[str, Dict[str, float]] = {}  # domain -> proposer -> score

    def add_proposer(self, proposer: Proposer) -> None:
        """Add a proposer to the ensemble."""
        self.proposers[proposer.id] = proposer

    def remove_proposer(self, proposer_id: str) -> None:
        """Remove a proposer from the ensemble."""
        self.proposers.pop(proposer_id, None)

    def propose_all(
        self,
        info_fragments: List[str],
        context: str = "",
    ) -> List[Proposal]:
        """Get proposals from all proposers."""
        all_proposals = []
        for proposer in self.proposers.values():
            proposals = proposer.propose(info_fragments, context)
            all_proposals.extend(proposals)
        return all_proposals

    def propose_routed(
        self,
        info_fragments: List[str],
        context: str = "",
        domain: str = None,
    ) -> List[Proposal]:
        """Route to best proposer(s) for this input."""
        if domain and domain in self.routing_scores:
            # Use routing scores to select best proposers
            scores = self.routing_scores[domain]
            best_id = max(scores, key=scores.get)
            proposer = self.proposers.get(best_id)
            if proposer:
                return proposer.propose(info_fragments, context)

        # Default: use all proposers but take fewer from each
        all_proposals = []
        for proposer in self.proposers.values():
            proposals = proposer.propose(info_fragments, context, num_proposals=1)
            all_proposals.extend(proposals)
        return all_proposals

    def propose_tournament(
        self,
        info_fragments: List[str],
        context: str = "",
        rounds: int = 3,
    ) -> List[Proposal]:
        """
        Tournament-style proposal generation.

        Multiple rounds where proposals compete, with best informing next round.
        """
        all_proposals = []
        current_context = context

        for round_num in range(rounds):
            round_proposals = self.propose_all(info_fragments, current_context)
            all_proposals.extend(round_proposals)

            # Update context with best proposals
            if round_proposals:
                best = max(round_proposals, key=lambda p: p.confidence)
                current_context += f"\nPrevious hypothesis: {best.content}"

        return all_proposals

    def update_routing(self, domain: str, proposer_id: str, score: float) -> None:
        """Update routing scores based on performance."""
        if domain not in self.routing_scores:
            self.routing_scores[domain] = {}

        current = self.routing_scores[domain].get(proposer_id, 0.5)
        # Exponential moving average
        self.routing_scores[domain][proposer_id] = 0.9 * current + 0.1 * score

    def get_best_proposers(self, domain: str = None, n: int = 3) -> List[Proposer]:
        """Get the n best proposers for a domain."""
        if domain and domain in self.routing_scores:
            scores = self.routing_scores[domain]
            sorted_ids = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)
            return [self.proposers[pid] for pid in sorted_ids[:n] if pid in self.proposers]

        # Return by calibration if no domain routing
        sorted_proposers = sorted(
            self.proposers.values(),
            key=lambda p: p.calibration_error
        )
        return sorted_proposers[:n]


def create_proposer_ensemble(
    domains: List[str] = None,
    model: str = "llama3.2",
    provider: str = "ollama",
) -> ProposerEnsemble:
    """Helper to create an ensemble with domain-specialized proposers."""
    ensemble = ProposerEnsemble()

    domains = domains or ["general", "reasoning", "math", "code", "factual"]

    for domain in domains:
        config = ProposerConfig(
            name=f"{domain}_proposer",
            model=model,
            provider=provider,
            domain=domain,
        )
        proposer = Proposer(config=config)
        ensemble.add_proposer(proposer)

    return ensemble
