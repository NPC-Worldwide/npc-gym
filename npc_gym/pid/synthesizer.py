"""
Synthesizer models for combining PID proposals.

Synthesizers take multiple proposals and combine them into a
single, improved hypothesis. They can use various strategies
from simple averaging to learned fusion models.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple
from enum import Enum


class SynthesisStrategy(Enum):
    """Strategies for combining proposals."""
    BEST_ONLY = "best_only"  # Just use highest-confidence proposal
    WEIGHTED_MERGE = "weighted_merge"  # Merge weighted by confidence
    CONSENSUS = "consensus"  # Find common elements
    LLM_FUSION = "llm_fusion"  # Use LLM to synthesize
    DEBATE = "debate"  # Iterative refinement through debate


@dataclass
class SynthesizerConfig:
    """Configuration for Synthesizer."""
    name: str = "synthesizer"
    model: str = "llama3.2"
    provider: str = "ollama"

    strategy: SynthesisStrategy = SynthesisStrategy.LLM_FUSION
    confidence_threshold: float = 0.3  # Min confidence to include in synthesis
    max_proposals: int = 5  # Max proposals to synthesize
    debate_rounds: int = 3  # For debate strategy


@dataclass
class Synthesis:
    """Result of synthesizing proposals."""
    content: str
    confidence: float
    contributing_proposals: List[str]
    strategy_used: str
    reasoning: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)


class Synthesizer:
    """
    Combines multiple proposals into a unified hypothesis.

    Takes the best elements from different proposals and
    synthesizes them into a more complete and accurate answer.

    Usage:
        synthesizer = Synthesizer(config=SynthesizerConfig())

        # Synthesize proposals
        result = synthesizer.synthesize(proposals, vote_results)
    """

    def __init__(
        self,
        config: SynthesizerConfig = None,
        npc: Any = None,
    ):
        self.config = config or SynthesizerConfig()
        self.npc = npc
        self.id = self.config.name

        # Track performance
        self.synthesis_history: List[Tuple[Synthesis, float]] = []

    @classmethod
    def from_npc(cls, npc: Any) -> "Synthesizer":
        """Create Synthesizer from npcpy NPC."""
        config = SynthesizerConfig(
            name=npc.name,
            model=getattr(npc, 'model', 'llama3.2'),
            provider=getattr(npc, 'provider', 'ollama'),
        )
        return cls(config=config, npc=npc)

    def synthesize(
        self,
        proposals: List[Any],
        vote_results: Dict[str, Any] = None,
        context: str = "",
    ) -> Synthesis:
        """
        Synthesize proposals into unified result.

        Args:
            proposals: List of Proposal objects
            vote_results: Optional voting results for weighting
            context: Optional context

        Returns:
            Synthesis object with combined result
        """
        if not proposals:
            return Synthesis(
                content="",
                confidence=0.0,
                contributing_proposals=[],
                strategy_used=self.config.strategy.value,
            )

        # Filter by confidence threshold
        filtered = [
            p for p in proposals
            if getattr(p, 'confidence', 0.5) >= self.config.confidence_threshold
        ]

        if not filtered:
            filtered = proposals[:1]  # Use at least one

        # Limit to max proposals
        filtered = filtered[:self.config.max_proposals]

        # Apply strategy
        if self.config.strategy == SynthesisStrategy.BEST_ONLY:
            return self._synthesize_best(filtered, vote_results)
        elif self.config.strategy == SynthesisStrategy.WEIGHTED_MERGE:
            return self._synthesize_weighted(filtered, vote_results)
        elif self.config.strategy == SynthesisStrategy.CONSENSUS:
            return self._synthesize_consensus(filtered)
        elif self.config.strategy == SynthesisStrategy.LLM_FUSION:
            return self._synthesize_llm(filtered, context)
        elif self.config.strategy == SynthesisStrategy.DEBATE:
            return self._synthesize_debate(filtered, context)

        return self._synthesize_best(filtered, vote_results)

    def _synthesize_best(
        self,
        proposals: List[Any],
        vote_results: Dict[str, Any],
    ) -> Synthesis:
        """Just return the best proposal."""
        # Use vote results if available
        if vote_results and "winner" in vote_results:
            best = vote_results["winner"]
        else:
            best = max(proposals, key=lambda p: getattr(p, 'confidence', 0.5))

        return Synthesis(
            content=getattr(best, 'content', str(best)),
            confidence=getattr(best, 'confidence', 0.5),
            contributing_proposals=[getattr(best, 'proposer_id', 'unknown')],
            strategy_used="best_only",
        )

    def _synthesize_weighted(
        self,
        proposals: List[Any],
        vote_results: Dict[str, Any],
    ) -> Synthesis:
        """Weighted merge of proposals."""
        # Get weights from vote results or confidence
        if vote_results and "final_scores" in vote_results:
            weights = vote_results["final_scores"]
        else:
            weights = [getattr(p, 'confidence', 0.5) for p in proposals]

        # Normalize weights
        total = sum(weights)
        if total > 0:
            weights = [w / total for w in weights]

        # Build weighted content (simple concatenation with weights)
        content_parts = []
        contributing = []
        total_confidence = 0

        for p, w in zip(proposals, weights):
            if w > 0.1:  # Only include significant contributors
                content = getattr(p, 'content', str(p))
                content_parts.append(f"[{w:.2f}] {content}")
                contributing.append(getattr(p, 'proposer_id', 'unknown'))
                total_confidence += getattr(p, 'confidence', 0.5) * w

        return Synthesis(
            content="\n".join(content_parts),
            confidence=total_confidence,
            contributing_proposals=contributing,
            strategy_used="weighted_merge",
        )

    def _synthesize_consensus(self, proposals: List[Any]) -> Synthesis:
        """Find consensus elements across proposals."""
        # Extract words from all proposals
        all_words = []
        for p in proposals:
            content = getattr(p, 'content', str(p)).lower()
            words = content.split()
            all_words.extend(words)

        # Count word frequency
        word_counts = {}
        for w in all_words:
            word_counts[w] = word_counts.get(w, 0) + 1

        # Find consensus words (appear in majority)
        threshold = len(proposals) / 2
        consensus_words = [w for w, c in word_counts.items() if c >= threshold]

        # Build consensus content
        if consensus_words:
            content = "Consensus elements: " + ", ".join(consensus_words[:20])
        else:
            # Fall back to best
            best = max(proposals, key=lambda p: getattr(p, 'confidence', 0.5))
            content = getattr(best, 'content', str(best))

        avg_confidence = sum(getattr(p, 'confidence', 0.5) for p in proposals) / len(proposals)

        return Synthesis(
            content=content,
            confidence=avg_confidence,
            contributing_proposals=[getattr(p, 'proposer_id', 'unknown') for p in proposals],
            strategy_used="consensus",
        )

    def _synthesize_llm(
        self,
        proposals: List[Any],
        context: str,
    ) -> Synthesis:
        """Use LLM to synthesize proposals."""
        if not self.npc:
            return self._synthesize_best(proposals, None)

        # Build prompt
        proposals_text = ""
        for i, p in enumerate(proposals):
            content = getattr(p, 'content', str(p))
            conf = getattr(p, 'confidence', 0.5)
            proposals_text += f"\n{i + 1}. [{conf:.2f}] {content}"

        prompt = f"""You have multiple hypothesis proposals about the same thing.
Synthesize them into a single, improved hypothesis that:
1. Combines the best elements from each
2. Resolves any contradictions
3. Is more specific and accurate than any single proposal

PROPOSALS:{proposals_text}

{f"CONTEXT: {context}" if context else ""}

Output JSON:
{{
    "synthesis": "your combined hypothesis",
    "confidence": 0.0 to 1.0,
    "reasoning": "how you combined the proposals",
    "key_elements": ["list", "of", "key", "elements", "retained"]
}}
"""

        try:
            response = self.npc.get_llm_response(prompt, format='json')
            result = response.get('response', {})

            return Synthesis(
                content=result.get('synthesis', ''),
                confidence=float(result.get('confidence', 0.5)),
                contributing_proposals=[getattr(p, 'proposer_id', 'unknown') for p in proposals],
                strategy_used="llm_fusion",
                reasoning=result.get('reasoning', ''),
                metadata={"key_elements": result.get('key_elements', [])},
            )

        except Exception:
            return self._synthesize_best(proposals, None)

    def _synthesize_debate(
        self,
        proposals: List[Any],
        context: str,
    ) -> Synthesis:
        """Iterative debate refinement."""
        if not self.npc:
            return self._synthesize_best(proposals, None)

        # Start with best proposal
        current = max(proposals, key=lambda p: getattr(p, 'confidence', 0.5))
        current_content = getattr(current, 'content', str(current))

        contributing = [getattr(current, 'proposer_id', 'unknown')]

        for round_num in range(self.config.debate_rounds):
            # Get critiques from other proposals
            critiques = []
            for p in proposals:
                if getattr(p, 'content', '') != current_content:
                    critiques.append(getattr(p, 'content', str(p)))

            if not critiques:
                break

            # Refine through debate
            prompt = f"""Current hypothesis: {current_content}

Alternative viewpoints:
{chr(10).join(f'- {c}' for c in critiques[:3])}

Considering these alternatives, refine the hypothesis.
Keep what's strong, address valid critiques.

Output JSON:
{{
    "refined": "improved hypothesis",
    "confidence": 0.0 to 1.0,
    "changes": "what was changed and why"
}}
"""

            try:
                response = self.npc.get_llm_response(prompt, format='json')
                result = response.get('response', {})
                current_content = result.get('refined', current_content)
            except Exception:
                break

        return Synthesis(
            content=current_content,
            confidence=getattr(current, 'confidence', 0.5),
            contributing_proposals=contributing,
            strategy_used="debate",
            reasoning=f"Refined through {self.config.debate_rounds} debate rounds",
        )

    def record_outcome(self, synthesis: Synthesis, actual_score: float) -> None:
        """Record outcome for training."""
        self.synthesis_history.append((synthesis, actual_score))
