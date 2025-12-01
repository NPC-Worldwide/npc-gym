"""
PID Optimizer for training sub-models on partial information tasks.

The optimizer manages the training loop for:
1. Proposers - getting better at hypothesis generation
2. Voters - getting better at identifying good proposals
3. Synthesizers - getting better at combining proposals
4. Information routing - deciding what info to share
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple
from enum import Enum
import time
import random


@dataclass
class InfoEfficiencyMetric:
    """Tracks information efficiency of the system."""
    total_info_available: int  # Total fragments available
    info_used: int  # Fragments actually used
    accuracy_achieved: float  # Task accuracy
    time_to_answer: float  # Response latency

    @property
    def efficiency_score(self) -> float:
        """Higher is better - accuracy with minimal information."""
        if self.total_info_available == 0:
            return 0.0

        info_ratio = 1 - (self.info_used / self.total_info_available)
        return self.accuracy_achieved * (0.5 + 0.5 * info_ratio)

    @property
    def bits_per_accuracy(self) -> float:
        """Information efficiency: how much info per accuracy point."""
        if self.accuracy_achieved == 0:
            return float('inf')
        return self.info_used / self.accuracy_achieved


@dataclass
class OptimizationConfig:
    """Configuration for PID optimization."""
    # Training
    num_epochs: int = 100
    batch_size: int = 10
    eval_frequency: int = 10

    # Information constraints
    max_info_per_proposer: int = 5  # Max fragments per proposer
    info_budget_total: int = 20  # Total fragments to distribute

    # Objectives (weights)
    accuracy_weight: float = 1.0
    efficiency_weight: float = 0.5
    calibration_weight: float = 0.3

    # Learning
    learning_rate: float = 0.01
    routing_exploration: float = 0.1  # Exploration rate for info routing


@dataclass
class TrainingTrace:
    """A trace of a PID training episode."""
    info_fragments: List[str]
    info_distribution: Dict[str, List[int]]  # proposer_id -> fragment indices
    proposals: List[Any]
    votes: Dict[str, Any]
    synthesis: Any
    ground_truth: Any
    final_score: float
    efficiency: InfoEfficiencyMetric
    timestamp: float = field(default_factory=time.time)


class PIDOptimizer:
    """
    Optimizes the PID system components together.

    Manages the training loop for proposers, voters, and synthesizers,
    optimizing for both accuracy and information efficiency.

    Usage:
        optimizer = PIDOptimizer(config=OptimizationConfig())

        # Add components
        optimizer.set_proposers(proposer_ensemble)
        optimizer.set_voters(voter_ensemble)
        optimizer.set_synthesizer(synthesizer)

        # Train on dataset
        results = optimizer.train(dataset, evaluator)
    """

    def __init__(self, config: OptimizationConfig = None):
        self.config = config or OptimizationConfig()

        # Components
        self.proposers: Any = None  # ProposerEnsemble
        self.voters: Any = None  # VotingEnsemble
        self.synthesizer: Any = None  # Synthesizer

        # Training state
        self.traces: List[TrainingTrace] = []
        self.epoch: int = 0
        self.best_efficiency: float = 0.0

        # Information routing policy
        self.routing_policy: Dict[str, float] = {}  # proposer_id -> info allocation weight

    def set_proposers(self, proposers: Any) -> None:
        """Set the proposer ensemble."""
        self.proposers = proposers
        # Initialize routing policy
        if hasattr(proposers, 'proposers'):
            for pid in proposers.proposers.keys():
                self.routing_policy[pid] = 1.0 / len(proposers.proposers)

    def set_voters(self, voters: Any) -> None:
        """Set the voter ensemble."""
        self.voters = voters

    def set_synthesizer(self, synthesizer: Any) -> None:
        """Set the synthesizer."""
        self.synthesizer = synthesizer

    def train(
        self,
        dataset: List[Tuple[str, Any]],  # List of (text, ground_truth)
        evaluator: Callable[[str, Any], float],
        callbacks: List[Callable] = None,
    ) -> Dict[str, Any]:
        """
        Train the PID system on a dataset.

        Args:
            dataset: List of (source_text, ground_truth) pairs
            evaluator: Function to evaluate hypothesis against ground truth
            callbacks: Optional callbacks for monitoring

        Returns:
            Training results and metrics
        """
        callbacks = callbacks or []
        results = {
            "epochs": [],
            "best_efficiency": 0.0,
            "final_accuracy": 0.0,
        }

        for epoch in range(self.config.num_epochs):
            self.epoch = epoch
            epoch_traces = []

            # Sample batch
            batch = random.sample(dataset, min(self.config.batch_size, len(dataset)))

            for text, ground_truth in batch:
                trace = self._train_episode(text, ground_truth, evaluator)
                epoch_traces.append(trace)
                self.traces.append(trace)

            # Compute epoch metrics
            epoch_accuracy = sum(t.final_score for t in epoch_traces) / len(epoch_traces)
            epoch_efficiency = sum(t.efficiency.efficiency_score for t in epoch_traces) / len(epoch_traces)

            # Update routing policy
            self._update_routing(epoch_traces)

            # Track best
            if epoch_efficiency > self.best_efficiency:
                self.best_efficiency = epoch_efficiency

            results["epochs"].append({
                "epoch": epoch,
                "accuracy": epoch_accuracy,
                "efficiency": epoch_efficiency,
            })

            # Callbacks
            for callback in callbacks:
                callback(epoch, epoch_accuracy, epoch_efficiency)

            # Evaluation
            if epoch % self.config.eval_frequency == 0:
                eval_results = self._evaluate(dataset[:20], evaluator)
                results["epochs"][-1]["eval"] = eval_results

        results["best_efficiency"] = self.best_efficiency
        results["final_accuracy"] = results["epochs"][-1]["accuracy"]

        return results

    def _train_episode(
        self,
        text: str,
        ground_truth: Any,
        evaluator: Callable,
    ) -> TrainingTrace:
        """Run one training episode."""
        # Chunk text into fragments
        fragments = self._chunk_text(text)

        # Distribute fragments to proposers
        distribution = self._distribute_info(fragments)

        # Get proposals
        proposals = self._collect_proposals(fragments, distribution)

        # Vote on proposals
        votes = {}
        if self.voters:
            votes = self.voters.vote(proposals, context=text[:200])

        # Synthesize
        synthesis = None
        if self.synthesizer:
            synthesis = self.synthesizer.synthesize(proposals, votes, context=text[:200])
            final_content = synthesis.content
        elif proposals:
            best = max(proposals, key=lambda p: getattr(p, 'confidence', 0.5))
            final_content = getattr(best, 'content', str(best))
        else:
            final_content = ""

        # Evaluate
        score = evaluator(final_content, ground_truth)

        # Calculate efficiency
        total_fragments = len(fragments)
        used_fragments = sum(len(idxs) for idxs in distribution.values())

        efficiency = InfoEfficiencyMetric(
            total_info_available=total_fragments,
            info_used=used_fragments,
            accuracy_achieved=score,
            time_to_answer=0,  # Could add timing
        )

        # Record outcomes for component training
        self._record_outcomes(proposals, votes, synthesis, score)

        return TrainingTrace(
            info_fragments=fragments,
            info_distribution=distribution,
            proposals=proposals,
            votes=votes,
            synthesis=synthesis,
            ground_truth=ground_truth,
            final_score=score,
            efficiency=efficiency,
        )

    def _chunk_text(self, text: str) -> List[str]:
        """Chunk text into information fragments."""
        # Simple sentence-based chunking
        import re
        sentences = re.split(r'(?<=[.!?])\s+', text)
        return [s.strip() for s in sentences if s.strip()]

    def _distribute_info(
        self,
        fragments: List[str],
    ) -> Dict[str, List[int]]:
        """Distribute fragments to proposers based on routing policy."""
        if not self.proposers or not hasattr(self.proposers, 'proposers'):
            return {}

        distribution = {pid: [] for pid in self.proposers.proposers.keys()}
        proposer_ids = list(distribution.keys())

        if not proposer_ids:
            return distribution

        # Budget per proposer
        budget = self.config.info_budget_total
        available = list(range(len(fragments)))

        # Shuffle for randomness
        random.shuffle(available)

        # Distribute according to policy
        for frag_idx in available[:budget]:
            # Exploration vs exploitation
            if random.random() < self.config.routing_exploration:
                # Random allocation
                pid = random.choice(proposer_ids)
            else:
                # Policy-based allocation
                weights = [self.routing_policy.get(pid, 1.0) for pid in proposer_ids]
                total = sum(weights)
                if total > 0:
                    r = random.random() * total
                    cumsum = 0
                    for pid, w in zip(proposer_ids, weights):
                        cumsum += w
                        if r <= cumsum:
                            break
                else:
                    pid = random.choice(proposer_ids)

            # Check budget per proposer
            if len(distribution[pid]) < self.config.max_info_per_proposer:
                distribution[pid].append(frag_idx)

        return distribution

    def _collect_proposals(
        self,
        fragments: List[str],
        distribution: Dict[str, List[int]],
    ) -> List[Any]:
        """Collect proposals from all proposers."""
        proposals = []

        if not self.proposers or not hasattr(self.proposers, 'proposers'):
            return proposals

        for pid, frag_indices in distribution.items():
            proposer = self.proposers.proposers.get(pid)
            if not proposer:
                continue

            # Get fragments for this proposer
            proposer_frags = [fragments[i] for i in frag_indices if i < len(fragments)]

            if proposer_frags:
                props = proposer.propose(proposer_frags, num_proposals=1)
                proposals.extend(props)

        return proposals

    def _record_outcomes(
        self,
        proposals: List[Any],
        votes: Dict[str, Any],
        synthesis: Any,
        score: float,
    ) -> None:
        """Record outcomes for component training."""
        # Record for proposers
        if self.proposers and hasattr(self.proposers, 'proposers'):
            for proposal in proposals:
                pid = getattr(proposal, 'proposer_id', None)
                if pid and pid in self.proposers.proposers:
                    self.proposers.proposers[pid].record_outcome(proposal, score)

        # Record for voters
        if self.voters and votes and "winner_idx" in votes:
            winner_idx = votes["winner_idx"]
            for voter_id, voter in self.voters.voters.items():
                if voter_id in votes.get("individual_votes", {}):
                    voter.record_outcome(votes["individual_votes"][voter_id], winner_idx)

        # Record for synthesizer
        if self.synthesizer and synthesis:
            self.synthesizer.record_outcome(synthesis, score)

    def _update_routing(self, traces: List[TrainingTrace]) -> None:
        """Update info routing policy based on traces."""
        if not self.proposers or not hasattr(self.proposers, 'proposers'):
            return

        # Calculate performance by proposer
        proposer_scores: Dict[str, List[float]] = {
            pid: [] for pid in self.proposers.proposers.keys()
        }

        for trace in traces:
            for proposal in trace.proposals:
                pid = getattr(proposal, 'proposer_id', None)
                if pid and pid in proposer_scores:
                    # Score based on proposal confidence vs actual outcome
                    prop_conf = getattr(proposal, 'confidence', 0.5)
                    calibration = 1 - abs(prop_conf - trace.final_score)
                    proposer_scores[pid].append(calibration * trace.final_score)

        # Update routing weights
        for pid, scores in proposer_scores.items():
            if scores:
                avg_score = sum(scores) / len(scores)
                current = self.routing_policy.get(pid, 0.5)
                self.routing_policy[pid] = (
                    (1 - self.config.learning_rate) * current +
                    self.config.learning_rate * avg_score
                )

        # Normalize
        total = sum(self.routing_policy.values())
        if total > 0:
            for pid in self.routing_policy:
                self.routing_policy[pid] /= total

    def _evaluate(
        self,
        dataset: List[Tuple[str, Any]],
        evaluator: Callable,
    ) -> Dict[str, float]:
        """Evaluate current system performance."""
        scores = []
        efficiencies = []

        for text, ground_truth in dataset:
            trace = self._train_episode(text, ground_truth, evaluator)
            scores.append(trace.final_score)
            efficiencies.append(trace.efficiency.efficiency_score)

        return {
            "accuracy": sum(scores) / len(scores) if scores else 0,
            "efficiency": sum(efficiencies) / len(efficiencies) if efficiencies else 0,
            "samples": len(scores),
        }

    def get_routing_stats(self) -> Dict[str, Any]:
        """Get information routing statistics."""
        return {
            "policy": self.routing_policy.copy(),
            "traces_collected": len(self.traces),
            "best_efficiency": self.best_efficiency,
            "current_epoch": self.epoch,
        }

    def export_traces(self) -> List[Dict[str, Any]]:
        """Export traces for analysis."""
        exported = []
        for trace in self.traces:
            exported.append({
                "info_fragments": trace.info_fragments,
                "distribution": trace.info_distribution,
                "num_proposals": len(trace.proposals),
                "final_score": trace.final_score,
                "efficiency_score": trace.efficiency.efficiency_score,
                "info_used": trace.efficiency.info_used,
                "total_info": trace.efficiency.total_info_available,
                "timestamp": trace.timestamp,
            })
        return exported


def create_pid_system(
    model: str = "llama3.2",
    provider: str = "ollama",
    num_proposers: int = 3,
) -> Tuple[Any, Any, Any, PIDOptimizer]:
    """
    Helper to create a complete PID system.

    Returns:
        Tuple of (proposer_ensemble, voter_ensemble, synthesizer, optimizer)
    """
    from npc_gym.pid.proposer import ProposerEnsemble, Proposer, ProposerConfig
    from npc_gym.pid.voter import VotingEnsemble, Voter, VoterConfig
    from npc_gym.pid.synthesizer import Synthesizer, SynthesizerConfig

    # Create proposers
    proposers = ProposerEnsemble()
    domains = ["reasoning", "factual", "analytical"][:num_proposers]
    for i, domain in enumerate(domains):
        config = ProposerConfig(
            name=f"proposer_{i}",
            model=model,
            provider=provider,
            domain=domain,
        )
        proposers.add_proposer(Proposer(config=config))

    # Create voters
    voters = VotingEnsemble()
    for i in range(2):
        config = VoterConfig(
            name=f"voter_{i}",
            model=model,
            provider=provider,
        )
        voters.add_voter(Voter(config=config))

    # Create synthesizer
    synth_config = SynthesizerConfig(
        name="synthesizer",
        model=model,
        provider=provider,
    )
    synthesizer = Synthesizer(config=synth_config)

    # Create optimizer
    optimizer = PIDOptimizer()
    optimizer.set_proposers(proposers)
    optimizer.set_voters(voters)
    optimizer.set_synthesizer(synthesizer)

    return proposers, voters, synthesizer, optimizer
