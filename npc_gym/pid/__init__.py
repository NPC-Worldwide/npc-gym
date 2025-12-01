"""
PID (Partial Information Decomposition) module for npc-gym.

Provides tools for training small sub-models that:
- Generate hypotheses from partial information
- Vote on and synthesize proposals
- Optimize for accuracy with minimal information

Key concepts:
- Proposer: Small model that generates candidate answers from fragments
- Voter: Evaluates and ranks proposals
- Synthesizer: Combines proposals into final answer
- InfoRouter: Decides which fragments to share
"""

from npc_gym.pid.proposer import (
    Proposer,
    ProposerConfig,
    ProposerEnsemble,
)
from npc_gym.pid.voter import (
    Voter,
    VoterConfig,
    VotingStrategy,
)
from npc_gym.pid.synthesizer import (
    Synthesizer,
    SynthesizerConfig,
)
from npc_gym.pid.optimizer import (
    PIDOptimizer,
    OptimizationConfig,
    InfoEfficiencyMetric,
)

__all__ = [
    # Proposers
    "Proposer",
    "ProposerConfig",
    "ProposerEnsemble",
    # Voters
    "Voter",
    "VoterConfig",
    "VotingStrategy",
    # Synthesizers
    "Synthesizer",
    "SynthesizerConfig",
    # Optimization
    "PIDOptimizer",
    "OptimizationConfig",
    "InfoEfficiencyMetric",
]
