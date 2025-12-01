"""
Training module for npc-gym.

Provides:
- TrainingLoop: Main orchestrator for episodes -> traces -> training
- TraceCollector: Collects and manages game traces
- ModelEvolver: Genetic evolution of model genomes
- Integrations with npcpy fine-tuning (SFT, DPO, etc.)
"""

from npc_gym.training.loop import TrainingLoop, TrainingConfig
from npc_gym.training.traces import TraceCollector, TraceBuffer
from npc_gym.training.evolution import ModelEvolver, GenePool

__all__ = [
    "TrainingLoop",
    "TrainingConfig",
    "TraceCollector",
    "TraceBuffer",
    "ModelEvolver",
    "GenePool",
]
