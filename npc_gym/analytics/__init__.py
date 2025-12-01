"""
Analytics and visualization for npc-gym.

Provides Plotly-based visualizations for:
- Training progress tracking
- Agent performance analysis
- Information flow visualization
- Model evolution metrics
"""

from npc_gym.analytics.plots import (
    TrainingPlotter,
    PerformanceDashboard,
    InfoFlowVisualizer,
)
from npc_gym.analytics.metrics import (
    MetricsCollector,
    AgentMetrics,
    TrainingMetrics,
)

__all__ = [
    "TrainingPlotter",
    "PerformanceDashboard",
    "InfoFlowVisualizer",
    "MetricsCollector",
    "AgentMetrics",
    "TrainingMetrics",
]
