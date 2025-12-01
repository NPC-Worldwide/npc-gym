"""
Metrics collection and aggregation for npc-gym analytics.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
from collections import defaultdict
import time
import json


@dataclass
class AgentMetrics:
    """Metrics for a single agent."""
    agent_id: str
    wins: int = 0
    losses: int = 0
    draws: int = 0
    total_reward: float = 0.0
    games_played: int = 0

    # Hypothesis quality (for PID games)
    hypothesis_scores: List[float] = field(default_factory=list)
    confidence_calibration: List[Tuple[float, float]] = field(default_factory=list)

    # Response timing
    response_times: List[float] = field(default_factory=list)

    # System 1/2 usage
    system1_uses: int = 0
    system2_uses: int = 0

    @property
    def win_rate(self) -> float:
        total = self.wins + self.losses + self.draws
        return self.wins / total if total > 0 else 0.0

    @property
    def avg_reward(self) -> float:
        return self.total_reward / self.games_played if self.games_played > 0 else 0.0

    @property
    def avg_response_time(self) -> float:
        return sum(self.response_times) / len(self.response_times) if self.response_times else 0.0

    @property
    def system1_ratio(self) -> float:
        total = self.system1_uses + self.system2_uses
        return self.system1_uses / total if total > 0 else 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "agent_id": self.agent_id,
            "wins": self.wins,
            "losses": self.losses,
            "draws": self.draws,
            "win_rate": self.win_rate,
            "total_reward": self.total_reward,
            "avg_reward": self.avg_reward,
            "games_played": self.games_played,
            "avg_hypothesis_score": sum(self.hypothesis_scores) / len(self.hypothesis_scores) if self.hypothesis_scores else 0,
            "avg_response_time": self.avg_response_time,
            "system1_ratio": self.system1_ratio,
        }


@dataclass
class EpochMetrics:
    """Metrics for a training epoch."""
    epoch: int
    timestamp: float = field(default_factory=time.time)

    # Performance
    avg_reward: float = 0.0
    max_reward: float = 0.0
    min_reward: float = 0.0

    # Games
    games_played: int = 0
    avg_game_length: float = 0.0

    # Evolution
    best_fitness: float = 0.0
    avg_fitness: float = 0.0
    num_genes: int = 0

    # Efficiency
    avg_info_efficiency: float = 0.0
    avg_response_time: float = 0.0

    # Custom metrics
    custom: Dict[str, float] = field(default_factory=dict)


@dataclass
class TrainingMetrics:
    """Aggregated training metrics."""
    epochs: List[EpochMetrics] = field(default_factory=list)
    agents: Dict[str, AgentMetrics] = field(default_factory=dict)

    # Global stats
    total_games: int = 0
    total_steps: int = 0
    start_time: float = field(default_factory=time.time)

    def add_epoch(self, metrics: EpochMetrics) -> None:
        self.epochs.append(metrics)

    def get_agent(self, agent_id: str) -> AgentMetrics:
        if agent_id not in self.agents:
            self.agents[agent_id] = AgentMetrics(agent_id=agent_id)
        return self.agents[agent_id]

    def get_metric_series(self, metric_name: str) -> List[float]:
        """Get a metric across all epochs."""
        values = []
        for epoch in self.epochs:
            if hasattr(epoch, metric_name):
                values.append(getattr(epoch, metric_name))
            elif metric_name in epoch.custom:
                values.append(epoch.custom[metric_name])
        return values

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_games": self.total_games,
            "total_steps": self.total_steps,
            "num_epochs": len(self.epochs),
            "runtime_seconds": time.time() - self.start_time,
            "agents": {aid: a.to_dict() for aid, a in self.agents.items()},
            "epochs": [
                {
                    "epoch": e.epoch,
                    "avg_reward": e.avg_reward,
                    "best_fitness": e.best_fitness,
                    "games_played": e.games_played,
                    **e.custom
                }
                for e in self.epochs
            ],
        }


class MetricsCollector:
    """
    Collects and aggregates metrics during training.

    Usage:
        collector = MetricsCollector()

        # During training
        collector.record_game_result(agent_id, reward, win=True)
        collector.record_action(agent_id, system="fast", response_time=0.05)
        collector.end_epoch(epoch_num)

        # Get metrics for plotting
        rewards = collector.get_series("avg_reward")
        dashboard = collector.create_dashboard()
    """

    def __init__(self):
        self.metrics = TrainingMetrics()
        self.current_epoch_data: Dict[str, List] = defaultdict(list)
        self.current_epoch: int = 0

    def reset(self) -> None:
        """Reset all metrics."""
        self.metrics = TrainingMetrics()
        self.current_epoch_data = defaultdict(list)
        self.current_epoch = 0

    def record_game_result(
        self,
        agent_id: str,
        reward: float,
        win: bool = None,
        loss: bool = None,
        draw: bool = None,
        hypothesis_score: float = None,
    ) -> None:
        """Record result of a game for an agent."""
        agent = self.metrics.get_agent(agent_id)
        agent.games_played += 1
        agent.total_reward += reward

        if win:
            agent.wins += 1
        elif loss:
            agent.losses += 1
        elif draw:
            agent.draws += 1

        if hypothesis_score is not None:
            agent.hypothesis_scores.append(hypothesis_score)

        self.current_epoch_data["rewards"].append(reward)
        self.metrics.total_games += 1

    def record_action(
        self,
        agent_id: str,
        system: str = None,  # "fast", "slow", "ensemble"
        response_time: float = None,
        confidence: float = None,
        actual_score: float = None,
    ) -> None:
        """Record an agent action."""
        agent = self.metrics.get_agent(agent_id)

        if system == "fast" or system == "system1":
            agent.system1_uses += 1
        elif system == "slow" or system == "system2":
            agent.system2_uses += 1

        if response_time is not None:
            agent.response_times.append(response_time)
            self.current_epoch_data["response_times"].append(response_time)

        if confidence is not None and actual_score is not None:
            agent.confidence_calibration.append((confidence, actual_score))

        self.metrics.total_steps += 1

    def record_evolution(
        self,
        best_fitness: float,
        avg_fitness: float,
        num_genes: int,
    ) -> None:
        """Record evolution metrics."""
        self.current_epoch_data["best_fitness"].append(best_fitness)
        self.current_epoch_data["avg_fitness"].append(avg_fitness)
        self.current_epoch_data["num_genes"].append(num_genes)

    def record_efficiency(
        self,
        info_efficiency: float,
        info_used: int = None,
        total_info: int = None,
    ) -> None:
        """Record information efficiency metrics."""
        self.current_epoch_data["info_efficiency"].append(info_efficiency)
        if info_used is not None:
            self.current_epoch_data["info_used"].append(info_used)
        if total_info is not None:
            self.current_epoch_data["total_info"].append(total_info)

    def record_custom(self, metric_name: str, value: float) -> None:
        """Record a custom metric."""
        self.current_epoch_data[metric_name].append(value)

    def end_epoch(self, epoch: int = None) -> EpochMetrics:
        """End current epoch and aggregate metrics."""
        if epoch is not None:
            self.current_epoch = epoch
        else:
            self.current_epoch += 1

        data = self.current_epoch_data

        # Aggregate
        rewards = data.get("rewards", [0])
        epoch_metrics = EpochMetrics(
            epoch=self.current_epoch,
            avg_reward=sum(rewards) / len(rewards) if rewards else 0,
            max_reward=max(rewards) if rewards else 0,
            min_reward=min(rewards) if rewards else 0,
            games_played=len(rewards),
            avg_game_length=sum(data.get("game_lengths", [0])) / len(data.get("game_lengths", [1])),
            best_fitness=max(data.get("best_fitness", [0])) if data.get("best_fitness") else 0,
            avg_fitness=sum(data.get("avg_fitness", [0])) / len(data.get("avg_fitness", [1])) if data.get("avg_fitness") else 0,
            num_genes=data.get("num_genes", [0])[-1] if data.get("num_genes") else 0,
            avg_info_efficiency=sum(data.get("info_efficiency", [0])) / len(data.get("info_efficiency", [1])) if data.get("info_efficiency") else 0,
            avg_response_time=sum(data.get("response_times", [0])) / len(data.get("response_times", [1])) if data.get("response_times") else 0,
        )

        # Add custom metrics
        for key, values in data.items():
            if key not in ["rewards", "game_lengths", "best_fitness", "avg_fitness",
                          "num_genes", "info_efficiency", "response_times", "info_used", "total_info"]:
                if values:
                    epoch_metrics.custom[key] = sum(values) / len(values)

        self.metrics.add_epoch(epoch_metrics)

        # Reset epoch data
        self.current_epoch_data = defaultdict(list)

        return epoch_metrics

    def get_series(self, metric_name: str) -> List[float]:
        """Get metric values across all epochs."""
        return self.metrics.get_metric_series(metric_name)

    def get_epochs(self) -> List[int]:
        """Get list of epoch numbers."""
        return [e.epoch for e in self.metrics.epochs]

    def get_agent_metrics(self, agent_id: str) -> AgentMetrics:
        """Get metrics for a specific agent."""
        return self.metrics.get_agent(agent_id)

    def get_all_agent_ids(self) -> List[str]:
        """Get all agent IDs."""
        return list(self.metrics.agents.keys())

    def export_json(self, filepath: str = None) -> str:
        """Export metrics as JSON."""
        data = self.metrics.to_dict()
        json_str = json.dumps(data, indent=2)

        if filepath:
            with open(filepath, 'w') as f:
                f.write(json_str)

        return json_str

    def summary(self) -> str:
        """Get text summary of metrics."""
        m = self.metrics

        lines = ["=== Training Summary ==="]
        lines.append(f"Total games: {m.total_games}")
        lines.append(f"Total steps: {m.total_steps}")
        lines.append(f"Epochs: {len(m.epochs)}")
        lines.append(f"Runtime: {time.time() - m.start_time:.1f}s")
        lines.append("")

        if m.epochs:
            last = m.epochs[-1]
            lines.append("Latest Epoch:")
            lines.append(f"  Avg reward: {last.avg_reward:.3f}")
            lines.append(f"  Best fitness: {last.best_fitness:.3f}")
            lines.append(f"  Info efficiency: {last.avg_info_efficiency:.3f}")
        lines.append("")

        lines.append("Agents:")
        for agent_id, agent in m.agents.items():
            lines.append(f"  {agent_id}:")
            lines.append(f"    Win rate: {agent.win_rate:.1%}")
            lines.append(f"    Avg reward: {agent.avg_reward:.3f}")
            lines.append(f"    System 1 ratio: {agent.system1_ratio:.1%}")

        return "\n".join(lines)
