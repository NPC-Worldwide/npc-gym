"""
Plotly-based visualization for npc-gym analytics.

Provides interactive plots and dashboards for monitoring
training progress, agent performance, and system behavior.
"""

from __future__ import annotations
from typing import Any, Dict, List, Optional, Tuple
import json

# Plotly is optional
try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False


def check_plotly():
    """Check if plotly is available."""
    if not PLOTLY_AVAILABLE:
        raise ImportError(
            "Plotly required for visualizations. Install with: pip install plotly"
        )


class TrainingPlotter:
    """
    Creates training progress plots.

    Usage:
        plotter = TrainingPlotter(metrics_collector)

        # Single metric plot
        fig = plotter.plot_metric("avg_reward")
        fig.show()

        # Multiple metrics
        fig = plotter.plot_multiple(["avg_reward", "best_fitness"])

        # Save to HTML
        plotter.save("training_progress.html")
    """

    def __init__(self, collector: Any = None):
        check_plotly()
        self.collector = collector
        self.figures: Dict[str, go.Figure] = {}

    def plot_metric(
        self,
        metric_name: str,
        title: str = None,
        y_label: str = None,
    ) -> go.Figure:
        """Plot a single metric over epochs."""
        if not self.collector:
            return self._empty_figure(title or metric_name)

        epochs = self.collector.get_epochs()
        values = self.collector.get_series(metric_name)

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=epochs,
            y=values,
            mode='lines+markers',
            name=metric_name,
            line=dict(width=2),
            marker=dict(size=6),
        ))

        fig.update_layout(
            title=title or f"{metric_name} over Training",
            xaxis_title="Epoch",
            yaxis_title=y_label or metric_name,
            template="plotly_dark",
            hovermode="x unified",
        )

        self.figures[metric_name] = fig
        return fig

    def plot_multiple(
        self,
        metrics: List[str],
        title: str = "Training Metrics",
        normalize: bool = False,
    ) -> go.Figure:
        """Plot multiple metrics on the same chart."""
        if not self.collector:
            return self._empty_figure(title)

        epochs = self.collector.get_epochs()
        fig = go.Figure()

        for metric in metrics:
            values = self.collector.get_series(metric)

            if normalize and values:
                max_val = max(abs(v) for v in values) or 1
                values = [v / max_val for v in values]

            fig.add_trace(go.Scatter(
                x=epochs,
                y=values,
                mode='lines+markers',
                name=metric,
                line=dict(width=2),
            ))

        fig.update_layout(
            title=title,
            xaxis_title="Epoch",
            yaxis_title="Value" + (" (normalized)" if normalize else ""),
            template="plotly_dark",
            hovermode="x unified",
            legend=dict(orientation="h", yanchor="bottom", y=1.02),
        )

        return fig

    def plot_agent_comparison(
        self,
        metric: str = "win_rate",
    ) -> go.Figure:
        """Compare agents on a metric."""
        if not self.collector:
            return self._empty_figure("Agent Comparison")

        agent_ids = self.collector.get_all_agent_ids()
        values = []

        for aid in agent_ids:
            agent_metrics = self.collector.get_agent_metrics(aid)
            if hasattr(agent_metrics, metric):
                values.append(getattr(agent_metrics, metric))
            else:
                values.append(0)

        fig = go.Figure(data=[
            go.Bar(
                x=agent_ids,
                y=values,
                marker_color=['#00d9ff', '#00ff88', '#ff6b6b', '#ffd93d'][:len(agent_ids)],
            )
        ])

        fig.update_layout(
            title=f"Agent {metric.replace('_', ' ').title()}",
            xaxis_title="Agent",
            yaxis_title=metric.replace('_', ' ').title(),
            template="plotly_dark",
        )

        return fig

    def plot_reward_distribution(self) -> go.Figure:
        """Plot distribution of rewards."""
        if not self.collector:
            return self._empty_figure("Reward Distribution")

        # Collect all rewards from agents
        all_rewards = []
        for aid in self.collector.get_all_agent_ids():
            agent = self.collector.get_agent_metrics(aid)
            # Approximate from total/games
            if agent.games_played > 0:
                avg = agent.avg_reward
                all_rewards.extend([avg] * min(agent.games_played, 100))

        fig = go.Figure(data=[
            go.Histogram(
                x=all_rewards,
                nbinsx=30,
                marker_color='#00d9ff',
            )
        ])

        fig.update_layout(
            title="Reward Distribution",
            xaxis_title="Reward",
            yaxis_title="Frequency",
            template="plotly_dark",
        )

        return fig

    def plot_system_usage(self) -> go.Figure:
        """Plot System 1 vs System 2 usage by agent."""
        if not self.collector:
            return self._empty_figure("System Usage")

        agent_ids = self.collector.get_all_agent_ids()
        s1_usage = []
        s2_usage = []

        for aid in agent_ids:
            agent = self.collector.get_agent_metrics(aid)
            s1_usage.append(agent.system1_uses)
            s2_usage.append(agent.system2_uses)

        fig = go.Figure(data=[
            go.Bar(name='System 1 (Fast)', x=agent_ids, y=s1_usage, marker_color='#00ff88'),
            go.Bar(name='System 2 (Slow)', x=agent_ids, y=s2_usage, marker_color='#00d9ff'),
        ])

        fig.update_layout(
            title="System 1/2 Usage by Agent",
            xaxis_title="Agent",
            yaxis_title="Number of Uses",
            barmode='stack',
            template="plotly_dark",
            legend=dict(orientation="h", yanchor="bottom", y=1.02),
        )

        return fig

    def _empty_figure(self, title: str) -> go.Figure:
        """Create empty figure with message."""
        fig = go.Figure()
        fig.add_annotation(
            text="No data available",
            xref="paper", yref="paper",
            x=0.5, y=0.5,
            showarrow=False,
            font=dict(size=20, color="#888"),
        )
        fig.update_layout(
            title=title,
            template="plotly_dark",
        )
        return fig

    def save(self, filepath: str, metrics: List[str] = None) -> None:
        """Save plot(s) to HTML file."""
        if metrics:
            fig = self.plot_multiple(metrics)
        elif self.figures:
            fig = list(self.figures.values())[0]
        else:
            fig = self.plot_metric("avg_reward")

        fig.write_html(filepath)


class PerformanceDashboard:
    """
    Multi-panel performance dashboard.

    Usage:
        dashboard = PerformanceDashboard(collector)
        fig = dashboard.create()
        fig.show()
    """

    def __init__(self, collector: Any = None):
        check_plotly()
        self.collector = collector

    def create(
        self,
        title: str = "npc-gym Training Dashboard",
    ) -> go.Figure:
        """Create the full dashboard."""
        # 2x3 grid
        fig = make_subplots(
            rows=2, cols=3,
            subplot_titles=(
                "Reward Over Time",
                "Best Fitness",
                "Agent Win Rates",
                "System 1/2 Usage",
                "Response Times",
                "Information Efficiency",
            ),
            specs=[
                [{"type": "scatter"}, {"type": "scatter"}, {"type": "bar"}],
                [{"type": "bar"}, {"type": "histogram"}, {"type": "scatter"}],
            ],
        )

        if self.collector:
            epochs = self.collector.get_epochs() or [0]

            # 1. Reward over time
            rewards = self.collector.get_series("avg_reward") or [0]
            fig.add_trace(
                go.Scatter(x=epochs, y=rewards, mode='lines+markers',
                          name='Avg Reward', line=dict(color='#00d9ff')),
                row=1, col=1
            )

            # 2. Best fitness
            fitness = self.collector.get_series("best_fitness") or [0]
            fig.add_trace(
                go.Scatter(x=epochs, y=fitness, mode='lines+markers',
                          name='Best Fitness', line=dict(color='#00ff88')),
                row=1, col=2
            )

            # 3. Agent win rates
            agent_ids = self.collector.get_all_agent_ids()
            win_rates = [self.collector.get_agent_metrics(a).win_rate for a in agent_ids]
            fig.add_trace(
                go.Bar(x=agent_ids, y=win_rates, name='Win Rate',
                      marker_color='#ff6b6b'),
                row=1, col=3
            )

            # 4. System usage
            s1 = [self.collector.get_agent_metrics(a).system1_uses for a in agent_ids]
            s2 = [self.collector.get_agent_metrics(a).system2_uses for a in agent_ids]
            fig.add_trace(
                go.Bar(x=agent_ids, y=s1, name='System 1', marker_color='#00ff88'),
                row=2, col=1
            )
            fig.add_trace(
                go.Bar(x=agent_ids, y=s2, name='System 2', marker_color='#00d9ff'),
                row=2, col=1
            )

            # 5. Response times
            all_times = []
            for a in agent_ids:
                all_times.extend(self.collector.get_agent_metrics(a).response_times)
            if all_times:
                fig.add_trace(
                    go.Histogram(x=all_times, nbinsx=20, name='Response Time',
                                marker_color='#ffd93d'),
                    row=2, col=2
                )

            # 6. Info efficiency
            efficiency = self.collector.get_series("avg_info_efficiency") or [0]
            fig.add_trace(
                go.Scatter(x=epochs, y=efficiency, mode='lines+markers',
                          name='Info Efficiency', line=dict(color='#ff6b6b')),
                row=2, col=3
            )

        fig.update_layout(
            title=title,
            template="plotly_dark",
            height=800,
            showlegend=True,
            legend=dict(orientation="h", yanchor="bottom", y=-0.15),
        )

        return fig

    def save(self, filepath: str) -> None:
        """Save dashboard to HTML."""
        fig = self.create()
        fig.write_html(filepath)


class InfoFlowVisualizer:
    """
    Visualize information flow in PID games.

    Shows how information fragments are distributed to proposers
    and how proposals flow through the system.
    """

    def __init__(self):
        check_plotly()

    def plot_info_distribution(
        self,
        distribution: Dict[str, List[int]],
        fragments: List[str] = None,
        title: str = "Information Distribution",
    ) -> go.Figure:
        """Visualize info distribution as a Sankey diagram."""
        # Build Sankey diagram
        sources = []
        targets = []
        values = []
        labels = []

        # Source labels (fragments)
        num_frags = max(max(idxs) for idxs in distribution.values() if idxs) + 1 if distribution else 0
        frag_labels = [f"Frag {i}" for i in range(num_frags)]
        labels.extend(frag_labels)

        # Target labels (proposers)
        proposer_labels = list(distribution.keys())
        labels.extend(proposer_labels)

        # Build links
        for proposer_idx, (proposer, frag_indices) in enumerate(distribution.items()):
            target_idx = num_frags + proposer_idx
            for frag_idx in frag_indices:
                sources.append(frag_idx)
                targets.append(target_idx)
                values.append(1)

        fig = go.Figure(data=[go.Sankey(
            node=dict(
                pad=15,
                thickness=20,
                line=dict(color="black", width=0.5),
                label=labels,
                color=["#00d9ff"] * num_frags + ["#00ff88"] * len(proposer_labels),
            ),
            link=dict(
                source=sources,
                target=targets,
                value=values,
                color="rgba(0, 217, 255, 0.3)",
            )
        )])

        fig.update_layout(
            title=title,
            template="plotly_dark",
            font=dict(size=10),
        )

        return fig

    def plot_proposal_flow(
        self,
        proposals: List[Any],
        votes: Dict[str, Any] = None,
        synthesis: Any = None,
        title: str = "Proposal Flow",
    ) -> go.Figure:
        """Visualize flow from proposals through voting to synthesis."""
        labels = []
        sources = []
        targets = []
        values = []

        # Proposals
        for i, p in enumerate(proposals):
            proposer_id = getattr(p, 'proposer_id', f'P{i}')
            conf = getattr(p, 'confidence', 0.5)
            labels.append(f"{proposer_id}\n({conf:.2f})")

        num_proposals = len(proposals)

        # Voting layer
        if votes and "final_scores" in votes:
            labels.append("Voting")
            vote_idx = num_proposals

            for i, score in enumerate(votes["final_scores"][:num_proposals]):
                sources.append(i)
                targets.append(vote_idx)
                values.append(max(score, 0.1))

        # Synthesis layer
        if synthesis:
            labels.append("Synthesis")
            synth_idx = len(labels) - 1

            if votes:
                sources.append(vote_idx)
                targets.append(synth_idx)
                values.append(1)
            else:
                for i in range(num_proposals):
                    sources.append(i)
                    targets.append(synth_idx)
                    values.append(1)

        fig = go.Figure(data=[go.Sankey(
            node=dict(
                pad=15,
                thickness=20,
                line=dict(color="black", width=0.5),
                label=labels,
                color=["#00d9ff"] * num_proposals + ["#00ff88", "#ff6b6b"][:len(labels) - num_proposals],
            ),
            link=dict(
                source=sources,
                target=targets,
                value=values,
                color="rgba(0, 255, 136, 0.3)",
            )
        )])

        fig.update_layout(
            title=title,
            template="plotly_dark",
        )

        return fig

    def plot_calibration(
        self,
        calibration_data: List[Tuple[float, float]],
        title: str = "Confidence Calibration",
    ) -> go.Figure:
        """Plot calibration curve (predicted confidence vs actual accuracy)."""
        if not calibration_data:
            fig = go.Figure()
            fig.add_annotation(text="No calibration data", x=0.5, y=0.5,
                             xref="paper", yref="paper", showarrow=False)
            fig.update_layout(title=title, template="plotly_dark")
            return fig

        # Bin by confidence
        bins = [0, 0.2, 0.4, 0.6, 0.8, 1.0]
        bin_confs = []
        bin_actuals = []

        for i in range(len(bins) - 1):
            low, high = bins[i], bins[i + 1]
            in_bin = [(c, a) for c, a in calibration_data if low <= c < high]
            if in_bin:
                avg_conf = sum(c for c, a in in_bin) / len(in_bin)
                avg_actual = sum(a for c, a in in_bin) / len(in_bin)
                bin_confs.append(avg_conf)
                bin_actuals.append(avg_actual)

        fig = go.Figure()

        # Perfect calibration line
        fig.add_trace(go.Scatter(
            x=[0, 1], y=[0, 1],
            mode='lines',
            name='Perfect Calibration',
            line=dict(dash='dash', color='#888'),
        ))

        # Actual calibration
        fig.add_trace(go.Scatter(
            x=bin_confs,
            y=bin_actuals,
            mode='lines+markers',
            name='Actual',
            line=dict(color='#00d9ff', width=2),
            marker=dict(size=10),
        ))

        fig.update_layout(
            title=title,
            xaxis_title="Predicted Confidence",
            yaxis_title="Actual Accuracy",
            template="plotly_dark",
            xaxis=dict(range=[0, 1]),
            yaxis=dict(range=[0, 1]),
        )

        return fig


def create_training_report(
    collector: Any,
    output_path: str = "training_report.html",
) -> str:
    """Create a comprehensive training report."""
    check_plotly()

    # Create dashboard
    dashboard = PerformanceDashboard(collector)
    fig = dashboard.create()

    # Save
    fig.write_html(output_path)

    return output_path
