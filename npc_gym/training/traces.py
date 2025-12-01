"""
Trace collection and management for npc-gym.

Traces are complete game records used for:
- Training data generation (DPO preference pairs, SFT examples)
- Analysis and visualization
- Replay and debugging
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Callable
import json
import os
from datetime import datetime


@dataclass
class TraceStats:
    """Statistics about a collection of traces."""
    total_traces: int = 0
    total_steps: int = 0
    total_rewards: float = 0.0
    wins_by_player: Dict[str, int] = field(default_factory=dict)
    avg_game_length: float = 0.0
    avg_reward: float = 0.0

    def update(self, trace: "Trace") -> None:
        """Update stats with a new trace."""
        from npc_gym.core.env import Trace

        self.total_traces += 1
        self.total_steps += len(trace.steps)
        self.total_rewards += sum(trace.final_rewards.values())

        if trace.winner:
            self.wins_by_player[trace.winner] = self.wins_by_player.get(trace.winner, 0) + 1

        if self.total_traces > 0:
            self.avg_game_length = self.total_steps / self.total_traces
            self.avg_reward = self.total_rewards / self.total_traces


class TraceBuffer:
    """
    Buffer for storing and managing traces.

    Supports:
    - FIFO buffer with max size
    - Filtering by criteria
    - Conversion to training data formats
    """

    def __init__(self, max_size: int = 10000):
        self.max_size = max_size
        self.traces: List[Any] = []  # Trace objects
        self.stats = TraceStats()

    def add(self, trace: Any) -> None:
        """Add a trace to the buffer."""
        self.traces.append(trace)
        self.stats.update(trace)

        # Evict oldest if over capacity
        while len(self.traces) > self.max_size:
            self.traces.pop(0)

    def get_recent(self, n: int = 100) -> List[Any]:
        """Get most recent n traces."""
        return self.traces[-n:]

    def filter(
        self,
        predicate: Callable[[Any], bool]
    ) -> List[Any]:
        """Filter traces by predicate."""
        return [t for t in self.traces if predicate(t)]

    def get_winning_traces(self, player_id: str = None) -> List[Any]:
        """Get traces where specified player (or any) won."""
        if player_id:
            return [t for t in self.traces if t.winner == player_id]
        return [t for t in self.traces if t.winner is not None]

    def get_high_reward_traces(self, threshold: float = 0.0) -> List[Any]:
        """Get traces with rewards above threshold."""
        return [
            t for t in self.traces
            if any(r > threshold for r in t.final_rewards.values())
        ]

    def to_preference_pairs(
        self,
        min_reward_gap: float = 0.2
    ) -> List[Dict[str, Any]]:
        """
        Convert all traces to DPO preference pairs.

        Returns list of {prompt, chosen, rejected, reward_gap}
        """
        pairs = []
        for trace in self.traces:
            if hasattr(trace, 'to_preference_pairs'):
                pairs.extend(trace.to_preference_pairs(min_reward_gap))
        return pairs

    def to_sft_examples(self) -> List[Dict[str, str]]:
        """
        Convert winning actions to SFT training examples.

        Returns list of {input, output}
        """
        examples = []

        for trace in self.traces:
            if not trace.winner:
                continue

            for step in trace.steps:
                winner_obs = step["observations"].get(trace.winner, {})
                winner_action = step["actions"].get(trace.winner, {})

                if winner_obs and winner_action:
                    examples.append({
                        "input": json.dumps(winner_obs),
                        "output": json.dumps(winner_action),
                    })

        return examples

    def clear(self) -> None:
        """Clear the buffer."""
        self.traces = []
        self.stats = TraceStats()

    def save(self, filepath: str) -> None:
        """Save traces to file."""
        data = {
            "traces": [self._serialize_trace(t) for t in self.traces],
            "stats": {
                "total_traces": self.stats.total_traces,
                "total_steps": self.stats.total_steps,
                "wins_by_player": self.stats.wins_by_player,
            }
        }

        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2, default=str)

    def load(self, filepath: str) -> None:
        """Load traces from file."""
        with open(filepath, 'r') as f:
            data = json.load(f)

        # Would need to deserialize traces properly
        # For now just store raw data
        self.traces = data.get("traces", [])

    def _serialize_trace(self, trace: Any) -> Dict:
        """Serialize a trace to dict."""
        return {
            "env_id": getattr(trace, 'env_id', 'unknown'),
            "player_ids": getattr(trace, 'player_ids', []),
            "steps": getattr(trace, 'steps', []),
            "final_rewards": getattr(trace, 'final_rewards', {}),
            "winner": getattr(trace, 'winner', None),
            "metadata": getattr(trace, 'metadata', {}),
        }

    def __len__(self) -> int:
        return len(self.traces)


class TraceCollector:
    """
    Collects traces from multiple game runs.

    Features:
    - Runs games and collects traces
    - Aggregates statistics
    - Exports training data
    """

    def __init__(
        self,
        buffer_size: int = 10000,
        save_dir: str = None
    ):
        self.buffer = TraceBuffer(max_size=buffer_size)
        self.save_dir = save_dir

        if save_dir and not os.path.exists(save_dir):
            os.makedirs(save_dir)

    def collect_from_game(
        self,
        env: Any,  # Environment
        agents: Dict[str, Any],  # player_id -> Agent
        num_games: int = 1,
        verbose: bool = False
    ) -> TraceStats:
        """
        Run games and collect traces.

        Args:
            env: The game environment
            agents: Dict mapping player_id to Agent
            num_games: Number of games to run
            verbose: Print progress

        Returns:
            Statistics about collected traces
        """
        from npc_gym.core.env import Action

        initial_count = len(self.buffer)

        for game_idx in range(num_games):
            if verbose:
                print(f"Game {game_idx + 1}/{num_games}")

            # Reset environment
            observations, info = env.reset()

            terminated = False
            truncated = False

            while not (terminated or truncated):
                # Get current player(s)
                current_player = info.get("current_player")

                if current_player and current_player in agents:
                    # Single player's turn
                    agent = agents[current_player]
                    obs = observations[current_player]

                    response = agent.act(obs.to_dict() if hasattr(obs, 'to_dict') else obs)

                    action = Action(
                        player_id=current_player,
                        action_type=response.action_type,
                        value=response.value,
                        reasoning=response.reasoning,
                        confidence=response.confidence,
                    )

                    observations, rewards, terminated, truncated, info = env.step(action)

                else:
                    # Simultaneous moves or no current player
                    actions = {}
                    for player_id, agent in agents.items():
                        if player_id in observations:
                            obs = observations[player_id]
                            response = agent.act(obs.to_dict() if hasattr(obs, 'to_dict') else obs)

                            actions[player_id] = Action(
                                player_id=player_id,
                                action_type=response.action_type,
                                value=response.value,
                                reasoning=response.reasoning,
                                confidence=response.confidence,
                            )

                    if actions:
                        observations, rewards, terminated, truncated, info = env.step(actions)
                    else:
                        break

            # Collect trace
            trace = env.get_trace()
            if trace:
                self.buffer.add(trace)

                if verbose:
                    print(f"  Winner: {trace.winner}, Rewards: {trace.final_rewards}")

        # Return stats for this collection run
        new_traces = len(self.buffer) - initial_count
        return TraceStats(total_traces=new_traces)

    def export_dpo_data(
        self,
        filepath: str = None,
        min_reward_gap: float = 0.2
    ) -> List[Dict]:
        """Export preference pairs for DPO training."""
        pairs = self.buffer.to_preference_pairs(min_reward_gap)

        if filepath:
            with open(filepath, 'w') as f:
                json.dump(pairs, f, indent=2)

        return pairs

    def export_sft_data(self, filepath: str = None) -> List[Dict]:
        """Export examples for SFT training."""
        examples = self.buffer.to_sft_examples()

        if filepath:
            with open(filepath, 'w') as f:
                json.dump(examples, f, indent=2)

        return examples

    def save_checkpoint(self, name: str = None) -> str:
        """Save current traces to checkpoint file."""
        if not self.save_dir:
            self.save_dir = "./traces"
            os.makedirs(self.save_dir, exist_ok=True)

        if name is None:
            name = datetime.now().strftime("%Y%m%d_%H%M%S")

        filepath = os.path.join(self.save_dir, f"traces_{name}.json")
        self.buffer.save(filepath)
        return filepath

    def get_stats(self) -> TraceStats:
        """Get current statistics."""
        return self.buffer.stats
