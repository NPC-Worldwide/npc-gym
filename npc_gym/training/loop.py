"""
Main training loop for npc-gym.

Orchestrates the full training pipeline:
1. Run games to collect traces
2. Generate training data (DPO pairs, SFT examples)
3. Fine-tune models
4. Evolve model genome
5. Repeat

Integrates with npcpy's fine-tuning capabilities.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Callable, Type
import os
import json
from datetime import datetime

from npc_gym.training.traces import TraceCollector, TraceBuffer
from npc_gym.training.evolution import ModelEvolver, GenePool, ModelGene, EvolutionConfig


@dataclass
class TrainingConfig:
    """Configuration for the training loop."""
    # Environment
    env_class: Type = None  # Environment class to use
    env_kwargs: Dict[str, Any] = field(default_factory=dict)

    # Agents
    num_agents: int = 4
    agent_class: Type = None  # Agent class to use

    # Training loop
    num_epochs: int = 10
    games_per_epoch: int = 100
    evolution_frequency: int = 1  # Evolve every N epochs
    training_frequency: int = 5   # Train models every N epochs

    # Data collection
    min_traces_for_training: int = 50
    preference_pair_min_gap: float = 0.2

    # Evolution
    evolution_config: EvolutionConfig = field(default_factory=EvolutionConfig)

    # Fine-tuning
    use_dpo: bool = True
    use_sft: bool = True
    dpo_config: Dict[str, Any] = field(default_factory=dict)
    sft_config: Dict[str, Any] = field(default_factory=dict)

    # Saving
    save_dir: str = "./training_runs"
    save_frequency: int = 5  # Save checkpoint every N epochs

    # Logging
    verbose: bool = True
    log_file: str = None


class TrainingLoop:
    """
    Main training orchestrator for npc-gym.

    The loop:
    1. Initialize environment and agents
    2. For each epoch:
       a. Run games, collect traces
       b. Evaluate agent/gene fitness
       c. Generate training data
       d. Fine-tune models (if schedule)
       e. Evolve gene pool (if schedule)
       f. Update agents with new genes/models
    3. Save final checkpoints
    """

    def __init__(self, config: TrainingConfig):
        self.config = config

        # Initialize components
        self.trace_collector = TraceCollector(
            save_dir=os.path.join(config.save_dir, "traces")
        )
        self.model_evolver = ModelEvolver(
            config=config.evolution_config,
            save_dir=os.path.join(config.save_dir, "evolution")
        )

        # State
        self.epoch = 0
        self.total_games = 0
        self.history: List[Dict] = []

        # Environment and agents (initialized in setup)
        self.env = None
        self.agents: Dict[str, Any] = {}

        # Create save directory
        os.makedirs(config.save_dir, exist_ok=True)

    def setup(self) -> None:
        """Initialize environment and agents."""
        # Create environment
        if self.config.env_class:
            self.env = self.config.env_class(**self.config.env_kwargs)
        else:
            raise ValueError("Must provide env_class in config")

        # Create agents
        player_ids = self.env.player_ids

        if self.config.agent_class:
            from npc_gym.core.agent import AgentConfig, HybridAgent

            for player_id in player_ids:
                agent_config = AgentConfig(
                    name=player_id,
                    model="llama3.2",
                    provider="ollama",
                )

                if self.config.agent_class == HybridAgent:
                    agent = HybridAgent(
                        config=agent_config,
                        model_genome=self.model_evolver.gene_pool.genes,
                    )
                else:
                    agent = self.config.agent_class(config=agent_config)

                self.agents[player_id] = agent
        else:
            # Default to random agents
            from npc_gym.core.agent import RandomAgent, AgentConfig

            for player_id in player_ids:
                agent_config = AgentConfig(name=player_id)
                self.agents[player_id] = RandomAgent(config=agent_config)

        # Initialize gene pool
        self.model_evolver.initialize_pool()

        if self.config.verbose:
            print(f"Initialized {len(self.agents)} agents")
            print(f"Gene pool size: {len(self.model_evolver.gene_pool)}")

    def run_epoch(self) -> Dict[str, Any]:
        """Run one training epoch."""
        self.epoch += 1

        if self.config.verbose:
            print(f"\n{'='*50}")
            print(f"EPOCH {self.epoch}")
            print(f"{'='*50}")

        # 1. Run games and collect traces
        stats = self.trace_collector.collect_from_game(
            env=self.env,
            agents=self.agents,
            num_games=self.config.games_per_epoch,
            verbose=self.config.verbose,
        )

        self.total_games += self.config.games_per_epoch

        if self.config.verbose:
            print(f"Collected {stats.total_traces} traces")

        # 2. Evaluate fitness
        traces = self.trace_collector.buffer.get_recent(self.config.games_per_epoch)

        # Map agents to genes (for now, simple mapping)
        agent_gene_mapping = {
            agent_id: agent_id.split("_")[-1]  # Extract any suffix
            for agent_id in self.agents
        }

        fitness_scores = self.model_evolver.evaluate_on_traces(
            traces, agent_gene_mapping
        )

        if self.config.verbose:
            print(f"Fitness scores: {fitness_scores}")

        # 3. Evolution (if scheduled)
        evolution_stats = {}
        if self.epoch % self.config.evolution_frequency == 0:
            evolution_stats = self.model_evolver.evolve_generation(fitness_scores)
            if self.config.verbose:
                print(f"Evolution: Gen {evolution_stats['generation']}, "
                      f"Best fitness: {evolution_stats['best_fitness']:.3f}")

            # Update agents with new genes
            self._update_agent_genes()

        # 4. Fine-tuning (if scheduled and enough data)
        training_stats = {}
        n_traces = len(self.trace_collector.buffer)

        if (self.epoch % self.config.training_frequency == 0 and
            n_traces >= self.config.min_traces_for_training):

            training_stats = self._run_training()

        # 5. Save checkpoint (if scheduled)
        if self.epoch % self.config.save_frequency == 0:
            self._save_checkpoint()

        # Record epoch stats
        epoch_stats = {
            "epoch": self.epoch,
            "total_games": self.total_games,
            "traces_collected": stats.total_traces,
            "buffer_size": n_traces,
            "fitness_scores": fitness_scores,
            "evolution": evolution_stats,
            "training": training_stats,
        }

        self.history.append(epoch_stats)

        return epoch_stats

    def run(self, num_epochs: int = None) -> List[Dict]:
        """
        Run the full training loop.

        Args:
            num_epochs: Override config.num_epochs

        Returns:
            List of epoch statistics
        """
        if num_epochs is None:
            num_epochs = self.config.num_epochs

        # Setup if not done
        if self.env is None:
            self.setup()

        # Run epochs
        for _ in range(num_epochs):
            self.run_epoch()

        # Final save
        self._save_checkpoint(name="final")

        if self.config.verbose:
            print(f"\nTraining complete! {self.total_games} games played over {self.epoch} epochs.")

        return self.history

    def _update_agent_genes(self) -> None:
        """Update agents with evolved gene pool."""
        from npc_gym.core.agent import HybridAgent

        for agent in self.agents.values():
            if isinstance(agent, HybridAgent):
                agent.model_genome = self.model_evolver.gene_pool.genes

    def _run_training(self) -> Dict[str, Any]:
        """Run fine-tuning on collected data."""
        stats = {}

        traces = list(self.trace_collector.buffer.traces)

        # DPO Training
        if self.config.use_dpo:
            dpo_pairs = self.trace_collector.export_dpo_data(
                min_reward_gap=self.config.preference_pair_min_gap
            )

            if len(dpo_pairs) >= 10:
                try:
                    dpo_result = self._run_dpo_training(dpo_pairs)
                    stats["dpo"] = dpo_result
                except Exception as e:
                    print(f"DPO training failed: {e}")
                    stats["dpo"] = {"error": str(e)}

        # SFT Training for top genes
        if self.config.use_sft:
            try:
                trained_paths = self.model_evolver.train_winning_genes(
                    traces, top_k=3
                )
                stats["sft"] = {"trained_models": trained_paths}
            except Exception as e:
                print(f"SFT training failed: {e}")
                stats["sft"] = {"error": str(e)}

        return stats

    def _run_dpo_training(self, pairs: List[Dict]) -> Dict[str, Any]:
        """Run DPO training on preference pairs."""
        try:
            from npcpy.ft.rl import train_with_dpo, RLConfig

            # Convert pairs to trace format
            traces = [
                {
                    "task_prompt": p["prompt"],
                    "final_output": p["chosen"],
                    "reward": p.get("chosen_score", 1.0),
                }
                for p in pairs
            ] + [
                {
                    "task_prompt": p["prompt"],
                    "final_output": p["rejected"],
                    "reward": p.get("rejected_score", 0.0),
                }
                for p in pairs
            ]

            config = RLConfig(**self.config.dpo_config)
            adapter_path = train_with_dpo(traces, config)

            return {"adapter_path": adapter_path, "num_pairs": len(pairs)}

        except Exception as e:
            return {"error": str(e)}

    def _save_checkpoint(self, name: str = None) -> None:
        """Save training checkpoint."""
        if name is None:
            name = f"epoch_{self.epoch}"

        checkpoint_dir = os.path.join(self.config.save_dir, f"checkpoint_{name}")
        os.makedirs(checkpoint_dir, exist_ok=True)

        # Save traces
        self.trace_collector.buffer.save(
            os.path.join(checkpoint_dir, "traces.json")
        )

        # Save gene pool
        self.model_evolver.gene_pool.save(
            os.path.join(checkpoint_dir, "gene_pool.json")
        )

        # Save history
        with open(os.path.join(checkpoint_dir, "history.json"), 'w') as f:
            json.dump(self.history, f, indent=2, default=str)

        # Save config
        with open(os.path.join(checkpoint_dir, "config.json"), 'w') as f:
            json.dump({
                "num_epochs": self.config.num_epochs,
                "games_per_epoch": self.config.games_per_epoch,
                "num_agents": self.config.num_agents,
            }, f, indent=2)

        if self.config.verbose:
            print(f"Saved checkpoint to {checkpoint_dir}")

    def load_checkpoint(self, checkpoint_dir: str) -> None:
        """Load from checkpoint."""
        # Load traces
        traces_path = os.path.join(checkpoint_dir, "traces.json")
        if os.path.exists(traces_path):
            self.trace_collector.buffer.load(traces_path)

        # Load gene pool
        gene_pool_path = os.path.join(checkpoint_dir, "gene_pool.json")
        if os.path.exists(gene_pool_path):
            self.model_evolver.gene_pool.load(gene_pool_path)

        # Load history
        history_path = os.path.join(checkpoint_dir, "history.json")
        if os.path.exists(history_path):
            with open(history_path, 'r') as f:
                self.history = json.load(f)
                if self.history:
                    self.epoch = self.history[-1].get("epoch", 0)
                    self.total_games = self.history[-1].get("total_games", 0)

    def get_best_agent(self) -> Any:
        """Get the current best-performing agent."""
        best_genes = self.model_evolver.get_best_genome()

        from npc_gym.core.agent import HybridAgent, AgentConfig

        config = AgentConfig(
            name="best_agent",
            model="llama3.2",
            provider="ollama",
        )

        return HybridAgent(
            config=config,
            model_genome=best_genes[:5],  # Top 5 genes
        )


def quick_train(
    env_class: Type,
    source_texts: List[str] = None,
    num_epochs: int = 10,
    games_per_epoch: int = 50,
    **env_kwargs
) -> TrainingLoop:
    """
    Quick helper to run training with minimal setup.

    Args:
        env_class: Environment class (e.g., InfoPoker, HypothesisBlackjack)
        source_texts: List of texts to use for training (cycles through them)
        num_epochs: Number of training epochs
        games_per_epoch: Games to play per epoch
        **env_kwargs: Additional environment kwargs

    Returns:
        Trained TrainingLoop instance
    """
    from npc_gym.core.agent import HybridAgent

    config = TrainingConfig(
        env_class=env_class,
        env_kwargs=env_kwargs,
        num_agents=4,
        agent_class=HybridAgent,
        num_epochs=num_epochs,
        games_per_epoch=games_per_epoch,
        verbose=True,
    )

    loop = TrainingLoop(config)

    # If source texts provided, cycle through them
    if source_texts:
        for epoch in range(num_epochs):
            text_idx = epoch % len(source_texts)
            loop.env = env_class(source_text=source_texts[text_idx], **env_kwargs)
            loop.run_epoch()
    else:
        loop.run()

    return loop
