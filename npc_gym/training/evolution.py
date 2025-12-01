"""
Genetic evolution of model genomes for npc-gym.

Evolves the "model genome" - a collection of specialized models
with trigger patterns and confidence thresholds that form the
System 1 "gut feeling" layer.

Based on npcpy's ft/model_ensembler.py and ft/ge.py
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Callable, Tuple
import copy
import random
import json
import os


@dataclass
class ModelGene:
    """
    A gene representing a specialized model.

    Attributes:
        specialization: What this model is trained for
        trigger_patterns: Text patterns that activate this model
        model_path: Path to fine-tuned model weights
        base_model: Base model architecture
        confidence_threshold: Min confidence to use this model's output
        fitness: Performance score (updated during evolution)
    """
    specialization: str
    trigger_patterns: List[str] = field(default_factory=list)
    model_path: Optional[str] = None
    base_model: str = "Qwen/Qwen3-0.6B"
    confidence_threshold: float = 0.7
    fitness: float = 0.0
    generation: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def matches(self, text: str) -> bool:
        """Check if this gene's patterns match the input."""
        text_lower = text.lower()
        return any(pattern in text_lower for pattern in self.trigger_patterns)

    def mutate(self, mutation_rate: float = 0.1) -> "ModelGene":
        """Create a mutated copy of this gene."""
        new_gene = copy.deepcopy(self)
        new_gene.generation += 1

        # Mutate confidence threshold
        if random.random() < mutation_rate:
            delta = random.gauss(0, 0.1)
            new_gene.confidence_threshold = max(0.5, min(0.95, new_gene.confidence_threshold + delta))

        # Mutate trigger patterns
        if random.random() < mutation_rate and new_gene.trigger_patterns:
            # Remove a pattern
            if len(new_gene.trigger_patterns) > 1 and random.random() < 0.5:
                new_gene.trigger_patterns.pop(random.randint(0, len(new_gene.trigger_patterns) - 1))
            # Add a pattern
            else:
                new_pattern = f"pattern_{random.randint(1, 1000)}"
                new_gene.trigger_patterns.append(new_pattern)

        return new_gene

    def crossover(self, other: "ModelGene") -> "ModelGene":
        """Create offspring by crossing with another gene."""
        child = ModelGene(
            specialization=f"{self.specialization}_{other.specialization[:10]}",
            trigger_patterns=self.trigger_patterns[:len(self.trigger_patterns)//2] +
                            other.trigger_patterns[len(other.trigger_patterns)//2:],
            model_path=self.model_path if random.random() < 0.5 else other.model_path,
            base_model=self.base_model,
            confidence_threshold=(self.confidence_threshold + other.confidence_threshold) / 2,
            generation=max(self.generation, other.generation) + 1,
        )
        return child

    def to_dict(self) -> Dict:
        return {
            "specialization": self.specialization,
            "trigger_patterns": self.trigger_patterns,
            "model_path": self.model_path,
            "base_model": self.base_model,
            "confidence_threshold": self.confidence_threshold,
            "fitness": self.fitness,
            "generation": self.generation,
        }

    @classmethod
    def from_dict(cls, data: Dict) -> "ModelGene":
        return cls(**data)


class GenePool:
    """
    A pool of model genes that can be evolved.

    This represents the "model genome" - the collection of specialized
    models that form an agent's System 1.
    """

    def __init__(
        self,
        genes: List[ModelGene] = None,
        max_size: int = 20
    ):
        self.genes = genes or []
        self.max_size = max_size
        self.generation = 0
        self.history: List[Dict] = []

    def add_gene(self, gene: ModelGene) -> None:
        """Add a gene to the pool."""
        self.genes.append(gene)
        if len(self.genes) > self.max_size:
            # Remove lowest fitness
            self.genes.sort(key=lambda g: g.fitness, reverse=True)
            self.genes = self.genes[:self.max_size]

    def get_matching_genes(self, text: str) -> List[ModelGene]:
        """Get all genes whose patterns match the text."""
        return [g for g in self.genes if g.matches(text)]

    def get_best_gene(self, text: str = None) -> Optional[ModelGene]:
        """Get the best gene (optionally filtered by text match)."""
        if text:
            matching = self.get_matching_genes(text)
            if matching:
                return max(matching, key=lambda g: g.fitness)
        if self.genes:
            return max(self.genes, key=lambda g: g.fitness)
        return None

    def update_fitness(self, gene_id: str, fitness_delta: float) -> None:
        """Update fitness for a gene by specialization."""
        for gene in self.genes:
            if gene.specialization == gene_id:
                gene.fitness += fitness_delta
                break

    def evolve(
        self,
        fitness_fn: Callable[[ModelGene], float] = None,
        mutation_rate: float = 0.15,
        crossover_rate: float = 0.7,
        elite_ratio: float = 0.1
    ) -> None:
        """
        Evolve the gene pool for one generation.

        Args:
            fitness_fn: Function to evaluate gene fitness (if not already set)
            mutation_rate: Probability of mutation per gene
            crossover_rate: Probability of crossover
            elite_ratio: Fraction of top performers to keep unchanged
        """
        if not self.genes:
            return

        # Evaluate fitness if function provided
        if fitness_fn:
            for gene in self.genes:
                gene.fitness = fitness_fn(gene)

        # Sort by fitness
        self.genes.sort(key=lambda g: g.fitness, reverse=True)

        # Record history
        self.history.append({
            "generation": self.generation,
            "best_fitness": self.genes[0].fitness,
            "avg_fitness": sum(g.fitness for g in self.genes) / len(self.genes),
            "best_gene": self.genes[0].specialization,
        })

        # Keep elites
        n_elite = max(1, int(len(self.genes) * elite_ratio))
        new_genes = self.genes[:n_elite]

        # Generate rest through crossover and mutation
        while len(new_genes) < self.max_size:
            if random.random() < crossover_rate and len(self.genes) >= 2:
                # Tournament selection
                parent1 = self._tournament_select()
                parent2 = self._tournament_select()
                child = parent1.crossover(parent2)
            else:
                parent = self._tournament_select()
                child = copy.deepcopy(parent)

            # Mutation
            if random.random() < mutation_rate:
                child = child.mutate(mutation_rate)

            new_genes.append(child)

        self.genes = new_genes
        self.generation += 1

    def _tournament_select(self, k: int = 3) -> ModelGene:
        """Select a gene using tournament selection."""
        tournament = random.sample(self.genes, min(k, len(self.genes)))
        return max(tournament, key=lambda g: g.fitness)

    def save(self, filepath: str) -> None:
        """Save gene pool to file."""
        data = {
            "generation": self.generation,
            "genes": [g.to_dict() for g in self.genes],
            "history": self.history,
        }
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)

    def load(self, filepath: str) -> None:
        """Load gene pool from file."""
        with open(filepath, 'r') as f:
            data = json.load(f)

        self.generation = data.get("generation", 0)
        self.genes = [ModelGene.from_dict(g) for g in data.get("genes", [])]
        self.history = data.get("history", [])

    def __len__(self) -> int:
        return len(self.genes)


@dataclass
class EvolutionConfig:
    """Configuration for model evolution."""
    population_size: int = 20
    generations: int = 50
    mutation_rate: float = 0.15
    crossover_rate: float = 0.7
    elite_ratio: float = 0.1
    tournament_size: int = 3
    fitness_weights: Dict[str, float] = field(default_factory=lambda: {
        "accuracy": 0.6,
        "speed": 0.2,
        "efficiency": 0.2,
    })


class ModelEvolver:
    """
    Evolves model genomes through gameplay.

    Process:
    1. Run games with current gene pool
    2. Evaluate gene fitness based on game outcomes
    3. Evolve gene pool (selection, crossover, mutation)
    4. Optionally fine-tune winning genes' models
    5. Repeat
    """

    def __init__(
        self,
        config: EvolutionConfig = None,
        save_dir: str = None
    ):
        self.config = config or EvolutionConfig()
        self.save_dir = save_dir
        self.gene_pool = GenePool(max_size=self.config.population_size)

        if save_dir and not os.path.exists(save_dir):
            os.makedirs(save_dir)

    def initialize_pool(
        self,
        specializations: List[str] = None,
        base_model: str = "Qwen/Qwen3-0.6B"
    ) -> None:
        """
        Initialize gene pool with starter genes.

        Args:
            specializations: List of specialization domains
            base_model: Base model for all genes
        """
        if specializations is None:
            specializations = [
                "math",
                "code",
                "reasoning",
                "factual",
                "creative",
            ]

        # Pattern templates for each specialization
        pattern_templates = {
            "math": ["calculate", "solve", "equation", "number", "sum", "product"],
            "code": ["function", "class", "bug", "debug", "code", "program"],
            "reasoning": ["why", "because", "therefore", "logic", "infer"],
            "factual": ["what is", "who is", "when did", "where is", "fact"],
            "creative": ["story", "poem", "imagine", "creative", "write"],
        }

        for spec in specializations:
            patterns = pattern_templates.get(spec, [spec])
            gene = ModelGene(
                specialization=spec,
                trigger_patterns=patterns,
                base_model=base_model,
                confidence_threshold=random.uniform(0.6, 0.9),
            )
            self.gene_pool.add_gene(gene)

    def evaluate_on_traces(
        self,
        traces: List[Any],
        agent_gene_mapping: Dict[str, str] = None
    ) -> Dict[str, float]:
        """
        Evaluate gene fitness based on game traces.

        Args:
            traces: List of game traces
            agent_gene_mapping: Maps agent IDs to gene specializations

        Returns:
            Dict of gene_id -> fitness score
        """
        fitness_scores = {g.specialization: 0.0 for g in self.gene_pool.genes}
        counts = {g.specialization: 0 for g in self.gene_pool.genes}

        for trace in traces:
            if not hasattr(trace, 'winner') or not trace.winner:
                continue

            # Map winner to gene
            winner_gene = None
            if agent_gene_mapping and trace.winner in agent_gene_mapping:
                winner_gene = agent_gene_mapping[trace.winner]
            else:
                # Assume winner ID matches gene specialization
                winner_gene = trace.winner

            if winner_gene in fitness_scores:
                # Reward for winning
                reward = trace.final_rewards.get(trace.winner, 0)
                fitness_scores[winner_gene] += max(0, reward)
                counts[winner_gene] += 1

        # Normalize by game count
        for gene_id in fitness_scores:
            if counts[gene_id] > 0:
                fitness_scores[gene_id] /= counts[gene_id]

        return fitness_scores

    def evolve_generation(
        self,
        fitness_scores: Dict[str, float] = None
    ) -> Dict[str, Any]:
        """
        Evolve the gene pool for one generation.

        Args:
            fitness_scores: Pre-computed fitness scores (gene_id -> score)

        Returns:
            Stats about this generation
        """
        # Update fitness from scores
        if fitness_scores:
            for gene in self.gene_pool.genes:
                if gene.specialization in fitness_scores:
                    gene.fitness = fitness_scores[gene.specialization]

        # Evolve
        self.gene_pool.evolve(
            mutation_rate=self.config.mutation_rate,
            crossover_rate=self.config.crossover_rate,
            elite_ratio=self.config.elite_ratio,
        )

        # Return stats
        return {
            "generation": self.gene_pool.generation,
            "best_fitness": max(g.fitness for g in self.gene_pool.genes) if self.gene_pool.genes else 0,
            "avg_fitness": sum(g.fitness for g in self.gene_pool.genes) / len(self.gene_pool.genes) if self.gene_pool.genes else 0,
            "pool_size": len(self.gene_pool),
        }

    def train_winning_genes(
        self,
        traces: List[Any],
        training_fn: Callable = None,
        top_k: int = 3
    ) -> List[str]:
        """
        Fine-tune models for top-performing genes.

        Args:
            traces: Traces to extract training data from
            training_fn: Function to train a model (gene, data) -> model_path
            top_k: Number of top genes to train

        Returns:
            List of trained model paths
        """
        # Get top genes
        top_genes = sorted(
            self.gene_pool.genes,
            key=lambda g: g.fitness,
            reverse=True
        )[:top_k]

        trained_paths = []

        for gene in top_genes:
            # Extract training data for this gene
            training_data = self._extract_training_data(traces, gene)

            if not training_data:
                continue

            if training_fn:
                # Custom training function
                model_path = training_fn(gene, training_data)
            else:
                # Use npcpy SFT
                model_path = self._default_training(gene, training_data)

            if model_path:
                gene.model_path = model_path
                trained_paths.append(model_path)

        return trained_paths

    def _extract_training_data(
        self,
        traces: List[Any],
        gene: ModelGene
    ) -> List[Dict]:
        """Extract training examples relevant to a gene."""
        examples = []

        for trace in traces:
            for step in trace.steps:
                for player_id, obs in step.get("observations", {}).items():
                    # Check if this observation matches gene's patterns
                    obs_text = str(obs)
                    if gene.matches(obs_text):
                        action = step.get("actions", {}).get(player_id, {})
                        if action:
                            examples.append({
                                "input": obs_text,
                                "output": str(action),
                            })

        return examples

    def _default_training(
        self,
        gene: ModelGene,
        training_data: List[Dict]
    ) -> Optional[str]:
        """Default training using npcpy SFT."""
        try:
            from npcpy.ft.sft import run_sft, SFTConfig

            X = [d["input"] for d in training_data]
            y = [d["output"] for d in training_data]

            config = SFTConfig(
                base_model_name=gene.base_model,
                output_model_path=f"models/gene_{gene.specialization}_{gene.generation}",
                num_train_epochs=10,
            )

            model_path = run_sft(X, y, config=config)
            return model_path

        except Exception as e:
            print(f"Training failed for {gene.specialization}: {e}")
            return None

    def save_checkpoint(self, name: str = None) -> str:
        """Save evolution state."""
        if not self.save_dir:
            self.save_dir = "./evolution"
            os.makedirs(self.save_dir, exist_ok=True)

        if name is None:
            name = f"gen_{self.gene_pool.generation}"

        filepath = os.path.join(self.save_dir, f"gene_pool_{name}.json")
        self.gene_pool.save(filepath)
        return filepath

    def load_checkpoint(self, filepath: str) -> None:
        """Load evolution state."""
        self.gene_pool.load(filepath)

    def get_best_genome(self) -> List[ModelGene]:
        """Get current best genes for deployment."""
        return sorted(
            self.gene_pool.genes,
            key=lambda g: g.fitness,
            reverse=True
        )
