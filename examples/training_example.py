"""
Training example for npc-gym.

Demonstrates the full training loop:
1. Run games to collect traces
2. Evolve model genome
3. Fine-tune models (optional)
4. Track progress
"""

import sys
sys.path.insert(0, '..')

from npc_gym.envs import InfoPoker, InfoPokerConfig
from npc_gym.training import TrainingLoop, TrainingConfig
from npc_gym.training.evolution import EvolutionConfig
from npc_gym.core.agent import HybridAgent, RandomAgent


# Sample training texts
TRAINING_TEXTS = [
    """
    The application crashes intermittently when processing large file uploads.
    Stack traces indicate memory allocation failures in the image processing module.
    The server has 16GB RAM and typically uses 60% under normal load. However,
    during peak hours, memory usage spikes to 95%. Garbage collection logs show
    frequent full GC cycles lasting 500ms or more.
    """,
    """
    User authentication is failing for approximately 5% of login attempts.
    The error logs show timeout exceptions when connecting to the LDAP server.
    Network latency between the application server and LDAP is normally 2ms
    but occasionally spikes to 500ms. The LDAP server shows no signs of overload.
    """,
    """
    Search queries are returning stale results despite recent database updates.
    The search index is configured to refresh every 30 seconds. However, some
    documents appear to be missing from results for up to 10 minutes. The
    indexing service logs show no errors. Cache invalidation events are firing correctly.
    """,
]


def main():
    print("=" * 60)
    print("NPC-GYM TRAINING EXAMPLE")
    print("=" * 60)
    print()

    # Training configuration
    training_config = TrainingConfig(
        env_class=InfoPoker,
        env_kwargs={
            "source_text": TRAINING_TEXTS[0],  # Will cycle through
            "num_players": 4,
        },
        agent_class=RandomAgent,  # Use HybridAgent for real training
        num_agents=4,
        num_epochs=5,  # Small for demo
        games_per_epoch=10,  # Small for demo
        evolution_frequency=1,
        training_frequency=10,  # Disabled for demo (would need more data)
        min_traces_for_training=100,
        save_dir="./training_output",
        verbose=True,
    )

    # Create and run training loop
    loop = TrainingLoop(training_config)
    loop.setup()

    print(f"\nStarting training with {len(TRAINING_TEXTS)} source texts...")
    print()

    # Run training, cycling through source texts
    for epoch in range(training_config.num_epochs):
        # Update source text for variety
        text_idx = epoch % len(TRAINING_TEXTS)
        loop.env.ground_truth = TRAINING_TEXTS[text_idx]
        loop.env.config.source_text = TRAINING_TEXTS[text_idx]

        # Re-create deck with new text
        loop.env.deck = loop.env._create_deck()

        # Run epoch
        stats = loop.run_epoch()

        print(f"\nEpoch {stats['epoch']} Summary:")
        print(f"  Games played: {stats['total_games']}")
        print(f"  Traces collected: {stats['traces_collected']}")
        print(f"  Buffer size: {stats['buffer_size']}")

        if stats.get('evolution'):
            print(f"  Evolution generation: {stats['evolution'].get('generation', 'N/A')}")
            print(f"  Best fitness: {stats['evolution'].get('best_fitness', 0):.3f}")

    print()
    print("=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)

    # Show final stats
    print(f"\nTotal games played: {loop.total_games}")
    print(f"Total traces collected: {len(loop.trace_collector.buffer)}")

    # Show gene pool
    print(f"\nEvolved Gene Pool ({len(loop.model_evolver.gene_pool)} genes):")
    for gene in loop.model_evolver.get_best_genome()[:5]:
        print(f"  {gene.specialization}: fitness={gene.fitness:.3f}, "
              f"confidence={gene.confidence_threshold:.2f}")

    # Export training data
    print("\nExporting training data...")

    dpo_pairs = loop.trace_collector.export_dpo_data(
        filepath="./training_output/dpo_pairs.json",
        min_reward_gap=0.1,
    )
    print(f"  DPO pairs: {len(dpo_pairs)}")

    sft_examples = loop.trace_collector.export_sft_data(
        filepath="./training_output/sft_examples.json"
    )
    print(f"  SFT examples: {len(sft_examples)}")

    print("\nTraining output saved to ./training_output/")


if __name__ == "__main__":
    main()
