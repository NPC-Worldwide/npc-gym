"""
Train a NEAT agent to play Slime Volleyball.

Uses npcpy's NEAT evolver with the npc-gym SlimeVolley environment.
Demonstrates multi-backend support — run on numpy, jax, mlx, or cuda.

Usage:
    python train_slime_neat.py                    # numpy (default)
    python train_slime_neat.py --engine jax       # JAX backend
    python train_slime_neat.py --engine mlx       # MLX backend
    python train_slime_neat.py --engine cuda      # PyTorch CUDA
"""

import argparse
import pickle
import numpy as np

from npcpy.ft.neat import NEATEvolver, NEATConfig, NEATNetwork
from npcpy.ft.engine import get_engine
from npc_gym.envs.slime_volleyball import SlimeVolleyEnv


def evaluate_network(
    network: NEATNetwork,
    env: SlimeVolleyEnv,
    num_episodes: int = 3,
) -> float:
    """
    Evaluate a NEAT network on SlimeVolley.

    Returns average reward across episodes.
    The network's raw outputs are used directly — no heuristic overrides.
    """
    total_reward = 0.0

    for _ in range(num_episodes):
        obs = env.reset()
        done = False
        episode_reward = 0.0
        steps = 0

        while not done and steps < 3000:
            # Network decides action — output is 3 values, threshold at 0
            raw_output = network.activate(obs)
            action = np.array([
                1.0 if raw_output[0] > 0 else 0.0,
                1.0 if raw_output[1] > 0 else 0.0,
                1.0 if raw_output[2] > 0 else 0.0,
            ])

            obs, reward, done, info = env.step(action)
            episode_reward += reward
            steps += 1

        total_reward += episode_reward

    return total_reward / num_episodes


def main():
    parser = argparse.ArgumentParser(description="Train NEAT on Slime Volleyball")
    parser.add_argument("--engine", default="numpy", choices=["numpy", "jax", "mlx", "cuda"])
    parser.add_argument("--generations", type=int, default=100)
    parser.add_argument("--population", type=int, default=150)
    parser.add_argument("--episodes", type=int, default=3, help="Episodes per fitness eval")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", default="best_slime_neat.pkl")
    args = parser.parse_args()

    print(f"Engine: {args.engine}")
    print(f"Population: {args.population}")
    print(f"Generations: {args.generations}")
    print()

    env = SlimeVolleyEnv(max_score=5)

    config = NEATConfig(
        population_size=args.population,
        weight_mutation_rate=0.8,
        weight_perturbation_std=0.2,
        add_node_rate=0.03,
        add_connection_rate=0.05,
        species_threshold=3.0,
        species_stagnation_limit=15,
    )

    evolver = NEATEvolver(
        input_size=12,  # SlimeVolley observation
        output_size=3,  # [forward, backward, jump]
        config=config,
        engine=args.engine,
        seed=args.seed,
    )

    def fitness_fn(network: NEATNetwork) -> float:
        return evaluate_network(network, env, num_episodes=args.episodes)

    def on_generation(gen: int, stats: dict):
        if gen % 5 == 0:
            best_net = evolver.get_network()
            # Quick eval with more episodes for reporting
            score = evaluate_network(best_net, env, num_episodes=5)
            print(f"  -> 5-episode eval: {score:.2f}")

    print("Starting evolution...")
    best_genome = evolver.run(
        fitness_fn,
        generations=args.generations,
        callback=on_generation,
        verbose=True,
    )

    # Save
    evolver.save(args.output, best_genome)
    print(f"\nBest genome saved to {args.output}")
    print(f"Hidden nodes: {best_genome.num_hidden}")
    print(f"Connections: {best_genome.num_enabled_connections}")

    # Final evaluation
    best_net = NEATNetwork(best_genome, get_engine(args.engine))
    final_score = evaluate_network(best_net, env, num_episodes=20)
    print(f"Final 20-episode avg reward: {final_score:.3f}")

    # Show history
    if evolver.history:
        print(f"\nEvolution summary:")
        print(f"  Start fitness: {evolver.history[0]['best_fitness']:.3f}")
        print(f"  End fitness:   {evolver.history[-1]['best_fitness']:.3f}")
        print(f"  Best ever:     {evolver.history[-1]['best_ever_fitness']:.3f}")


if __name__ == "__main__":
    main()
