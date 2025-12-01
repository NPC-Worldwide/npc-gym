"""
Basic InfoPoker example.

Demonstrates:
1. Creating an InfoPoker environment
2. Running a game with hybrid agents
3. Collecting traces for training
"""

import sys
sys.path.insert(0, '..')

from npc_gym.envs import InfoPoker, InfoPokerConfig
from npc_gym.core.agent import HybridAgent, AgentConfig, RandomAgent
from npc_gym.core.env import Action


def main():
    # Sample text to decompose
    source_text = """
    The system experiences intermittent latency spikes, primarily between
    14:00 and 15:00 UTC. Initial logs from the application server show no
    corresponding CPU or memory anomalies. The database monitoring tools,
    however, report a surge in read operations during this window, though
    they remain within acceptable performance thresholds. Network diagnostics
    have cleared the internal infrastructure of any packet loss or hardware
    failure. A recently deployed microservice, service-alpha, handles user
    authentication and is most active during the specified time frame. This
    service was designed to be stateless and heavily cache user sessions to
    minimize database interaction.
    """

    # Create environment config
    config = InfoPokerConfig(
        source_text=source_text,
        chunk_by="word",
        hole_cards=5,  # 5 words per player as private info
        community_schedule=[10, 5, 5],  # Reveal 10, then 5, then 5 words
        num_judges=3,
        use_llm_judges=False,  # Use simple scoring for speed
    )

    # Create environment
    env = InfoPoker(config=config, num_players=3)

    print("=" * 60)
    print("INFO POKER - Partial Information Game")
    print("=" * 60)
    print(f"Players: {env.player_ids}")
    print(f"Source text length: {len(source_text.split())} words")
    print()

    # Create agents (using Random for demo, use HybridAgent with LLM for real games)
    agents = {}
    for player_id in env.player_ids:
        agent_config = AgentConfig(
            name=player_id,
            model="llama3.2",
            provider="ollama",
        )
        # Use RandomAgent for quick demo, HybridAgent for actual training
        agents[player_id] = RandomAgent(config=agent_config)

    # Run the game
    observations, info = env.reset()

    print("Initial deal complete!")
    print(f"Phase: {info['phase']}")
    print()

    step = 0
    while True:
        step += 1
        current_player = info.get("current_player")

        if not current_player:
            break

        obs = observations.get(current_player)
        if not obs:
            break

        print(f"Step {step}: {current_player}'s turn")
        print(f"  Private info: {obs.info_partition.private[:3]}...")  # First 3 words
        print(f"  Valid actions: {obs.valid_actions}")

        # Get agent action
        agent = agents[current_player]
        response = agent.act(obs.to_dict())

        # Create action
        action = Action(
            player_id=current_player,
            action_type=response.action_type,
            value=response.value,
            reasoning=f"Hypothesis: The issue is related to {obs.info_partition.private[0]}",
            confidence=0.7,
        )

        print(f"  Action: {action.action_type}")

        # Step environment
        observations, rewards, terminated, truncated, info = env.step(action)

        if terminated or truncated:
            break

    # Game over
    print()
    print("=" * 60)
    print("GAME OVER")
    print("=" * 60)

    trace = env.get_trace()
    if trace:
        print(f"Winner: {trace.winner}")
        print(f"Final rewards: {trace.final_rewards}")
        print(f"Total steps: {len(trace.steps)}")

        # Show hypothesis scores if available
        if "hypothesis_scores" in trace.metadata:
            print()
            print("Hypothesis Scores:")
            for player_id, score in trace.metadata["hypothesis_scores"].items():
                print(f"  {player_id}: {score:.3f}")

    # Render final state
    print()
    print(env.render(mode="text"))


if __name__ == "__main__":
    main()
