#!/usr/bin/env python3
"""
Example: Navigate a GridWorld with partial observability.

This demonstrates:
- Spatial navigation environment
- Partial observability (fog of war)
- Multi-agent support

Run: python examples/play_gridworld.py
"""

import random
import npc_gym
from npc_gym.core.env import Action
from npc_gym.envs.grid_world import GridWorld, GridWorldConfig


def random_agent(observation):
    """Simple random agent."""
    valid = observation.valid_actions
    if valid:
        return Action(
            player_id=observation.player_id,
            action_type=random.choice(valid),
        )
    return None


def greedy_agent(observation):
    """Agent that moves toward the goal."""
    game_state = observation.game_state
    goal_dir = game_state.get("goal_direction", "stay")

    # Map direction to action
    dir_map = {
        "north": "up",
        "south": "down",
        "east": "right",
        "west": "left",
        "here": "stay",
    }

    preferred = dir_map.get(goal_dir, "stay")

    if preferred in observation.valid_actions:
        return Action(
            player_id=observation.player_id,
            action_type=preferred,
            reasoning=f"Moving {preferred} toward goal ({goal_dir})",
        )

    # Fallback to random valid action
    return random_agent(observation)


def human_agent(observation):
    """Interactive human agent."""
    print("\n" + "=" * 50)
    print(observation.game_state.get("local_view", ""))
    print(f"\nPosition: {observation.game_state.get('position')}")
    print(f"Score: {observation.game_state.get('score'):.1f}")
    print(f"Items collected: {observation.game_state.get('items_collected')}")
    print(f"Goal direction: {observation.game_state.get('goal_direction')}")
    print(f"Nearby: {observation.game_state.get('nearby')}")
    print(f"\nValid actions: {observation.valid_actions}")

    while True:
        choice = input("Enter action (up/down/left/right/stay): ").strip().lower()
        if choice in observation.valid_actions:
            return Action(
                player_id=observation.player_id,
                action_type=choice,
            )
        # Allow shortcuts
        shortcuts = {"u": "up", "d": "down", "l": "left", "r": "right", "s": "stay"}
        if choice in shortcuts and shortcuts[choice] in observation.valid_actions:
            return Action(
                player_id=observation.player_id,
                action_type=shortcuts[choice],
            )
        print("Invalid action, try again.")


def play_game(agent_type="greedy", num_agents=1, verbose=True):
    """Play a GridWorld game."""
    # Create configuration
    config = GridWorldConfig(
        width=10,
        height=10,
        wall_density=0.15,
        num_items=5,
        num_traps=3,
        view_radius=2,
        fog_of_war=True,
        max_steps=100,
    )

    player_ids = [f"Agent{i+1}" for i in range(num_agents)]
    env = GridWorld(config=config, player_ids=player_ids)

    # Select agent type
    if agent_type == "human":
        agent_fn = human_agent
    elif agent_type == "greedy":
        agent_fn = greedy_agent
    else:
        agent_fn = random_agent

    # Reset
    observations, info = env.reset()

    if verbose:
        print("\n=== GridWorld Game Started ===")
        print(f"Grid size: {config.width}x{config.height}")
        print(f"Goal: Reach G, collect items (I), avoid traps (T)")
        print(f"Legend: @ = you, # = wall, . = empty, ? = unexplored")
        print(env._render_text())

    done = False
    step = 0

    while not done:
        step += 1

        # Each agent takes action
        actions = {}
        for player_id in player_ids:
            obs = observations[player_id]
            action = agent_fn(obs)
            if action:
                actions[player_id] = action

        if not actions:
            break

        # Step
        observations, rewards, terminated, truncated, info = env.step(actions)
        done = terminated or truncated

        if verbose and step % 5 == 0:  # Print every 5 steps
            print(f"\n--- Step {step} ---")
            print(env._render_text())

    # Game over
    if verbose:
        print("\n" + "=" * 50)
        print("GAME OVER!")
        print(env._render_text())
        print(f"\nSteps taken: {step}")
        print("\nFinal scores:")
        for player_id in player_ids:
            agent = env.agents[player_id]
            print(f"  {player_id}: {agent.score:.1f} (items: {agent.items_collected}, reached_goal: {agent.reached_goal})")

    return {pid: env.agents[pid].score for pid in player_ids}


def compare_agents(num_games=20):
    """Compare different agent strategies."""
    print("\n=== Comparing Agent Strategies ===\n")

    results = {"random": [], "greedy": []}

    for agent_type in results.keys():
        print(f"Testing {agent_type} agent...")
        for i in range(num_games):
            scores = play_game(agent_type=agent_type, verbose=False)
            results[agent_type].append(list(scores.values())[0])

    print("\n=== Results ===")
    for agent_type, scores in results.items():
        avg = sum(scores) / len(scores)
        best = max(scores)
        print(f"{agent_type:10s}: avg={avg:6.1f}, best={best:6.1f}")


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        if sys.argv[1] == "human":
            play_game(agent_type="human")
        elif sys.argv[1] == "compare":
            n = int(sys.argv[2]) if len(sys.argv) > 2 else 20
            compare_agents(n)
        elif sys.argv[1] == "multi":
            play_game(agent_type="greedy", num_agents=2)
        else:
            print("Usage: python play_gridworld.py [human|compare [n]|multi]")
    else:
        # Default: single greedy game
        play_game(agent_type="greedy")
