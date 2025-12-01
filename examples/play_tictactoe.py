#!/usr/bin/env python3
"""
Example: Play TicTacToe with random agents or interactively.

This demonstrates:
- Creating an environment
- Running a game loop
- Processing observations and actions

Run: python examples/play_tictactoe.py
"""

import random
import npc_gym
from npc_gym.core.env import Action


def random_agent(observation):
    """Simple random agent that picks a valid action."""
    valid = observation.valid_actions
    # Filter to just index format for simplicity
    valid_indices = [a for a in valid if a.isdigit()]
    if valid_indices:
        choice = random.choice(valid_indices)
        return Action(
            player_id=observation.player_id,
            action_type=choice,
            reasoning="Random choice",
            confidence=0.5,
        )
    return None


def human_agent(observation):
    """Interactive human agent."""
    print("\n" + "=" * 40)
    print(observation.game_state.get("board_text", ""))
    print(f"\nYou are: {observation.game_state.get('my_symbol', '?')}")
    print(f"Valid moves: {[a for a in observation.valid_actions if a.isdigit()]}")

    while True:
        choice = input("Enter position (0-8): ").strip()
        if choice in observation.valid_actions:
            return Action(
                player_id=observation.player_id,
                action_type=choice,
            )
        print("Invalid move, try again.")


def play_game(agent1="random", agent2="random", verbose=True):
    """Play a game of TicTacToe."""
    # Create environment with two players
    env = npc_gym.make("TicTacToe-v1", player_ids=["Player1", "Player2"])

    # Select agents
    agents = {
        "Player1": human_agent if agent1 == "human" else random_agent,
        "Player2": human_agent if agent2 == "human" else random_agent,
    }

    # Reset and get initial observations
    observations, info = env.reset()

    if verbose:
        print("\n=== TicTacToe Game Started ===")
        print("Board positions:")
        print(" 0 | 1 | 2 ")
        print("-----------")
        print(" 3 | 4 | 5 ")
        print("-----------")
        print(" 6 | 7 | 8 ")

    done = False
    step = 0

    while not done:
        step += 1
        current_player = env.state.current_player
        obs = observations[current_player]

        # Get action from appropriate agent
        agent_fn = agents[current_player]
        action = agent_fn(obs)

        if action is None:
            print("No valid actions!")
            break

        if verbose:
            print(f"\n[Step {step}] {current_player} plays position {action.action_type}")

        # Take the step (returns 5 values per Gymnasium API)
        observations, rewards, terminated, truncated, info = env.step({current_player: action})
        done = terminated or truncated

        # Print current board
        if verbose:
            print(env._render_board())

    # Game over
    if verbose:
        print("\n" + "=" * 40)
        print("GAME OVER!")
        print(env._render_text())

        # Show final rewards
        print("\nFinal rewards:")
        for player_id, reward in rewards.items():
            result = "WIN" if reward > 0 else "LOSE" if reward < 0 else "DRAW"
            print(f"  {player_id}: {result} ({reward})")

    return rewards


def tournament(num_games=10):
    """Run a tournament between random agents."""
    print(f"\n=== Running {num_games} game tournament ===\n")

    wins = {"Player1": 0, "Player2": 0, "Draw": 0}

    for i in range(num_games):
        rewards = play_game(agent1="random", agent2="random", verbose=False)

        if rewards["Player1"] > rewards["Player2"]:
            wins["Player1"] += 1
        elif rewards["Player2"] > rewards["Player1"]:
            wins["Player2"] += 1
        else:
            wins["Draw"] += 1

        print(f"Game {i+1}: P1={rewards['Player1']:.0f}, P2={rewards['Player2']:.0f}")

    print(f"\n=== Tournament Results ({num_games} games) ===")
    print(f"Player1 wins: {wins['Player1']}")
    print(f"Player2 wins: {wins['Player2']}")
    print(f"Draws: {wins['Draw']}")


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        if sys.argv[1] == "human":
            play_game(agent1="human", agent2="random")
        elif sys.argv[1] == "tournament":
            n = int(sys.argv[2]) if len(sys.argv) > 2 else 10
            tournament(n)
        else:
            print("Usage: python play_tictactoe.py [human|tournament [n]]")
    else:
        # Default: single random game
        play_game()
