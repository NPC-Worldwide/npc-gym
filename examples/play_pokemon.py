#!/usr/bin/env python3
"""
Example: Play Pokemon with an LLM-guided agent.

This demonstrates:
- Emulator integration via PyBoy
- Vision processing for screen understanding
- LLM-based decision making
- Reward shaping for game progression

Requirements:
- PyBoy: pip install pyboy
- A Pokemon ROM file (not included)

Run: python examples/play_pokemon.py --rom path/to/pokemon.gbc
"""

import argparse
import random
import time

from npc_gym.core.env import Action


def random_agent(observation):
    """Simple random agent for testing."""
    valid = observation.valid_actions
    if valid:
        # Bias toward movement
        movement = ["up", "down", "left", "right"]
        movement_valid = [a for a in movement if a in valid]
        if movement_valid and random.random() < 0.7:
            choice = random.choice(movement_valid)
        else:
            choice = random.choice(valid)

        return Action(
            player_id=observation.player_id,
            action_type=choice,
            reasoning="Random exploration",
        )
    return None


def exploration_agent(observation):
    """Agent that prioritizes exploration."""
    game_state = observation.game_state
    valid = observation.valid_actions

    # Get current state
    in_battle = game_state.get("in_battle", False)

    if in_battle:
        # In battle - use A to attack, occasionally B to run
        if random.random() < 0.9:
            return Action(
                player_id=observation.player_id,
                action_type="a",
                reasoning="Attack in battle",
            )
        else:
            return Action(
                player_id=observation.player_id,
                action_type="b",
                reasoning="Try to run",
            )

    # Not in battle - explore
    movement = ["up", "down", "left", "right"]
    movement_valid = [a for a in movement if a in valid]

    if movement_valid:
        # Prefer directions we haven't gone recently
        choice = random.choice(movement_valid)
        return Action(
            player_id=observation.player_id,
            action_type=choice,
            reasoning=f"Exploring {choice}",
        )

    # Fallback
    if "a" in valid:
        return Action(
            player_id=observation.player_id,
            action_type="a",
            reasoning="Interact",
        )

    return Action(
        player_id=observation.player_id,
        action_type="none",
    )


def llm_agent(observation, npc=None):
    """
    LLM-guided agent using npcpy.

    Uses vision description + game state to make decisions.
    """
    if npc is None:
        # Fallback to exploration
        return exploration_agent(observation)

    # Build prompt from observation
    screen_desc = observation.game_state.get("screen_description", "")
    game_info = observation.info_partition.as_text()
    valid_actions = observation.valid_actions

    prompt = f"""You are playing Pokemon. Based on the current game state, choose the best action.

{game_info}

Screen description: {screen_desc}

Valid actions: {', '.join(valid_actions)}

Choose one action and explain briefly. Format: ACTION: <action>"""

    try:
        response = npc.get_llm_response(prompt)
        text = response.get('response', '')

        # Parse action from response
        for action in valid_actions:
            if action.lower() in text.lower():
                return Action(
                    player_id=observation.player_id,
                    action_type=action,
                    reasoning=text[:100],
                )
    except Exception as e:
        print(f"LLM error: {e}")

    return exploration_agent(observation)


def play_game(rom_path: str = None, agent_type: str = "exploration", verbose: bool = True):
    """Play Pokemon with specified agent."""
    # Import here to allow demo without pyboy
    from npc_gym.envs.emulator.pokemon import PokemonEnv, PokemonConfig, PokemonGame

    # Create config
    config = PokemonConfig(
        rom_path=rom_path or "",
        game_version=PokemonGame.CRYSTAL,
        window_type="SDL2" if rom_path else "null",
        emulation_speed=1 if rom_path else 0,  # Normal speed with display
        max_steps=1000,
        frames_per_step=5,
    )

    env = PokemonEnv(config=config)

    # Select agent
    if agent_type == "random":
        agent_fn = random_agent
    elif agent_type == "exploration":
        agent_fn = exploration_agent
    elif agent_type == "llm":
        try:
            from npcpy import NPC
            npc = NPC(name="pokemon_player", model="llama3.2")
            agent_fn = lambda obs: llm_agent(obs, npc)
        except ImportError:
            print("npcpy not installed, falling back to exploration agent")
            agent_fn = exploration_agent
    else:
        agent_fn = random_agent

    # Reset
    observations, info = env.reset()

    if verbose:
        print("\n=== Pokemon Game Started ===")
        if rom_path:
            print(f"ROM: {rom_path}")
        else:
            print("Running in mock mode (no ROM)")
        print(f"Agent: {agent_type}")
        print()

    done = False
    step = 0
    total_reward = 0

    try:
        while not done:
            step += 1
            player_id = env.player_ids[0]
            obs = observations[player_id]

            # Get action
            action = agent_fn(obs)
            if action is None:
                action = Action(player_id=player_id, action_type="none")

            # Step
            observations, rewards, terminated, truncated, info = env.step({player_id: action})
            done = terminated or truncated
            total_reward += rewards.get(player_id, 0)

            # Print progress
            if verbose and step % 50 == 0:
                print(f"Step {step}: {action.action_type}")
                print(env._render_text())
                print(f"Total reward: {total_reward:.2f}")
                print()

            # Small delay for visual display
            if rom_path and config.emulation_speed == 1:
                time.sleep(0.05)

    except KeyboardInterrupt:
        print("\nGame interrupted by user")

    finally:
        env.close()

    if verbose:
        print("\n=== Game Over ===")
        print(f"Total steps: {step}")
        print(f"Total reward: {total_reward:.2f}")
        print(f"Badges earned: {env.badges_earned}")
        print(f"Pokemon caught: {env.pokemon_caught}")
        print(f"Locations explored: {len(env.visited_locations)}")

    return total_reward


def demo_mock():
    """Demo without actual ROM."""
    print("\n=== Mock Pokemon Demo (No ROM) ===")
    print("This demonstrates the environment structure without requiring a ROM.\n")

    from npc_gym.envs.emulator.pokemon import PokemonEnv, PokemonConfig, PokemonGame

    config = PokemonConfig(
        game_version=PokemonGame.CRYSTAL,
        max_steps=20,
    )

    env = PokemonEnv(config=config)
    observations, info = env.reset()

    print("Environment created successfully!")
    print(f"Action space: {env.BUTTONS}")
    print()

    # Take a few steps
    for i in range(5):
        player_id = env.player_ids[0]
        action = Action(
            player_id=player_id,
            action_type=random.choice(env.BUTTONS),
        )

        observations, rewards, terminated, truncated, info = env.step({player_id: action})

        obs = observations[player_id]
        print(f"Step {i+1}: Action = {action.action_type}")
        print(f"  Screen: {obs.game_state.get('screen_description', 'N/A')[:50]}...")
        print(f"  Frame: {env.frame_count}")
        print()

    env.close()
    print("Mock demo complete!")


def demo_vision():
    """Demo vision processing capabilities."""
    print("\n=== Vision Processing Demo ===\n")

    from PIL import Image
    from npc_gym.envs.emulator.vision import VisionProcessor, VisionConfig, VisionBackend

    # Create a test image
    img = Image.new('RGB', (160, 144))
    pixels = img.load()

    # Draw some patterns
    for y in range(144):
        for x in range(160):
            if x < 80 and y < 72:
                pixels[x, y] = (0, 128, 0)  # Green (grass)
            elif x >= 80 and y < 72:
                pixels[x, y] = (0, 0, 200)  # Blue (water)
            elif y >= 100:
                pixels[x, y] = (128, 128, 128)  # Gray (menu)
            else:
                pixels[x, y] = (200, 200, 100)  # Yellow (sand)

    # Process with simple backend
    processor = VisionProcessor(VisionConfig(backend=VisionBackend.SIMPLE))
    scene = processor.process(img)

    print("Simple vision analysis:")
    print(scene.as_text())
    print()

    # Show regions
    print("Region analysis:")
    for region, desc in scene.regions.items():
        print(f"  {region}: {desc}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Play Pokemon with npc-gym")
    parser.add_argument("--rom", type=str, help="Path to Pokemon ROM file")
    parser.add_argument(
        "--agent",
        type=str,
        default="exploration",
        choices=["random", "exploration", "llm"],
        help="Agent type"
    )
    parser.add_argument("--mock", action="store_true", help="Run mock demo without ROM")
    parser.add_argument("--vision", action="store_true", help="Demo vision processing")

    args = parser.parse_args()

    if args.vision:
        demo_vision()
    elif args.mock or not args.rom:
        demo_mock()
    else:
        play_game(rom_path=args.rom, agent_type=args.agent)
