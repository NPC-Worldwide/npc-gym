#!/usr/bin/env python3
"""
Example: Streaming Partial Information Decomposition (PID) game.

This demonstrates:
- Text as cards dealt to agents
- Hypothesis formation from partial info
- Voting and synthesis

Run: python examples/streaming_pid_demo.py
"""

from npc_gym.streaming.processor import TextStream, StreamDeck, ChunkStrategy
from npc_gym.streaming.env import StreamingPIDEnv, StreamingConfig
from npc_gym.core.env import Action


def simple_hypothesis_agent(observation):
    """Agent that forms hypothesis from available fragments."""
    hand_text = observation.game_state.get("hand_text", "")
    public_text = observation.game_state.get("public_text", "")

    # Simple hypothesis: combine what we know
    combined = f"{hand_text} {public_text}".strip()

    if "hypothesize" in observation.valid_actions:
        return Action(
            player_id=observation.player_id,
            action_type="hypothesize",
            value=f"Based on fragments: {combined[:100]}",
            reasoning="Combining available information",
            confidence=0.6,
        )
    elif observation.valid_actions:
        return Action(
            player_id=observation.player_id,
            action_type=observation.valid_actions[0],
        )
    return None


def demo_text_stream():
    """Demonstrate text streaming."""
    print("=== Text Stream Demo ===\n")

    text = """
    The quick brown fox jumps over the lazy dog.
    This sentence contains all letters of the alphabet.
    Pangrams are useful for testing fonts and keyboards.
    """

    # Process by sentence
    stream = TextStream(strategy=ChunkStrategy.SENTENCE)
    chunks = list(stream.process(text))

    print(f"Original text ({len(text)} chars):")
    print(text[:100] + "...")
    print(f"\nChunked into {len(chunks)} sentences:")
    for i, chunk in enumerate(chunks):
        print(f"  {i+1}. [{chunk.chunk_type}] {chunk.content}")


def demo_stream_deck():
    """Demonstrate dealing text to players."""
    print("\n=== Stream Deck Demo ===\n")

    text = """
    In a hole in the ground there lived a hobbit.
    Not a nasty dirty wet hole filled with the ends of worms.
    Nor yet a dry bare sandy hole with nothing in it to sit down on.
    It was a hobbit hole and that means comfort.
    """

    player_ids = ["Alice", "Bob", "Charlie"]
    deck = StreamDeck(player_ids=player_ids)

    # Create deck from text
    num_chunks = deck.from_text(text, strategy=ChunkStrategy.SENTENCE)
    print(f"Created deck with {num_chunks} chunks from text")

    # Shuffle and deal
    deck.shuffle()
    deck.deal_all()

    print("\nAfter dealing:")
    for player_id in player_ids:
        hand = deck.get_hand(player_id)
        print(f"\n{player_id}'s hand ({len(hand)} fragments):")
        for chunk in hand:
            print(f"  - {chunk.content[:50]}...")


def demo_pid_game():
    """Demonstrate a PID game."""
    print("\n=== Streaming PID Game Demo ===\n")

    # Source text (the "truth" to be inferred)
    source_text = """
    The Amazon rainforest produces 20% of the world's oxygen.
    It is home to over 10 million species of plants and animals.
    Deforestation threatens this critical ecosystem.
    Scientists warn we must act now to protect it.
    """

    ground_truth = "Amazon rainforest is important for oxygen and biodiversity but faces deforestation threat"

    # Create environment
    config = StreamingConfig(
        chunk_strategy=ChunkStrategy.SENTENCE,
        chunks_per_round=1,
        rounds_before_hypothesis=3,
        hypothesis_rounds=2,
    )

    env = StreamingPIDEnv(
        text=source_text,
        ground_truth=ground_truth,
        config=config,
        player_ids=["Agent1", "Agent2", "Agent3"],
    )

    # Reset
    observations, info = env.reset()

    print(f"Game started with {len(env.player_ids)} players")
    print(f"Source has {len(source_text.split('.'))-1} sentences")
    print(f"Ground truth: {ground_truth}\n")

    # Run game
    done = False
    step = 0

    while not done:
        step += 1
        print(f"--- Round {step} (Phase: {env.current_phase.value}) ---")

        # Each agent acts
        actions = {}
        for player_id in env.player_ids:
            obs = observations[player_id]
            action = simple_hypothesis_agent(obs)
            if action:
                actions[player_id] = action

            # Show what each agent knows
            hand = obs.game_state.get("hand_chunks", [])
            print(f"  {player_id}: {len(hand)} fragments")
            if obs.game_state.get("my_hypotheses"):
                print(f"    Hypothesis: {obs.game_state['my_hypotheses'][-1][:60]}...")

        if not actions:
            break

        # StreamingPIDEnv returns 4 values (obs, rewards, done, info)
        result = env.step(actions)
        if len(result) == 5:
            observations, rewards, terminated, truncated, info = result
            done = terminated or truncated
        else:
            observations, rewards, done, info = result

        if step > 10:  # Safety limit
            break

    print("\n=== Game Over ===")
    print(f"Total rounds: {step}")
    print("\nFinal scores:")
    for player_id, score in env.scores.items():
        print(f"  {player_id}: {score:.2f}")


def demo_pid_pipeline():
    """Demonstrate the full PID proposer/voter/synthesizer pipeline."""
    print("\n=== PID Pipeline Demo ===\n")

    from npc_gym.pid.proposer import Proposer, ProposerConfig, ProposerEnsemble, Proposal
    from npc_gym.pid.voter import Voter, VoterConfig, VotingEnsemble
    from npc_gym.pid.synthesizer import Synthesizer, SynthesizerConfig, SynthesisStrategy

    # Create proposers (each gets different info fragments)
    fragments = [
        ["climate change", "rising temperatures"],
        ["sea levels", "coastal flooding"],
        ["extreme weather", "hurricanes"],
    ]

    # Simulate proposals (in reality these would come from LLMs)
    proposals = [
        Proposal(
            content="Climate change is causing global warming and temperature rise",
            confidence=0.8,
            proposer_id="proposer_0",
            info_used=fragments[0],
        ),
        Proposal(
            content="Rising sea levels are threatening coastal communities",
            confidence=0.7,
            proposer_id="proposer_1",
            info_used=fragments[1],
        ),
        Proposal(
            content="Climate change is increasing extreme weather events",
            confidence=0.75,
            proposer_id="proposer_2",
            info_used=fragments[2],
        ),
    ]

    print("Proposals from fragments:")
    for p in proposals:
        print(f"  [{p.confidence:.2f}] {p.content}")
        print(f"         from: {p.info_used}")

    # Vote on proposals
    voters = VotingEnsemble()
    voters.add_voter(Voter(VoterConfig(name="voter_1")))

    vote_result = voters.vote(proposals)
    print(f"\nVoting result: Winner index = {vote_result['winner_idx']}")
    print(f"Final scores: {vote_result['final_scores']}")

    # Synthesize
    synth = Synthesizer(SynthesizerConfig(name="synth", strategy=SynthesisStrategy.CONSENSUS))
    synthesis = synth.synthesize(proposals, vote_result)

    print(f"\nSynthesized result:")
    print(f"  Content: {synthesis.content}")
    print(f"  Confidence: {synthesis.confidence:.2f}")
    print(f"  Contributors: {synthesis.contributing_proposals}")


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        if sys.argv[1] == "stream":
            demo_text_stream()
        elif sys.argv[1] == "deck":
            demo_stream_deck()
        elif sys.argv[1] == "game":
            demo_pid_game()
        elif sys.argv[1] == "pipeline":
            demo_pid_pipeline()
        else:
            print("Usage: python streaming_pid_demo.py [stream|deck|game|pipeline]")
    else:
        # Run all demos
        demo_text_stream()
        demo_stream_deck()
        demo_pid_game()
        demo_pid_pipeline()
