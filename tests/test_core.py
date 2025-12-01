"""Tests for core npc-gym functionality."""

import pytest
from npc_gym.core.env import Environment, GameState, Observation, Action, Phase
from npc_gym.core.spaces import (
    DiscreteSpace, ContinuousSpace, TextSpace,
    DeckSpace, CardSpace, CompositeSpace, Card
)
from npc_gym.core.info import InfoPartition, InformationStructure
from npc_gym.core.agent import Agent, AgentConfig, AgentResponse


class TestSpaces:
    """Test space implementations."""

    def test_discrete_space(self):
        # With n
        space = DiscreteSpace(n=5)
        assert len(space.choices) == 5
        assert space.contains(0)
        assert space.contains(4)
        assert not space.contains(5)
        assert space.sample() in range(5)

        # With choices
        space = DiscreteSpace(choices=["fold", "call", "raise"])
        assert "fold" in space.choices
        assert space.contains("call")
        assert not space.contains("all-in")

    def test_continuous_space(self):
        space = ContinuousSpace(low=0, high=100, shape=(2,))
        sample = space.sample()
        assert len(sample) == 2
        assert all(0 <= v <= 100 for v in sample)
        assert space.contains([50, 50])
        assert not space.contains([150, 50])

    def test_text_space(self):
        # Without vocab restriction
        space = TextSpace(max_length=100)
        assert space.contains("hello world")
        assert space.contains("any text works")

        # With vocab restriction
        space_vocab = TextSpace(max_length=100, vocab=["hello", "world"])
        assert space_vocab.contains("hello world")
        assert not space_vocab.contains("hello test")  # "test" not in vocab

    def test_card(self):
        # Basic card
        card = Card(value="A", suit="hearts")
        assert card.value == "A"
        assert card.suit == "hearts"

        # Standard deck
        deck = Card.standard_deck()
        assert len(deck) == 52

        # From text
        cards = Card.from_text("Hello world. This is a test.", chunk_by="sentence")
        assert len(cards) == 2
        assert cards[0].value == "Hello world"

    def test_deck_space(self):
        # Standard deck
        deck = DeckSpace(card_type="standard")
        assert len(deck.cards) == 52

        # Deal cards
        deck.shuffle()
        hand = deck.deal(5)
        assert len(hand) == 5
        assert len(deck.cards) == 47

        # Draw single card
        card = deck.draw()
        assert card is not None
        assert len(deck.cards) == 46

        # Contains
        assert deck.contains(deck.cards[0])

    def test_deck_from_text(self):
        text = "Hello world. This is a test. Final sentence."
        deck = DeckSpace.from_text(text, chunk_by="sentence")
        assert len(deck.cards) == 3
        assert deck.cards[0].value == "Hello world"

    def test_card_space(self):
        space = CardSpace(card_type="standard")
        card = space.sample()
        assert card.value in Card.RANKS
        assert card.suit in Card.SUITS

    def test_composite_space(self):
        space = CompositeSpace({
            "action": DiscreteSpace(choices=["fold", "call"]),
            "amount": ContinuousSpace(0, 100, shape=(1,)),
        })

        sample = space.sample()
        assert "action" in sample
        assert "amount" in sample

        # Contains check - need to match shapes
        import numpy as np
        assert space.contains({"action": "fold", "amount": np.array([50.0], dtype=np.float32)})


class TestInfoPartition:
    """Test information structures."""

    def test_info_partition(self):
        partition = InfoPartition(
            player_id="p1",
            private=["my hand is AA"],
            public=["pot is $100"],
        )

        assert "my hand is AA" in partition.private
        assert "pot is $100" in partition.public

    def test_information_structure(self):
        info = InformationStructure(player_ids=["p1", "p2"])
        # Add some info through the API
        assert info is not None
        assert len(info.player_ids) == 2


class TestGameState:
    """Test game state."""

    def test_game_state_creation(self):
        state = GameState(
            phase=Phase.SETUP,
            step=0,
            current_player="p1",
            player_order=["p1", "p2", "p3"],
        )

        assert state.phase == Phase.SETUP
        assert state.current_player == "p1"
        assert len(state.player_order) == 3

    def test_game_state_copy(self):
        state = GameState(
            phase=Phase.PLAYING,
            step=5,
            current_player="p2",
            player_order=["p1", "p2"],
            player_states={"p1": {"chips": 100}, "p2": {"chips": 200}},
        )

        # Use clone() method instead of copy()
        cloned = state.clone()
        cloned.player_states["p1"]["chips"] = 50

        # Original should be unchanged
        assert state.player_states["p1"]["chips"] == 100


class TestObservation:
    """Test observation structure."""

    def test_observation_creation(self):
        partition = InfoPartition(player_id="p1", private=["test"])
        obs = Observation(
            player_id="p1",
            info_partition=partition,
            game_state={"pot": 100},
            valid_actions=["fold", "call", "raise"],
            step=1,
        )

        assert obs.player_id == "p1"
        assert "fold" in obs.valid_actions
        assert obs.game_state["pot"] == 100


class TestAction:
    """Test action structure."""

    def test_action_creation(self):
        action = Action(
            player_id="p1",
            action_type="raise",
            value=50,
            reasoning="Strong hand",
            confidence=0.8,
        )

        assert action.player_id == "p1"
        assert action.action_type == "raise"
        assert action.value == 50
        assert action.confidence == 0.8


class TestAgent:
    """Test agent implementations."""

    def test_agent_config(self):
        config = AgentConfig(
            name="TestAgent",
            model="gpt-4",
            provider="openai",
        )
        assert config.name == "TestAgent"

    def test_agent_response(self):
        response = AgentResponse(
            action_type="call",
            value=None,
            reasoning="Pot odds are good",
            confidence=0.7,
            system_used="llm",
            response_time=0.5,
        )

        assert response.action_type == "call"
        assert response.system_used == "llm"


class TestPhases:
    """Test game phases."""

    def test_phase_values(self):
        # Phase enum has: SETUP, PLAYING, SHOWDOWN, TERMINAL
        assert Phase.SETUP.value == "setup"
        assert Phase.PLAYING.value == "playing"
        assert Phase.SHOWDOWN.value == "showdown"
        assert Phase.TERMINAL.value == "terminal"
