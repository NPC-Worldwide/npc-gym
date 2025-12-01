"""Tests for npc-gym environments."""

import pytest
from npc_gym.core.env import Action, Phase


class TestGridWorld:
    """Test GridWorld environment."""

    def test_gridworld_creation(self):
        from npc_gym.envs.grid_world import GridWorld, GridWorldConfig

        config = GridWorldConfig(width=5, height=5, num_items=2, num_traps=1)
        env = GridWorld(config=config, player_ids=["agent1"])

        obs, info = env.reset()
        assert "agent1" in obs
        assert obs["agent1"].player_id == "agent1"
        assert "position" in obs["agent1"].game_state

    def test_gridworld_movement(self):
        from npc_gym.envs.grid_world import GridWorld, GridWorldConfig

        config = GridWorldConfig(width=10, height=10)
        env = GridWorld(config=config, player_ids=["agent1"])
        env.reset()

        # Get initial position
        initial_pos = env.agents["agent1"].position

        # Take a valid action
        valid_actions = env._get_valid_actions("agent1")
        if "down" in valid_actions:
            action = Action(player_id="agent1", action_type="down")
            # Step returns 5 values: obs, rewards, terminated, truncated, info
            obs, rewards, terminated, truncated, info = env.step({"agent1": action})
            assert isinstance(obs, dict)

    def test_maze_variant(self):
        from npc_gym.envs.grid_world import Maze

        env = Maze(player_ids=["agent1"])
        obs, info = env.reset()
        assert env.config.view_radius == 1  # Limited visibility


class TestTicTacToe:
    """Test TicTacToe environment."""

    def test_tictactoe_creation(self):
        from npc_gym.envs.tictactoe import TicTacToe

        env = TicTacToe(player_ids=["x_player", "o_player"])
        obs, info = env.reset()

        assert len(obs) == 2
        assert "x_player" in obs
        assert "o_player" in obs

    def test_tictactoe_valid_actions(self):
        from npc_gym.envs.tictactoe import TicTacToe

        env = TicTacToe(player_ids=["p1", "p2"])
        obs, _ = env.reset()

        # Initially all 9 positions should be available (in 2 formats)
        valid = obs["p1"].valid_actions
        assert len(valid) == 18  # 9 positions * 2 formats (index and coords)

    def test_tictactoe_gameplay(self):
        from npc_gym.envs.tictactoe import TicTacToe

        env = TicTacToe(player_ids=["p1", "p2"])
        env.reset()

        # Play a few moves
        moves = ["4", "0", "1", "2", "7"]
        done = False

        for i, move in enumerate(moves):
            if done:
                break
            current = env.state.current_player
            action = Action(player_id=current, action_type=move)
            # Step returns 5 values
            obs, rewards, terminated, truncated, info = env.step({current: action})
            done = terminated or truncated

        assert env.state.step <= 9

    def test_tictactoe_win_detection(self):
        from npc_gym.envs.tictactoe import TicTacToe, CellState

        env = TicTacToe(player_ids=["p1", "p2"])
        env.reset()

        # Manually set up a winning position
        env.board[0][0] = CellState.X
        env.board[0][1] = CellState.X
        env.board[0][2] = CellState.X

        winner = env._check_winner()
        assert winner == CellState.X


class TestConnectFour:
    """Test ConnectFour environment."""

    def test_connectfour_creation(self):
        from npc_gym.envs.tictactoe import ConnectFour

        env = ConnectFour(player_ids=["p1", "p2"])
        obs, info = env.reset()

        assert len(obs) == 2
        assert env.config.board_size == 7
        assert env.config.board_height == 6

    def test_connectfour_gravity(self):
        from npc_gym.envs.tictactoe import ConnectFour, CellState

        env = ConnectFour(player_ids=["p1", "p2"])
        env.reset()

        # Drop in column 3
        action = Action(player_id="p1", action_type="3")
        env.step({"p1": action})

        # Piece should be at bottom
        bottom_row = env.config.board_height - 1
        assert env.board[bottom_row][3] != CellState.EMPTY


class TestStreamingEnv:
    """Test streaming PID environment."""

    def test_streaming_creation(self):
        from npc_gym.streaming.env import StreamingPIDEnv, StreamingConfig

        text = "The quick brown fox jumps over the lazy dog."
        env = StreamingPIDEnv(
            text=text,
            ground_truth="A fox jumping",
            config=StreamingConfig(chunks_per_round=1),
            player_ids=["p1", "p2"],
        )

        obs, info = env.reset()
        assert len(obs) == 2

    def test_streaming_text_processing(self):
        from npc_gym.streaming.processor import TextStream, ChunkStrategy

        text = "First sentence. Second sentence. Third one."
        stream = TextStream(strategy=ChunkStrategy.SENTENCE)

        chunks = list(stream.process(text))
        assert len(chunks) == 3
        assert chunks[0].content == "First sentence."

    def test_stream_deck(self):
        from npc_gym.streaming.processor import StreamDeck, ChunkStrategy

        deck = StreamDeck(player_ids=["p1", "p2", "p3"])
        text = "Word1 word2 word3 word4 word5 word6"
        num = deck.from_text(text, strategy=ChunkStrategy.WORD)

        assert num == 6
        assert deck.deck_size() == 6

        # Deal
        deck.deal_round()
        assert deck.deck_size() == 3  # 6 - 3 players


class TestEnvironmentRegistry:
    """Test environment registration and creation."""

    def test_make_function(self):
        import npc_gym

        env = npc_gym.make("TicTacToe-v1")
        assert env is not None
        assert env.env_id == "TicTacToe-v1"

    def test_list_envs(self):
        import npc_gym

        envs = npc_gym.list_envs()
        assert "TicTacToe-v1" in envs
        assert "GridWorld-v1" in envs
        assert "InfoPoker-v1" in envs

    def test_invalid_env(self):
        import npc_gym

        with pytest.raises(ValueError):
            npc_gym.make("NonexistentEnv-v1")
