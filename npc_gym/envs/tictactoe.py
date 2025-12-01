"""
TicTacToe and board game environments for npc-gym.

Classic competitive games for testing:
- Pattern recognition (System 1)
- Strategic reasoning (System 2)
- Opponent modeling
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
from enum import Enum

from npc_gym.core.env import Environment, GameState, Observation, Action, Phase, Trace
from npc_gym.core.spaces import DiscreteSpace
from npc_gym.core.info import InfoPartition


class CellState(Enum):
    """Board cell states."""
    EMPTY = "."
    X = "X"
    O = "O"


@dataclass
class BoardGameConfig:
    """Configuration for board games."""
    board_size: int = 3
    win_length: int = 3  # How many in a row to win
    first_player_random: bool = True


class TicTacToe(Environment):
    """
    Classic TicTacToe for two players.

    A simple but complete competitive game that tests:
    - Pattern recognition for winning/blocking moves
    - Strategic planning
    - Opponent modeling

    Hybrid agents can use System 1 for common patterns
    (fork, block, center) and System 2 for complex positions.
    """

    env_id = "TicTacToe-v1"
    min_players = 2
    max_players = 2

    def __init__(
        self,
        config: BoardGameConfig = None,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.config = config or BoardGameConfig()

        # Board state
        self.board: List[List[CellState]] = []
        self.player_symbols: Dict[str, CellState] = {}
        self.move_history: List[Tuple[str, int, int]] = []

        # Action space: positions 0-8 for 3x3 board
        size = self.config.board_size
        self.action_space = DiscreteSpace(n=size * size)

        # Also allow "row,col" format
        self.valid_action_formats = ["index", "coordinates"]

    def _setup_game(self) -> GameState:
        """Initialize the board."""
        size = self.config.board_size
        self.board = [[CellState.EMPTY for _ in range(size)] for _ in range(size)]
        self.move_history = []

        # Assign symbols
        if self.config.first_player_random:
            import random
            symbols = [CellState.X, CellState.O]
            random.shuffle(symbols)
        else:
            symbols = [CellState.X, CellState.O]

        self.player_symbols = {
            self.player_ids[0]: symbols[0],
            self.player_ids[1]: symbols[1],
        }

        state = GameState(
            phase=Phase.PLAYING,
            step=0,
            current_player=self.player_ids[0],  # X goes first
            player_order=self.player_ids.copy(),
            player_states={
                pid: {"symbol": sym.value, "moves": 0}
                for pid, sym in self.player_symbols.items()
            },
            metadata={"board_size": size, "win_length": self.config.win_length}
        )

        return state

    def _get_observation(self, player_id: str) -> Observation:
        """Get observation for a player."""
        symbol = self.player_symbols[player_id]
        opponent_symbol = CellState.X if symbol == CellState.O else CellState.O

        # Board as text
        board_text = self._render_board()

        # Strategic analysis
        analysis = self._analyze_position(player_id)

        # Valid moves
        valid_moves = self._get_empty_positions()

        info_partition = InfoPartition(
            player_id=player_id,
            private=[f"You are {symbol.value}"],
            public=[board_text, f"Turn {self.state.step + 1}"],
        )

        game_state = {
            "board": [[c.value for c in row] for row in self.board],
            "my_symbol": symbol.value,
            "opponent_symbol": opponent_symbol.value,
            "move_count": len(self.move_history),
            "analysis": analysis,
            "board_text": board_text,
        }

        # Format valid actions as both indices and coordinates
        valid_actions = []
        for row, col in valid_moves:
            idx = row * self.config.board_size + col
            valid_actions.append(str(idx))
            valid_actions.append(f"{row},{col}")

        return Observation(
            player_id=player_id,
            info_partition=info_partition,
            game_state=game_state,
            valid_actions=valid_actions,
            step=self.state.step,
        )

    def _analyze_position(self, player_id: str) -> Dict[str, Any]:
        """Analyze the current position for strategic hints."""
        symbol = self.player_symbols[player_id]
        opponent_symbol = CellState.X if symbol == CellState.O else CellState.O

        analysis = {
            "winning_moves": [],
            "blocking_moves": [],
            "fork_opportunities": [],
            "center_available": False,
            "corner_available": [],
        }

        empty = self._get_empty_positions()
        size = self.config.board_size

        # Check for winning/blocking moves
        for row, col in empty:
            # Check if this move wins
            if self._would_win(row, col, symbol):
                analysis["winning_moves"].append((row, col))

            # Check if this move blocks opponent
            if self._would_win(row, col, opponent_symbol):
                analysis["blocking_moves"].append((row, col))

        # Center available?
        center = size // 2
        if self.board[center][center] == CellState.EMPTY:
            analysis["center_available"] = True

        # Corners available?
        corners = [(0, 0), (0, size-1), (size-1, 0), (size-1, size-1)]
        for r, c in corners:
            if self.board[r][c] == CellState.EMPTY:
                analysis["corner_available"].append((r, c))

        return analysis

    def _would_win(self, row: int, col: int, symbol: CellState) -> bool:
        """Check if placing symbol at (row, col) would win."""
        # Temporarily place
        original = self.board[row][col]
        self.board[row][col] = symbol

        # Check for win
        winner = self._check_winner()

        # Restore
        self.board[row][col] = original

        return winner == symbol

    def _get_empty_positions(self) -> List[Tuple[int, int]]:
        """Get list of empty board positions."""
        empty = []
        for row in range(self.config.board_size):
            for col in range(self.config.board_size):
                if self.board[row][col] == CellState.EMPTY:
                    empty.append((row, col))
        return empty

    def _get_valid_actions(self, player_id: str) -> List[str]:
        """Get valid actions as strings."""
        actions = []
        for row, col in self._get_empty_positions():
            idx = row * self.config.board_size + col
            actions.append(str(idx))
        return actions

    def _apply_action(self, action: Action) -> None:
        """Apply a player's move."""
        player_id = action.player_id
        symbol = self.player_symbols[player_id]

        # Parse action (can be index "4" or coordinates "1,1")
        action_str = action.action_type

        if "," in action_str:
            row, col = map(int, action_str.split(","))
        else:
            idx = int(action_str)
            row = idx // self.config.board_size
            col = idx % self.config.board_size

        # Validate and place
        if self.board[row][col] == CellState.EMPTY:
            self.board[row][col] = symbol
            self.move_history.append((player_id, row, col))
            self.state.player_states[player_id]["moves"] += 1

        # Switch player
        self._next_player()

    def _next_player(self) -> None:
        """Switch to other player."""
        current_idx = self.player_ids.index(self.state.current_player)
        next_idx = 1 - current_idx
        self.state.current_player = self.player_ids[next_idx]

    def _compute_rewards(self) -> Dict[str, float]:
        """Compute rewards based on game outcome."""
        winner = self._check_winner()

        if winner:
            # Find who has this symbol
            winner_id = None
            loser_id = None
            for pid, sym in self.player_symbols.items():
                if sym == winner:
                    winner_id = pid
                else:
                    loser_id = pid

            return {winner_id: 1.0, loser_id: -1.0}

        # Draw or ongoing
        if self._is_terminal():
            return {pid: 0.0 for pid in self.player_ids}

        return {pid: 0.0 for pid in self.player_ids}

    def _check_winner(self) -> Optional[CellState]:
        """Check if there's a winner."""
        size = self.config.board_size
        win_len = self.config.win_length

        # Check rows
        for row in range(size):
            for col in range(size - win_len + 1):
                cells = [self.board[row][col + i] for i in range(win_len)]
                if cells[0] != CellState.EMPTY and all(c == cells[0] for c in cells):
                    return cells[0]

        # Check columns
        for col in range(size):
            for row in range(size - win_len + 1):
                cells = [self.board[row + i][col] for i in range(win_len)]
                if cells[0] != CellState.EMPTY and all(c == cells[0] for c in cells):
                    return cells[0]

        # Check diagonals
        for row in range(size - win_len + 1):
            for col in range(size - win_len + 1):
                # Down-right diagonal
                cells = [self.board[row + i][col + i] for i in range(win_len)]
                if cells[0] != CellState.EMPTY and all(c == cells[0] for c in cells):
                    return cells[0]

        for row in range(size - win_len + 1):
            for col in range(win_len - 1, size):
                # Down-left diagonal
                cells = [self.board[row + i][col - i] for i in range(win_len)]
                if cells[0] != CellState.EMPTY and all(c == cells[0] for c in cells):
                    return cells[0]

        return None

    def _is_terminal(self) -> bool:
        """Check if game is over."""
        # Winner exists
        if self._check_winner():
            return True

        # Board full (draw)
        if not self._get_empty_positions():
            return True

        return False

    def _render_board(self) -> str:
        """Render board as text."""
        size = self.config.board_size
        lines = []

        # Column headers
        header = "    " + "   ".join(str(i) for i in range(size))
        lines.append(header)
        lines.append("  " + "-" * (size * 4 - 1))

        for row in range(size):
            cells = " | ".join(self.board[row][col].value for col in range(size))
            lines.append(f"{row} | {cells} |")
            if row < size - 1:
                lines.append("  " + "-" * (size * 4 - 1))

        lines.append("  " + "-" * (size * 4 - 1))

        return "\n".join(lines)

    def _render_text(self) -> str:
        """Full text render."""
        lines = [f"=== TicTacToe (Move {len(self.move_history)}) ==="]
        lines.append(self._render_board())
        lines.append("")

        for pid, sym in self.player_symbols.items():
            marker = " <--" if pid == self.state.current_player else ""
            lines.append(f"{pid}: {sym.value}{marker}")

        winner = self._check_winner()
        if winner:
            winner_id = [p for p, s in self.player_symbols.items() if s == winner][0]
            lines.append(f"\n*** {winner_id} WINS! ***")
        elif self._is_terminal():
            lines.append("\n*** DRAW ***")

        return "\n".join(lines)


@dataclass
class ConnectFourConfig(BoardGameConfig):
    """Configuration for Connect Four."""
    board_size: int = 7  # Width
    board_height: int = 6
    win_length: int = 4


class ConnectFour(Environment):
    """
    Connect Four: gravity-based board game.

    Pieces fall to the lowest available row in each column.
    First to connect 4 in a row wins.

    Good for testing:
    - Spatial pattern recognition
    - Multi-step planning
    - Threat detection
    """

    env_id = "ConnectFour-v1"
    min_players = 2
    max_players = 2

    def __init__(
        self,
        config: ConnectFourConfig = None,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.config = config or ConnectFourConfig()

        # Board state
        self.board: List[List[CellState]] = []
        self.player_symbols: Dict[str, CellState] = {}
        self.column_heights: List[int] = []  # Track top of each column
        self.move_history: List[Tuple[str, int]] = []  # (player, column)

        # Action space: column indices
        self.action_space = DiscreteSpace(n=self.config.board_size)

    def _setup_game(self) -> GameState:
        """Initialize the board."""
        width = self.config.board_size
        height = self.config.board_height

        self.board = [[CellState.EMPTY for _ in range(width)] for _ in range(height)]
        self.column_heights = [0] * width
        self.move_history = []

        # Assign symbols
        if self.config.first_player_random:
            import random
            symbols = [CellState.X, CellState.O]
            random.shuffle(symbols)
        else:
            symbols = [CellState.X, CellState.O]

        self.player_symbols = {
            self.player_ids[0]: symbols[0],
            self.player_ids[1]: symbols[1],
        }

        state = GameState(
            phase=Phase.PLAYING,
            step=0,
            current_player=self.player_ids[0],
            player_order=self.player_ids.copy(),
            player_states={
                pid: {"symbol": sym.value, "moves": 0}
                for pid, sym in self.player_symbols.items()
            },
            metadata={"width": width, "height": height, "win_length": self.config.win_length}
        )

        return state

    def _get_observation(self, player_id: str) -> Observation:
        """Get observation for a player."""
        symbol = self.player_symbols[player_id]
        opponent_symbol = CellState.X if symbol == CellState.O else CellState.O

        board_text = self._render_board()
        analysis = self._analyze_position(player_id)
        valid_cols = self._get_valid_columns()

        info_partition = InfoPartition(
            player_id=player_id,
            private=[f"You are {symbol.value}"],
            public=[board_text],
        )

        game_state = {
            "board": [[c.value for c in row] for row in self.board],
            "my_symbol": symbol.value,
            "opponent_symbol": opponent_symbol.value,
            "column_heights": self.column_heights.copy(),
            "move_count": len(self.move_history),
            "analysis": analysis,
            "board_text": board_text,
        }

        return Observation(
            player_id=player_id,
            info_partition=info_partition,
            game_state=game_state,
            valid_actions=[str(c) for c in valid_cols],
            step=self.state.step,
        )

    def _analyze_position(self, player_id: str) -> Dict[str, Any]:
        """Analyze current position."""
        symbol = self.player_symbols[player_id]
        opponent = CellState.X if symbol == CellState.O else CellState.O

        analysis = {
            "winning_moves": [],
            "blocking_moves": [],
            "threats": [],
            "center_control": 0,
        }

        valid_cols = self._get_valid_columns()

        for col in valid_cols:
            # Check winning/blocking
            if self._would_win(col, symbol):
                analysis["winning_moves"].append(col)
            if self._would_win(col, opponent):
                analysis["blocking_moves"].append(col)

        # Center column control
        center = self.config.board_size // 2
        for row in range(self.config.board_height):
            if self.board[row][center] == symbol:
                analysis["center_control"] += 1
            elif self.board[row][center] == opponent:
                analysis["center_control"] -= 1

        return analysis

    def _would_win(self, col: int, symbol: CellState) -> bool:
        """Check if dropping in this column would win."""
        if self.column_heights[col] >= self.config.board_height:
            return False

        row = self.config.board_height - 1 - self.column_heights[col]

        # Temporarily place
        self.board[row][col] = symbol

        winner = self._check_winner()

        # Restore
        self.board[row][col] = CellState.EMPTY

        return winner == symbol

    def _get_valid_columns(self) -> List[int]:
        """Get columns that aren't full."""
        return [c for c in range(self.config.board_size)
                if self.column_heights[c] < self.config.board_height]

    def _get_valid_actions(self, player_id: str) -> List[str]:
        """Get valid actions."""
        return [str(c) for c in self._get_valid_columns()]

    def _apply_action(self, action: Action) -> None:
        """Drop a piece in the chosen column."""
        player_id = action.player_id
        symbol = self.player_symbols[player_id]
        col = int(action.action_type)

        if col in self._get_valid_columns():
            # Find lowest empty row
            row = self.config.board_height - 1 - self.column_heights[col]
            self.board[row][col] = symbol
            self.column_heights[col] += 1
            self.move_history.append((player_id, col))
            self.state.player_states[player_id]["moves"] += 1

        # Switch player
        current_idx = self.player_ids.index(self.state.current_player)
        self.state.current_player = self.player_ids[1 - current_idx]

    def _compute_rewards(self) -> Dict[str, float]:
        """Compute rewards."""
        winner = self._check_winner()

        if winner:
            winner_id = None
            loser_id = None
            for pid, sym in self.player_symbols.items():
                if sym == winner:
                    winner_id = pid
                else:
                    loser_id = pid
            return {winner_id: 1.0, loser_id: -1.0}

        if self._is_terminal():
            return {pid: 0.0 for pid in self.player_ids}

        return {pid: 0.0 for pid in self.player_ids}

    def _check_winner(self) -> Optional[CellState]:
        """Check for a winner."""
        height = self.config.board_height
        width = self.config.board_size
        win_len = self.config.win_length

        # Check horizontal
        for row in range(height):
            for col in range(width - win_len + 1):
                cells = [self.board[row][col + i] for i in range(win_len)]
                if cells[0] != CellState.EMPTY and all(c == cells[0] for c in cells):
                    return cells[0]

        # Check vertical
        for col in range(width):
            for row in range(height - win_len + 1):
                cells = [self.board[row + i][col] for i in range(win_len)]
                if cells[0] != CellState.EMPTY and all(c == cells[0] for c in cells):
                    return cells[0]

        # Check diagonal (down-right)
        for row in range(height - win_len + 1):
            for col in range(width - win_len + 1):
                cells = [self.board[row + i][col + i] for i in range(win_len)]
                if cells[0] != CellState.EMPTY and all(c == cells[0] for c in cells):
                    return cells[0]

        # Check diagonal (down-left)
        for row in range(height - win_len + 1):
            for col in range(win_len - 1, width):
                cells = [self.board[row + i][col - i] for i in range(win_len)]
                if cells[0] != CellState.EMPTY and all(c == cells[0] for c in cells):
                    return cells[0]

        return None

    def _is_terminal(self) -> bool:
        """Check if game is over."""
        if self._check_winner():
            return True

        # All columns full
        if not self._get_valid_columns():
            return True

        return False

    def _render_board(self) -> str:
        """Render board as text."""
        lines = []
        width = self.config.board_size

        # Column headers
        header = " " + " ".join(str(i) for i in range(width))
        lines.append(header)

        for row in self.board:
            cells = "|" + "|".join(c.value for c in row) + "|"
            lines.append(cells)

        # Bottom border
        lines.append("+" + "-" * (width * 2 - 1) + "+")

        return "\n".join(lines)

    def _render_text(self) -> str:
        """Full text render."""
        lines = [f"=== Connect Four (Move {len(self.move_history)}) ==="]
        lines.append(self._render_board())
        lines.append("")

        for pid, sym in self.player_symbols.items():
            marker = " <--" if pid == self.state.current_player else ""
            lines.append(f"{pid}: {sym.value}{marker}")

        winner = self._check_winner()
        if winner:
            winner_id = [p for p, s in self.player_symbols.items() if s == winner][0]
            lines.append(f"\n*** {winner_id} WINS! ***")
        elif self._is_terminal():
            lines.append("\n*** DRAW ***")

        return "\n".join(lines)
