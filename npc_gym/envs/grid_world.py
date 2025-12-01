"""
GridWorld: Spatial navigation environment for hybrid agents.

A classic RL environment adapted for LLM+ML agents:
- Grid-based navigation with partial observability
- Goals, obstacles, items to collect
- Multi-agent support (cooperative or competitive)
- Text descriptions of visual state for LLM reasoning
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Set
from enum import Enum
import random

from npc_gym.core.env import Environment, GameState, Observation, Action, Phase, Trace
from npc_gym.core.spaces import DiscreteSpace, CompositeSpace, ContinuousSpace
from npc_gym.core.info import InfoPartition


class CellType(Enum):
    """Types of grid cells."""
    EMPTY = "."
    WALL = "#"
    GOAL = "G"
    ITEM = "I"
    TRAP = "T"
    START = "S"
    FOG = "?"


class Direction(Enum):
    """Movement directions."""
    UP = "up"
    DOWN = "down"
    LEFT = "left"
    RIGHT = "right"
    STAY = "stay"


@dataclass
class GridWorldConfig:
    """Configuration for GridWorld environment."""
    # Grid dimensions
    width: int = 10
    height: int = 10

    # Generation
    wall_density: float = 0.2
    num_items: int = 3
    num_traps: int = 2
    random_start: bool = True

    # Visibility
    view_radius: int = 2  # How far each agent can see
    fog_of_war: bool = True  # Hide unseen areas

    # Rewards
    goal_reward: float = 100.0
    item_reward: float = 10.0
    trap_penalty: float = -20.0
    step_penalty: float = -0.1
    wall_penalty: float = -1.0

    # Game rules
    max_steps: int = 200
    multi_agent_mode: str = "competitive"  # "competitive", "cooperative", "independent"
    respawn_items: bool = False


@dataclass
class AgentState:
    """State for an agent in the grid."""
    player_id: str
    x: int = 0
    y: int = 0
    score: float = 0.0
    items_collected: int = 0
    steps_taken: int = 0
    reached_goal: bool = False
    visited: Set[Tuple[int, int]] = field(default_factory=set)

    def position(self) -> Tuple[int, int]:
        return (self.x, self.y)


class GridWorld(Environment):
    """
    GridWorld environment for spatial reasoning.

    Agents navigate a grid, collecting items and avoiding traps
    while trying to reach the goal. Supports partial observability
    where agents only see cells within their view radius.

    Observations include both grid state and text descriptions
    for hybrid System 1/2 agents.
    """

    env_id = "GridWorld-v1"
    min_players = 1
    max_players = 4

    def __init__(
        self,
        config: GridWorldConfig = None,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.config = config or GridWorldConfig()

        # Grid state
        self.grid: List[List[CellType]] = []
        self.agents: Dict[str, AgentState] = {}
        self.goal_pos: Tuple[int, int] = (0, 0)
        self.item_positions: Set[Tuple[int, int]] = set()
        self.trap_positions: Set[Tuple[int, int]] = set()

        # Action space
        self.action_space = DiscreteSpace(
            choices=[d.value for d in Direction]
        )

        # Observation space
        self.observation_space = CompositeSpace({
            "position": ContinuousSpace(0, max(self.config.width, self.config.height), shape=(2,)),
            "local_view": DiscreteSpace(n=(2 * self.config.view_radius + 1) ** 2),
            "direction_to_goal": DiscreteSpace(choices=["north", "south", "east", "west", "here"]),
        })

    def _setup_game(self) -> GameState:
        """Initialize the grid world."""
        # Generate grid
        self._generate_grid()

        # Place agents
        self.agents = {}
        start_positions = self._get_start_positions(len(self.player_ids))

        for i, player_id in enumerate(self.player_ids):
            pos = start_positions[i]
            self.agents[player_id] = AgentState(
                player_id=player_id,
                x=pos[0],
                y=pos[1],
            )
            self.agents[player_id].visited.add(pos)

        state = GameState(
            phase=Phase.PLAYING,
            step=0,
            current_player=self.player_ids[0],
            player_order=self.player_ids.copy(),
            player_states={pid: {"x": a.x, "y": a.y, "score": a.score}
                          for pid, a in self.agents.items()},
            metadata={"max_steps": self.config.max_steps}
        )

        return state

    def _generate_grid(self) -> None:
        """Generate the grid with walls, items, traps, and goal."""
        w, h = self.config.width, self.config.height

        # Start with empty grid
        self.grid = [[CellType.EMPTY for _ in range(w)] for _ in range(h)]

        # Add walls around border
        for x in range(w):
            self.grid[0][x] = CellType.WALL
            self.grid[h-1][x] = CellType.WALL
        for y in range(h):
            self.grid[y][0] = CellType.WALL
            self.grid[y][w-1] = CellType.WALL

        # Add random internal walls
        num_walls = int((w - 2) * (h - 2) * self.config.wall_density)
        empty_cells = [(x, y) for x in range(1, w-1) for y in range(1, h-1)]
        random.shuffle(empty_cells)

        for i in range(min(num_walls, len(empty_cells))):
            x, y = empty_cells[i]
            self.grid[y][x] = CellType.WALL

        # Get remaining empty cells
        empty_cells = [(x, y) for x in range(1, w-1) for y in range(1, h-1)
                      if self.grid[y][x] == CellType.EMPTY]
        random.shuffle(empty_cells)

        # Place goal
        if empty_cells:
            gx, gy = empty_cells.pop()
            self.goal_pos = (gx, gy)
            self.grid[gy][gx] = CellType.GOAL

        # Place items
        self.item_positions = set()
        for _ in range(min(self.config.num_items, len(empty_cells))):
            if empty_cells:
                ix, iy = empty_cells.pop()
                self.item_positions.add((ix, iy))
                self.grid[iy][ix] = CellType.ITEM

        # Place traps
        self.trap_positions = set()
        for _ in range(min(self.config.num_traps, len(empty_cells))):
            if empty_cells:
                tx, ty = empty_cells.pop()
                self.trap_positions.add((tx, ty))
                self.grid[ty][tx] = CellType.TRAP

    def _get_start_positions(self, num_agents: int) -> List[Tuple[int, int]]:
        """Get starting positions for agents."""
        empty_cells = [
            (x, y) for x in range(1, self.config.width - 1)
            for y in range(1, self.config.height - 1)
            if self.grid[y][x] == CellType.EMPTY
        ]

        if self.config.random_start:
            random.shuffle(empty_cells)

        return empty_cells[:num_agents]

    def _get_observation(self, player_id: str) -> Observation:
        """Get observation for a player."""
        agent = self.agents[player_id]

        # Get local view
        local_view = self._get_local_view(agent)

        # Get visible cells as text
        visible_description = self._describe_view(agent, local_view)

        # Direction to goal
        goal_direction = self._get_direction_to_goal(agent)

        # Nearby threats and opportunities
        nearby = self._describe_nearby(agent)

        info_partition = InfoPartition(
            player_id=player_id,
            private=[f"Position: ({agent.x}, {agent.y})", f"Score: {agent.score}"],
            public=[f"Goal direction: {goal_direction}"],
        )

        game_state = {
            "position": (agent.x, agent.y),
            "score": agent.score,
            "items_collected": agent.items_collected,
            "steps_taken": agent.steps_taken,
            "goal_direction": goal_direction,
            "local_view": visible_description,
            "nearby": nearby,
            "grid_size": (self.config.width, self.config.height),
        }

        # Add other agents' positions if visible
        other_agents = {}
        for pid, other in self.agents.items():
            if pid != player_id:
                dist = abs(other.x - agent.x) + abs(other.y - agent.y)
                if dist <= self.config.view_radius:
                    other_agents[pid] = {
                        "position": (other.x, other.y),
                        "relative": (other.x - agent.x, other.y - agent.y),
                    }
        game_state["visible_agents"] = other_agents

        return Observation(
            player_id=player_id,
            info_partition=info_partition,
            game_state=game_state,
            valid_actions=self._get_valid_actions(player_id),
            step=self.state.step if self.state else 0,
        )

    def _get_local_view(self, agent: AgentState) -> List[List[str]]:
        """Get the local grid view for an agent."""
        r = self.config.view_radius
        view = []

        for dy in range(-r, r + 1):
            row = []
            for dx in range(-r, r + 1):
                x, y = agent.x + dx, agent.y + dy

                if 0 <= x < self.config.width and 0 <= y < self.config.height:
                    if self.config.fog_of_war and (x, y) not in agent.visited:
                        row.append(CellType.FOG.value)
                    else:
                        # Check for other agents
                        has_agent = any(
                            a.x == x and a.y == y
                            for pid, a in self.agents.items()
                            if pid != agent.player_id
                        )
                        if has_agent:
                            row.append("A")
                        elif dx == 0 and dy == 0:
                            row.append("@")  # Self
                        else:
                            row.append(self.grid[y][x].value)
                else:
                    row.append(CellType.WALL.value)
            view.append(row)

        return view

    def _describe_view(self, agent: AgentState, view: List[List[str]]) -> str:
        """Generate text description of the view."""
        lines = ["Local view:"]
        for row in view:
            lines.append("  " + " ".join(row))

        # Legend
        lines.append("")
        lines.append("Legend: @ = you, G = goal, I = item, T = trap, # = wall, ? = unexplored")

        return "\n".join(lines)

    def _describe_nearby(self, agent: AgentState) -> str:
        """Describe nearby features in natural language."""
        descriptions = []
        r = self.config.view_radius

        # Check each direction
        directions = {
            "north": (0, -1),
            "south": (0, 1),
            "east": (1, 0),
            "west": (-1, 0),
        }

        for dir_name, (dx, dy) in directions.items():
            # Look in this direction
            features = []
            for dist in range(1, r + 1):
                x, y = agent.x + dx * dist, agent.y + dy * dist
                if 0 <= x < self.config.width and 0 <= y < self.config.height:
                    cell = self.grid[y][x]
                    if cell == CellType.WALL:
                        features.append(f"wall at distance {dist}")
                        break
                    elif cell == CellType.GOAL:
                        features.append(f"GOAL at distance {dist}")
                    elif cell == CellType.ITEM:
                        features.append(f"item at distance {dist}")
                    elif cell == CellType.TRAP:
                        features.append(f"trap at distance {dist}")

            if features:
                descriptions.append(f"{dir_name.capitalize()}: {', '.join(features)}")

        return "; ".join(descriptions) if descriptions else "Clear surroundings"

    def _get_direction_to_goal(self, agent: AgentState) -> str:
        """Get cardinal direction to goal."""
        dx = self.goal_pos[0] - agent.x
        dy = self.goal_pos[1] - agent.y

        if dx == 0 and dy == 0:
            return "here"

        # Primary direction
        if abs(dx) > abs(dy):
            return "east" if dx > 0 else "west"
        else:
            return "south" if dy > 0 else "north"

    def _get_valid_actions(self, player_id: str) -> List[str]:
        """Get valid actions for a player."""
        agent = self.agents[player_id]
        valid = [Direction.STAY.value]

        moves = {
            Direction.UP: (0, -1),
            Direction.DOWN: (0, 1),
            Direction.LEFT: (-1, 0),
            Direction.RIGHT: (1, 0),
        }

        for direction, (dx, dy) in moves.items():
            nx, ny = agent.x + dx, agent.y + dy
            if 0 <= nx < self.config.width and 0 <= ny < self.config.height:
                if self.grid[ny][nx] != CellType.WALL:
                    valid.append(direction.value)

        return valid

    def _apply_action(self, action: Action) -> None:
        """Apply a player action."""
        player_id = action.player_id
        agent = self.agents[player_id]
        action_type = action.action_type.lower()

        # Calculate new position
        moves = {
            Direction.UP.value: (0, -1),
            Direction.DOWN.value: (0, 1),
            Direction.LEFT.value: (-1, 0),
            Direction.RIGHT.value: (1, 0),
            Direction.STAY.value: (0, 0),
        }

        dx, dy = moves.get(action_type, (0, 0))
        nx, ny = agent.x + dx, agent.y + dy

        # Validate move
        if 0 <= nx < self.config.width and 0 <= ny < self.config.height:
            if self.grid[ny][nx] == CellType.WALL:
                agent.score += self.config.wall_penalty
            else:
                # Move agent
                agent.x, agent.y = nx, ny
                agent.visited.add((nx, ny))
                agent.steps_taken += 1
                agent.score += self.config.step_penalty

                # Check cell effects
                cell = self.grid[ny][nx]

                if cell == CellType.GOAL:
                    agent.reached_goal = True
                    agent.score += self.config.goal_reward

                elif cell == CellType.ITEM:
                    agent.items_collected += 1
                    agent.score += self.config.item_reward
                    self.item_positions.discard((nx, ny))
                    self.grid[ny][nx] = CellType.EMPTY

                elif cell == CellType.TRAP:
                    agent.score += self.config.trap_penalty
                    self.trap_positions.discard((nx, ny))
                    self.grid[ny][nx] = CellType.EMPTY

        # Advance to next player
        self._next_player()

    def _next_player(self) -> None:
        """Move to next player."""
        current_idx = self.player_ids.index(self.state.current_player)
        next_idx = (current_idx + 1) % len(self.player_ids)
        self.state.current_player = self.player_ids[next_idx]

    def _compute_rewards(self) -> Dict[str, float]:
        """Compute rewards for all players."""
        return {pid: agent.score for pid, agent in self.agents.items()}

    def _is_terminal(self) -> bool:
        """Check if game is over."""
        # Check if any agent reached goal
        if any(a.reached_goal for a in self.agents.values()):
            return True

        # Check step limit
        if self.state.step >= self.config.max_steps:
            return True

        # Check if all items collected (for cooperative mode)
        if self.config.multi_agent_mode == "cooperative":
            if not self.item_positions:
                return True

        return False

    def _render_text(self) -> str:
        """Render the full grid as text."""
        lines = [f"=== GridWorld (Step {self.state.step}) ==="]

        # Create display grid
        display = [[self.grid[y][x].value for x in range(self.config.width)]
                   for y in range(self.config.height)]

        # Add agents
        for pid, agent in self.agents.items():
            display[agent.y][agent.x] = pid[0].upper()  # First letter of player ID

        # Render
        for row in display:
            lines.append(" ".join(row))

        # Stats
        lines.append("")
        for pid, agent in self.agents.items():
            lines.append(f"{pid}: pos=({agent.x},{agent.y}) score={agent.score:.1f} items={agent.items_collected}")

        return "\n".join(lines)


@dataclass
class MazeConfig(GridWorldConfig):
    """Configuration for maze variant."""
    wall_density: float = 0.3
    view_radius: int = 1  # Limited visibility
    fog_of_war: bool = True


class Maze(GridWorld):
    """Maze variant with more walls and limited visibility."""
    env_id = "Maze-v1"

    def __init__(self, **kwargs):
        if "config" not in kwargs:
            kwargs["config"] = MazeConfig()
        super().__init__(**kwargs)


@dataclass
class CollectorConfig(GridWorldConfig):
    """Configuration for item collection variant."""
    num_items: int = 10
    num_traps: int = 0
    goal_reward: float = 0  # No goal bonus
    item_reward: float = 20.0
    max_steps: int = 100


class ItemCollector(GridWorld):
    """Variant focused on collecting items."""
    env_id = "ItemCollector-v1"

    def __init__(self, **kwargs):
        if "config" not in kwargs:
            kwargs["config"] = CollectorConfig()
        super().__init__(**kwargs)

    def _is_terminal(self) -> bool:
        """End when all items collected or time runs out."""
        if not self.item_positions:
            return True
        return super()._is_terminal()
