"""
Information structures for partial observability games.

Core concept: Partial Information Decomposition (PID)
- Total information is partitioned across agents
- Each agent sees private + public subsets
- Hidden information is revealed according to a schedule
- Agents must infer from incomplete data
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Callable
from enum import Enum
import copy


class Visibility(Enum):
    """Information visibility levels."""
    HIDDEN = "hidden"       # No one can see
    PRIVATE = "private"     # Only owner can see
    PUBLIC = "public"       # Everyone can see
    REVEALED = "revealed"   # Was hidden, now public


@dataclass
class InfoItem:
    """A single piece of information with visibility tracking."""
    content: Any
    visibility: Visibility = Visibility.HIDDEN
    owner: Optional[str] = None  # Player ID for private info
    revealed_at: Optional[int] = None  # Step when revealed
    metadata: Dict[str, Any] = field(default_factory=dict)

    def reveal(self, step: int = None) -> None:
        """Make this info public."""
        self.visibility = Visibility.REVEALED
        self.revealed_at = step

    def is_visible_to(self, player_id: str) -> bool:
        """Check if a player can see this info."""
        if self.visibility in (Visibility.PUBLIC, Visibility.REVEALED):
            return True
        if self.visibility == Visibility.PRIVATE and self.owner == player_id:
            return True
        return False


@dataclass
class InfoPartition:
    """
    Represents how information is partitioned for a single player.

    This is what an agent "sees" - their view of the game state.
    """
    player_id: str
    private: List[Any] = field(default_factory=list)      # Only this player sees
    public: List[Any] = field(default_factory=list)       # Everyone sees
    known_hidden: int = 0                                  # Count of hidden items
    other_players: Dict[str, int] = field(default_factory=dict)  # Other players' private counts

    def to_observation(self) -> Dict[str, Any]:
        """Convert to observation dict for agent."""
        return {
            "private_info": self.private,
            "public_info": self.public,
            "hidden_count": self.known_hidden,
            "other_players": self.other_players,
        }

    def as_text(self, separator: str = " | ") -> str:
        """Render as text for LLM consumption."""
        parts = []
        if self.private:
            private_str = separator.join(str(x) for x in self.private)
            parts.append(f"[Private] {private_str}")
        if self.public:
            public_str = separator.join(str(x) for x in self.public)
            parts.append(f"[Public] {public_str}")
        if self.known_hidden > 0:
            parts.append(f"[Hidden] {self.known_hidden} items unrevealed")
        return "\n".join(parts)


class InformationStructure:
    """
    Manages the complete information state of a game.

    Tracks:
    - All information items and their visibility
    - Who owns what private info
    - Revelation schedule
    - Information flow between players
    """

    def __init__(self, player_ids: List[str]):
        self.player_ids = player_ids
        self.items: List[InfoItem] = []
        self._step = 0
        self._revelation_schedule: List[Callable[["InformationStructure", int], None]] = []

    def add_item(
        self,
        content: Any,
        visibility: Visibility = Visibility.HIDDEN,
        owner: str = None,
        metadata: Dict = None
    ) -> InfoItem:
        """Add an information item to the structure."""
        item = InfoItem(
            content=content,
            visibility=visibility,
            owner=owner,
            metadata=metadata or {}
        )
        self.items.append(item)
        return item

    def add_items(
        self,
        contents: List[Any],
        visibility: Visibility = Visibility.HIDDEN,
        owner: str = None
    ) -> List[InfoItem]:
        """Add multiple items with same visibility."""
        return [self.add_item(c, visibility, owner) for c in contents]

    def deal_private(
        self,
        contents: List[Any],
        player_id: str
    ) -> List[InfoItem]:
        """Deal items as private to a specific player."""
        return self.add_items(contents, Visibility.PRIVATE, owner=player_id)

    def reveal_public(self, contents: List[Any]) -> List[InfoItem]:
        """Add items as immediately public."""
        return self.add_items(contents, Visibility.PUBLIC)

    def get_partition(self, player_id: str) -> InfoPartition:
        """
        Get the information partition for a specific player.

        This is the player's "view" of the game.
        """
        partition = InfoPartition(player_id=player_id)

        for item in self.items:
            if item.visibility in (Visibility.PUBLIC, Visibility.REVEALED):
                partition.public.append(item.content)
            elif item.visibility == Visibility.PRIVATE:
                if item.owner == player_id:
                    partition.private.append(item.content)
                else:
                    # Track that another player has private info
                    owner = item.owner or "unknown"
                    partition.other_players[owner] = partition.other_players.get(owner, 0) + 1
            elif item.visibility == Visibility.HIDDEN:
                partition.known_hidden += 1

        return partition

    def reveal_items(
        self,
        count: int = 1,
        from_hidden: bool = True,
        predicate: Callable[[InfoItem], bool] = None
    ) -> List[InfoItem]:
        """
        Reveal hidden items to make them public.

        Args:
            count: Number of items to reveal
            from_hidden: Only reveal HIDDEN items (not PRIVATE)
            predicate: Optional filter for which items to reveal
        """
        revealed = []
        for item in self.items:
            if len(revealed) >= count:
                break

            if from_hidden and item.visibility != Visibility.HIDDEN:
                continue

            if predicate and not predicate(item):
                continue

            item.reveal(step=self._step)
            revealed.append(item)

        return revealed

    def step(self) -> None:
        """Advance the information state by one step."""
        self._step += 1

        # Apply scheduled revelations
        for schedule_fn in self._revelation_schedule:
            schedule_fn(self, self._step)

    def add_revelation_schedule(
        self,
        schedule_fn: Callable[["InformationStructure", int], None]
    ) -> None:
        """
        Add a function to be called each step to reveal information.

        Example:
            def reveal_on_flop(info_struct, step):
                if step == 2:  # Flop
                    info_struct.reveal_items(3)
        """
        self._revelation_schedule.append(schedule_fn)

    def get_all_public(self) -> List[Any]:
        """Get all currently public information."""
        return [
            item.content for item in self.items
            if item.visibility in (Visibility.PUBLIC, Visibility.REVEALED)
        ]

    def get_all_hidden(self) -> List[Any]:
        """Get all hidden information (for debugging/evaluation)."""
        return [
            item.content for item in self.items
            if item.visibility == Visibility.HIDDEN
        ]

    def hidden_count(self) -> int:
        """Count of hidden items."""
        return sum(1 for item in self.items if item.visibility == Visibility.HIDDEN)

    def clone(self) -> "InformationStructure":
        """Create a deep copy of the information structure."""
        new_struct = InformationStructure(self.player_ids.copy())
        new_struct.items = [copy.deepcopy(item) for item in self.items]
        new_struct._step = self._step
        new_struct._revelation_schedule = self._revelation_schedule.copy()
        return new_struct

    def __repr__(self) -> str:
        counts = {
            "hidden": sum(1 for i in self.items if i.visibility == Visibility.HIDDEN),
            "private": sum(1 for i in self.items if i.visibility == Visibility.PRIVATE),
            "public": sum(1 for i in self.items if i.visibility in (Visibility.PUBLIC, Visibility.REVEALED)),
        }
        return f"InformationStructure(players={self.player_ids}, items={counts})"


class PokerStyleInfo(InformationStructure):
    """
    Poker-style information structure with:
    - Hole cards (private per player)
    - Community cards (revealed over rounds)
    - Deck (hidden)

    Revelation schedule: Preflop -> Flop (3) -> Turn (1) -> River (1)
    """

    def __init__(
        self,
        player_ids: List[str],
        deck_contents: List[Any],
        hole_cards_per_player: int = 2,
        community_schedule: List[int] = None  # [3, 1, 1] for Texas Hold'em
    ):
        super().__init__(player_ids)

        self.deck_contents = deck_contents
        self.hole_cards_per_player = hole_cards_per_player
        self.community_schedule = community_schedule or [3, 1, 1]
        self._community_index = 0

        # Add all deck contents as hidden
        self.add_items(deck_contents, Visibility.HIDDEN)

        # Set up revelation schedule
        self.add_revelation_schedule(self._community_reveal)

    def deal_hole_cards(self) -> Dict[str, List[Any]]:
        """Deal hole cards to all players."""
        dealt = {}
        hidden_items = [i for i in self.items if i.visibility == Visibility.HIDDEN]

        for player_id in self.player_ids:
            player_cards = []
            for _ in range(self.hole_cards_per_player):
                if hidden_items:
                    item = hidden_items.pop(0)
                    item.visibility = Visibility.PRIVATE
                    item.owner = player_id
                    player_cards.append(item.content)
            dealt[player_id] = player_cards

        return dealt

    def _community_reveal(self, info_struct: InformationStructure, step: int) -> None:
        """Reveal community cards according to schedule."""
        # Steps: 1=preflop (no reveal), 2=flop, 3=turn, 4=river
        schedule_idx = step - 2  # Adjust for preflop

        if 0 <= schedule_idx < len(self.community_schedule):
            count = self.community_schedule[schedule_idx]
            self.reveal_items(count)

    def get_community_cards(self) -> List[Any]:
        """Get currently revealed community cards."""
        return self.get_all_public()


class TextPIDInfo(InformationStructure):
    """
    Text-based Partial Information Decomposition.

    Takes a source text, chunks it, and distributes as information.
    Used for hypothesis games where agents infer from text fragments.
    """

    def __init__(
        self,
        player_ids: List[str],
        source_text: str,
        chunk_by: str = "word",
        private_ratio: float = 0.2,  # Fraction dealt as private
        public_ratio: float = 0.3,   # Fraction revealed as public over time
    ):
        super().__init__(player_ids)

        self.source_text = source_text
        self.chunk_by = chunk_by
        self.private_ratio = private_ratio
        self.public_ratio = public_ratio

        # Chunk the text
        from npc_gym.core.spaces import Card
        self.chunks = [c.value for c in Card.from_text(source_text, chunk_by)]

        # Add all chunks as hidden initially
        self.add_items(self.chunks, Visibility.HIDDEN)

    def deal_fragments(self) -> Dict[str, List[str]]:
        """
        Deal text fragments to players.

        Returns dict of player_id -> their private fragments
        """
        import random

        hidden_items = [i for i in self.items if i.visibility == Visibility.HIDDEN]
        random.shuffle(hidden_items)

        n_private_per_player = max(1, int(len(self.chunks) * self.private_ratio / len(self.player_ids)))

        dealt = {}
        for player_id in self.player_ids:
            player_fragments = []
            for _ in range(n_private_per_player):
                if hidden_items:
                    item = hidden_items.pop(0)
                    item.visibility = Visibility.PRIVATE
                    item.owner = player_id
                    player_fragments.append(item.content)
            dealt[player_id] = player_fragments

        return dealt

    def reveal_fragments(self, count: int = None) -> List[str]:
        """Reveal some hidden fragments as public."""
        if count is None:
            count = max(1, int(len(self.chunks) * self.public_ratio / 4))

        revealed = self.reveal_items(count)
        return [item.content for item in revealed]

    def get_ground_truth(self) -> str:
        """Get the original source text (for evaluation)."""
        return self.source_text
