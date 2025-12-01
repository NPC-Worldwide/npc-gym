"""
Action and observation spaces for npc-gym.

Extends Gymnasium-style spaces with:
- TextSpace: For natural language observations/actions
- DeckSpace: For card-like discrete element pools
- CardSpace: For individual cards/tokens
- CompositeSpace: For combining multiple spaces
"""

from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union, Sequence
import random
import numpy as np


class Space(ABC):
    """Base class for all spaces."""

    def __init__(self, dtype: type = None, seed: int = None):
        self.dtype = dtype
        self._np_random = np.random.default_rng(seed)

    @abstractmethod
    def sample(self) -> Any:
        """Randomly sample an element from this space."""
        pass

    @abstractmethod
    def contains(self, x: Any) -> bool:
        """Check if x is a valid member of this space."""
        pass

    def seed(self, seed: int) -> None:
        """Seed the random number generator."""
        self._np_random = np.random.default_rng(seed)

    @property
    @abstractmethod
    def shape(self) -> Tuple:
        """Shape of elements in this space."""
        pass


class DiscreteSpace(Space):
    """
    A discrete space with n possible values: {0, 1, ..., n-1}
    or named actions: {"fold", "call", "raise"}
    """

    def __init__(
        self,
        n: int = None,
        choices: List[str] = None,
        seed: int = None
    ):
        super().__init__(dtype=int if n else str, seed=seed)

        if choices:
            self.choices = choices
            self.n = len(choices)
            self._choice_to_idx = {c: i for i, c in enumerate(choices)}
        elif n:
            self.n = n
            self.choices = list(range(n))
            self._choice_to_idx = {i: i for i in range(n)}
        else:
            raise ValueError("Must provide either n or choices")

    def sample(self) -> Union[int, str]:
        idx = self._np_random.integers(0, self.n)
        return self.choices[idx]

    def contains(self, x: Any) -> bool:
        if isinstance(x, int):
            return 0 <= x < self.n
        return x in self.choices

    def index(self, x: Any) -> int:
        """Get index of a choice."""
        return self._choice_to_idx.get(x, -1)

    @property
    def shape(self) -> Tuple:
        return (self.n,)

    def __repr__(self) -> str:
        if all(isinstance(c, int) for c in self.choices):
            return f"DiscreteSpace(n={self.n})"
        return f"DiscreteSpace(choices={self.choices})"


class ContinuousSpace(Space):
    """
    A continuous space in R^n bounded by [low, high].
    """

    def __init__(
        self,
        low: Union[float, np.ndarray],
        high: Union[float, np.ndarray],
        shape: Tuple[int, ...] = None,
        seed: int = None
    ):
        super().__init__(dtype=float, seed=seed)

        if shape is None:
            if np.isscalar(low):
                shape = (1,)
            else:
                shape = np.array(low).shape

        self._shape = shape
        self.low = np.full(shape, low, dtype=np.float32)
        self.high = np.full(shape, high, dtype=np.float32)

    def sample(self) -> np.ndarray:
        return self._np_random.uniform(
            self.low, self.high, size=self._shape
        ).astype(np.float32)

    def contains(self, x: Any) -> bool:
        x = np.asarray(x)
        return (
            x.shape == self._shape and
            np.all(x >= self.low) and
            np.all(x <= self.high)
        )

    @property
    def shape(self) -> Tuple:
        return self._shape

    def __repr__(self) -> str:
        return f"ContinuousSpace(low={self.low.min()}, high={self.high.max()}, shape={self._shape})"


class TextSpace(Space):
    """
    A space for natural language text.

    Can be bounded by:
    - max_length: Maximum character/token length
    - vocab: Restricted vocabulary
    - pattern: Regex pattern for valid text
    """

    def __init__(
        self,
        max_length: int = 1024,
        vocab: List[str] = None,
        pattern: str = None,
        seed: int = None
    ):
        super().__init__(dtype=str, seed=seed)
        self.max_length = max_length
        self.vocab = vocab
        self.pattern = pattern
        self._compiled_pattern = None

        if pattern:
            import re
            self._compiled_pattern = re.compile(pattern)

    def sample(self) -> str:
        """Sample random text (primarily for testing)."""
        if self.vocab:
            # Sample words from vocab
            n_words = self._np_random.integers(1, min(20, len(self.vocab)))
            words = self._np_random.choice(self.vocab, size=n_words)
            return " ".join(words)
        else:
            # Generate random characters
            length = self._np_random.integers(1, min(100, self.max_length))
            chars = [chr(self._np_random.integers(97, 123)) for _ in range(length)]
            return "".join(chars)

    def contains(self, x: Any) -> bool:
        if not isinstance(x, str):
            return False
        if len(x) > self.max_length:
            return False
        if self.vocab:
            words = x.split()
            if not all(w in self.vocab for w in words):
                return False
        if self._compiled_pattern:
            if not self._compiled_pattern.match(x):
                return False
        return True

    @property
    def shape(self) -> Tuple:
        return (self.max_length,)

    def __repr__(self) -> str:
        return f"TextSpace(max_length={self.max_length}, vocab_size={len(self.vocab) if self.vocab else 'unlimited'})"


@dataclass
class Card:
    """
    A generic card/token that can represent:
    - Playing cards (suit, rank)
    - Information fragments (text chunks)
    - Any discrete game element
    """
    value: Any
    suit: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    # Standard playing card support
    SUITS = ["hearts", "diamonds", "clubs", "spades"]
    RANKS = ["2", "3", "4", "5", "6", "7", "8", "9", "10", "J", "Q", "K", "A"]

    @classmethod
    def standard_deck(cls) -> List["Card"]:
        """Generate a standard 52-card deck."""
        return [
            cls(value=rank, suit=suit)
            for suit in cls.SUITS
            for rank in cls.RANKS
        ]

    @classmethod
    def from_text(cls, text: str, chunk_by: str = "sentence") -> List["Card"]:
        """
        Convert text into cards (information fragments).

        Args:
            text: Source text to chunk
            chunk_by: "sentence", "word", "paragraph", or "char:N"
        """
        import re

        if chunk_by == "sentence":
            chunks = [s.strip() for s in re.split(r'[.!?]+', text) if s.strip()]
        elif chunk_by == "word":
            chunks = text.split()
        elif chunk_by == "paragraph":
            chunks = [p.strip() for p in text.split('\n\n') if p.strip()]
        elif chunk_by.startswith("char:"):
            n = int(chunk_by.split(":")[1])
            chunks = [text[i:i+n] for i in range(0, len(text), n)]
        else:
            chunks = [text]

        return [cls(value=chunk, metadata={"index": i}) for i, chunk in enumerate(chunks)]

    def __hash__(self):
        return hash((self.value, self.suit))

    def __eq__(self, other):
        if not isinstance(other, Card):
            return False
        return self.value == other.value and self.suit == other.suit

    def __repr__(self):
        if self.suit:
            return f"{self.value}{self.suit[0].upper()}"
        elif isinstance(self.value, str) and len(self.value) > 30:
            return f"Card('{self.value[:30]}...')"
        return f"Card({self.value})"


class CardSpace(Space):
    """
    Space for individual cards from a defined set.
    """

    def __init__(
        self,
        cards: List[Card] = None,
        card_type: str = "standard",
        seed: int = None
    ):
        super().__init__(dtype=Card, seed=seed)

        if cards:
            self.cards = cards
        elif card_type == "standard":
            self.cards = Card.standard_deck()
        else:
            self.cards = []

        self._card_set = set(self.cards)

    def sample(self) -> Card:
        idx = self._np_random.integers(0, len(self.cards))
        return self.cards[idx]

    def contains(self, x: Any) -> bool:
        return x in self._card_set

    @property
    def shape(self) -> Tuple:
        return (len(self.cards),)

    def __repr__(self) -> str:
        return f"CardSpace(n_cards={len(self.cards)})"


class DeckSpace(Space):
    """
    Space representing a deck of cards that can be:
    - Shuffled
    - Dealt (removed from deck)
    - Drawn from

    This is a mutable space that tracks deck state.
    """

    def __init__(
        self,
        cards: List[Card] = None,
        card_type: str = "standard",
        seed: int = None
    ):
        super().__init__(dtype=list, seed=seed)

        if cards:
            self._initial_cards = cards.copy()
        elif card_type == "standard":
            self._initial_cards = Card.standard_deck()
        else:
            self._initial_cards = []

        self.cards = self._initial_cards.copy()

    def reset(self) -> None:
        """Reset deck to initial state."""
        self.cards = self._initial_cards.copy()

    def shuffle(self) -> None:
        """Shuffle the deck in place."""
        self._np_random.shuffle(self.cards)

    def deal(self, n: int = 1) -> List[Card]:
        """Deal n cards from the deck."""
        if n > len(self.cards):
            raise ValueError(f"Cannot deal {n} cards, only {len(self.cards)} remaining")
        dealt = self.cards[:n]
        self.cards = self.cards[n:]
        return dealt

    def draw(self) -> Card:
        """Draw a single card."""
        return self.deal(1)[0]

    def peek(self, n: int = 1) -> List[Card]:
        """Look at top n cards without removing."""
        return self.cards[:n]

    def remaining(self) -> int:
        """Number of cards remaining."""
        return len(self.cards)

    def sample(self) -> List[Card]:
        """Sample a random subset of remaining cards."""
        n = self._np_random.integers(1, len(self.cards) + 1)
        indices = self._np_random.choice(len(self.cards), size=n, replace=False)
        return [self.cards[i] for i in indices]

    def contains(self, x: Any) -> bool:
        if isinstance(x, Card):
            return x in self.cards
        if isinstance(x, list):
            return all(c in self.cards for c in x)
        return False

    @property
    def shape(self) -> Tuple:
        return (len(self.cards),)

    @classmethod
    def from_text(
        cls,
        text: str,
        chunk_by: str = "word",
        seed: int = None
    ) -> "DeckSpace":
        """
        Create a deck from text chunks.

        This is the core of Partial Information Decomposition:
        text becomes cards that can be dealt to agents.
        """
        cards = Card.from_text(text, chunk_by=chunk_by)
        return cls(cards=cards, seed=seed)

    def __repr__(self) -> str:
        return f"DeckSpace(remaining={len(self.cards)}, total={len(self._initial_cards)})"


class CompositeSpace(Space):
    """
    Combines multiple spaces into a structured observation/action space.

    Example:
        space = CompositeSpace({
            "hand": CardSpace(),
            "stack": ContinuousSpace(0, 10000),
            "action": DiscreteSpace(choices=["fold", "call", "raise"]),
            "reasoning": TextSpace(max_length=500)
        })
    """

    def __init__(
        self,
        spaces: Dict[str, Space],
        seed: int = None
    ):
        super().__init__(dtype=dict, seed=seed)
        self.spaces = spaces

    def sample(self) -> Dict[str, Any]:
        return {key: space.sample() for key, space in self.spaces.items()}

    def contains(self, x: Any) -> bool:
        if not isinstance(x, dict):
            return False
        if set(x.keys()) != set(self.spaces.keys()):
            return False
        return all(
            self.spaces[key].contains(x[key])
            for key in self.spaces
        )

    @property
    def shape(self) -> Dict[str, Tuple]:
        return {key: space.shape for key, space in self.spaces.items()}

    def __getitem__(self, key: str) -> Space:
        return self.spaces[key]

    def __repr__(self) -> str:
        space_strs = ", ".join(f"{k}: {v}" for k, v in self.spaces.items())
        return f"CompositeSpace({{{space_strs}}})"


# Convenience aliases
Box = ContinuousSpace
Discrete = DiscreteSpace
Text = TextSpace
