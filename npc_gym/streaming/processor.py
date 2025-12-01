"""
Text stream processing for real-time NLP games.

Processes incoming text as streams of tokens, sentences, or chunks
that can be distributed to agents as they arrive.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Generator, Iterator, List, Optional, Tuple
from enum import Enum
import re
import time


class ChunkStrategy(Enum):
    """How to chunk text into pieces."""
    WORD = "word"
    SENTENCE = "sentence"
    PARAGRAPH = "paragraph"
    FIXED_SIZE = "fixed_size"
    SEMANTIC = "semantic"  # Requires NLP model
    LINE = "line"
    TOKEN = "token"  # Whitespace-separated


@dataclass
class StreamChunk:
    """A chunk from a text stream."""
    content: str
    index: int  # Position in stream
    timestamp: float  # When it was produced
    chunk_type: str = "text"  # word, sentence, etc.
    metadata: Dict[str, Any] = field(default_factory=dict)

    # Contextual info
    previous_context: str = ""  # What came before
    is_first: bool = False
    is_last: bool = False

    def __str__(self) -> str:
        return self.content


class TextStream:
    """
    Process text as a stream of chunks.

    Can operate in two modes:
    - Batch: Process complete text into chunks
    - Streaming: Process text as it arrives

    Usage:
        # Batch mode
        stream = TextStream(strategy=ChunkStrategy.SENTENCE)
        chunks = list(stream.process(full_text))

        # Streaming mode
        stream = TextStream(strategy=ChunkStrategy.WORD)
        for word in incoming_words:
            chunk = stream.add(word)
            if chunk:
                handle_chunk(chunk)
    """

    def __init__(
        self,
        strategy: ChunkStrategy = ChunkStrategy.SENTENCE,
        chunk_size: int = 50,  # For FIXED_SIZE
        overlap: int = 0,  # Overlap between chunks
        min_chunk_size: int = 1,
    ):
        self.strategy = strategy
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.min_chunk_size = min_chunk_size

        # Streaming state
        self.buffer: str = ""
        self.chunks_produced: int = 0
        self.start_time: float = time.time()
        self.previous_context: str = ""

    def reset(self) -> None:
        """Reset stream state."""
        self.buffer = ""
        self.chunks_produced = 0
        self.start_time = time.time()
        self.previous_context = ""

    def process(self, text: str) -> Generator[StreamChunk, None, None]:
        """Process complete text into chunks."""
        self.reset()
        chunks = self._split_text(text)

        for i, chunk_text in enumerate(chunks):
            chunk = StreamChunk(
                content=chunk_text,
                index=i,
                timestamp=time.time() - self.start_time,
                chunk_type=self.strategy.value,
                is_first=(i == 0),
                is_last=(i == len(chunks) - 1),
                previous_context=self.previous_context,
            )

            self.previous_context = chunk_text
            self.chunks_produced += 1

            yield chunk

    def add(self, text: str) -> Optional[StreamChunk]:
        """
        Add text to stream buffer, return chunk if complete.

        For streaming mode - call repeatedly as text arrives.
        """
        self.buffer += text

        # Try to extract a complete chunk
        chunk_text = self._try_extract_chunk()

        if chunk_text:
            chunk = StreamChunk(
                content=chunk_text,
                index=self.chunks_produced,
                timestamp=time.time() - self.start_time,
                chunk_type=self.strategy.value,
                previous_context=self.previous_context,
            )

            self.previous_context = chunk_text
            self.chunks_produced += 1

            return chunk

        return None

    def flush(self) -> Optional[StreamChunk]:
        """Flush any remaining buffer content as final chunk."""
        if self.buffer.strip():
            chunk = StreamChunk(
                content=self.buffer.strip(),
                index=self.chunks_produced,
                timestamp=time.time() - self.start_time,
                chunk_type=self.strategy.value,
                is_last=True,
                previous_context=self.previous_context,
            )

            self.buffer = ""
            self.chunks_produced += 1

            return chunk

        return None

    def _split_text(self, text: str) -> List[str]:
        """Split text according to strategy."""
        if self.strategy == ChunkStrategy.WORD:
            return text.split()

        elif self.strategy == ChunkStrategy.TOKEN:
            return re.split(r'\s+', text)

        elif self.strategy == ChunkStrategy.SENTENCE:
            # Split on sentence boundaries
            sentences = re.split(r'(?<=[.!?])\s+', text)
            return [s.strip() for s in sentences if s.strip()]

        elif self.strategy == ChunkStrategy.PARAGRAPH:
            paragraphs = text.split('\n\n')
            return [p.strip() for p in paragraphs if p.strip()]

        elif self.strategy == ChunkStrategy.LINE:
            lines = text.split('\n')
            return [l.strip() for l in lines if l.strip()]

        elif self.strategy == ChunkStrategy.FIXED_SIZE:
            chunks = []
            for i in range(0, len(text), self.chunk_size - self.overlap):
                chunk = text[i:i + self.chunk_size]
                if len(chunk) >= self.min_chunk_size:
                    chunks.append(chunk)
            return chunks

        elif self.strategy == ChunkStrategy.SEMANTIC:
            # Fallback to sentence for now
            # Full implementation would use NLP model
            return re.split(r'(?<=[.!?])\s+', text)

        return [text]

    def _try_extract_chunk(self) -> Optional[str]:
        """Try to extract a complete chunk from buffer."""
        if self.strategy == ChunkStrategy.WORD:
            if ' ' in self.buffer:
                parts = self.buffer.split(' ', 1)
                self.buffer = parts[1] if len(parts) > 1 else ""
                return parts[0]

        elif self.strategy == ChunkStrategy.SENTENCE:
            match = re.search(r'^(.*?[.!?])\s+', self.buffer)
            if match:
                sentence = match.group(1)
                self.buffer = self.buffer[match.end():]
                return sentence

        elif self.strategy == ChunkStrategy.LINE:
            if '\n' in self.buffer:
                parts = self.buffer.split('\n', 1)
                self.buffer = parts[1]
                return parts[0].strip()

        elif self.strategy == ChunkStrategy.FIXED_SIZE:
            if len(self.buffer) >= self.chunk_size:
                chunk = self.buffer[:self.chunk_size]
                self.buffer = self.buffer[self.chunk_size - self.overlap:]
                return chunk

        return None


@dataclass
class DealConfig:
    """Configuration for dealing chunks to players."""
    deal_rate: float = 1.0  # Chunks per second
    round_robin: bool = True  # Alternate between players
    duplicate_public: bool = False  # Give same chunks to all players
    max_hand_size: int = 5  # Max chunks per player at once
    discard_oldest: bool = True  # When hand full, discard oldest


class StreamDeck:
    """
    Distributes text chunks to multiple players like a deck of cards.

    Manages which chunks go to which players and when, supporting
    various dealing strategies for PID games.

    Usage:
        deck = StreamDeck(player_ids=["p1", "p2", "p3"])
        stream = TextStream(strategy=ChunkStrategy.SENTENCE)

        for chunk in stream.process(document):
            deck.add_chunk(chunk)

        # Deal to players
        deck.deal_round()

        # Get each player's hand
        p1_chunks = deck.get_hand("p1")
    """

    def __init__(
        self,
        player_ids: List[str],
        config: DealConfig = None,
    ):
        self.player_ids = player_ids
        self.config = config or DealConfig()

        # Deck (undealt chunks)
        self.deck: List[StreamChunk] = []

        # Player hands
        self.hands: Dict[str, List[StreamChunk]] = {
            pid: [] for pid in player_ids
        }

        # Public chunks (visible to all)
        self.public: List[StreamChunk] = []

        # Discard pile
        self.discarded: List[StreamChunk] = []

        # Dealing state
        self.current_dealer_idx: int = 0
        self.chunks_dealt: int = 0
        self.last_deal_time: float = 0

    def add_chunk(self, chunk: StreamChunk) -> None:
        """Add a chunk to the deck."""
        self.deck.append(chunk)

    def add_chunks(self, chunks: List[StreamChunk]) -> None:
        """Add multiple chunks to deck."""
        self.deck.extend(chunks)

    def from_text(
        self,
        text: str,
        strategy: ChunkStrategy = ChunkStrategy.SENTENCE,
    ) -> int:
        """Create deck from text, return number of chunks."""
        stream = TextStream(strategy=strategy)
        chunks = list(stream.process(text))
        self.add_chunks(chunks)
        return len(chunks)

    def shuffle(self) -> None:
        """Shuffle the deck."""
        import random
        random.shuffle(self.deck)

    def deal_one(self, player_id: str = None) -> Optional[StreamChunk]:
        """Deal one chunk to a specific player or next in rotation."""
        if not self.deck:
            return None

        # Determine recipient
        if player_id is None:
            player_id = self.player_ids[self.current_dealer_idx]
            self.current_dealer_idx = (self.current_dealer_idx + 1) % len(self.player_ids)

        # Manage hand size
        if len(self.hands[player_id]) >= self.config.max_hand_size:
            if self.config.discard_oldest:
                discarded = self.hands[player_id].pop(0)
                self.discarded.append(discarded)
            else:
                return None  # Can't deal, hand full

        # Deal chunk
        chunk = self.deck.pop(0)
        self.hands[player_id].append(chunk)
        self.chunks_dealt += 1
        self.last_deal_time = time.time()

        return chunk

    def deal_round(self) -> Dict[str, StreamChunk]:
        """Deal one chunk to each player."""
        dealt = {}
        for pid in self.player_ids:
            chunk = self.deal_one(pid)
            if chunk:
                dealt[pid] = chunk
        return dealt

    def deal_all(self) -> Dict[str, List[StreamChunk]]:
        """Deal all chunks in deck."""
        dealt = {pid: [] for pid in self.player_ids}

        while self.deck:
            for pid in self.player_ids:
                chunk = self.deal_one(pid)
                if chunk:
                    dealt[pid].append(chunk)
                if not self.deck:
                    break

        return dealt

    def deal_to_public(self, n: int = 1) -> List[StreamChunk]:
        """Move chunks to public area (visible to all)."""
        revealed = []
        for _ in range(min(n, len(self.deck))):
            chunk = self.deck.pop(0)
            self.public.append(chunk)
            revealed.append(chunk)
        return revealed

    def get_hand(self, player_id: str) -> List[StreamChunk]:
        """Get a player's current hand."""
        return self.hands.get(player_id, [])

    def get_hand_text(self, player_id: str, separator: str = " ") -> str:
        """Get player's hand as concatenated text."""
        chunks = self.get_hand(player_id)
        return separator.join(c.content for c in chunks)

    def get_public_text(self, separator: str = " ") -> str:
        """Get public chunks as text."""
        return separator.join(c.content for c in self.public)

    def discard_from_hand(self, player_id: str, chunk_index: int = 0) -> Optional[StreamChunk]:
        """Discard a chunk from player's hand."""
        hand = self.hands.get(player_id, [])
        if 0 <= chunk_index < len(hand):
            chunk = hand.pop(chunk_index)
            self.discarded.append(chunk)
            return chunk
        return None

    def hand_sizes(self) -> Dict[str, int]:
        """Get size of each player's hand."""
        return {pid: len(hand) for pid, hand in self.hands.items()}

    def deck_size(self) -> int:
        """Get remaining deck size."""
        return len(self.deck)

    def reset(self) -> None:
        """Reset deck state."""
        # Return all cards to deck
        for hand in self.hands.values():
            self.deck.extend(hand)
            hand.clear()

        self.deck.extend(self.public)
        self.public.clear()

        self.deck.extend(self.discarded)
        self.discarded.clear()

        self.current_dealer_idx = 0
        self.chunks_dealt = 0


class StreamingTextSource:
    """
    Simulates a streaming text source.

    Useful for testing and simulating real-time text input
    (e.g., speech transcription, live document editing).
    """

    def __init__(
        self,
        text: str,
        words_per_second: float = 2.0,
        jitter: float = 0.2,  # Random variation in timing
    ):
        self.full_text = text
        self.words = text.split()
        self.wps = words_per_second
        self.jitter = jitter

        self.position: int = 0
        self.start_time: Optional[float] = None

    def start(self) -> None:
        """Start the stream."""
        self.position = 0
        self.start_time = time.time()

    def get_available(self) -> str:
        """Get text available so far based on elapsed time."""
        if self.start_time is None:
            return ""

        elapsed = time.time() - self.start_time
        words_available = int(elapsed * self.wps)
        words_available = min(words_available, len(self.words))

        return " ".join(self.words[:words_available])

    def get_new(self) -> str:
        """Get newly available text since last call."""
        if self.start_time is None:
            return ""

        elapsed = time.time() - self.start_time
        words_available = int(elapsed * self.wps)
        words_available = min(words_available, len(self.words))

        if words_available > self.position:
            new_text = " ".join(self.words[self.position:words_available])
            self.position = words_available
            return new_text

        return ""

    def is_complete(self) -> bool:
        """Check if all text has been streamed."""
        return self.position >= len(self.words)

    def __iter__(self) -> Iterator[str]:
        """Iterate over words with timing."""
        import random

        for word in self.words:
            yield word

            # Wait with jitter
            delay = 1.0 / self.wps
            delay *= (1 + random.uniform(-self.jitter, self.jitter))
            time.sleep(delay)
