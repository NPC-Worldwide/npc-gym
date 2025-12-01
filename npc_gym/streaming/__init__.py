"""
Streaming/real-time NLP processing for npc-gym.

Provides:
- TextStream: Process incoming text as a stream of tokens/chunks
- StreamDeck: Distribute text chunks to agents in real-time
- StreamingPIDEnv: Environment wrapper for streaming PID games
"""

from npc_gym.streaming.processor import (
    TextStream,
    StreamChunk,
    StreamDeck,
    ChunkStrategy,
)
from npc_gym.streaming.env import StreamingPIDEnv, StreamingConfig

__all__ = [
    "TextStream",
    "StreamChunk",
    "StreamDeck",
    "ChunkStrategy",
    "StreamingPIDEnv",
    "StreamingConfig",
]
