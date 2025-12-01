"""
Wrappers for npc-gym integration with other frameworks.

Provides:
- npcpy integration (NPC agents, teams)
- Gymnasium compatibility layer
"""

from npc_gym.wrappers.npcpy_wrapper import NPCWrapper, TeamWrapper

__all__ = ["NPCWrapper", "TeamWrapper"]
