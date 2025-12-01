"""
Wrappers for integrating npcpy NPCs and Teams with npc-gym.
"""

from __future__ import annotations
from typing import Any, Dict, List, Optional

from npc_gym.core.agent import Agent, AgentConfig, AgentResponse


class NPCWrapper(Agent):
    """
    Wraps an npcpy NPC to work as an npc-gym agent.

    Usage:
        from npcpy.npc_compiler import NPC

        npc = NPC(name="Claude", model="gpt-4", provider="openai")
        agent = NPCWrapper(npc)

        response = agent.act(observation)
    """

    def __init__(self, npc: Any, config: AgentConfig = None):
        """
        Args:
            npc: An npcpy NPC instance
            config: Optional agent config (derived from NPC if not provided)
        """
        if config is None:
            config = AgentConfig(
                name=npc.name,
                model=getattr(npc, 'model', 'llama3.2'),
                provider=getattr(npc, 'provider', 'ollama'),
            )

        super().__init__(config)
        self.npc = npc

    def act(self, observation: Dict[str, Any]) -> AgentResponse:
        """Get action from NPC."""
        import time
        start_time = time.time()

        # Build prompt
        prompt = self._build_prompt(observation)

        # Get NPC response
        response = self.npc.get_llm_response(prompt, format='json')
        llm_output = response.get('response', {})

        response_time = time.time() - start_time

        # Parse response
        if isinstance(llm_output, dict):
            action_type = llm_output.get('action', llm_output.get('action_type', 'pass'))
            value = llm_output.get('value', llm_output.get('amount'))
            reasoning = llm_output.get('reasoning', '')
            confidence = float(llm_output.get('confidence', 0.5))
        else:
            # Try to extract action from text
            action_type = self._extract_action(str(llm_output), observation)
            value = None
            reasoning = str(llm_output)
            confidence = 0.5

        agent_response = AgentResponse(
            action_type=action_type,
            value=value,
            reasoning=reasoning,
            confidence=confidence,
            system_used="npc",
            response_time=response_time,
        )

        self.action_history.append(agent_response)
        return agent_response

    def _build_prompt(self, observation: Dict[str, Any]) -> str:
        """Build prompt from observation."""
        valid_actions = observation.get("valid_actions", [])
        private_info = observation.get("private_info", [])
        public_info = observation.get("public_info", [])
        game_state = observation.get("game_state", {})

        prompt = f"""You are playing a game. Here is your current situation:

PRIVATE INFORMATION (only you can see):
{private_info}

PUBLIC INFORMATION (everyone sees):
{public_info}

GAME STATE:
{game_state}

VALID ACTIONS: {', '.join(valid_actions)}

Choose your action carefully. Consider:
1. What can you infer from the available information?
2. What is your hypothesis about the underlying truth?
3. How confident are you?

Respond with a JSON object:
{{
    "action": "your_chosen_action",
    "value": null_or_numeric_value_if_applicable,
    "reasoning": "Your hypothesis and reasoning",
    "confidence": 0.0_to_1.0
}}
"""
        return prompt

    def _extract_action(self, text: str, observation: Dict[str, Any]) -> str:
        """Extract action from text response."""
        valid_actions = observation.get("valid_actions", ["pass"])
        text_lower = text.lower()

        for action in valid_actions:
            if action.lower() in text_lower:
                return action

        return valid_actions[0] if valid_actions else "pass"


class TeamWrapper:
    """
    Wraps an npcpy Team to work with npc-gym environments.

    Maps team members to game players and coordinates their actions.

    Usage:
        from npcpy.npc_compiler import Team

        team = Team(name="MyTeam", npcs=[npc1, npc2, npc3])
        wrapper = TeamWrapper(team)

        # Get agents dict for game
        agents = wrapper.get_agents()
    """

    def __init__(self, team: Any):
        """
        Args:
            team: An npcpy Team instance
        """
        self.team = team
        self._agents: Dict[str, NPCWrapper] = {}

        # Create wrapped agents for each team member
        for npc_name, npc in team.npcs.items():
            self._agents[npc_name] = NPCWrapper(npc)

    def get_agents(self) -> Dict[str, Agent]:
        """Get dict of wrapped agents for use with npc-gym."""
        return self._agents

    def get_agent(self, name: str) -> Optional[Agent]:
        """Get a specific agent by name."""
        return self._agents.get(name)

    def map_to_players(self, player_ids: List[str]) -> Dict[str, Agent]:
        """
        Map team members to player IDs.

        If more player slots than NPCs, reuses NPCs.
        If more NPCs than slots, uses first N.

        Args:
            player_ids: List of player IDs from environment

        Returns:
            Dict mapping player_id to Agent
        """
        agents = {}
        npc_list = list(self._agents.values())

        for i, player_id in enumerate(player_ids):
            npc_idx = i % len(npc_list)
            agents[player_id] = npc_list[npc_idx]

        return agents


def create_npc_agent(
    name: str,
    model: str = "llama3.2",
    provider: str = "ollama",
    primary_directive: str = None
) -> NPCWrapper:
    """
    Helper to create an NPC-backed agent.

    Args:
        name: Agent name
        model: LLM model name
        provider: LLM provider
        primary_directive: System prompt for NPC

    Returns:
        NPCWrapper agent
    """
    try:
        from npcpy.npc_compiler import NPC

        directive = primary_directive or f"You are {name}, a strategic game player."

        npc = NPC(
            name=name,
            primary_directive=directive,
            model=model,
            provider=provider,
        )

        return NPCWrapper(npc)

    except ImportError:
        raise ImportError("npcpy required. Install with: pip install npcpy")


def create_team_agents(
    team_theme: str,
    num_agents: int,
    model: str = "llama3.2",
    provider: str = "ollama"
) -> Dict[str, NPCWrapper]:
    """
    Create a team of NPC agents with generated personas.

    Args:
        team_theme: Theme for generating personas
        num_agents: Number of agents to create
        model: LLM model name
        provider: LLM provider

    Returns:
        Dict of agent_name -> NPCWrapper
    """
    try:
        from npcpy.llm_funcs import get_llm_response
        from npcpy.npc_compiler import NPC

        agents = {}

        for i in range(num_agents):
            # Generate persona
            prompt = f"""Generate a unique persona for a game-playing AI agent.
Theme: {team_theme}
Agent number: {i + 1}

Return JSON:
{{"name": "unique_name", "directive": "personality and strategy description"}}
"""
            response = get_llm_response(prompt, model=model, provider=provider, format='json')
            persona = response.get('response', {})

            name = persona.get('name', f'Agent_{i}')
            directive = persona.get('directive', f'A strategic player focusing on {team_theme}')

            npc = NPC(
                name=name,
                primary_directive=directive,
                model=model,
                provider=provider,
            )

            agents[name] = NPCWrapper(npc)

        return agents

    except ImportError:
        raise ImportError("npcpy required. Install with: pip install npcpy")
