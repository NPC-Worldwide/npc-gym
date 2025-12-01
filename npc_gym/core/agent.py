"""
Agent classes for npc-gym.

Implements the hybrid System 1/System 2 architecture:
- System 1 (Fast): Pattern-matched responses from small trained models
- System 2 (Slow): Full LLM reasoning when uncertain
"""

from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Callable, Tuple
import time


@dataclass
class AgentConfig:
    """Configuration for an agent."""
    name: str
    model: str = "llama3.2"
    provider: str = "ollama"
    temperature: float = 0.7
    fast_threshold: float = 0.8      # Confidence needed to use System 1
    ensemble_threshold: float = 0.6  # Confidence needed to use ensemble
    use_system1: bool = True
    use_ensemble: bool = True
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AgentResponse:
    """Response from an agent."""
    action_type: str
    value: Any = None
    reasoning: Optional[str] = None
    confidence: float = 0.5
    system_used: str = "system2"  # "system1", "ensemble", or "system2"
    response_time: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)


class Agent(ABC):
    """
    Base class for all agents.

    Agents receive observations and produce actions.
    """

    def __init__(self, config: AgentConfig):
        self.config = config
        self.name = config.name
        self.action_history: List[AgentResponse] = []
        self.total_tokens = 0
        self.total_cost = 0.0

    @abstractmethod
    def act(self, observation: Dict[str, Any]) -> AgentResponse:
        """
        Choose an action given an observation.

        Args:
            observation: Dict containing game state info

        Returns:
            AgentResponse with action and metadata
        """
        pass

    def reset(self) -> None:
        """Reset agent state for new game."""
        self.action_history = []

    def update(self, reward: float, terminal: bool = False) -> None:
        """
        Update agent after receiving reward.

        Override for learning agents.
        """
        pass


class RandomAgent(Agent):
    """Agent that takes random valid actions."""

    def act(self, observation: Dict[str, Any]) -> AgentResponse:
        import random

        valid_actions = observation.get("valid_actions", ["pass"])
        action = random.choice(valid_actions)

        response = AgentResponse(
            action_type=action,
            confidence=1.0 / len(valid_actions),
            system_used="random",
        )
        self.action_history.append(response)
        return response


class LLMAgent(Agent):
    """Agent that uses an LLM for all decisions (pure System 2)."""

    def __init__(self, config: AgentConfig, npc: Any = None):
        super().__init__(config)
        self.npc = npc
        self._llm_fn = None

    def set_llm_function(self, fn: Callable) -> None:
        """Set the LLM function to use."""
        self._llm_fn = fn

    def act(self, observation: Dict[str, Any]) -> AgentResponse:
        start_time = time.time()

        # Build prompt from observation
        prompt = self._build_prompt(observation)

        # Get LLM response
        if self.npc:
            response = self.npc.get_llm_response(prompt, format='json')
            llm_output = response.get('response', {})
        elif self._llm_fn:
            llm_output = self._llm_fn(prompt)
        else:
            # Fallback to npcpy
            from npcpy.llm_funcs import get_llm_response
            response = get_llm_response(
                prompt,
                model=self.config.model,
                provider=self.config.provider,
                format='json'
            )
            llm_output = response.get('response', {})

        response_time = time.time() - start_time

        # Parse response
        if isinstance(llm_output, dict):
            action_type = llm_output.get('action', llm_output.get('action_type', 'pass'))
            value = llm_output.get('value', llm_output.get('amount'))
            reasoning = llm_output.get('reasoning', '')
            confidence = float(llm_output.get('confidence', 0.5))
        else:
            action_type = str(llm_output)
            value = None
            reasoning = ""
            confidence = 0.5

        agent_response = AgentResponse(
            action_type=action_type,
            value=value,
            reasoning=reasoning,
            confidence=confidence,
            system_used="system2",
            response_time=response_time,
        )
        self.action_history.append(agent_response)
        return agent_response

    def _build_prompt(self, observation: Dict[str, Any]) -> str:
        """Build prompt from observation."""
        valid_actions = observation.get("valid_actions", [])

        prompt = f"""You are {self.name} playing a game.

OBSERVATION:
{self._format_observation(observation)}

VALID ACTIONS: {', '.join(valid_actions)}

Choose an action and explain your reasoning.
Respond with a JSON object:
{{
    "action": "your_chosen_action",
    "value": null_or_numeric_value,
    "reasoning": "why you chose this action",
    "confidence": 0.0_to_1.0
}}
"""
        return prompt

    def _format_observation(self, observation: Dict[str, Any]) -> str:
        """Format observation for prompt."""
        lines = []
        if "private_info" in observation:
            lines.append(f"Private Info: {observation['private_info']}")
        if "public_info" in observation:
            lines.append(f"Public Info: {observation['public_info']}")
        if "game_state" in observation:
            for k, v in observation["game_state"].items():
                lines.append(f"{k}: {v}")
        return "\n".join(lines)


@dataclass
class ModelGene:
    """
    A specialized model with trigger patterns.

    Used for System 1 fast responses.
    """
    specialization: str
    trigger_patterns: List[str]
    model_path: Optional[str] = None
    base_model: str = "Qwen/Qwen3-0.6B"
    confidence_threshold: float = 0.7
    metadata: Dict[str, Any] = field(default_factory=dict)

    def matches(self, text: str) -> bool:
        """Check if this gene's patterns match the input."""
        text_lower = text.lower()
        return any(pattern in text_lower for pattern in self.trigger_patterns)


class HybridAgent(Agent):
    """
    Hybrid System 1/System 2 agent.

    Routes decisions through:
    1. System 1 (Fast Path): Pattern-matched gut responses from fine-tuned models
    2. Ensemble: Vote across multiple specialized models
    3. System 2 (Slow Path): Full LLM reasoning

    The routing is based on confidence thresholds.
    """

    def __init__(
        self,
        config: AgentConfig,
        model_genome: List[ModelGene] = None,
        npc: Any = None
    ):
        super().__init__(config)
        self.model_genome = model_genome or []
        self.npc = npc

        # Stats
        self.system1_uses = 0
        self.ensemble_uses = 0
        self.system2_uses = 0

        # Loaded models cache
        self._loaded_models: Dict[str, Any] = {}

    def add_gene(self, gene: ModelGene) -> None:
        """Add a specialized model gene."""
        self.model_genome.append(gene)

    def act(self, observation: Dict[str, Any]) -> AgentResponse:
        start_time = time.time()

        # Convert observation to text for pattern matching
        obs_text = self._observation_to_text(observation)

        # Try System 1 (fast path)
        if self.config.use_system1:
            system1_response = self._try_system1(obs_text, observation)
            if system1_response and system1_response.confidence >= self.config.fast_threshold:
                system1_response.response_time = time.time() - start_time
                self.system1_uses += 1
                self.action_history.append(system1_response)
                return system1_response

        # Try Ensemble
        if self.config.use_ensemble and len(self.model_genome) > 1:
            ensemble_response = self._try_ensemble(obs_text, observation)
            if ensemble_response and ensemble_response.confidence >= self.config.ensemble_threshold:
                ensemble_response.response_time = time.time() - start_time
                self.ensemble_uses += 1
                self.action_history.append(ensemble_response)
                return ensemble_response

        # Fall back to System 2 (full reasoning)
        system2_response = self._system2_reasoning(observation)
        system2_response.response_time = time.time() - start_time
        self.system2_uses += 1
        self.action_history.append(system2_response)
        return system2_response

    def _observation_to_text(self, observation: Dict[str, Any]) -> str:
        """Convert observation dict to text for pattern matching."""
        parts = []
        if "private_info" in observation:
            parts.append(str(observation["private_info"]))
        if "public_info" in observation:
            parts.append(str(observation["public_info"]))
        if "game_state" in observation:
            parts.append(str(observation["game_state"]))
        return " ".join(parts)

    def _try_system1(
        self,
        obs_text: str,
        observation: Dict[str, Any]
    ) -> Optional[AgentResponse]:
        """
        Try to get a fast System 1 response.

        Checks if any model gene patterns match, then uses that model.
        """
        for gene in self.model_genome:
            if gene.matches(obs_text):
                # Found matching pattern
                if gene.model_path:
                    response = self._run_fast_model(gene, obs_text, observation)
                    if response:
                        return response

        return None

    def _run_fast_model(
        self,
        gene: ModelGene,
        obs_text: str,
        observation: Dict[str, Any]
    ) -> Optional[AgentResponse]:
        """Run a fast specialized model."""
        try:
            # Try to load and run the model
            if gene.model_path not in self._loaded_models:
                from npcpy.ft.sft import load_sft_model
                model, tokenizer = load_sft_model(gene.model_path)
                self._loaded_models[gene.model_path] = (model, tokenizer)

            model, tokenizer = self._loaded_models[gene.model_path]

            # Simple inference
            from npcpy.ft.sft import predict_sft
            prompt = self._build_fast_prompt(observation)
            response_text = predict_sft(model, tokenizer, prompt, temperature=0.1)

            # Parse response
            action = self._parse_fast_response(response_text, observation)

            return AgentResponse(
                action_type=action,
                confidence=gene.confidence_threshold,
                system_used="system1",
                reasoning=f"Fast path via {gene.specialization}",
                metadata={"gene": gene.specialization}
            )
        except Exception as e:
            # Fall through to next option
            return None

    def _try_ensemble(
        self,
        obs_text: str,
        observation: Dict[str, Any]
    ) -> Optional[AgentResponse]:
        """
        Try ensemble voting across multiple models.
        """
        votes = []
        confidences = []

        for gene in self.model_genome:
            if gene.model_path:
                response = self._run_fast_model(gene, obs_text, observation)
                if response:
                    votes.append(response.action_type)
                    confidences.append(response.confidence)

        if not votes:
            return None

        # Majority vote
        from collections import Counter
        vote_counts = Counter(votes)
        winner, count = vote_counts.most_common(1)[0]
        confidence = count / len(votes)

        return AgentResponse(
            action_type=winner,
            confidence=confidence,
            system_used="ensemble",
            reasoning=f"Ensemble vote: {vote_counts}",
            metadata={"votes": dict(vote_counts)}
        )

    def _system2_reasoning(self, observation: Dict[str, Any]) -> AgentResponse:
        """Full System 2 LLM reasoning."""
        # Use LLMAgent logic
        llm_agent = LLMAgent(self.config, npc=self.npc)
        response = llm_agent.act(observation)
        response.system_used = "system2"
        return response

    def _build_fast_prompt(self, observation: Dict[str, Any]) -> str:
        """Build a concise prompt for fast models."""
        valid_actions = observation.get("valid_actions", [])
        private = observation.get("private_info", "")
        public = observation.get("public_info", "")

        return f"Private: {private}\nPublic: {public}\nActions: {valid_actions}\nAction:"

    def _parse_fast_response(
        self,
        response_text: str,
        observation: Dict[str, Any]
    ) -> str:
        """Parse response from fast model."""
        valid_actions = observation.get("valid_actions", ["pass"])
        response_lower = response_text.lower().strip()

        # Check for valid action in response
        for action in valid_actions:
            if action.lower() in response_lower:
                return action

        # Default to first valid action
        return valid_actions[0]

    def get_stats(self) -> Dict[str, Any]:
        """Get usage statistics."""
        total = self.system1_uses + self.ensemble_uses + self.system2_uses
        return {
            "system1_uses": self.system1_uses,
            "ensemble_uses": self.ensemble_uses,
            "system2_uses": self.system2_uses,
            "system1_ratio": self.system1_uses / total if total > 0 else 0,
            "ensemble_ratio": self.ensemble_uses / total if total > 0 else 0,
            "system2_ratio": self.system2_uses / total if total > 0 else 0,
        }

    def reset(self) -> None:
        """Reset agent state."""
        super().reset()
        # Don't reset stats - accumulate across games


class NPCAgent(Agent):
    """
    Agent that wraps an npcpy NPC for compatibility.
    """

    def __init__(self, npc: Any, config: AgentConfig = None):
        if config is None:
            config = AgentConfig(
                name=npc.name,
                model=getattr(npc, 'model', 'llama3.2'),
                provider=getattr(npc, 'provider', 'ollama'),
            )
        super().__init__(config)
        self.npc = npc

    def act(self, observation: Dict[str, Any]) -> AgentResponse:
        """Use NPC to choose action."""
        start_time = time.time()

        # Build prompt
        prompt = self._build_prompt(observation)

        # Get response from NPC
        response = self.npc.get_llm_response(prompt, format='json')
        llm_output = response.get('response', {})
        response_time = time.time() - start_time

        # Parse
        if isinstance(llm_output, dict):
            action_type = llm_output.get('action', 'pass')
            value = llm_output.get('value')
            reasoning = llm_output.get('reasoning', '')
            confidence = float(llm_output.get('confidence', 0.5))
        else:
            action_type = str(llm_output).split()[0] if llm_output else 'pass'
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
        valid_actions = observation.get("valid_actions", [])

        prompt = f"""GAME OBSERVATION:
Private Info: {observation.get('private_info', 'None')}
Public Info: {observation.get('public_info', 'None')}
Game State: {observation.get('game_state', {})}

VALID ACTIONS: {', '.join(valid_actions)}

Choose your action. Respond with JSON:
{{"action": "your_action", "value": null_or_number, "reasoning": "why", "confidence": 0.0_to_1.0}}
"""
        return prompt
