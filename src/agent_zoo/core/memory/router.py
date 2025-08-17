"""
LLM-based memory router for intelligent memory classification.

Uses an LLM to determine which memory type is appropriate for each observation.
"""

import hashlib
import json
from typing import Any, Optional
from pydantic import BaseModel, Field

from agent_zoo.core.memory.items import MemoryType


class MemoryRouteDecision(BaseModel):
    """Decision about where to route a memory item."""
    
    memory_type: MemoryType
    confidence: float = Field(ge=0.0, le=1.0, description="Confidence in the routing decision")
    reasoning: str | None = Field(default=None, description="Explanation for the decision")


class LLMMemoryRouter:
    """
    LLM-based router for classifying memories into appropriate types.
    
    This router uses an LLM to intelligently determine which memory
    system should store each piece of information.
    """
    
    def __init__(self, llm_client: Any, cache_decisions: bool = True):
        """
        Initialize the LLM router.
        
        Args:
            llm_client: LLM client for generating routing decisions
            cache_decisions: Whether to cache routing decisions
        """
        self.llm = llm_client
        self.cache_decisions = cache_decisions
        self.routing_cache = {}
        
        # Routing prompt template
        self.prompt_template = """You are a memory routing system for an AI agent. Your task is to classify content into the appropriate memory type.

Content to classify: {content}

Context information:
- Current task: {task}
- Source: {source}
- Content type: {content_type}
- Agent state: {agent_state}

Memory Types:
1. WORKING: Active conversations, current tasks, temporary information needed for ongoing work. Things the agent is actively thinking about or using right now.

2. SEMANTIC: Facts, knowledge, definitions, rules, and general information that is always true. Things like "Paris is the capital of France" or "Water boils at 100°C".

3. EPISODIC: Personal experiences, events that happened, stories with temporal context. Things with "when" and "what happened" - memories of specific occurrences.

4. PROCEDURAL: Instructions, how-to knowledge, skills, step-by-step procedures. Knowledge about how to do things, like "How to make coffee" or "Steps to debug code".

Analyze the content and respond with your classification in JSON format:
{{
    "type": "MEMORY_TYPE",
    "confidence": 0.0-1.0,
    "reasoning": "Brief explanation of why this content belongs in this memory type"
}}

Examples:
- "User: What's the weather?" → WORKING (active conversation)
- "The speed of light is 299,792,458 m/s" → SEMANTIC (fact)
- "Yesterday I learned Python at the workshop" → EPISODIC (personal experience)
- "To reset password: 1. Click forgot 2. Enter email" → PROCEDURAL (how-to steps)

Respond with valid JSON only."""
    
    async def route_storage(
        self,
        content: Any,
        context: dict[str, Any] | None = None
    ) -> MemoryRouteDecision:
        """
        Route content to appropriate memory type using LLM.
        
        Args:
            content: Content to classify
            context: Additional context for routing decision
            
        Returns:
            Routing decision with confidence and reasoning
        """
        # Check cache if enabled
        if self.cache_decisions:
            cache_key = self._get_cache_key(content, context)
            if cache_key in self.routing_cache:
                return self.routing_cache[cache_key]
        
        # Prepare context
        if context is None:
            context = {}
        
        # Truncate content if too long
        content_str = str(content)
        if len(content_str) > 1000:
            content_str = content_str[:997] + "..."
        
        # Determine content type
        if isinstance(content, dict):
            content_type = "structured_data"
        elif isinstance(content, str):
            content_type = "text"
        elif isinstance(content, (list, tuple)):
            content_type = "sequence"
        else:
            content_type = "other"
        
        # Format prompt
        prompt = self.prompt_template.format(
            content=content_str,
            task=context.get("current_task", "general_operation"),
            source=context.get("source", "unknown"),
            content_type=content_type,
            agent_state=context.get("agent_state", "active")
        )
        
        try:
            # Get LLM response
            response = await self.llm.generate(
                prompt,
                max_tokens=150,
                temperature=0.3  # Lower temperature for consistency
            )
            
            # Parse JSON response
            decision_data = json.loads(response)
            
            # Create decision object (handle both uppercase and lowercase)
            memory_type_str = decision_data["type"].lower()
            decision = MemoryRouteDecision(
                memory_type=MemoryType(memory_type_str),
                confidence=float(decision_data["confidence"]),
                reasoning=decision_data.get("reasoning")
            )
            
        except (json.JSONDecodeError, KeyError, ValueError) as e:
            # Fallback to working memory on error
            decision = MemoryRouteDecision(
                memory_type=MemoryType.WORKING,
                confidence=0.5,
                reasoning=f"Defaulted to working memory due to routing error: {str(e)}"
            )
        
        # Cache decision if enabled
        if self.cache_decisions and cache_key:
            self.routing_cache[cache_key] = decision
        
        return decision
    
    async def route_query(
        self,
        query: str,
        context: dict[str, Any] | None = None
    ) -> list[MemoryType]:
        """
        Determine which memory types to search for a query.
        
        Args:
            query: The search query
            context: Additional context
            
        Returns:
            List of memory types to search
        """
        if context is None:
            context = {}
        
        # Query routing prompt
        prompt = f"""You are a memory routing system. Determine which memory types should be searched for this query.

Query: {query}

Context:
- Current task: {context.get("current_task", "general_operation")}
- Agent state: {context.get("agent_state", "active")}

Memory Types:
1. WORKING: Current tasks, active conversations, temporary information
2. SEMANTIC: Facts, knowledge, definitions, general information
3. EPISODIC: Past experiences, events, temporal memories
4. PROCEDURAL: How-to knowledge, instructions, procedures

Analyze the query and determine which memory types are relevant. You can select multiple types.

Examples:
- "What did the user just say?" → ["WORKING"]
- "What is the capital of France?" → ["SEMANTIC"]
- "What happened yesterday?" → ["EPISODIC", "WORKING"]
- "How do I make coffee?" → ["PROCEDURAL", "SEMANTIC"]
- "Tell me everything about Paris" → ["SEMANTIC", "EPISODIC", "PROCEDURAL"]

Respond with JSON:
{{
    "types": ["MEMORY_TYPE1", "MEMORY_TYPE2"],
    "reasoning": "Brief explanation"
}}"""
        
        try:
            # Get LLM response
            response = await self.llm.generate(
                prompt,
                max_tokens=150,
                temperature=0.3
            )
            
            # Parse response (handle both uppercase and lowercase)
            result = json.loads(response)
            memory_types = [MemoryType(t.lower()) for t in result["types"]]
            
            # Ensure at least one type
            if not memory_types:
                memory_types = [MemoryType.WORKING, MemoryType.SEMANTIC]
            
            return memory_types
            
        except (json.JSONDecodeError, KeyError, ValueError):
            # Fallback to searching working and semantic
            return [MemoryType.WORKING, MemoryType.SEMANTIC]
    
    def clear_cache(self) -> None:
        """Clear the routing cache."""
        self.routing_cache.clear()
    
    def get_cache_stats(self) -> dict[str, Any]:
        """Get statistics about cache usage."""
        if not self.cache_decisions:
            return {"enabled": False}
        
        # Analyze cache entries
        type_counts = {}
        for decision in self.routing_cache.values():
            memory_type = decision.memory_type.value
            type_counts[memory_type] = type_counts.get(memory_type, 0) + 1
        
        return {
            "enabled": True,
            "size": len(self.routing_cache),
            "type_distribution": type_counts
        }
    
    def _get_cache_key(self, content: Any, context: dict[str, Any] | None) -> str | None:
        """
        Generate cache key for content and context.
        
        Args:
            content: Content to hash
            context: Context to include in hash
            
        Returns:
            Cache key or None if unhashable
        """
        try:
            # Create hashable representation
            content_str = str(content)[:500]  # Limit size
            context_str = json.dumps(context or {}, sort_keys=True)[:200]
            
            # Generate hash
            combined = f"{content_str}|{context_str}"
            return hashlib.md5(combined.encode()).hexdigest()
        
        except:
            # Return None if unhashable
            return None


class SimpleRouter:
    """
    Simple fallback router using basic pattern matching.
    
    This can be used when LLM is unavailable or for testing.
    """
    
    def __init__(self):
        """Initialize the simple router."""
        pass
    
    async def route_storage(
        self,
        content: Any,
        context: dict[str, Any] | None = None
    ) -> MemoryRouteDecision:
        """
        Route content using simple pattern matching.
        
        Args:
            content: Content to classify
            context: Additional context
            
        Returns:
            Routing decision
        """
        content_str = str(content).lower()
        
        # Check for procedural patterns
        procedural_patterns = [
            "how to", "steps:", "procedure:", "instructions:",
            "first", "then", "finally", "step 1", "step 2"
        ]
        if any(pattern in content_str for pattern in procedural_patterns):
            return MemoryRouteDecision(
                memory_type=MemoryType.PROCEDURAL,
                confidence=0.7,
                reasoning="Contains procedural keywords"
            )
        
        # Check for semantic patterns
        semantic_patterns = [
            "fact:", "definition:", "is defined as", "means",
            "always", "never", "rule:", "principle:"
        ]
        if any(pattern in content_str for pattern in semantic_patterns):
            return MemoryRouteDecision(
                memory_type=MemoryType.SEMANTIC,
                confidence=0.7,
                reasoning="Contains factual/definitional keywords"
            )
        
        # Check for episodic patterns
        episodic_patterns = [
            "remember when", "happened", "occurred", "experienced",
            "yesterday", "last week", "ago", "event:"
        ]
        if any(pattern in content_str for pattern in episodic_patterns):
            return MemoryRouteDecision(
                memory_type=MemoryType.EPISODIC,
                confidence=0.6,
                reasoning="Contains temporal/experiential keywords"
            )
        
        # Check context hints
        if context:
            if context.get("is_conversation"):
                return MemoryRouteDecision(
                    memory_type=MemoryType.WORKING,
                    confidence=0.8,
                    reasoning="Active conversation context"
                )
            
            if context.get("is_learning"):
                return MemoryRouteDecision(
                    memory_type=MemoryType.SEMANTIC,
                    confidence=0.7,
                    reasoning="Learning context suggests semantic storage"
                )
        
        # Default to working memory
        return MemoryRouteDecision(
            memory_type=MemoryType.WORKING,
            confidence=0.5,
            reasoning="Default classification"
        )
    
    async def route_query(
        self,
        query: str,
        context: dict[str, Any] | None = None
    ) -> list[MemoryType]:
        """
        Determine which memory types to search using simple patterns.
        
        Args:
            query: The search query
            context: Additional context
            
        Returns:
            List of memory types to search
        """
        query_lower = query.lower()
        memory_types = []
        
        # Check for working memory patterns
        if any(word in query_lower for word in ["just", "recent", "current", "now", "said"]):
            memory_types.append(MemoryType.WORKING)
        
        # Check for semantic patterns
        if any(word in query_lower for word in ["what is", "define", "fact", "mean", "explain"]):
            memory_types.append(MemoryType.SEMANTIC)
        
        # Check for episodic patterns
        if any(word in query_lower for word in ["remember", "yesterday", "happened", "when", "event"]):
            memory_types.append(MemoryType.EPISODIC)
        
        # Check for procedural patterns
        if any(word in query_lower for word in ["how to", "how do", "steps", "procedure", "instructions"]):
            memory_types.append(MemoryType.PROCEDURAL)
        
        # Default to working and semantic if no patterns match
        if not memory_types:
            memory_types = [MemoryType.WORKING, MemoryType.SEMANTIC]
        
        return memory_types