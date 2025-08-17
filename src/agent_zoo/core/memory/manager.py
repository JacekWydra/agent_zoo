"""
Memory Manager for high-level memory operations.

Provides a simple interface for agents to use the memory system without
worrying about the underlying complexity.
"""

import time
from typing import Any

from pydantic import BaseModel, Field

from agent_zoo.core.memory.items import (
    BaseMemoryItem,
    EpisodicMemoryItem,
    MemoryType,
    ProceduralMemoryItem,
    SemanticMemoryItem,
    WorkingMemoryItem,
)
from agent_zoo.core.memory.memory import Memory, MemorySearchResult
from agent_zoo.core.memory.router import LLMMemoryRouter, MemoryRouteDecision, SimpleRouter
from agent_zoo.interfaces.messages import Message, MessageRole


class MemoryManagerConfig(BaseModel):
    """Configuration for MemoryManager."""

    # Storage settings
    persist_directory: str | None = Field(
        default=None, description="Directory for persistent storage"
    )

    # Routing settings
    use_llm_router: bool = Field(
        default=True, description="Use LLM for routing (vs simple patterns)"
    )
    cache_routing_decisions: bool = Field(default=True, description="Cache routing decisions")

    # Context settings
    max_context_tokens: int = Field(default=4000, description="Maximum tokens for context")
    default_search_results: int = Field(default=10, description="Default number of search results")

    # Auto-capture settings
    auto_capture: bool = Field(default=True, description="Automatically capture observations")
    capture_messages: bool = Field(default=True, description="Capture Message objects")

    # Consolidation settings
    auto_consolidate: bool = Field(default=True, description="Automatically consolidate memory")
    consolidation_interval_seconds: int = Field(
        default=300, description="Time between consolidations"
    )


class MemoryManager:
    """
    High-level interface for memory operations.

    This manager simplifies memory usage by:
    - Automatically routing memories to appropriate types using LLM
    - Providing simple observe/recall interface
    - Managing memory lifecycle and consolidation
    - Building context for agents

    Basic usage:
        # Store observations (auto-routed)
        await manager.observe("The sky is blue")

        # Recall raw memories (auto-routed retrieval)
        memories = await manager.recall("What color is the sky?")

        # Get formatted context for agent (builds on recall)
        context = await manager.get_context("What color is the sky?")

        # Or force specific types when needed
        await manager.observe("Task started", memory_type=MemoryType.WORKING)
        memories = await manager.recall("Recent events", memory_types=[MemoryType.EPISODIC])
    """

    def __init__(self, config: MemoryManagerConfig | None = None, llm_client: Any | None = None):
        """
        Initialize the memory manager.

        Args:
            config: Configuration settings
            llm_client: LLM client for routing (required if use_llm_router=True)
        """
        self.config = config or MemoryManagerConfig()

        # Initialize core memory system
        self.memory = Memory(persist_directory=self.config.persist_directory)

        # Initialize router
        if self.config.use_llm_router:
            if not llm_client:
                raise ValueError("LLM client required when use_llm_router=True")
            self.router = LLMMemoryRouter(
                llm_client, cache_decisions=self.config.cache_routing_decisions
            )
        else:
            self.router = SimpleRouter()

        # Tracking
        self._last_consolidation = time.time()
        self._total_observations = 0
        self._total_recalls = 0
        self._current_task_id = None

    async def observe(
        self,
        observation: Any,
        context: dict[str, Any] | None = None,
        memory_type: MemoryType | None = None,
    ) -> str | None:
        """
        Process and store an observation.

        The observation is automatically routed to the appropriate memory type
        unless explicitly specified.

        Args:
            observation: Any observable data
            context: Additional context for routing
            memory_type: Optional - force specific memory type (bypasses router)

        Returns:
            ID of stored memory item (None if not captured)

        Examples:
            # Automatic routing (recommended)
            await manager.observe("Paris is the capital of France")

            # Force specific type when needed
            await manager.observe("Remember this", memory_type=MemoryType.WORKING)
        """
        if not self.config.auto_capture:
            return None

        self._total_observations += 1

        # Check if we should capture this type
        if isinstance(observation, Message) and not self.config.capture_messages:
            return None

        # Build routing context
        routing_context = {
            "current_task": self._current_task_id,
            "source": self._detect_source(observation),
            "agent_state": "active",
            **(context or {}),
        }

        # Determine memory type
        if memory_type is None:
            # Use router to decide
            route_decision = await self.router.route_storage(observation, routing_context)
            memory_type = route_decision.memory_type
        else:
            # User specified - create manual decision
            route_decision = MemoryRouteDecision(
                memory_type=memory_type, confidence=1.0, reasoning="User-specified memory type"
            )

        # Create appropriate memory item
        memory_item = self._create_memory_item(observation, memory_type, route_decision)

        # Store in memory
        item_id = await self.memory.store(memory_item, memory_type)

        # Maybe consolidate
        if self.config.auto_consolidate:
            await self._maybe_consolidate()

        return item_id

    async def recall(
        self,
        query: str,
        context: dict[str, Any] | None = None,
        memory_types: list[MemoryType] | None = None,
        max_tokens: int | None = None,
    ) -> list[BaseMemoryItem]:
        """
        Recall relevant memories for a query.

        Memory types to search are automatically determined unless specified.

        Args:
            query: Search query
            context: Additional context for routing
            memory_types: Optional - force specific memory types to search
            max_tokens: Maximum tokens (defaults to config)

        Returns:
            List of relevant memory items

        Examples:
            # Automatic routing (recommended)
            memories = await manager.recall("What is the capital of France?")

            # Force specific types when needed
            memories = await manager.recall("Recent events", memory_types=[MemoryType.EPISODIC])
        """
        self._total_recalls += 1

        max_tokens = max_tokens or self.config.max_context_tokens

        # Determine which memory types to search
        if memory_types is None:
            # Build routing context
            routing_context = {
                "current_task": self._current_task_id,
                "agent_state": "active",
                **(context or {}),
            }
            # Use router to decide
            memory_types = await self.router.route_query(query, routing_context)

        # Retrieve from memory
        results = await self.memory.retrieve(
            query=query,
            n_results=self.config.default_search_results * 2,  # Get extra for filtering
            memory_types=memory_types,
            max_tokens=max_tokens,
        )

        # Extract and return items
        return [result.item for result in results[: self.config.default_search_results]]

    async def get_context(
        self,
        query: str | Message,
        context: dict[str, Any] | None = None,
        memory_types: list[MemoryType] | None = None,
        max_tokens: int | None = None,
    ) -> list[dict[str, Any]]:
        """
        Get formatted context for an agent or LLM.

        This builds on recall() to provide context formatted for use in prompts.

        Args:
            query: Query string or Message to get context for
            context: Additional context for routing
            memory_types: Optional - force specific memory types to search
            max_tokens: Maximum tokens (defaults to config)

        Returns:
            List of context items formatted for LLM

        Examples:
            # Get context for agent processing
            context = await manager.get_context("What is the capital of France?")

            # With a Message object
            context = await manager.get_context(user_message)

            # Use in LLM prompt
            prompt = f"Context: {context}\\n\\nQuestion: {query}"
        """
        # Extract query string if Message
        if isinstance(query, Message):
            query_str = query.content
            # Add message metadata to context
            if context is None:
                context = {}
            context["message_role"] = str(query.role)
        else:
            query_str = query

        # Get raw memories using recall
        memories = await self.recall(query_str, context, memory_types, max_tokens)

        # Format for LLM
        formatted_context = []
        for memory in memories:
            context_item = {
                "content": memory.content,
                "type": memory.__class__.__name__,
                "importance": memory.importance,
                "timestamp": memory.timestamp.isoformat(),
            }

            # Add type-specific fields
            if isinstance(memory, WorkingMemoryItem):
                context_item["is_active"] = memory.is_active
                context_item["source"] = memory.source
            elif isinstance(memory, SemanticMemoryItem):
                context_item["confidence"] = memory.confidence
                context_item["verified"] = memory.verified
            elif isinstance(memory, EpisodicMemoryItem):
                context_item["event_type"] = memory.event_type
                context_item["significance"] = memory.significance
            elif isinstance(memory, ProceduralMemoryItem):
                context_item["procedure"] = memory.procedure_name
                context_item["steps_count"] = len(memory.steps)

            formatted_context.append(context_item)

        return formatted_context

    async def start_task(self, task_id: str) -> None:
        """
        Signal the start of a new task.

        This helps with memory organization and routing.

        Args:
            task_id: Unique task identifier
        """
        self._current_task_id = task_id

        # Store task start in working memory (force working memory for task management)
        task_memory = WorkingMemoryItem(
            content=f"Started task: {task_id}",
            task_id=task_id,
            source="task_management",
            importance=7.0,
            priority=8.0,
        )

        await self.memory.store(task_memory, MemoryType.WORKING)

    async def complete_task(self, task_id: str | None = None) -> None:
        """
        Signal task completion.

        This triggers memory consolidation and cleanup.

        Args:
            task_id: Task to complete (None = current task)
        """
        task_id = task_id or self._current_task_id

        if task_id:
            # Store task completion (force working memory)
            completion_memory = WorkingMemoryItem(
                content=f"Completed task: {task_id}",
                task_id=task_id,
                source="task_management",
                importance=6.0,
            )
            await self.memory.store(completion_memory, MemoryType.WORKING)

            if task_id == self._current_task_id:
                self._current_task_id = None

        # Trigger consolidation
        await self.memory.consolidate()

    async def clear_working_memory(self) -> None:
        """Clear all working memory."""
        await self.memory.clear(MemoryType.WORKING)

    async def clear_all(self) -> None:
        """Clear all memory."""
        await self.memory.clear()
        if hasattr(self.router, "clear_cache"):
            self.router.clear_cache()
        self._current_task_id = None
        self._total_observations = 0
        self._total_recalls = 0

    def get_stats(self) -> dict[str, Any]:
        """Get memory statistics."""
        memory_stats = self.memory.get_stats() if hasattr(self.memory, "get_stats") else {}
        stats = {
            "total_observations": self._total_observations,
            "total_recalls": self._total_recalls,
            "current_task": self._current_task_id,
            "memory_stats": memory_stats,
        }

        if hasattr(self.router, "get_cache_stats"):
            stats["router_cache"] = self.router.get_cache_stats()

        return stats

    def _create_memory_item(
        self, observation: Any, memory_type: MemoryType, route_decision: MemoryRouteDecision
    ) -> BaseMemoryItem:
        """
        Create appropriate memory item from observation.

        Args:
            observation: The observation
            memory_type: Target memory type
            route_decision: Routing decision

        Returns:
            Memory item
        """
        # Extract content and metadata
        if isinstance(observation, Message):
            content = observation.content
            metadata = {
                "role": observation.role.value
                if hasattr(observation.role, "value")
                else str(observation.role),
                "routing_confidence": route_decision.confidence,
                "routing_reasoning": route_decision.reasoning,
            }
            importance = 8.0 if observation.role == MessageRole.USER else 6.0
            source = f"message_{observation.role}"

        elif isinstance(observation, dict):
            content = observation.get("content", observation)
            metadata = {
                "routing_confidence": route_decision.confidence,
                "routing_reasoning": route_decision.reasoning,
                **observation.get("metadata", {}),
            }
            importance = observation.get("importance", 5.0)
            source = observation.get("source", "observation")

        else:
            content = str(observation)
            metadata = {
                "routing_confidence": route_decision.confidence,
                "routing_reasoning": route_decision.reasoning,
            }
            importance = 5.0
            source = "generic"

        # Create appropriate item type based on memory type
        if memory_type == MemoryType.WORKING:
            return WorkingMemoryItem(
                content=content,
                metadata=metadata,
                importance=importance,
                source=source,
                task_id=self._current_task_id,
                priority=importance,
                token_count=self._estimate_tokens(content),
            )

        elif memory_type == MemoryType.SEMANTIC:
            return SemanticMemoryItem(
                content=content,
                metadata=metadata,
                importance=importance,
                source=source,
                confidence=route_decision.confidence,
            )

        elif memory_type == MemoryType.EPISODIC:
            return EpisodicMemoryItem(
                content=content, metadata=metadata, importance=importance, significance=importance
            )

        elif memory_type == MemoryType.PROCEDURAL:
            # Try to extract procedure name from content
            procedure_name = "Unnamed Procedure"
            if isinstance(content, str) and ":" in content:
                procedure_name = content.split(":")[0].strip()

            return ProceduralMemoryItem(
                content=content,
                metadata=metadata,
                importance=importance,
                procedure_name=procedure_name,
                complexity=5.0,
            )

        else:
            # Fallback to base memory item
            return BaseMemoryItem(content=content, metadata=metadata, importance=importance)

    def _detect_source(self, observation: Any) -> str:
        """Detect the source of an observation."""
        if isinstance(observation, Message):
            return f"message_{observation.role.value if hasattr(observation.role, 'value') else observation.role}"
        elif isinstance(observation, dict) and "source" in observation:
            return observation["source"]
        elif hasattr(observation, "__class__"):
            return observation.__class__.__name__
        else:
            return "unknown"

    def _estimate_tokens(self, content: Any) -> int:
        """Estimate token count for content."""
        content_str = str(content)
        # Rough estimate: 1 token per 4 characters
        return len(content_str) // 4

    async def _maybe_consolidate(self) -> None:
        """Check if consolidation is needed."""
        if not self.config.auto_consolidate:
            return

        time_since_last = time.time() - self._last_consolidation
        if time_since_last >= self.config.consolidation_interval_seconds:
            await self.memory.consolidate()
            self._last_consolidation = time.time()
