"""
Base interfaces and abstractions for agent implementations.
"""

import asyncio
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any, Generic, TypeVar

import structlog
from pydantic import BaseModel, Field

from agent_zoo.core.memory.manager import MemoryManager, MemoryManagerConfig
from agent_zoo.interfaces.messages import Message
from agent_zoo.interfaces.state import AgentState, AgentStatus

logger = structlog.get_logger()

T = TypeVar("T")


class AgentConfig(BaseModel):
    """Base configuration for agents."""

    name: str = Field(default="agent", description="Agent name for identification")
    description: str = Field(default="", description="Agent description")
    max_iterations: int = Field(default=10, ge=1, description="Maximum iterations for processing")
    timeout_seconds: float = Field(default=300, gt=0, description="Timeout for agent execution")
    enable_monitoring: bool = Field(default=True, description="Enable monitoring and logging")
    enable_caching: bool = Field(default=False, description="Enable response caching")
    retry_attempts: int = Field(default=3, ge=0, description="Number of retry attempts on failure")

    # Memory configuration
    memory_config: MemoryManagerConfig | None = Field(
        default=None, description="Memory manager configuration"
    )

    # Fallback history settings (when no memory manager)
    max_history_items: int = Field(
        default=100, ge=1, description="Maximum conversation history items"
    )
    history_token_limit: int | None = Field(
        default=None, description="Optional token limit for history"
    )

    class Config:
        extra = "allow"  # Allow additional fields for specific agent configs


class BaseAgent(ABC, Generic[T]):
    """
    Abstract base class for all agent implementations.

    This class defines the core interface that all agents must implement,
    providing a consistent API for agent interaction regardless of the
    underlying architecture (ReAct, ToT, GoT, etc.).

    Supports two modes:
    1. With MemoryManager: Full intelligent memory management
    2. Without MemoryManager: Simple conversation history as fallback
    """

    def __init__(self, config: AgentConfig | None = None):
        """
        Initialize the agent with configuration.

        Args:
            config: Agent configuration
        """
        self.config = config or AgentConfig()
        self._state = AgentState(status=AgentStatus.IDLE, max_steps=self.config.max_iterations)

        # Initialize memory manager if configured
        self.memory: MemoryManager | None = None
        if self.config.memory_config:
            # Note: Derived classes should pass llm_client if using LLM router
            self.memory = MemoryManager(self.config.memory_config)

        # Fallback: conversation history when no memory manager
        self.conversation_history: list[Message] = []

        if self.config.enable_monitoring:
            self._setup_monitoring()

    async def process(self, input_data: Any) -> T:
        """
        Main processing method with automatic memory/context management.

        This method handles memory management automatically and calls
        the derived class's _process method with appropriate context.

        Args:
            input_data: Input data for processing (Message or dict)

        Returns:
            Processed result of type T
        """
        # Convert input to Message if needed
        if isinstance(input_data, dict):
            message = Message(
                role="user", content=input_data.get("content", str(input_data)), metadata=input_data
            )
        elif isinstance(input_data, Message):
            message = input_data
        else:
            message = Message(role="user", content=str(input_data))

        if self.memory:
            # WITH MEMORY MANAGER: Full intelligent memory
            await self.memory.observe(message)

            # Get relevant context based on current message
            context = await self.memory.get_context(message.content)

            # Call derived class implementation
            response = await self._process(message, context)

            # Store response in memory
            if response:
                await self.memory.observe(response)

            # Memory manager handles consolidation internally

        else:
            # WITHOUT MEMORY MANAGER: Simple history as context
            # Add incoming message to history
            self.conversation_history.append(message)

            # Trim history if too long
            if len(self.conversation_history) > self.config.max_history_items:
                excess = len(self.conversation_history) - self.config.max_history_items
                self.conversation_history = self.conversation_history[excess:]

            # Provide conversation history as context
            response = await self._process(message, self.conversation_history)

            # Add response to history if it's a Message
            if isinstance(response, Message):
                self.conversation_history.append(response)

                # Trim again if needed
                if len(self.conversation_history) > self.config.max_history_items:
                    excess = len(self.conversation_history) - self.config.max_history_items
                    self.conversation_history = self.conversation_history[excess:]

        return response

    @abstractmethod
    async def _process(self, message: Message, context: list[Any]) -> T:
        """
        Core processing logic implemented by derived classes.

        Args:
            message: Current message to process
            context: Either list[MemoryItem] (with MemoryManager)
                    or list[Message] (without MemoryManager)

        Returns:
            Processed result of type T
        """
        pass

    @abstractmethod
    async def think(self, context: dict[str, Any]) -> dict[str, Any]:
        """
        Reasoning/thinking phase of the agent.

        Args:
            context: Current context for reasoning

        Returns:
            Thought or reasoning result
        """
        pass

    @abstractmethod
    async def act(self, thought: dict[str, Any]) -> dict[str, Any]:
        """
        Action phase of the agent.

        Args:
            thought: Result from thinking phase

        Returns:
            Action result
        """
        pass

    async def run(self, input_data: Any) -> T:
        """
        Run the agent with monitoring and error handling.

        Args:
            input_data: Input data for processing

        Returns:
            Processed result
        """
        self._state.start_time = datetime.now()
        self._state.iteration_count = 0
        self._state.status = AgentStatus.THINKING

        try:
            # Apply timeout if configured
            if self.config.timeout_seconds > 0:
                result = await asyncio.wait_for(
                    self.process(input_data), timeout=self.config.timeout_seconds
                )
            else:
                result = await self.process(input_data)

            self._state.status = AgentStatus.COMPLETED
            self._state.end_time = datetime.now()
            self._record_metrics(success=True)
            return result

        except asyncio.TimeoutError:
            self._state.status = AgentStatus.ERROR
            self._state.end_time = datetime.now()
            self._record_metrics(success=False, error="timeout")
            logger.error(f"Agent {self.config.name} timed out after {self.config.timeout_seconds}s")
            raise

        except Exception as e:
            self._state.status = AgentStatus.ERROR
            self._state.end_time = datetime.now()
            self._record_metrics(success=False, error=str(e))
            logger.error(f"Agent {self.config.name} failed", error=str(e))
            raise

    def clear_history(self) -> None:
        """Clear conversation history (when not using MemoryManager)."""
        self.conversation_history = []

    async def clear_memory(self) -> None:
        """Clear all memory (both MemoryManager and history)."""
        if self.memory:
            await self.memory.clear_all()
        self.conversation_history = []

    @property
    def state(self) -> AgentState:
        """Get the current state of the agent."""
        return self._state

    def set_state(self, state: AgentState) -> None:
        """
        Set the agent state.

        Args:
            state: AgentState to set
        """
        self._state = state

    def reset(self) -> None:
        """Reset the agent to initial state."""
        self._state.reset()
        self._state.max_steps = self.config.max_iterations
        # Don't clear memory/history on reset - that's a separate operation

    def get_full_snapshot(self) -> dict[str, Any]:
        """
        Get a full snapshot of the agent including config and state.

        Returns:
            Dictionary with config, state, and memory/history
        """
        snapshot = {
            "config": self.config.model_dump(),
            "state": self._state.model_dump(),
        }

        if self.memory:
            snapshot["memory"] = self.memory.get_snapshot()
        else:
            snapshot["conversation_history"] = [
                msg.model_dump() for msg in self.conversation_history
            ]

        return snapshot

    def load_full_snapshot(self, snapshot: dict[str, Any]) -> None:
        """
        Load agent from a full snapshot.

        Args:
            snapshot: Dictionary with config, state, and memory/history
        """
        self.config = AgentConfig.model_validate(snapshot["config"])
        self._state = AgentState.model_validate(snapshot["state"])

        if "memory" in snapshot and self.memory:
            self.memory.load_snapshot(snapshot["memory"])
        elif "conversation_history" in snapshot:
            self.conversation_history = [
                Message.model_validate(msg) for msg in snapshot["conversation_history"]
            ]

    def _setup_monitoring(self) -> None:
        """Setup monitoring for the agent."""
        logger.info(f"Agent {self.config.name} initialized with monitoring")

    def _record_metrics(self, success: bool, error: str | None = None) -> None:
        """Record agent execution metrics."""
        duration = (
            (datetime.now() - self._state.start_time).total_seconds()
            if self._state.start_time
            else 0
        )

        self._state.metrics = {
            "duration_seconds": duration,
            "iterations": self._state.iteration_count,
            "success": success,
            "error": error,
            "timestamp": datetime.now().isoformat(),
        }

        if self.config.enable_monitoring:
            logger.info(f"Agent {self.config.name} execution completed", **self._state.metrics)

    def __repr__(self) -> str:
        """String representation of the agent."""
        memory_type = "MemoryManager" if self.memory else "History"
        return (
            f"{self.__class__.__name__}(name={self.config.name}, "
            f"status={self._state.status.value}, memory={memory_type})"
        )


class ComposableAgent(BaseAgent[T]):
    """
    Base class for agents that can be composed with other agents.

    This allows building complex agents from simpler components.
    """

    def __init__(self, config: AgentConfig | None = None):
        super().__init__(config)
        self.components: list[BaseAgent] = []

    def add_component(self, component: BaseAgent) -> None:
        """
        Add a component agent.

        Args:
            component: Component agent to add
        """
        self.components.append(component)

    def remove_component(self, component: BaseAgent) -> None:
        """
        Remove a component agent.

        Args:
            component: Component agent to remove
        """
        if component in self.components:
            self.components.remove(component)

    async def process_with_components(
        self, input_data: dict[str, Any], parallel: bool = False
    ) -> list[Any]:
        """
        Process input through all component agents.

        Args:
            input_data: Input data for processing
            parallel: Whether to process components in parallel

        Returns:
            list of results from each component
        """
        if parallel:
            tasks = [component.run(input_data) for component in self.components]
            results = await asyncio.gather(*tasks, return_exceptions=True)
        else:
            results = []
            for component in self.components:
                result = await component.run(input_data)
                results.append(result)

        return results


class StatefulAgent(BaseAgent[T]):
    """
    Base class for agents with persistent state management.
    """

    def __init__(self, config: AgentConfig | None = None):
        super().__init__(config)
        self._persistent_state: dict[str, Any] = {}
        self._state_history: list[dict[str, Any]] = []

    def save_state(self) -> None:
        """Save current state to history."""
        self._state_history.append(
            {
                "timestamp": datetime.now().isoformat(),
                "state": self._state.model_copy(deep=True),
                "persistent_state": self._persistent_state.copy(),
            }
        )

    def restore_state(self, index: int = -1) -> None:
        """
        Restore state from history.

        Args:
            index: History index to restore from (default: most recent)
        """
        if self._state_history:
            historical = self._state_history[index]
            self._state = historical["state"]
            self._persistent_state = historical["persistent_state"].copy()

    def get_persistent_state(self) -> dict[str, Any]:
        """Get persistent state that survives resets."""
        return self._persistent_state.copy()

    def set_persistent_state(self, key: str, value: Any) -> None:
        """Set a persistent state value."""
        self._persistent_state[key] = value

    def clear_history(self) -> None:
        """Clear state history."""
        self._state_history = []
