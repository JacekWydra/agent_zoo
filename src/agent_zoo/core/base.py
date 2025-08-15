"""
Base interfaces and abstractions for agent implementations.
"""

import asyncio
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any, Generic, TypeVar

import structlog
from pydantic import BaseModel, Field

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

    class Config:
        extra = "allow"  # Allow additional fields for specific agent configs


class BaseAgent(ABC, Generic[T]):
    """
    Abstract base class for all agent implementations.

    This class defines the core interface that all agents must implement,
    providing a consistent API for agent interaction regardless of the
    underlying architecture (ReAct, ToT, GoT, etc.).
    """

    def __init__(self, config: AgentConfig | None = None):
        """
        Initialize the agent with configuration.

        Args:
            config: Agent configuration
        """
        self.config = config or AgentConfig()
        self._state = AgentState(status=AgentStatus.IDLE, max_steps=self.config.max_iterations)

        if self.config.enable_monitoring:
            self._setup_monitoring()

    @abstractmethod
    async def process(self, input_data: dict[str, Any]) -> T:
        """
        Main processing method for the agent.

        This is the core method that each agent must implement,
        defining how it processes input and generates output.

        Args:
            input_data: Input data for processing

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

    async def run(self, input_data: dict[str, Any]) -> T:
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

    def get_full_snapshot(self) -> dict[str, Any]:
        """
        Get a full snapshot of the agent including config and state.

        Returns:
            Dictionary with config and state
        """
        return {
            "config": self.config.model_dump(),
            "state": self._state.model_dump(),
        }

    def load_full_snapshot(self, snapshot: dict[str, Any]) -> None:
        """
        Load agent from a full snapshot.

        Args:
            snapshot: Dictionary with config and state
        """
        self.config = AgentConfig.model_validate(snapshot["config"])
        self._state = AgentState.model_validate(snapshot["state"])

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
        return (
            f"{self.__class__.__name__}(name={self.config.name}, status={self._state.status.value})"
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
