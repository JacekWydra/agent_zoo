"""
Unit tests for base agent classes and configurations.
"""

import asyncio
from typing import Any
from unittest.mock import patch

import pytest
from pydantic import ValidationError

from agent_zoo.core.base import AgentConfig, BaseAgent
from agent_zoo.interfaces.messages import Message, MessageRole
from agent_zoo.interfaces.state import AgentState, AgentStatus


class TestAgentConfig:
    """Tests for AgentConfig."""

    def test_default_config(self):
        """Test default configuration values."""
        config = AgentConfig()
        assert config.name == "agent"
        assert config.description == ""
        assert config.max_iterations == 10
        assert config.timeout_seconds == 300
        assert config.enable_monitoring is True
        assert config.enable_caching is False
        assert config.retry_attempts == 3

    def test_custom_config(self):
        """Test custom configuration values."""
        config = AgentConfig(
            name="custom_agent",
            description="A custom test agent",
            max_iterations=20,
            timeout_seconds=60.0,
            enable_monitoring=False,
            enable_caching=True,
            retry_attempts=5,
        )
        assert config.name == "custom_agent"
        assert config.description == "A custom test agent"
        assert config.max_iterations == 20
        assert config.timeout_seconds == 60.0
        assert config.enable_monitoring is False
        assert config.enable_caching is True
        assert config.retry_attempts == 5

    def test_extra_fields_allowed(self):
        """Test that extra fields are allowed in config."""
        config = AgentConfig(
            name="test",
            custom_field="custom_value",
            another_field=42,
        )
        assert config.name == "test"
        assert config.custom_field == "custom_value"
        assert config.another_field == 42

    def test_config_validation(self):
        """Test configuration validation."""
        # Test negative max_iterations
        with pytest.raises(ValidationError):
            AgentConfig(max_iterations=-1)

        # Test negative timeout
        with pytest.raises(ValidationError):
            AgentConfig(timeout_seconds=-1.0)

        # Test negative retry attempts
        with pytest.raises(ValidationError):
            AgentConfig(retry_attempts=-1)


class ConcreteAgent(BaseAgent):
    """Concrete implementation of BaseAgent for testing."""

    async def process(self, input_data: dict[str, Any]) -> Message:
        """Process input data."""
        # Handle both dict and Message input
        if isinstance(input_data, Message):
            message = input_data
        else:
            message = Message(
                role=MessageRole.USER, content=input_data.get("content", str(input_data))
            )
        return Message(
            role=MessageRole.ASSISTANT,
            content=f"Processed: {message.content}",
        )

    async def think(self, context: dict[str, Any]) -> dict[str, Any]:
        """Think about a query."""
        query = context.get("query", "")
        return {"thought": f"Thinking about: {query}"}

    async def act(self, thought: dict[str, Any]) -> dict[str, Any]:
        """Execute an action."""
        action = thought.get("action", "default")
        return {"action": action, "result": "success"}

    async def _reflect(self) -> str:
        """Reflect on current state."""
        return "Reflection complete"


class TestBaseAgent:
    """Tests for BaseAgent."""

    @pytest.fixture
    def concrete_agent(self, agent_config):
        """Create a concrete agent instance."""
        return ConcreteAgent(config=agent_config)

    def test_agent_initialization(self, concrete_agent, agent_config):
        """Test agent initialization."""
        assert concrete_agent.config == agent_config
        assert isinstance(concrete_agent.state, AgentState)
        assert concrete_agent.state.status == AgentStatus.IDLE
        assert concrete_agent.state.iteration_count == 0
        assert len(concrete_agent.state.messages) == 0

    def test_agent_with_default_config(self):
        """Test agent with default configuration."""
        agent = ConcreteAgent()
        assert agent.config.name == "agent"
        assert isinstance(agent.state, AgentState)

    @pytest.mark.asyncio
    async def test_reset(self, concrete_agent):
        """Test agent reset functionality."""
        # Modify state
        concrete_agent.state.status = AgentStatus.THINKING
        concrete_agent.state.iteration_count = 5
        concrete_agent.state.messages.append({"role": "user", "content": "test"})

        # Reset
        concrete_agent.reset()

        # Check state is reset
        assert concrete_agent.state.status == AgentStatus.IDLE
        assert concrete_agent.state.iteration_count == 0
        assert len(concrete_agent.state.messages) == 0

    @pytest.mark.asyncio
    async def test_process_message(self, concrete_agent):
        """Test message processing."""
        input_data = {"content": "Hello, agent!"}
        result = await concrete_agent.process(input_data)

        assert result.role == MessageRole.ASSISTANT
        assert result.content == "Processed: Hello, agent!"

    @pytest.mark.asyncio
    async def test_think(self, concrete_agent):
        """Test thinking functionality."""
        context = {"query": "What is 2+2?"}
        result = await concrete_agent.think(context)
        assert result == {"thought": "Thinking about: What is 2+2?"}

    @pytest.mark.asyncio
    async def test_act(self, concrete_agent):
        """Test action execution."""
        thought = {"action": "calculate"}
        result = await concrete_agent.act(thought)
        assert result == {"action": "calculate", "result": "success"}

    @pytest.mark.asyncio
    async def test_reflect(self, concrete_agent):
        """Test reflection functionality."""
        result = await concrete_agent._reflect()
        assert result == "Reflection complete"

    @pytest.mark.asyncio
    async def test_run_with_max_iterations(self, concrete_agent):
        """Test run method respects max_iterations."""
        concrete_agent.config.max_iterations = 2

        # Mock the process method to track calls
        call_count = 0

        async def mock_process(input_data):
            nonlocal call_count
            call_count += 1
            if call_count >= 2:
                concrete_agent.state.status = AgentStatus.COMPLETED
            return Message(role=MessageRole.ASSISTANT, content=f"Response {call_count}")

        concrete_agent.process = mock_process

        input_msg = Message(role=MessageRole.USER, content="Test")
        result = await concrete_agent.run(input_msg)

        assert isinstance(result, Message)
        assert concrete_agent.state.iteration_count <= 2

    @pytest.mark.asyncio
    async def test_run_with_timeout(self, concrete_agent):
        """Test run method respects timeout."""
        concrete_agent.config.timeout_seconds = 0.1

        # Mock process to take too long
        async def slow_process(input_data):
            await asyncio.sleep(1.0)
            return Message(role=MessageRole.ASSISTANT, content="Too slow")

        concrete_agent.process = slow_process

        input_msg = Message(role=MessageRole.USER, content="Test")

        with pytest.raises(asyncio.TimeoutError):
            await concrete_agent.run(input_msg)

    def test_state_property(self, concrete_agent):
        """Test state property access."""
        assert isinstance(concrete_agent.state, AgentState)
        assert concrete_agent.state.status == AgentStatus.IDLE

        # Modify state
        concrete_agent.state.status = AgentStatus.THINKING
        assert concrete_agent.state.status == AgentStatus.THINKING

    def test_set_state(self, concrete_agent):
        """Test set_state method."""
        new_state = AgentState(
            status=AgentStatus.ACTING,
            iteration_count=3,
            current_task="New task",
            goals=["Goal 1", "Goal 2"],
        )
        concrete_agent.set_state(new_state)

        assert concrete_agent.state == new_state
        assert concrete_agent.state.status == AgentStatus.ACTING
        assert concrete_agent.state.iteration_count == 3
        assert concrete_agent.state.current_task == "New task"
        assert concrete_agent.state.goals == ["Goal 1", "Goal 2"]

    @patch("agent_zoo.core.base.logger")
    def test_setup_monitoring(self, mock_logger, agent_config):
        """Test monitoring setup."""
        agent_config.enable_monitoring = True
        agent = ConcreteAgent(config=agent_config)

        # Check that logger was called during initialization
        mock_logger.info.assert_called()

    def test_agent_representation(self, concrete_agent):
        """Test agent string representation."""
        repr_str = repr(concrete_agent)
        assert "ConcreteAgent" in repr_str
        assert "test_agent" in repr_str

    @pytest.mark.asyncio
    async def test_error_handling(self, concrete_agent):
        """Test error handling in agent operations."""

        async def failing_process(input_data):
            raise ValueError("Processing failed")

        concrete_agent.process = failing_process

        input_msg = Message(role=MessageRole.USER, content="Test")

        with pytest.raises(ValueError, match="Processing failed"):
            await concrete_agent.run(input_msg)

        # State should indicate error
        assert concrete_agent.state.status == AgentStatus.ERROR


class TestGenericTypes:
    """Test generic type handling in BaseAgent."""

    def test_generic_type_preservation(self):
        """Test that generic types are preserved."""

        class TypedAgent(BaseAgent[str]):
            async def process(self, input_data: dict[str, Any]) -> Message:
                return Message(role=MessageRole.ASSISTANT, content="test")

            async def think(self, context: dict[str, Any]) -> dict[str, Any]:
                return {"thought": "thinking"}

            async def act(self, thought: dict[str, Any]) -> dict[str, Any]:
                return {"action": "acting"}

            async def _reflect(self) -> str:
                return "reflecting"

        agent = TypedAgent()
        assert agent is not None
        # Type system should preserve the generic type
