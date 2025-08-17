"""
Unit tests for BaseAgent integration with memory system.

Tests both memory-enabled and fallback modes.
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from agent_zoo.core.base import AgentConfig, BaseAgent
from agent_zoo.core.memory.manager import MemoryManagerConfig
from agent_zoo.interfaces.messages import Message, MessageRole
from agent_zoo.interfaces.state import AgentState, AgentStatus


class ConcreteAgent(BaseAgent):
    """Concrete agent implementation for testing."""
    
    async def _process(self, message: Message, context: list) -> Message:
        """Simple processing that returns a response."""
        return Message(
            role=MessageRole.ASSISTANT,
            content=f"Processed: {message.content} with {len(context)} context items"
        )
    
    async def think(self, context: dict) -> dict:
        """Simple thinking implementation."""
        return {"thought": "thinking"}
    
    async def act(self, thought: dict) -> dict:
        """Simple action implementation."""
        return {"action": "acting"}


@pytest.fixture
def agent_with_memory(memory_manager_config, mock_llm_client):
    """Create an agent with memory manager."""
    config = AgentConfig(
        name="test_agent",
        memory_config=memory_manager_config
    )
    
    with patch("agent_zoo.core.base.MemoryManager") as MockMemoryManager:
        agent = ConcreteAgent(config)
        
        # Mock the memory manager
        agent.memory = AsyncMock()
        agent.memory.observe = AsyncMock(return_value="obs_id")
        agent.memory.get_context = AsyncMock(return_value=[
            {"content": "Memory 1", "type": "WorkingMemoryItem"},
            {"content": "Memory 2", "type": "SemanticMemoryItem"}
        ])
        
        return agent


@pytest.fixture
def agent_without_memory():
    """Create an agent without memory manager (fallback mode)."""
    config = AgentConfig(
        name="test_agent",
        memory_config=None,  # No memory config
        max_history_items=10
    )
    
    agent = ConcreteAgent(config)
    return agent


class TestAgentWithMemory:
    """Test agent with MemoryManager enabled."""

    def test_initialization_with_memory(self, memory_manager_config):
        """Test agent initializes with memory manager when configured."""
        config = AgentConfig(
            name="test_agent",
            memory_config=memory_manager_config
        )
        
        with patch("agent_zoo.core.base.MemoryManager") as MockMemoryManager:
            agent = ConcreteAgent(config)
            
            # Memory manager should be created
            MockMemoryManager.assert_called_once_with(memory_manager_config)
            assert agent.memory is not None

    @pytest.mark.asyncio
    async def test_process_with_memory_observe(self, agent_with_memory):
        """Test that process method observes incoming messages."""
        input_message = Message(role=MessageRole.USER, content="Test input")
        
        response = await agent_with_memory.process(input_message)
        
        # Should observe the input message
        agent_with_memory.memory.observe.assert_called()
        observed = agent_with_memory.memory.observe.call_args_list[0][0][0]
        assert observed.content == "Test input"
        assert observed.role == MessageRole.USER

    @pytest.mark.asyncio
    async def test_process_with_memory_context(self, agent_with_memory):
        """Test that process method gets context from memory."""
        input_message = Message(role=MessageRole.USER, content="Test input")
        
        response = await agent_with_memory.process(input_message)
        
        # Should get context based on input
        agent_with_memory.memory.get_context.assert_called_once_with("Test input")
        
        # Response should indicate context was used
        assert "with 2 context items" in response.content

    @pytest.mark.asyncio
    async def test_process_stores_response(self, agent_with_memory):
        """Test that agent response is stored in memory."""
        input_message = Message(role=MessageRole.USER, content="Test input")
        
        response = await agent_with_memory.process(input_message)
        
        # Should observe both input and response
        assert agent_with_memory.memory.observe.call_count == 2
        
        # Check response was stored
        response_call = agent_with_memory.memory.observe.call_args_list[1][0][0]
        assert response_call.content == response.content

    @pytest.mark.asyncio
    async def test_process_dict_input(self, agent_with_memory):
        """Test processing dictionary input."""
        input_data = {"content": "Test content", "metadata": "value"}
        
        response = await agent_with_memory.process(input_data)
        
        # Should convert dict to Message
        observed = agent_with_memory.memory.observe.call_args_list[0][0][0]
        assert observed.content == "Test content"
        assert observed.metadata == input_data

    @pytest.mark.asyncio
    async def test_process_string_input(self, agent_with_memory):
        """Test processing string input."""
        input_data = "Plain string input"
        
        response = await agent_with_memory.process(input_data)
        
        # Should convert string to Message
        observed = agent_with_memory.memory.observe.call_args_list[0][0][0]
        assert observed.content == "Plain string input"

    @pytest.mark.asyncio
    async def test_clear_memory(self, agent_with_memory):
        """Test clearing all memory."""
        await agent_with_memory.clear_memory()
        
        agent_with_memory.memory.clear_all.assert_called_once()
        # History should also be cleared
        assert len(agent_with_memory.conversation_history) == 0


class TestAgentWithoutMemory:
    """Test agent in fallback mode without MemoryManager."""

    def test_initialization_without_memory(self):
        """Test agent initializes without memory when not configured."""
        config = AgentConfig(name="test_agent", memory_config=None)
        
        agent = ConcreteAgent(config)
        
        assert agent.memory is None
        assert agent.conversation_history == []

    @pytest.mark.asyncio
    async def test_process_uses_conversation_history(self, agent_without_memory):
        """Test that process uses conversation history as context."""
        input_message = Message(role=MessageRole.USER, content="Test input")
        
        # Add some history
        agent_without_memory.conversation_history = [
            Message(role=MessageRole.USER, content="Previous input"),
            Message(role=MessageRole.ASSISTANT, content="Previous response")
        ]
        
        response = await agent_without_memory.process(input_message)
        
        # Should use history as context (3 items: 2 history + 1 current)
        assert "with 3 context items" in response.content

    @pytest.mark.asyncio
    async def test_process_adds_to_history(self, agent_without_memory):
        """Test that messages are added to conversation history."""
        input_message = Message(role=MessageRole.USER, content="Test input")
        
        response = await agent_without_memory.process(input_message)
        
        # Both input and response should be in history
        assert len(agent_without_memory.conversation_history) == 2
        assert agent_without_memory.conversation_history[0].content == "Test input"
        assert agent_without_memory.conversation_history[1] == response

    @pytest.mark.asyncio
    async def test_history_trimming(self, agent_without_memory):
        """Test that conversation history is trimmed to max_history_items."""
        agent_without_memory.config.max_history_items = 3
        
        # Process multiple messages
        for i in range(5):
            await agent_without_memory.process(f"Message {i}")
        
        # History should be trimmed to 3 items
        assert len(agent_without_memory.conversation_history) <= 3
        
        # Should keep most recent messages
        last_message = agent_without_memory.conversation_history[-1]
        assert "Message 4" in last_message.content or "Processed: Message 4" in last_message.content

    def test_clear_history(self, agent_without_memory):
        """Test clearing conversation history."""
        agent_without_memory.conversation_history = [
            Message(role=MessageRole.USER, content="Test")
        ]
        
        agent_without_memory.clear_history()
        
        assert len(agent_without_memory.conversation_history) == 0

    @pytest.mark.asyncio
    async def test_clear_memory_without_manager(self, agent_without_memory):
        """Test clear_memory when no memory manager exists."""
        agent_without_memory.conversation_history = [
            Message(role=MessageRole.USER, content="Test")
        ]
        
        await agent_without_memory.clear_memory()
        
        # Should still clear conversation history
        assert len(agent_without_memory.conversation_history) == 0


class TestAgentLifecycle:
    """Test agent lifecycle methods with memory."""

    @pytest.mark.asyncio
    async def test_run_with_memory(self, agent_with_memory):
        """Test running agent with memory integration."""
        input_data = "Test input"
        
        result = await agent_with_memory.run(input_data)
        
        # Should process through memory system
        agent_with_memory.memory.observe.assert_called()
        agent_with_memory.memory.get_context.assert_called()
        
        # State should be updated
        assert agent_with_memory.state.status == AgentStatus.COMPLETED

    @pytest.mark.asyncio
    async def test_run_without_memory(self, agent_without_memory):
        """Test running agent without memory (fallback mode)."""
        input_data = "Test input"
        
        result = await agent_without_memory.run(input_data)
        
        # Should still work with conversation history
        assert len(agent_without_memory.conversation_history) > 0
        assert agent_without_memory.state.status == AgentStatus.COMPLETED

    def test_reset_preserves_memory(self, agent_with_memory):
        """Test that reset doesn't clear memory."""
        agent_with_memory.conversation_history = [
            Message(role=MessageRole.USER, content="Test")
        ]
        
        agent_with_memory.reset()
        
        # State should be reset but memory preserved
        assert agent_with_memory.state.iteration_count == 0
        assert agent_with_memory.state.status == AgentStatus.IDLE
        assert len(agent_with_memory.conversation_history) == 1

    def test_get_snapshot_with_memory(self, agent_with_memory):
        """Test getting snapshot includes memory state."""
        agent_with_memory.memory.get_snapshot = MagicMock(return_value={"memory": "state"})
        
        snapshot = agent_with_memory.get_full_snapshot()
        
        assert "memory" in snapshot
        assert snapshot["memory"] == {"memory": "state"}
        agent_with_memory.memory.get_snapshot.assert_called_once()

    def test_get_snapshot_without_memory(self, agent_without_memory):
        """Test getting snapshot includes conversation history."""
        agent_without_memory.conversation_history = [
            Message(role=MessageRole.USER, content="Test")
        ]
        
        snapshot = agent_without_memory.get_full_snapshot()
        
        assert "conversation_history" in snapshot
        assert len(snapshot["conversation_history"]) == 1

    def test_load_snapshot_with_memory(self, agent_with_memory):
        """Test loading snapshot restores memory state."""
        snapshot = {
            "config": agent_with_memory.config.model_dump(),
            "state": agent_with_memory.state.model_dump(),
            "memory": {"memory": "state"}
        }
        
        agent_with_memory.memory.load_snapshot = MagicMock()
        
        agent_with_memory.load_full_snapshot(snapshot)
        
        agent_with_memory.memory.load_snapshot.assert_called_once_with({"memory": "state"})

    def test_load_snapshot_without_memory(self, agent_without_memory):
        """Test loading snapshot restores conversation history."""
        snapshot = {
            "config": agent_without_memory.config.model_dump(),
            "state": agent_without_memory.state.model_dump(),
            "conversation_history": [
                {"role": "user", "content": "Test"}
            ]
        }
        
        agent_without_memory.load_full_snapshot(snapshot)
        
        assert len(agent_without_memory.conversation_history) == 1
        assert agent_without_memory.conversation_history[0].content == "Test"

    def test_repr_with_memory(self, agent_with_memory):
        """Test string representation with memory."""
        repr_str = repr(agent_with_memory)
        
        assert "ConcreteAgent" in repr_str
        assert "test_agent" in repr_str
        assert "memory=MemoryManager" in repr_str

    def test_repr_without_memory(self, agent_without_memory):
        """Test string representation without memory."""
        repr_str = repr(agent_without_memory)
        
        assert "ConcreteAgent" in repr_str
        assert "test_agent" in repr_str
        assert "memory=History" in repr_str


class TestEdgeCases:
    """Test edge cases and error handling."""

    @pytest.mark.asyncio
    async def test_process_none_response(self, agent_with_memory):
        """Test handling when _process returns None."""
        # Override _process to return None
        agent_with_memory._process = AsyncMock(return_value=None)
        
        await agent_with_memory.process("Test")
        
        # Should handle None response gracefully
        # Only input should be observed, not None response
        assert agent_with_memory.memory.observe.call_count == 1

    @pytest.mark.asyncio
    async def test_memory_observe_fails(self, agent_with_memory):
        """Test handling when memory observe fails."""
        agent_with_memory.memory.observe.side_effect = Exception("Storage failed")
        
        # Should not crash the agent
        with pytest.raises(Exception, match="Storage failed"):
            await agent_with_memory.process("Test")

    @pytest.mark.asyncio
    async def test_memory_get_context_fails(self, agent_with_memory):
        """Test handling when memory get_context fails."""
        agent_with_memory.memory.get_context.side_effect = Exception("Retrieval failed")
        
        # Should not crash the agent
        with pytest.raises(Exception, match="Retrieval failed"):
            await agent_with_memory.process("Test")