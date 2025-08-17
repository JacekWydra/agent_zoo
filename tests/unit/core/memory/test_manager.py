"""
Unit tests for MemoryManager orchestration layer.

Tests high-level memory operations and agent integration.
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from agent_zoo.core.memory.items import (
    BaseMemoryItem,
    MemoryType,
    WorkingMemoryItem,
)
from agent_zoo.core.memory.manager import MemoryManager, MemoryManagerConfig
from agent_zoo.core.memory.router import MemoryRouteDecision
from agent_zoo.interfaces.messages import Message, MessageRole


@pytest.fixture
def memory_manager(memory_manager_config, mock_llm_client):
    """Create a MemoryManager with mocked dependencies."""
    with patch("agent_zoo.core.memory.manager.Memory") as MockMemory:
        with patch("agent_zoo.core.memory.manager.LLMMemoryRouter") as MockRouter:
            # Create manager
            manager = MemoryManager(memory_manager_config, mock_llm_client)
            
            # Mock the memory and router
            manager.memory = AsyncMock()
            manager.router = AsyncMock()
            
            # Set up default router behavior
            manager.router.route_storage.return_value = MemoryRouteDecision(
                memory_type=MemoryType.WORKING,
                confidence=0.8,
                reasoning="Default routing"
            )
            manager.router.route_query.return_value = [MemoryType.WORKING, MemoryType.SEMANTIC]
            
            return manager


class TestMemoryManagerInitialization:
    """Test MemoryManager initialization."""

    def test_init_with_llm_router(self, memory_manager_config, mock_llm_client):
        """Test initialization with LLM router."""
        with patch("agent_zoo.core.memory.manager.Memory"):
            with patch("agent_zoo.core.memory.manager.LLMMemoryRouter") as MockRouter:
                manager = MemoryManager(memory_manager_config, mock_llm_client)
                
                MockRouter.assert_called_once_with(
                    mock_llm_client,
                    cache_decisions=memory_manager_config.cache_routing_decisions
                )

    def test_init_without_llm_client_raises(self, memory_manager_config):
        """Test that missing LLM client raises error when required."""
        with patch("agent_zoo.core.memory.manager.Memory"):
            with pytest.raises(ValueError, match="LLM client required"):
                MemoryManager(memory_manager_config, llm_client=None)

    def test_init_with_simple_router(self, mock_llm_client):
        """Test initialization with simple router."""
        config = MemoryManagerConfig(use_llm_router=False)
        
        with patch("agent_zoo.core.memory.manager.Memory"):
            with patch("agent_zoo.core.memory.manager.SimpleRouter") as MockRouter:
                manager = MemoryManager(config, mock_llm_client)
                
                MockRouter.assert_called_once()


class TestMemoryObserve:
    """Test observation storage."""

    @pytest.mark.asyncio
    async def test_observe_with_auto_routing(self, memory_manager):
        """Test observation with automatic routing."""
        observation = "Paris is the capital of France"
        
        # Set up router to return semantic
        memory_manager.router.route_storage.return_value = MemoryRouteDecision(
            memory_type=MemoryType.SEMANTIC,
            confidence=0.9,
            reasoning="Factual content"
        )
        memory_manager.memory.store.return_value = "item_123"
        
        item_id = await memory_manager.observe(observation)
        
        assert item_id == "item_123"
        memory_manager.router.route_storage.assert_called_once()
        memory_manager.memory.store.assert_called_once()
        
        # Check that correct item type was created
        stored_item = memory_manager.memory.store.call_args[0][0]
        assert stored_item.content == observation

    @pytest.mark.asyncio
    async def test_observe_with_manual_override(self, memory_manager):
        """Test observation with manual memory type override."""
        observation = "Important task"
        memory_manager.memory.store.return_value = "item_456"
        
        item_id = await memory_manager.observe(
            observation,
            memory_type=MemoryType.WORKING
        )
        
        assert item_id == "item_456"
        # Router should not be called when type is specified
        memory_manager.router.route_storage.assert_not_called()
        
        # Check correct memory type was used
        call_args = memory_manager.memory.store.call_args[0]
        assert call_args[1] == MemoryType.WORKING

    @pytest.mark.asyncio
    async def test_observe_message_object(self, memory_manager):
        """Test observing a Message object."""
        message = Message(role=MessageRole.USER, content="Test message")
        memory_manager.memory.store.return_value = "msg_123"
        
        item_id = await memory_manager.observe(message)
        
        assert item_id == "msg_123"
        # Check that message content was extracted
        stored_item = memory_manager.memory.store.call_args[0][0]
        assert stored_item.content == "Test message"

    @pytest.mark.asyncio
    async def test_observe_respects_auto_capture(self, memory_manager):
        """Test that auto_capture setting is respected."""
        memory_manager.config.auto_capture = False
        
        item_id = await memory_manager.observe("Content")
        
        assert item_id is None
        memory_manager.memory.store.assert_not_called()

    @pytest.mark.asyncio
    async def test_observe_respects_capture_messages(self, memory_manager):
        """Test that capture_messages setting is respected."""
        memory_manager.config.capture_messages = False
        message = Message(role=MessageRole.ASSISTANT, content="Response")
        
        item_id = await memory_manager.observe(message)
        
        assert item_id is None
        memory_manager.memory.store.assert_not_called()

    @pytest.mark.asyncio
    async def test_observe_increments_counter(self, memory_manager):
        """Test that observation counter is incremented."""
        initial_count = memory_manager._total_observations
        memory_manager.memory.store.return_value = "item"
        
        await memory_manager.observe("Content")
        
        assert memory_manager._total_observations == initial_count + 1


class TestMemoryRecall:
    """Test memory recall operations."""

    @pytest.mark.asyncio
    async def test_recall_with_auto_routing(self, memory_manager):
        """Test recall with automatic routing."""
        query = "What is the capital?"
        
        # Mock memory retrieve
        mock_results = [
            MagicMock(item=BaseMemoryItem(content="Paris is capital"))
        ]
        memory_manager.memory.retrieve.return_value = mock_results
        
        memories = await memory_manager.recall(query)
        
        assert len(memories) == 1
        assert memories[0].content == "Paris is capital"
        memory_manager.router.route_query.assert_called_once()

    @pytest.mark.asyncio
    async def test_recall_with_manual_types(self, memory_manager):
        """Test recall with manual memory type selection."""
        query = "Recent tasks"
        memory_types = [MemoryType.WORKING]
        
        mock_results = [
            MagicMock(item=WorkingMemoryItem(content="Task 1"))
        ]
        memory_manager.memory.retrieve.return_value = mock_results
        
        memories = await memory_manager.recall(query, memory_types=memory_types)
        
        # Router should not be called
        memory_manager.router.route_query.assert_not_called()
        
        # Check correct types were searched
        call_args = memory_manager.memory.retrieve.call_args[1]
        assert call_args["memory_types"] == memory_types

    @pytest.mark.asyncio
    async def test_recall_respects_max_tokens(self, memory_manager):
        """Test that recall respects token limits."""
        query = "Test query"
        max_tokens = 500
        
        memory_manager.memory.retrieve.return_value = []
        
        await memory_manager.recall(query, max_tokens=max_tokens)
        
        call_args = memory_manager.memory.retrieve.call_args[1]
        assert call_args["max_tokens"] == max_tokens

    @pytest.mark.asyncio
    async def test_recall_increments_counter(self, memory_manager):
        """Test that recall counter is incremented."""
        initial_count = memory_manager._total_recalls
        memory_manager.memory.retrieve.return_value = []
        
        await memory_manager.recall("Query")
        
        assert memory_manager._total_recalls == initial_count + 1


class TestMemoryContext:
    """Test context generation for LLMs."""

    @pytest.mark.asyncio
    async def test_get_context_formats_for_llm(self, memory_manager, sample_working_item):
        """Test that get_context formats memories for LLM consumption."""
        query = "What are current tasks?"
        
        # Mock recall to return sample item
        memory_manager.memory.retrieve.return_value = [
            MagicMock(item=sample_working_item)
        ]
        
        context = await memory_manager.get_context(query)
        
        assert isinstance(context, list)
        assert len(context) == 1
        
        item = context[0]
        assert item["content"] == sample_working_item.content
        assert item["type"] == "WorkingMemoryItem"
        assert item["importance"] == sample_working_item.importance
        assert item["is_active"] == sample_working_item.is_active
        assert item["source"] == sample_working_item.source

    @pytest.mark.asyncio
    async def test_get_context_with_message(self, memory_manager):
        """Test get_context with Message object."""
        message = Message(role=MessageRole.USER, content="Query")
        memory_manager.memory.retrieve.return_value = []
        
        context = await memory_manager.get_context(message)
        
        # Should extract content from message
        call_args = memory_manager.memory.retrieve.call_args
        assert call_args.args[0] == "Query" if call_args.args else call_args.kwargs.get("query") == "Query"

    @pytest.mark.asyncio
    async def test_get_context_includes_type_specific_fields(
        self, memory_manager, sample_semantic_item, sample_episodic_item, sample_procedural_item
    ):
        """Test that context includes type-specific fields."""
        memory_manager.memory.retrieve.return_value = [
            MagicMock(item=sample_semantic_item),
            MagicMock(item=sample_episodic_item),
            MagicMock(item=sample_procedural_item),
        ]
        
        context = await memory_manager.get_context("Query")
        
        # Semantic fields
        assert context[0]["confidence"] == sample_semantic_item.confidence
        assert context[0]["verified"] == sample_semantic_item.verified
        
        # Episodic fields
        assert context[1]["event_type"] == sample_episodic_item.event_type
        assert context[1]["significance"] == sample_episodic_item.significance
        
        # Procedural fields
        assert context[2]["procedure"] == sample_procedural_item.procedure_name
        assert context[2]["steps_count"] == len(sample_procedural_item.steps)


class TestTaskManagement:
    """Test task-related memory operations."""

    @pytest.mark.asyncio
    async def test_start_task(self, memory_manager):
        """Test starting a new task."""
        task_id = "task_001"
        memory_manager.memory.store.return_value = "task_mem_id"
        
        await memory_manager.start_task(task_id)
        
        assert memory_manager._current_task_id == task_id
        
        # Check task memory was stored
        memory_manager.memory.store.assert_called_once()
        stored_item = memory_manager.memory.store.call_args[0][0]
        assert isinstance(stored_item, WorkingMemoryItem)
        assert f"Started task: {task_id}" in stored_item.content
        assert stored_item.task_id == task_id

    @pytest.mark.asyncio
    async def test_complete_task(self, memory_manager):
        """Test completing a task."""
        task_id = "task_001"
        memory_manager._current_task_id = task_id
        memory_manager.memory.store.return_value = "completion_id"
        memory_manager.memory.consolidate.return_value = {}
        
        await memory_manager.complete_task(task_id)
        
        assert memory_manager._current_task_id is None
        
        # Check completion was stored
        stored_item = memory_manager.memory.store.call_args[0][0]
        assert f"Completed task: {task_id}" in stored_item.content
        
        # Check consolidation was triggered
        memory_manager.memory.consolidate.assert_called_once()

    @pytest.mark.asyncio
    async def test_complete_current_task(self, memory_manager):
        """Test completing current task without specifying ID."""
        memory_manager._current_task_id = "current_task"
        memory_manager.memory.store.return_value = "completion_id"
        memory_manager.memory.consolidate.return_value = {}
        
        await memory_manager.complete_task()
        
        assert memory_manager._current_task_id is None


class TestMemoryManagement:
    """Test memory management operations."""

    @pytest.mark.asyncio
    async def test_clear_working_memory(self, memory_manager):
        """Test clearing working memory only."""
        await memory_manager.clear_working_memory()
        
        memory_manager.memory.clear.assert_called_once_with(MemoryType.WORKING)

    @pytest.mark.asyncio
    async def test_clear_all(self, memory_manager):
        """Test clearing all memory."""
        # Add mock router with clear_cache
        memory_manager.router.clear_cache = MagicMock()
        
        await memory_manager.clear_all()
        
        memory_manager.memory.clear.assert_called_once_with()
        memory_manager.router.clear_cache.assert_called_once()

    def test_get_stats(self, memory_manager):
        """Test getting memory statistics."""
        memory_manager._total_observations = 10
        memory_manager._total_recalls = 5
        memory_manager._current_task_id = "active_task"
        memory_manager.memory.get_stats = MagicMock(return_value={"items": 100})
        memory_manager.router.get_cache_stats = MagicMock(return_value={"size": 20})
        
        stats = memory_manager.get_stats()
        
        assert stats["total_observations"] == 10
        assert stats["total_recalls"] == 5
        assert stats["current_task"] == "active_task"
        assert stats["memory_stats"]["items"] == 100
        assert stats["router_cache"]["size"] == 20


class TestInternalMethods:
    """Test internal helper methods."""

    def test_detect_source_from_message(self, memory_manager):
        """Test source detection from Message object."""
        message = Message(role=MessageRole.USER, content="Test")
        
        source = memory_manager._detect_source(message)
        
        assert source == "message_user"

    def test_detect_source_from_dict(self, memory_manager):
        """Test source detection from dictionary."""
        observation = {"source": "api_call", "content": "Data"}
        
        source = memory_manager._detect_source(observation)
        
        assert source == "api_call"

    def test_detect_source_from_class(self, memory_manager):
        """Test source detection from class name."""
        class CustomClass:
            pass
        
        source = memory_manager._detect_source(CustomClass())
        
        assert source == "CustomClass"

    def test_detect_source_unknown(self, memory_manager):
        """Test source detection fallback."""
        source = memory_manager._detect_source(123)
        
        assert source == "int"  # Returns class name for objects

    def test_estimate_tokens(self, memory_manager):
        """Test token estimation."""
        content = "This is a test string"  # 21 characters
        
        tokens = memory_manager._estimate_tokens(content)
        
        assert tokens == 21 // 4  # 5 tokens

    @pytest.mark.asyncio
    async def test_maybe_consolidate(self, memory_manager):
        """Test conditional consolidation based on interval."""
        import time
        
        memory_manager.config.auto_consolidate = True
        memory_manager.config.consolidation_interval_seconds = 0.1
        # Ensure consolidate returns a value
        memory_manager.memory.consolidate.return_value = {}
        
        # Reset the last consolidation time to ensure first call triggers
        memory_manager._last_consolidation = 0
        
        # First call - should consolidate
        await memory_manager._maybe_consolidate()
        memory_manager.memory.consolidate.assert_called_once()
        
        # Immediate second call - should not consolidate
        memory_manager.memory.consolidate.reset_mock()
        await memory_manager._maybe_consolidate()
        memory_manager.memory.consolidate.assert_not_called()
        
        # Wait and call again - should consolidate
        time.sleep(0.2)
        await memory_manager._maybe_consolidate()
        memory_manager.memory.consolidate.assert_called_once()