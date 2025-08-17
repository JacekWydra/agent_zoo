"""
Unit tests for memory routing system.

Tests LLM-based and simple pattern-based routing.
"""

import json
from unittest.mock import AsyncMock, MagicMock

import pytest

from agent_zoo.core.memory.items import MemoryType
from agent_zoo.core.memory.router import (
    LLMMemoryRouter,
    MemoryRouteDecision,
    SimpleRouter,
)


class TestLLMMemoryRouter:
    """Test LLM-based memory routing."""

    @pytest.fixture
    def llm_router(self, mock_llm_client):
        """Create an LLM router with mock client."""
        return LLMMemoryRouter(mock_llm_client, cache_decisions=True)

    @pytest.mark.asyncio
    async def test_route_storage_to_semantic(self, llm_router):
        """Test routing factual content to semantic memory."""
        content = "Paris is the capital of France"
        context = {"source": "encyclopedia"}
        
        decision = await llm_router.route_storage(content, context)
        
        assert decision.memory_type == MemoryType.SEMANTIC
        assert decision.confidence == 0.9
        assert "factual" in decision.reasoning.lower()

    @pytest.mark.asyncio
    async def test_route_storage_to_episodic(self, llm_router):
        """Test routing experiential content to episodic memory."""
        content = "I remember when we discussed the project yesterday"
        
        decision = await llm_router.route_storage(content)
        
        assert decision.memory_type == MemoryType.EPISODIC
        assert decision.confidence == 0.8
        assert "experiential" in decision.reasoning.lower()

    @pytest.mark.asyncio
    async def test_route_storage_to_procedural(self, llm_router):
        """Test routing procedural content to procedural memory."""
        content = "How to reset password: 1. Click forgot 2. Enter email"
        
        decision = await llm_router.route_storage(content)
        
        assert decision.memory_type == MemoryType.PROCEDURAL
        assert decision.confidence == 0.85
        assert "procedural" in decision.reasoning.lower()

    @pytest.mark.asyncio
    async def test_route_storage_defaults_to_working(self, llm_router):
        """Test default routing to working memory."""
        content = "Current task status"
        
        decision = await llm_router.route_storage(content)
        
        assert decision.memory_type == MemoryType.WORKING
        assert decision.confidence == 0.7

    @pytest.mark.asyncio
    async def test_route_storage_truncates_long_content(self, llm_router):
        """Test that long content is truncated."""
        content = "x" * 2000  # Very long content
        
        decision = await llm_router.route_storage(content)
        
        # Should still route successfully
        assert isinstance(decision.memory_type, MemoryType)

    @pytest.mark.asyncio
    async def test_route_storage_caching(self, llm_router):
        """Test that routing decisions are cached."""
        content = "Test content for caching"
        context = {"task": "test"}
        
        # First call
        decision1 = await llm_router.route_storage(content, context)
        
        # Second call - should use cache
        decision2 = await llm_router.route_storage(content, context)
        
        assert decision1.memory_type == decision2.memory_type
        assert decision1.confidence == decision2.confidence
        assert len(llm_router.routing_cache) > 0

    @pytest.mark.asyncio
    async def test_route_storage_error_handling(self):
        """Test handling of LLM errors."""
        # Create router with failing LLM
        class FailingLLM:
            async def generate(self, *args, **kwargs):
                return "invalid json"
        
        router = LLMMemoryRouter(FailingLLM(), cache_decisions=False)
        
        decision = await router.route_storage("content")
        
        # Should default to working memory on error
        assert decision.memory_type == MemoryType.WORKING
        assert decision.confidence == 0.5
        assert "error" in decision.reasoning.lower()

    @pytest.mark.asyncio
    async def test_route_query(self, llm_router):
        """Test routing queries to appropriate memory types."""
        query = "What is the capital of France?"
        
        memory_types = await llm_router.route_query(query)
        
        assert isinstance(memory_types, list)
        assert len(memory_types) > 0
        assert all(isinstance(mt, MemoryType) for mt in memory_types)

    @pytest.mark.asyncio
    async def test_route_query_error_handling(self):
        """Test query routing error handling."""
        class FailingLLM:
            async def generate(self, *args, **kwargs):
                return "invalid"
        
        router = LLMMemoryRouter(FailingLLM(), cache_decisions=False)
        
        memory_types = await router.route_query("query")
        
        # Should default to working and semantic
        assert MemoryType.WORKING in memory_types
        assert MemoryType.SEMANTIC in memory_types

    def test_clear_cache(self, llm_router):
        """Test clearing the routing cache."""
        # Add some cached entries
        llm_router.routing_cache["key1"] = MagicMock()
        llm_router.routing_cache["key2"] = MagicMock()
        
        llm_router.clear_cache()
        
        assert len(llm_router.routing_cache) == 0

    def test_get_cache_stats(self, llm_router):
        """Test getting cache statistics."""
        # Add some cached decisions
        llm_router.routing_cache["key1"] = MemoryRouteDecision(
            memory_type=MemoryType.SEMANTIC,
            confidence=0.9,
            reasoning="test"
        )
        llm_router.routing_cache["key2"] = MemoryRouteDecision(
            memory_type=MemoryType.WORKING,
            confidence=0.8,
            reasoning="test"
        )
        
        stats = llm_router.get_cache_stats()
        
        assert stats["enabled"] is True
        assert stats["size"] == 2
        assert stats["type_distribution"]["semantic"] == 1
        assert stats["type_distribution"]["working"] == 1

    def test_get_cache_stats_disabled(self):
        """Test cache stats when caching is disabled."""
        router = LLMMemoryRouter(MagicMock(), cache_decisions=False)
        
        stats = router.get_cache_stats()
        
        assert stats == {"enabled": False}

    def test_get_cache_key(self, llm_router):
        """Test cache key generation."""
        content = "Test content"
        context = {"task": "test", "source": "user"}
        
        key = llm_router._get_cache_key(content, context)
        
        assert key is not None
        assert isinstance(key, str)
        assert len(key) == 32  # MD5 hash length

    def test_get_cache_key_unhashable(self, llm_router):
        """Test cache key generation with unhashable content."""
        # Create unhashable content (e.g., with exception in __str__)
        class Unhashable:
            def __str__(self):
                raise Exception("Cannot hash")
        
        key = llm_router._get_cache_key(Unhashable(), {})
        
        assert key is None


class TestSimpleRouter:
    """Test simple pattern-based routing."""

    @pytest.fixture
    def simple_router(self):
        """Create a simple router."""
        return SimpleRouter()

    @pytest.mark.asyncio
    async def test_route_storage_procedural_patterns(self, simple_router):
        """Test procedural pattern detection."""
        contents = [
            "How to make coffee",
            "Steps: 1. First do this 2. Then do that",
            "Procedure: Reset the system",
            "Instructions: Follow these steps"
        ]
        
        for content in contents:
            decision = await simple_router.route_storage(content)
            assert decision.memory_type == MemoryType.PROCEDURAL
            assert "procedural" in decision.reasoning.lower()

    @pytest.mark.asyncio
    async def test_route_storage_semantic_patterns(self, simple_router):
        """Test semantic pattern detection."""
        contents = [
            "Fact: Water boils at 100°C",
            "Definition: AI is defined as...",
            "Python means a programming language",
            "Rule: Always validate input"
        ]
        
        for content in contents:
            decision = await simple_router.route_storage(content)
            assert decision.memory_type == MemoryType.SEMANTIC
            assert "factual" in decision.reasoning.lower() or "definitional" in decision.reasoning.lower()

    @pytest.mark.asyncio
    async def test_route_storage_episodic_patterns(self, simple_router):
        """Test episodic pattern detection."""
        contents = [
            "Remember when we discussed this",
            "Yesterday I learned Python",
            "An event occurred last week",
            "I experienced something interesting"
        ]
        
        for content in contents:
            decision = await simple_router.route_storage(content)
            assert decision.memory_type == MemoryType.EPISODIC
            assert "temporal" in decision.reasoning.lower() or "experiential" in decision.reasoning.lower()

    @pytest.mark.asyncio
    async def test_route_storage_context_hints(self, simple_router):
        """Test routing based on context hints."""
        # Conversation context
        context = {"is_conversation": True}
        decision = await simple_router.route_storage("Any content", context)
        assert decision.memory_type == MemoryType.WORKING
        assert "conversation" in decision.reasoning.lower()
        
        # Learning context
        context = {"is_learning": True}
        decision = await simple_router.route_storage("Any content", context)
        assert decision.memory_type == MemoryType.SEMANTIC
        assert "learning" in decision.reasoning.lower()

    @pytest.mark.asyncio
    async def test_route_storage_default(self, simple_router):
        """Test default routing when no patterns match."""
        content = "Random content without patterns"
        
        decision = await simple_router.route_storage(content)
        
        assert decision.memory_type == MemoryType.WORKING
        assert decision.confidence == 0.5
        assert "default" in decision.reasoning.lower()

    @pytest.mark.asyncio
    async def test_route_query_working_patterns(self, simple_router):
        """Test query routing for working memory patterns."""
        queries = [
            "What did you just say?",
            "Recent activities",
            "Current status",
            "What's happening now?"
        ]
        
        for query in queries:
            memory_types = await simple_router.route_query(query)
            assert MemoryType.WORKING in memory_types

    @pytest.mark.asyncio
    async def test_route_query_semantic_patterns(self, simple_router):
        """Test query routing for semantic patterns."""
        queries = [
            "What is Python?",
            "Define machine learning",
            "Explain the fact",
            "What does this mean?"
        ]
        
        for query in queries:
            memory_types = await simple_router.route_query(query)
            assert MemoryType.SEMANTIC in memory_types

    @pytest.mark.asyncio
    async def test_route_query_episodic_patterns(self, simple_router):
        """Test query routing for episodic patterns."""
        queries = [
            "Do you remember our conversation?",
            "What happened yesterday?",
            "When did we discuss this?",
            "Past events"
        ]
        
        for query in queries:
            memory_types = await simple_router.route_query(query)
            assert MemoryType.EPISODIC in memory_types

    @pytest.mark.asyncio
    async def test_route_query_procedural_patterns(self, simple_router):
        """Test query routing for procedural patterns."""
        queries = [
            "How to reset password?",
            "How do I configure this?",
            "Steps to complete",
            "Procedure for setup",
            "Instructions please"
        ]
        
        for query in queries:
            memory_types = await simple_router.route_query(query)
            assert MemoryType.PROCEDURAL in memory_types

    @pytest.mark.asyncio
    async def test_route_query_default(self, simple_router):
        """Test default query routing when no patterns match."""
        query = "Generic query"
        
        memory_types = await simple_router.route_query(query)
        
        # Should default to working and semantic
        assert MemoryType.WORKING in memory_types
        assert MemoryType.SEMANTIC in memory_types
        assert len(memory_types) == 2


class TestMemoryRouteDecision:
    """Test MemoryRouteDecision model."""

    def test_create_decision(self):
        """Test creating a routing decision."""
        decision = MemoryRouteDecision(
            memory_type=MemoryType.SEMANTIC,
            confidence=0.95,
            reasoning="Contains factual information"
        )
        
        assert decision.memory_type == MemoryType.SEMANTIC
        assert decision.confidence == 0.95
        assert decision.reasoning == "Contains factual information"

    def test_confidence_validation(self):
        """Test confidence value validation."""
        # Valid confidence
        decision = MemoryRouteDecision(
            memory_type=MemoryType.WORKING,
            confidence=0.5
        )
        assert decision.confidence == 0.5
        
        # Test boundaries
        decision = MemoryRouteDecision(
            memory_type=MemoryType.WORKING,
            confidence=0.0
        )
        assert decision.confidence == 0.0
        
        decision = MemoryRouteDecision(
            memory_type=MemoryType.WORKING,
            confidence=1.0
        )
        assert decision.confidence == 1.0
        
        # Invalid confidence should raise
        with pytest.raises(ValueError):
            MemoryRouteDecision(
                memory_type=MemoryType.WORKING,
                confidence=1.5
            )

    def test_optional_reasoning(self):
        """Test that reasoning is optional."""
        decision = MemoryRouteDecision(
            memory_type=MemoryType.EPISODIC,
            confidence=0.7
        )
        
        assert decision.reasoning is None