"""
Unit tests for the Memory class with ChromaDB integration.

Tests storage, retrieval, and lifecycle management.
"""

import json
from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch

import pytest

from agent_zoo.core.memory.items import (
    BaseMemoryItem,
    EpisodicMemoryItem,
    MemoryType,
    SemanticMemoryItem,
    WorkingMemoryItem,
)
from agent_zoo.core.memory.memory import Memory, MemorySearchResult


@pytest.fixture
def memory_instance(mock_embedding_function):
    """Create a Memory instance with mocked ChromaDB."""
    with patch("agent_zoo.core.memory.memory.CHROMADB_AVAILABLE", True):
        with patch("agent_zoo.core.memory.memory.chromadb") as mock_chromadb:
            # Mock ChromaDB client
            mock_client = MagicMock()
            mock_chromadb.EphemeralClient.return_value = mock_client
            
            # Mock collections
            mock_collections = {}
            for memory_type in MemoryType:
                mock_collection = MagicMock()
                mock_collection.name = f"{memory_type.value}_memory"
                mock_collections[memory_type] = mock_collection
                
            def get_or_create_collection(name, **kwargs):
                for memory_type in MemoryType:
                    if name == f"{memory_type.value}_memory":
                        return mock_collections[memory_type]
                return MagicMock()
            
            mock_client.get_or_create_collection = get_or_create_collection
            
            # Create memory instance
            memory = Memory(embedding_function=mock_embedding_function)
            memory.collections = mock_collections
            
            return memory


class TestMemoryInitialization:
    """Test Memory class initialization."""

    def test_chromadb_not_available(self):
        """Test error when ChromaDB is not installed."""
        with patch("agent_zoo.core.memory.memory.CHROMADB_AVAILABLE", False):
            with pytest.raises(ImportError, match="ChromaDB is not installed"):
                Memory()

    def test_creates_collections_for_each_type(self, memory_instance):
        """Test that collections are created for each memory type."""
        assert len(memory_instance.collections) == 4
        assert MemoryType.WORKING in memory_instance.collections
        assert MemoryType.SEMANTIC in memory_instance.collections
        assert MemoryType.EPISODIC in memory_instance.collections
        assert MemoryType.PROCEDURAL in memory_instance.collections


class TestMemoryStore:
    """Test memory storage operations."""

    @pytest.mark.asyncio
    async def test_store_with_type_validation(self, memory_instance, sample_working_item):
        """Test that store validates item type matches memory type."""
        # Correct type should work
        item_id = await memory_instance.store(sample_working_item, MemoryType.WORKING)
        assert item_id == sample_working_item.id

    @pytest.mark.asyncio
    async def test_store_wrong_type_raises_error(self, memory_instance, sample_semantic_item):
        """Test storing wrong item type raises error."""
        with pytest.raises(ValueError, match="doesn't match memory type"):
            await memory_instance.store(sample_semantic_item, MemoryType.WORKING)

    @pytest.mark.asyncio
    async def test_store_base_item_converts_to_specific(self, memory_instance):
        """Test that base items are converted to specific types."""
        base_item = BaseMemoryItem(content="Test content")
        
        # Should convert base item to WorkingMemoryItem
        item_id = await memory_instance.store(base_item, MemoryType.WORKING)
        assert item_id == base_item.id

    @pytest.mark.asyncio
    async def test_store_with_custom_embedding(self, memory_instance):
        """Test storing with pre-computed embedding."""
        item = WorkingMemoryItem(content="Test")
        embedding = [0.1] * 384
        
        await memory_instance.store(item, MemoryType.WORKING, embedding=embedding)
        
        collection = memory_instance.collections[MemoryType.WORKING]
        collection.add.assert_called_once()
        call_args = collection.add.call_args[1]
        assert call_args["embeddings"] == [embedding]


class TestMemoryRetrieve:
    """Test memory retrieval operations."""

    @pytest.mark.asyncio
    async def test_retrieve_from_specific_types(self, memory_instance):
        """Test retrieving from specific memory types."""
        # Mock collection query response
        mock_results = {
            "ids": [["item1", "item2"]],
            "distances": [[0.1, 0.2]],
            "metadatas": [[
                {
                    "_item_data": json.dumps({
                        "_type": "WorkingMemoryItem",
                        "content": "Result 1",
                        "is_active": True,
                        "priority": 5.0
                    }),
                    "importance": 7.0
                },
                {
                    "_item_data": json.dumps({
                        "_type": "WorkingMemoryItem",
                        "content": "Result 2",
                        "is_active": True,
                        "priority": 6.0
                    }),
                    "importance": 8.0
                }
            ]],
            "documents": [["Result 1", "Result 2"]]
        }
        
        memory_instance.collections[MemoryType.WORKING].query.return_value = mock_results
        
        results = await memory_instance.retrieve(
            query="test query",
            n_results=2,
            memory_types=[MemoryType.WORKING]
        )
        
        assert len(results) == 2
        assert isinstance(results[0], MemorySearchResult)
        assert isinstance(results[0].item, WorkingMemoryItem)

    @pytest.mark.asyncio
    async def test_retrieve_with_token_limit(self, memory_instance):
        """Test retrieval respects token limits for working memory."""
        # Mock results with different token counts
        mock_results = {
            "ids": [["item1", "item2", "item3"]],
            "distances": [[0.1, 0.2, 0.3]],
            "metadatas": [[
                {
                    "_item_data": json.dumps({
                        "_type": "WorkingMemoryItem",
                        "content": "Small",
                        "token_count": 10
                    }),
                    "importance": 5.0
                },
                {
                    "_item_data": json.dumps({
                        "_type": "WorkingMemoryItem",
                        "content": "Large",
                        "token_count": 100
                    }),
                    "importance": 5.0
                },
                {
                    "_item_data": json.dumps({
                        "_type": "WorkingMemoryItem",
                        "content": "Medium",
                        "token_count": 20
                    }),
                    "importance": 5.0
                }
            ]],
            "documents": [["Small", "Large", "Medium"]]
        }
        
        memory_instance.collections[MemoryType.WORKING].query.return_value = mock_results
        
        results = await memory_instance.retrieve(
            query="test",
            memory_types=[MemoryType.WORKING],
            max_tokens=35  # Should include items 1 and 3, but not 2
        )
        
        # Should filter out the large item
        assert len(results) <= 2
        total_tokens = sum(r.item.token_count for r in results if hasattr(r.item, 'token_count'))
        assert total_tokens <= 35

    @pytest.mark.asyncio
    async def test_retrieve_filters_inactive_working_memory(self, memory_instance):
        """Test that inactive working memory items are filtered by default."""
        memory_instance.collections[MemoryType.WORKING].query.return_value = {
            "ids": [[]],
            "distances": [[]],
            "metadatas": [[]],
            "documents": [[]]
        }
        
        await memory_instance.retrieve(
            query="test",
            memory_types=[MemoryType.WORKING]
        )
        
        # Check that is_active filter was applied
        call_args = memory_instance.collections[MemoryType.WORKING].query.call_args[1]
        assert call_args["where"]["is_active"] is True


class TestMemoryLifecycle:
    """Test memory lifecycle operations."""

    @pytest.mark.asyncio
    async def test_update_item(self, memory_instance, sample_working_item):
        """Test updating an existing memory item."""
        await memory_instance.update(sample_working_item, MemoryType.WORKING)
        
        collection = memory_instance.collections[MemoryType.WORKING]
        # Should delete and re-add
        collection.delete.assert_called_once_with(ids=[sample_working_item.id])

    @pytest.mark.asyncio
    async def test_delete_item_from_specific_type(self, memory_instance):
        """Test deleting item from specific memory type."""
        memory_instance.collections[MemoryType.WORKING].delete.return_value = None
        
        result = await memory_instance.delete("item_id", MemoryType.WORKING)
        
        assert result is True
        memory_instance.collections[MemoryType.WORKING].delete.assert_called_once_with(ids=["item_id"])

    @pytest.mark.asyncio
    async def test_delete_item_searches_all_types(self, memory_instance):
        """Test deleting item searches all types when none specified."""
        # Make first collections raise exception, last one succeed
        for memory_type in list(MemoryType)[:-1]:
            memory_instance.collections[memory_type].delete.side_effect = Exception("Not found")
        memory_instance.collections[MemoryType.PROCEDURAL].delete.return_value = None
        
        result = await memory_instance.delete("item_id")
        
        assert result is True

    @pytest.mark.asyncio
    async def test_get_by_id(self, memory_instance):
        """Test retrieving item by ID."""
        mock_results = {
            "ids": ["item_id"],
            "metadatas": [{
                "_item_data": json.dumps({
                    "_type": "SemanticMemoryItem",
                    "content": "Fact",
                    "source": "test",
                    "confidence": 0.9
                })
            }]
        }
        
        memory_instance.collections[MemoryType.SEMANTIC].get.return_value = mock_results
        
        item = await memory_instance.get_by_id("item_id", MemoryType.SEMANTIC)
        
        assert isinstance(item, SemanticMemoryItem)
        assert item.content == "Fact"
        assert item.confidence == 0.9

    @pytest.mark.asyncio
    async def test_consolidate(self, memory_instance):
        """Test memory consolidation process."""
        # Mock old active items to deactivate
        old_active = {
            "ids": [],
            "metadatas": [{
                "_item_data": json.dumps({
                    "_type": "WorkingMemoryItem",
                    "id": "old_item",
                    "content": "Old active",
                    "is_active": True
                }),
                "id": "old_item"
            }]
        }
        
        # Mock important inactive items to migrate
        important_inactive = {
            "ids": [],
            "metadatas": [{
                "id": "important_item",
                "importance": 8.0,
                "_item_data": json.dumps({
                    "_type": "WorkingMemoryItem",
                    "content": "Important"
                })
            }],
            "documents": ["Important content"]
        }
        
        # Mock expired items
        expired = {
            "ids": ["expired_item"]
        }
        
        memory_instance.collections[MemoryType.WORKING].get.side_effect = [
            old_active,
            important_inactive,
            expired
        ]
        
        stats = await memory_instance.consolidate()
        
        assert "deactivated" in stats
        assert "migrated" in stats
        assert "deleted" in stats
        assert "timestamp" in stats

    @pytest.mark.asyncio
    async def test_clear_specific_type(self, memory_instance):
        """Test clearing specific memory type."""
        await memory_instance.clear(MemoryType.WORKING)
        
        # Should delete and recreate the collection
        memory_instance.client.delete_collection.assert_called()

    @pytest.mark.asyncio
    async def test_clear_all_types(self, memory_instance):
        """Test clearing all memory types."""
        await memory_instance.clear()
        
        # Should delete all collections
        assert memory_instance.client.delete_collection.call_count >= 4

    def test_get_stats(self, memory_instance):
        """Test getting memory statistics."""
        for memory_type in MemoryType:
            memory_instance.collections[memory_type].count.return_value = 10
        
        stats = memory_instance.get_stats()
        
        assert stats["total_items"] == 40
        assert stats["by_type"][MemoryType.WORKING.value] == 10
        assert stats["distance_metric"] == "cosine"


class TestRelevanceCalculation:
    """Test relevance and scoring calculations."""

    def test_distance_to_score_cosine(self, memory_instance):
        """Test cosine distance to score conversion."""
        memory_instance.distance_metric = "cosine"
        
        score = memory_instance._distance_to_score(0.0)  # Perfect match
        assert score == 1.0
        
        score = memory_instance._distance_to_score(2.0)  # Maximum distance
        assert score == 0.0
        
        score = memory_instance._distance_to_score(1.0)  # Middle distance
        assert score == 0.5

    def test_distance_to_score_l2(self, memory_instance):
        """Test L2 distance to score conversion."""
        memory_instance.distance_metric = "l2"
        
        score = memory_instance._distance_to_score(0.0)  # Perfect match
        assert score == 1.0
        
        score = memory_instance._distance_to_score(1.0)
        assert score == 0.5  # 1/(1+1)

    def test_distance_to_score_inner_product(self, memory_instance):
        """Test inner product distance to score conversion."""
        memory_instance.distance_metric = "ip"
        
        score = memory_instance._distance_to_score(10.0)
        assert score == 10.0
        
        score = memory_instance._distance_to_score(-5.0)
        assert score == 0.0  # Negative clamped to 0

    def test_calculate_relevance(self, memory_instance):
        """Test combined relevance calculation."""
        item = WorkingMemoryItem(
            content="Test",
            importance=8.0,
            last_accessed=datetime.now()
        )
        
        result = MemorySearchResult(
            item=item,
            score=0.9,  # High similarity
            distance=0.2,
            memory_type=MemoryType.WORKING
        )
        
        relevance = memory_instance._calculate_relevance(result)
        
        # Should combine similarity * importance * recency * type_weight
        assert relevance > 0
        assert relevance <= 10.8  # Max possible: 1.0 * 1.0 * 1.0 * 1.2 * 10

    def test_convert_to_specific_type(self, memory_instance):
        """Test converting base item to specific types."""
        base_item = BaseMemoryItem(
            content="Test",
            importance=7.0,
            metadata={"key": "value"}
        )
        
        # Convert to WorkingMemoryItem
        working = memory_instance._convert_to_specific_type(base_item, MemoryType.WORKING)
        assert isinstance(working, WorkingMemoryItem)
        assert working.content == "Test"
        assert working.importance == 7.0
        
        # Convert to SemanticMemoryItem
        semantic = memory_instance._convert_to_specific_type(base_item, MemoryType.SEMANTIC)
        assert isinstance(semantic, SemanticMemoryItem)
        assert semantic.source == "converted"
        
        # Convert to EpisodicMemoryItem
        episodic = memory_instance._convert_to_specific_type(base_item, MemoryType.EPISODIC)
        assert isinstance(episodic, EpisodicMemoryItem)
        
        # Convert to ProceduralMemoryItem
        procedural = memory_instance._convert_to_specific_type(base_item, MemoryType.PROCEDURAL)
        assert isinstance(procedural, procedural.__class__)
        assert procedural.procedure_name == "Unnamed"