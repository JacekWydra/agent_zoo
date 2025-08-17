"""
Unit tests for memory item types.

Tests all memory item classes and their specific behaviors.
"""

import json
from datetime import datetime, timedelta

import pytest

from agent_zoo.core.memory.items import (
    BaseMemoryItem,
    EpisodicMemoryItem,
    MemoryType,
    ProceduralMemoryItem,
    SemanticMemoryItem,
    WorkingMemoryItem,
    deserialize_memory_item,
)


class TestBaseMemoryItem:
    """Test BaseMemoryItem functionality."""

    def test_creation_with_defaults(self):
        """Test creating a base memory item with default values."""
        item = BaseMemoryItem(content="Test content")
        
        assert item.content == "Test content"
        assert item.id is not None
        assert isinstance(item.timestamp, datetime)
        assert item.metadata == {}
        assert item.importance == 5.0
        assert item.access_count == 0
        assert item.last_accessed is None
        assert item.embedding is None

    def test_update_access(self):
        """Test access tracking updates."""
        item = BaseMemoryItem(content="Test")
        original_count = item.access_count
        
        item.update_access()
        
        assert item.access_count == original_count + 1
        assert item.last_accessed is not None
        assert isinstance(item.last_accessed, datetime)

    def test_get_metadata(self):
        """Test metadata generation for ChromaDB."""
        item = BaseMemoryItem(
            content="Test",
            importance=7.5,
            metadata={"custom": "value"}
        )
        item.update_access()
        
        metadata = item.get_metadata()
        
        assert metadata["importance"] == 7.5
        assert metadata["access_count"] == 1
        assert metadata["_type"] == "BaseMemoryItem"
        assert "_item_data" in metadata
        assert metadata["custom"] == "value"
        assert "last_accessed" in metadata


class TestWorkingMemoryItem:
    """Test WorkingMemoryItem functionality."""

    def test_creation_with_specific_fields(self, sample_working_item):
        """Test working memory item has all specific fields."""
        assert sample_working_item.is_active is True
        assert sample_working_item.priority == 8.0
        assert sample_working_item.task_id == "task_123"
        assert sample_working_item.token_count == 10
        assert sample_working_item.activation_count == 1
        assert sample_working_item.source == "user_input"
        assert len(sample_working_item.related_items) == 2

    def test_deactivate(self):
        """Test deactivation lifecycle."""
        item = WorkingMemoryItem(
            content="Active task",
            importance=8.0,
            is_active=True
        )
        
        item.deactivate()
        
        assert item.is_active is False
        assert item.deactivated_at is not None
        assert item.importance == 8.0 * 0.8  # Decayed
        assert item.importance >= 0.1  # Never below minimum

    def test_reactivate(self):
        """Test reactivation of inactive memory."""
        item = WorkingMemoryItem(
            content="Task",
            importance=5.0,
            is_active=False,
            activation_count=1
        )
        
        item.reactivate()
        
        assert item.is_active is True
        assert item.deactivated_at is None
        assert item.activation_count == 2
        assert item.importance == 5.0 * 1.2  # Boosted
        assert item.importance <= 10.0  # Never above maximum
        assert item.access_count == 1

    def test_metadata_includes_working_fields(self, sample_working_item):
        """Test metadata includes all working memory specific fields."""
        metadata = sample_working_item.get_metadata()
        
        assert metadata["is_active"] is True
        assert metadata["priority"] == 8.0
        assert metadata["task_id"] == "task_123"
        assert metadata["token_count"] == 10
        assert metadata["activation_count"] == 1
        assert metadata["source"] == "user_input"
        assert "related_items" in metadata


class TestSemanticMemoryItem:
    """Test SemanticMemoryItem functionality."""

    def test_creation_with_specific_fields(self, sample_semantic_item):
        """Test semantic memory item has all specific fields."""
        assert sample_semantic_item.concepts == ["Paris", "France", "capital"]
        assert sample_semantic_item.confidence == 1.0
        assert sample_semantic_item.verified is True
        assert sample_semantic_item.source == "encyclopedia"
        assert sample_semantic_item.source_reliability == 0.95
        assert sample_semantic_item.domain == "geography"
        assert sample_semantic_item.category == "capitals"

    def test_metadata_includes_semantic_fields(self, sample_semantic_item):
        """Test metadata includes all semantic specific fields."""
        metadata = sample_semantic_item.get_metadata()
        
        assert metadata["confidence"] == 1.0
        assert metadata["verified"] is True
        assert metadata["source"] == "encyclopedia"
        assert metadata["source_reliability"] == 0.95
        assert metadata["domain"] == "geography"
        assert metadata["category"] == "capitals"
        assert "concepts" in metadata


class TestEpisodicMemoryItem:
    """Test EpisodicMemoryItem functionality."""

    def test_creation_with_specific_fields(self, sample_episodic_item):
        """Test episodic memory item has all specific fields."""
        assert sample_episodic_item.event_type == "conversation"
        assert isinstance(sample_episodic_item.event_time, datetime)
        assert sample_episodic_item.duration_seconds == 30.5
        assert sample_episodic_item.sequence_number == 1
        assert sample_episodic_item.participants == ["user", "assistant"]
        assert sample_episodic_item.location == "chat_session"
        assert sample_episodic_item.emotional_valence == 0.2
        assert sample_episodic_item.significance == 6.0
        assert sample_episodic_item.outcomes == ["weather_info_provided"]
        assert sample_episodic_item.lessons_learned == ["user_prefers_detailed_forecasts"]

    def test_default_event_id_is_timestamp(self):
        """Test that event_id defaults to timestamp."""
        item = EpisodicMemoryItem(content="Event")
        
        # Event ID should be a timestamp string
        assert item.event_id is not None
        float(item.event_id)  # Should not raise

    def test_metadata_includes_episodic_fields(self, sample_episodic_item):
        """Test metadata includes all episodic specific fields."""
        metadata = sample_episodic_item.get_metadata()
        
        assert metadata["event_type"] == "conversation"
        assert "event_time" in metadata
        assert metadata["duration_seconds"] == 30.5
        assert metadata["sequence_number"] == 1
        assert metadata["location"] == "chat_session"
        assert metadata["emotional_valence"] == 0.2
        assert metadata["significance"] == 6.0
        assert "participants" in metadata
        assert "outcomes" in metadata


class TestProceduralMemoryItem:
    """Test ProceduralMemoryItem functionality."""

    def test_creation_with_specific_fields(self, sample_procedural_item):
        """Test procedural memory item has all specific fields."""
        assert sample_procedural_item.procedure_name == "Make Coffee"
        assert len(sample_procedural_item.steps) == 5
        assert sample_procedural_item.prerequisites == ["coffee_beans", "hot_water"]
        assert sample_procedural_item.required_tools == ["coffee_maker", "mug"]
        assert sample_procedural_item.complexity == 3.0
        assert sample_procedural_item.estimated_duration == 300.0
        assert sample_procedural_item.success_rate == 0.95
        assert sample_procedural_item.execution_count == 20
        assert sample_procedural_item.domain == "culinary"
        assert sample_procedural_item.skill_type == "beverage_preparation"

    def test_add_execution_success(self):
        """Test recording a successful execution."""
        item = ProceduralMemoryItem(
            content="Test procedure",
            procedure_name="Test",
            success_rate=0.8,
            execution_count=4
        )
        
        item.add_execution(success=True)
        
        assert item.execution_count == 5
        # Success rate should be updated: (0.8 * 4 + 1.0) / 5 = 0.84
        assert pytest.approx(item.success_rate) == 0.84
        assert item.access_count == 1

    def test_add_execution_failure(self):
        """Test recording a failed execution."""
        item = ProceduralMemoryItem(
            content="Test procedure",
            procedure_name="Test",
            success_rate=0.8,
            execution_count=4
        )
        
        item.add_execution(success=False)
        
        assert item.execution_count == 5
        # Success rate should be updated: (0.8 * 4 + 0.0) / 5 = 0.64
        assert pytest.approx(item.success_rate) == 0.64
        assert item.access_count == 1

    def test_first_execution(self):
        """Test first execution sets success rate directly."""
        item = ProceduralMemoryItem(
            content="New procedure",
            procedure_name="New",
            success_rate=0.0,
            execution_count=0
        )
        
        item.add_execution(success=True)
        assert item.success_rate == 1.0
        
        item2 = ProceduralMemoryItem(
            content="New procedure 2",
            procedure_name="New2",
            success_rate=0.0,
            execution_count=0
        )
        
        item2.add_execution(success=False)
        assert item2.success_rate == 0.0

    def test_metadata_includes_procedural_fields(self, sample_procedural_item):
        """Test metadata includes all procedural specific fields."""
        metadata = sample_procedural_item.get_metadata()
        
        assert metadata["procedure_name"] == "Make Coffee"
        assert metadata["num_steps"] == 5
        assert metadata["complexity"] == 3.0
        assert metadata["estimated_duration"] == 300.0
        assert metadata["success_rate"] == 0.95
        assert metadata["execution_count"] == 20
        assert metadata["domain"] == "culinary"
        assert metadata["skill_type"] == "beverage_preparation"
        assert "prerequisites" in metadata
        assert "required_tools" in metadata


class TestDeserialization:
    """Test memory item deserialization."""

    def test_deserialize_base_item(self):
        """Test deserializing a base memory item."""
        data = {
            "_type": "BaseMemoryItem",
            "id": "test_id",
            "content": "Test content",
            "importance": 7.0,
            "metadata": {"key": "value"}
        }
        
        item = deserialize_memory_item(data)
        
        assert isinstance(item, BaseMemoryItem)
        assert item.id == "test_id"
        assert item.content == "Test content"
        assert item.importance == 7.0

    def test_deserialize_working_item(self):
        """Test deserializing a working memory item."""
        data = {
            "_type": "WorkingMemoryItem",
            "content": "Working content",
            "is_active": True,
            "priority": 9.0,
            "token_count": 50
        }
        
        item = deserialize_memory_item(data)
        
        assert isinstance(item, WorkingMemoryItem)
        assert item.is_active is True
        assert item.priority == 9.0
        assert item.token_count == 50

    def test_deserialize_semantic_item(self):
        """Test deserializing a semantic memory item."""
        data = {
            "_type": "SemanticMemoryItem",
            "content": "Fact",
            "confidence": 0.9,
            "verified": True,
            "source": "test"
        }
        
        item = deserialize_memory_item(data)
        
        assert isinstance(item, SemanticMemoryItem)
        assert item.confidence == 0.9
        assert item.verified is True

    def test_deserialize_episodic_item(self):
        """Test deserializing an episodic memory item."""
        data = {
            "_type": "EpisodicMemoryItem",
            "content": "Event",
            "event_type": "test_event",
            "significance": 8.0
        }
        
        item = deserialize_memory_item(data)
        
        assert isinstance(item, EpisodicMemoryItem)
        assert item.event_type == "test_event"
        assert item.significance == 8.0

    def test_deserialize_procedural_item(self):
        """Test deserializing a procedural memory item."""
        data = {
            "_type": "ProceduralMemoryItem",
            "content": "Procedure",
            "procedure_name": "Test Procedure",
            "steps": [{"step": 1}],
            "complexity": 5.0
        }
        
        item = deserialize_memory_item(data)
        
        assert isinstance(item, ProceduralMemoryItem)
        assert item.procedure_name == "Test Procedure"
        assert len(item.steps) == 1
        assert item.complexity == 5.0

    def test_deserialize_unknown_type_defaults_to_base(self):
        """Test unknown type defaults to BaseMemoryItem."""
        data = {
            "_type": "UnknownType",
            "content": "Unknown"
        }
        
        item = deserialize_memory_item(data)
        
        assert isinstance(item, BaseMemoryItem)
        assert item.content == "Unknown"

    def test_deserialize_missing_type_defaults_to_base(self):
        """Test missing _type field defaults to BaseMemoryItem."""
        data = {
            "content": "No type"
        }
        
        item = deserialize_memory_item(data)
        
        assert isinstance(item, BaseMemoryItem)
        assert item.content == "No type"


class TestMemoryType:
    """Test MemoryType enum."""

    def test_memory_types_exist(self):
        """Test all memory types are defined."""
        assert MemoryType.WORKING == "working"
        assert MemoryType.SEMANTIC == "semantic"
        assert MemoryType.EPISODIC == "episodic"
        assert MemoryType.PROCEDURAL == "procedural"

    def test_memory_type_iteration(self):
        """Test we can iterate over memory types."""
        types = list(MemoryType)
        assert len(types) == 4
        assert MemoryType.WORKING in types
        assert MemoryType.SEMANTIC in types
        assert MemoryType.EPISODIC in types
        assert MemoryType.PROCEDURAL in types