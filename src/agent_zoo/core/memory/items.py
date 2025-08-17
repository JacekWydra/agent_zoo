"""
Memory item types for different memory systems.

Each memory type has specific metadata fields that are stored in ChromaDB.
"""

import uuid
from datetime import datetime
from enum import Enum
from typing import Any, Optional
from pydantic import BaseModel, Field


class MemoryType(str, Enum):
    """Types of memory systems."""
    
    WORKING = "working"  # Short-term, limited capacity
    SEMANTIC = "semantic"  # Long-term facts and knowledge
    EPISODIC = "episodic"  # Experiences with temporal context
    PROCEDURAL = "procedural"  # Skills and procedures


class BaseMemoryItem(BaseModel):
    """
    Base class for all memory items.
    
    Contains common fields shared across all memory types.
    All type-specific fields are stored as metadata in ChromaDB.
    """
    
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    content: Any = Field(description="The actual content stored")
    timestamp: datetime = Field(default_factory=datetime.now)
    metadata: dict[str, Any] = Field(default_factory=dict)
    
    # Importance and usage tracking
    importance: float = Field(default=5.0, ge=0.0, le=10.0, description="Static importance/priority score")
    access_count: int = Field(default=0, description="Number of times accessed")
    last_accessed: datetime | None = Field(default=None)
    
    # Embedding for similarity search (optional, computed on storage)
    embedding: list[float] | None = Field(default=None, description="Vector embedding for similarity search")
    
    def update_access(self) -> None:
        """Update access tracking."""
        self.access_count += 1
        self.last_accessed = datetime.now()
    
    def get_metadata(self) -> dict[str, Any]:
        """
        Get metadata for ChromaDB storage.
        
        Base implementation provides common metadata.
        Subclasses extend this with type-specific fields.
        """
        return {
            "importance": self.importance,
            "access_count": self.access_count,
            "last_accessed": self.last_accessed.isoformat() if self.last_accessed else "",
            "_type": self.__class__.__name__,
            "_item_data": self.model_dump_json(),
            **self.metadata  # Include any custom metadata
        }


class WorkingMemoryItem(BaseMemoryItem):
    """
    Short-term, task-relevant memories.
    
    These are active memories currently being used for ongoing tasks.
    They have a lifecycle and can become inactive over time.
    """
    
    # Activity management
    is_active: bool = Field(default=True, description="Is this memory currently relevant?")
    priority: float = Field(default=5.0, ge=0.0, le=10.0, description="Retention priority")
    
    # Task association
    task_id: str | None = Field(default=None, description="Associated task identifier")
    expires_at: datetime | None = Field(default=None, description="When this memory expires")
    
    # Token management for context
    token_count: int = Field(default=0, description="Number of tokens this item uses")
    
    # Lifecycle tracking
    activation_count: int = Field(default=1, description="Times this has been activated")
    deactivated_at: datetime | None = Field(default=None, description="When item became inactive")
    
    # Source tracking
    source: str | None = Field(default=None, description="Where this memory came from")
    related_items: list[str] = Field(default_factory=list, description="Related memory IDs")
    
    def deactivate(self) -> None:
        """Mark this memory as inactive."""
        self.is_active = False
        self.deactivated_at = datetime.now()
        self.importance = max(0.1, self.importance * 0.8)  # Decay importance
    
    def reactivate(self) -> None:
        """Reactivate an inactive memory."""
        self.is_active = True
        self.deactivated_at = None
        self.activation_count += 1
        self.importance = min(10.0, self.importance * 1.2)  # Boost importance
        self.update_access()
    
    def get_metadata(self) -> dict[str, Any]:
        """Get metadata for ChromaDB storage."""
        metadata = super().get_metadata()
        metadata.update({
            "is_active": self.is_active,
            "priority": self.priority,
            "task_id": self.task_id or "",
            "expires_at": self.expires_at.isoformat() if self.expires_at else "",
            "token_count": self.token_count,
            "activation_count": self.activation_count,
            "deactivated_at": self.deactivated_at.isoformat() if self.deactivated_at else "",
            "source": self.source or "",
            "related_items": ",".join(self.related_items) if self.related_items else "",
        })
        return metadata


class SemanticMemoryItem(BaseMemoryItem):
    """
    Long-term facts and knowledge.
    
    These are stable pieces of information that represent
    general knowledge, facts, rules, and definitions.
    """
    
    # Knowledge representation
    concepts: list[str] = Field(default_factory=list, description="Key concepts/entities")
    relationships: dict[str, Any] = Field(default_factory=dict, description="Relations to other facts")
    
    # Confidence and verification
    confidence: float = Field(default=0.5, ge=0.0, le=1.0, description="Confidence in this fact")
    verified: bool = Field(default=False, description="Has this fact been verified?")
    contradicts: list[str] = Field(default_factory=list, description="IDs of contradicting facts")
    
    # Source and provenance
    source: str = Field(default="unknown", description="Where this fact came from")
    source_reliability: float = Field(default=0.5, ge=0.0, le=1.0, description="Source reliability")
    
    # Domain and category
    domain: str | None = Field(default=None, description="Knowledge domain")
    category: str | None = Field(default=None, description="Fact category")
    
    def get_metadata(self) -> dict[str, Any]:
        """Get metadata for ChromaDB storage."""
        metadata = super().get_metadata()
        metadata.update({
            "concepts": ",".join(self.concepts) if self.concepts else "",
            "confidence": self.confidence,
            "verified": self.verified,
            "source": self.source,
            "source_reliability": self.source_reliability,
            "domain": self.domain or "",
            "category": self.category or "",
            "contradicts": ",".join(self.contradicts) if self.contradicts else "",
        })
        return metadata


class EpisodicMemoryItem(BaseMemoryItem):
    """
    Experiences with temporal and emotional context.
    
    These memories represent events, experiences, and stories
    that have temporal context and often emotional significance.
    """
    
    # Event identification
    event_id: str = Field(default_factory=lambda: str(datetime.now().timestamp()), 
                         description="Unique event identifier")
    event_type: str | None = Field(default=None, description="Type of event")
    
    # Temporal context
    event_time: datetime = Field(default_factory=datetime.now, description="When event occurred")
    duration_seconds: float | None = Field(default=None, description="Event duration")
    sequence_number: int | None = Field(default=None, description="Position in event sequence")
    
    # Participants and location
    participants: list[str] = Field(default_factory=list, description="Who was involved")
    location: str | None = Field(default=None, description="Where it happened")
    
    # Emotional and significance
    emotional_valence: float = Field(default=0.0, ge=-1.0, le=1.0, 
                                    description="Emotional tone (-1 negative, +1 positive)")
    significance: float = Field(default=5.0, ge=0.0, le=10.0, 
                               description="How significant was this event")
    
    # Outcomes and lessons
    outcomes: list[str] = Field(default_factory=list, description="What resulted from this")
    lessons_learned: list[str] = Field(default_factory=list, description="What was learned")
    
    def get_metadata(self) -> dict[str, Any]:
        """Get metadata for ChromaDB storage."""
        metadata = super().get_metadata()
        metadata.update({
            "event_id": self.event_id,
            "event_type": self.event_type or "",
            "event_time": self.event_time.isoformat(),
            "duration_seconds": self.duration_seconds or 0,
            "sequence_number": self.sequence_number or 0,
            "participants": ",".join(self.participants) if self.participants else "",
            "location": self.location or "",
            "emotional_valence": self.emotional_valence,
            "significance": self.significance,
            "outcomes": ",".join(self.outcomes) if self.outcomes else "",
            "lessons_learned": ",".join(self.lessons_learned) if self.lessons_learned else "",
        })
        return metadata


class ProceduralMemoryItem(BaseMemoryItem):
    """
    Skills, procedures, and how-to knowledge.
    
    These memories represent procedural knowledge - knowing how
    to do things, skills, and step-by-step procedures.
    """
    
    # Procedure definition
    procedure_name: str = Field(description="Name of the procedure/skill")
    steps: list[dict[str, Any]] = Field(default_factory=list, description="Procedure steps")
    
    # Prerequisites and requirements
    prerequisites: list[str] = Field(default_factory=list, description="Required knowledge/skills")
    required_tools: list[str] = Field(default_factory=list, description="Tools/resources needed")
    
    # Complexity and performance
    complexity: float = Field(default=5.0, ge=0.0, le=10.0, description="Difficulty level")
    estimated_duration: float | None = Field(default=None, description="Time to complete (seconds)")
    success_rate: float = Field(default=0.0, ge=0.0, le=1.0, description="Historical success rate")
    execution_count: int = Field(default=0, description="Times this procedure was executed")
    
    # Domain and categorization
    domain: str = Field(default="general", description="Skill domain")
    skill_type: str | None = Field(default=None, description="Type of skill")
    
    # Variations and alternatives
    variations: list[str] = Field(default_factory=list, description="Alternative approaches")
    common_errors: list[str] = Field(default_factory=list, description="Common mistakes to avoid")
    
    def add_execution(self, success: bool) -> None:
        """Record an execution of this procedure."""
        self.execution_count += 1
        # Update success rate with running average
        if self.execution_count == 1:
            self.success_rate = 1.0 if success else 0.0
        else:
            self.success_rate = ((self.success_rate * (self.execution_count - 1)) + 
                               (1.0 if success else 0.0)) / self.execution_count
        self.update_access()
    
    def get_metadata(self) -> dict[str, Any]:
        """Get metadata for ChromaDB storage."""
        metadata = super().get_metadata()
        metadata.update({
            "procedure_name": self.procedure_name,
            "num_steps": len(self.steps),
            "prerequisites": ",".join(self.prerequisites) if self.prerequisites else "",
            "required_tools": ",".join(self.required_tools) if self.required_tools else "",
            "complexity": self.complexity,
            "estimated_duration": self.estimated_duration or 0,
            "success_rate": self.success_rate,
            "execution_count": self.execution_count,
            "domain": self.domain,
            "skill_type": self.skill_type or "",
            "variations": ",".join(self.variations) if self.variations else "",
            "common_errors": ",".join(self.common_errors) if self.common_errors else "",
        })
        return metadata


# Type mapping for deserialization
MEMORY_TYPE_MAP = {
    "BaseMemoryItem": BaseMemoryItem,
    "WorkingMemoryItem": WorkingMemoryItem,
    "SemanticMemoryItem": SemanticMemoryItem,
    "EpisodicMemoryItem": EpisodicMemoryItem,
    "ProceduralMemoryItem": ProceduralMemoryItem,
}


def deserialize_memory_item(data: dict[str, Any]) -> BaseMemoryItem:
    """
    Deserialize a memory item from stored data.
    
    Args:
        data: Dictionary containing item data and type
        
    Returns:
        Appropriate memory item instance
    """
    item_type = data.get("_type", "BaseMemoryItem")
    item_class = MEMORY_TYPE_MAP.get(item_type, BaseMemoryItem)
    return item_class.model_validate(data)