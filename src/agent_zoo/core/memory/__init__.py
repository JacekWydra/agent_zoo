"""
Memory systems for LLM agents.

This module provides a unified memory architecture with:
- ChromaDB as the single source of truth
- Four memory types: Working, Semantic, Episodic, Procedural
- LLM-based intelligent routing
- High-level MemoryManager for orchestration
"""

from agent_zoo.core.memory.items import (
    MemoryType,
    BaseMemoryItem,
    WorkingMemoryItem,
    SemanticMemoryItem,
    EpisodicMemoryItem,
    ProceduralMemoryItem,
    deserialize_memory_item,
)
from agent_zoo.core.memory.memory import (
    Memory,
    MemorySearchResult,
)
from agent_zoo.core.memory.router import (
    LLMMemoryRouter,
    SimpleRouter,
    MemoryRouteDecision,
)
from agent_zoo.core.memory.manager import (
    MemoryManager,
    MemoryManagerConfig,
)

__all__ = [
    # Memory types enum
    "MemoryType",
    # Memory items
    "BaseMemoryItem",
    "WorkingMemoryItem",
    "SemanticMemoryItem",
    "EpisodicMemoryItem",
    "ProceduralMemoryItem",
    "deserialize_memory_item",
    # Core memory system
    "Memory",
    "MemorySearchResult",
    # Routing
    "LLMMemoryRouter",
    "SimpleRouter",
    "MemoryRouteDecision",
    # High-level manager
    "MemoryManager",
    "MemoryManagerConfig",
]