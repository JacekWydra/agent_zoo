"""
Main Memory class using ChromaDB as the single source of truth.

All memory operations go through ChromaDB collections with vector search.
"""

import json
from datetime import datetime, timedelta
from typing import Any, Optional

try:
    import chromadb
    from chromadb.config import Settings
    from chromadb.utils import embedding_functions

    CHROMADB_AVAILABLE = True
except ImportError:
    CHROMADB_AVAILABLE = False

from agent_zoo.core.memory.items import (
    BaseMemoryItem,
    EpisodicMemoryItem,
    MemoryType,
    ProceduralMemoryItem,
    SemanticMemoryItem,
    WorkingMemoryItem,
    deserialize_memory_item,
)


class MemorySearchResult:
    """Result from memory search."""

    def __init__(
        self, item: BaseMemoryItem, score: float, distance: float, memory_type: MemoryType
    ):
        self.item = item
        self.score = score  # Similarity score (higher is better)
        self.distance = distance  # Raw distance from ChromaDB
        self.memory_type = memory_type


class Memory:
    """
    Main memory system using ChromaDB as the single source of truth.

    Each memory type has its own collection in ChromaDB. All storage,
    retrieval, and management goes through the vector database.
    """

    def __init__(
        self,
        persist_directory: str | None = None,
        embedding_function: Any | None = None,
        distance_metric: str = "cosine",
    ):
        """
        Initialize the memory system.

        Args:
            persist_directory: Directory to persist ChromaDB data (None for in-memory)
            embedding_function: Custom embedding function (None for default)
            distance_metric: Distance metric for similarity ("cosine", "l2", or "ip")
        """
        if not CHROMADB_AVAILABLE:
            raise ImportError("ChromaDB is not installed. Install it with: pip install chromadb")

        # Initialize ChromaDB client
        if persist_directory:
            self.client = chromadb.PersistentClient(
                path=persist_directory, settings=Settings(anonymized_telemetry=False)
            )
        else:
            self.client = chromadb.EphemeralClient(settings=Settings(anonymized_telemetry=False))

        # Set up embedding function
        if embedding_function is None:
            # Use default sentence transformer
            self.embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(
                model_name="all-MiniLM-L6-v2"  # Fast and good quality
            )
        else:
            self.embedding_function = embedding_function

        self.distance_metric = distance_metric

        # Create collections for each memory type
        self.collections = {}
        for memory_type in MemoryType:
            collection_name = f"{memory_type.value}_memory"
            self.collections[memory_type] = self.client.get_or_create_collection(
                name=collection_name,
                embedding_function=self.embedding_function,
                metadata={"hnsw:space": distance_metric},
            )

    async def store(
        self, item: BaseMemoryItem, memory_type: MemoryType, embedding: list[float] | None = None
    ) -> str:
        """
        Store a memory item in the appropriate collection.
        
        Validates that the item type matches the memory type.

        Args:
            item: Memory item to store
            memory_type: Which memory system to store in
            embedding: Pre-computed embedding (optional)

        Returns:
            ID of the stored item
            
        Raises:
            ValueError: If item type doesn't match memory type
        """
        # Validate item type matches memory type
        expected_types = {
            MemoryType.WORKING: WorkingMemoryItem,
            MemoryType.SEMANTIC: SemanticMemoryItem,
            MemoryType.EPISODIC: EpisodicMemoryItem,
            MemoryType.PROCEDURAL: ProceduralMemoryItem,
        }
        
        expected_type = expected_types.get(memory_type)
        if expected_type and not isinstance(item, expected_type):
            # Try to convert if it's a base item
            if type(item) == BaseMemoryItem:
                item = self._convert_to_specific_type(item, memory_type)
            else:
                raise ValueError(
                    f"Item type {type(item).__name__} doesn't match "
                    f"memory type {memory_type.value} (expected {expected_type.__name__})"
                )
        
        collection = self.collections[memory_type]

        # Prepare document text for embedding
        if isinstance(item.content, str):
            document = item.content
        else:
            document = json.dumps(item.content)

        # Get all metadata including type-specific fields
        metadata = item.get_metadata()
        metadata["memory_type"] = memory_type.value

        # Add to collection
        if embedding:
            collection.add(
                ids=[item.id], embeddings=[embedding], documents=[document], metadatas=[metadata]
            )
        else:
            collection.add(ids=[item.id], documents=[document], metadatas=[metadata])

        return item.id

    async def retrieve(
        self,
        query: str,
        n_results: int = 10,
        memory_types: list[MemoryType] | None = None,
        filters: dict[str, Any] | None = None,
        max_tokens: int | None = None,
    ) -> list[MemorySearchResult]:
        """
        Retrieve memories using vector similarity search.

        Args:
            query: Search query text
            n_results: Maximum number of results
            memory_types: Which memory types to search (None = all)
            filters: Metadata filters to apply
            max_tokens: Maximum token budget for results

        Returns:
            List of search results sorted by relevance
        """
        if memory_types is None:
            memory_types = list(MemoryType)

        all_results = []
        remaining_tokens = max_tokens or float("inf")

        for memory_type in memory_types:
            collection = self.collections[memory_type]

            # Build where clause for filtering
            where_clause = filters.copy() if filters else {}

            # Add memory-type specific filters
            if memory_type == MemoryType.WORKING:
                # For working memory, default to active items only
                if "is_active" not in where_clause:
                    where_clause["is_active"] = True

            # Query the collection
            results = collection.query(
                query_texts=[query],
                n_results=n_results * 2,  # Get more to account for token filtering
                where=where_clause if where_clause else None,
                include=["metadatas", "distances", "documents"],
            )

            if not results["ids"][0]:
                continue

            # Process results
            for i in range(len(results["ids"][0])):
                metadata = results["metadatas"][0][i]

                # Deserialize the memory item to the correct type
                item_data = json.loads(metadata["_item_data"])
                # deserialize_memory_item already returns the correct type based on _type field
                item = deserialize_memory_item(item_data)
                
                # Ensure it's the right type for this memory type
                expected_types = {
                    MemoryType.WORKING: WorkingMemoryItem,
                    MemoryType.SEMANTIC: SemanticMemoryItem,
                    MemoryType.EPISODIC: EpisodicMemoryItem,
                    MemoryType.PROCEDURAL: ProceduralMemoryItem,
                }
                expected_type = expected_types.get(memory_type)
                if expected_type and not isinstance(item, expected_type):
                    # This shouldn't happen if data is consistent, but log warning
                    print(f"Warning: Retrieved {type(item).__name__} from {memory_type.value} collection")

                # Check token budget for working memory items
                if isinstance(item, WorkingMemoryItem):
                    if item.token_count > remaining_tokens:
                        continue
                    remaining_tokens -= item.token_count

                # Calculate similarity score
                distance = results["distances"][0][i]
                score = self._distance_to_score(distance)

                # Update access tracking
                item.update_access()
                await self.update(item, memory_type)

                all_results.append(
                    MemorySearchResult(
                        item=item, score=score, distance=distance, memory_type=memory_type
                    )
                )

        # Sort by combined relevance score
        all_results.sort(key=lambda x: self._calculate_relevance(x), reverse=True)

        return all_results[:n_results]

    async def update(self, item: BaseMemoryItem, memory_type: MemoryType) -> None:
        """
        Update an existing memory item.

        Args:
            item: Updated memory item
            memory_type: Which memory system it's in
        """
        collection = self.collections[memory_type]

        # ChromaDB doesn't have direct update, so delete and re-add
        try:
            collection.delete(ids=[item.id])
        except:
            pass  # Item might not exist

        await self.store(item, memory_type)

    async def delete(self, item_id: str, memory_type: MemoryType | None = None) -> bool:
        """
        Delete a memory item.

        Args:
            item_id: ID of item to delete
            memory_type: Memory type (None to search all)

        Returns:
            True if deleted, False if not found
        """
        if memory_type:
            collections = [self.collections[memory_type]]
        else:
            collections = self.collections.values()

        for collection in collections:
            try:
                collection.delete(ids=[item_id])
                return True
            except:
                continue

        return False

    async def get_by_id(
        self, item_id: str, memory_type: MemoryType | None = None
    ) -> BaseMemoryItem | None:
        """
        Get a specific memory item by ID.

        Args:
            item_id: Item ID
            memory_type: Memory type (None to search all)

        Returns:
            Memory item or None if not found
        """
        if memory_type:
            collections = [self.collections[memory_type]]
        else:
            collections = self.collections.values()

        for collection in collections:
            results = collection.get(ids=[item_id], include=["metadatas"])

            if results["ids"]:
                metadata = results["metadatas"][0]
                item_data = json.loads(metadata["_item_data"])
                return deserialize_memory_item(item_data)

        return None

    async def consolidate(self) -> dict[str, Any]:
        """
        Consolidate memory by managing lifecycle and migration.

        This handles:
        - Deactivating old working memories
        - Migrating important items between memory types
        - Cleaning up expired items

        Returns:
            Consolidation statistics
        """
        stats = {
            "deactivated": 0,
            "migrated": 0,
            "deleted": 0,
            "timestamp": datetime.now().isoformat(),
        }

        # Process working memory
        working_collection = self.collections[MemoryType.WORKING]

        # Find old active items
        cutoff_time = (datetime.now() - timedelta(minutes=5)).isoformat()

        old_active = working_collection.get(
            where={"$and": [{"is_active": True}, {"last_accessed": {"$lt": cutoff_time}}]},
            include=["metadatas"],
        )

        # Deactivate old items
        for metadata in old_active["metadatas"]:
            item_data = json.loads(metadata["_item_data"])
            item = WorkingMemoryItem.model_validate(item_data)
            item.deactivate()
            await self.update(item, MemoryType.WORKING)
            stats["deactivated"] += 1

        # Find important inactive items to migrate
        important_inactive = working_collection.get(
            where={"$and": [{"is_active": False}, {"importance": {"$gte": 7.0}}]},
            include=["metadatas", "documents"],
        )

        # Migrate to semantic memory
        for i, metadata in enumerate(important_inactive["metadatas"]):
            # Create semantic memory item
            semantic_item = SemanticMemoryItem(
                content=important_inactive["documents"][i],
                importance=metadata["importance"],
                source="migrated_from_working",
                confidence=0.7,
            )

            await self.store(semantic_item, MemoryType.SEMANTIC)

            # Delete from working memory
            await self.delete(metadata["id"], MemoryType.WORKING)

            stats["migrated"] += 1

        # Clean up expired items
        now = datetime.now().isoformat()
        expired = working_collection.get(where={"expires_at": {"$lt": now}}, include=["ids"])

        for item_id in expired["ids"]:
            await self.delete(item_id, MemoryType.WORKING)
            stats["deleted"] += 1

        return stats

    async def clear(self, memory_type: MemoryType | None = None) -> None:
        """
        Clear all memories or specific memory type.

        Args:
            memory_type: Specific type to clear (None for all)
        """
        if memory_type:
            # Clear specific type
            collection_name = f"{memory_type.value}_memory"
            try:
                self.client.delete_collection(collection_name)
            except:
                pass
            self.collections[memory_type] = self.client.get_or_create_collection(
                name=collection_name,
                embedding_function=self.embedding_function,
                metadata={"hnsw:space": self.distance_metric},
            )
        else:
            # Clear all types
            for mem_type in list(MemoryType):
                collection_name = f"{mem_type.value}_memory"
                try:
                    self.client.delete_collection(collection_name)
                except:
                    pass
                self.collections[mem_type] = self.client.get_or_create_collection(
                    name=collection_name,
                    embedding_function=self.embedding_function,
                    metadata={"hnsw:space": self.distance_metric},
                )

    def get_stats(self) -> dict[str, Any]:
        """Get statistics about memory usage."""
        stats = {"total_items": 0, "by_type": {}, "distance_metric": self.distance_metric}

        for memory_type, collection in self.collections.items():
            count = collection.count()
            stats["by_type"][memory_type.value] = count
            stats["total_items"] += count

        return stats

    def _distance_to_score(self, distance: float) -> float:
        """
        Convert distance to similarity score.

        Args:
            distance: Raw distance from ChromaDB

        Returns:
            Similarity score (0 to 1, higher is better)
        """
        if self.distance_metric == "cosine":
            # Cosine distance is 0-2, convert to 0-1 similarity
            return 1.0 - (distance / 2.0)
        elif self.distance_metric == "l2":
            # L2 distance, use inverse
            return 1.0 / (1.0 + distance)
        else:  # inner product
            # Higher is better for inner product
            return max(0.0, distance)

    def _calculate_relevance(self, result: MemorySearchResult) -> float:
        """
        Calculate combined relevance score for sorting.

        Args:
            result: Search result to score

        Returns:
            Combined relevance score
        """
        # Base similarity score
        similarity = result.score

        # Importance factor
        importance = result.item.importance / 10.0

        # Recency factor
        if result.item.last_accessed:
            age_hours = (datetime.now() - result.item.last_accessed).total_seconds() / 3600
            recency = 1.0 / (1.0 + age_hours * 0.1)
        else:
            recency = 0.5

        # Memory type weight
        type_weights = {
            MemoryType.WORKING: 1.2,  # Boost current context
            MemoryType.SEMANTIC: 1.0,  # Neutral
            MemoryType.EPISODIC: 0.9,  # Slightly lower
            MemoryType.PROCEDURAL: 0.8,  # Lower priority
        }
        type_weight = type_weights.get(result.memory_type, 1.0)

        # Combine factors
        return similarity * importance * recency * type_weight
    
    def _convert_to_specific_type(self, item: BaseMemoryItem, memory_type: MemoryType) -> BaseMemoryItem:
        """
        Convert a base memory item to a specific type.
        
        Args:
            item: Base memory item to convert
            memory_type: Target memory type
            
        Returns:
            Converted item of the appropriate type
        """
        base_data = {
            "id": item.id,
            "content": item.content,
            "timestamp": item.timestamp,
            "metadata": item.metadata,
            "importance": item.importance,
            "access_count": item.access_count,
            "last_accessed": item.last_accessed,
            "embedding": item.embedding,
        }
        
        if memory_type == MemoryType.WORKING:
            return WorkingMemoryItem(**base_data)
        elif memory_type == MemoryType.SEMANTIC:
            return SemanticMemoryItem(**base_data, source="converted")
        elif memory_type == MemoryType.EPISODIC:
            return EpisodicMemoryItem(**base_data)
        elif memory_type == MemoryType.PROCEDURAL:
            return ProceduralMemoryItem(**base_data, procedure_name="Unnamed")
        else:
            return item
