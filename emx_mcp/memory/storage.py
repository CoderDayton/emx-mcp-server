"""Hierarchical memory storage with full integration."""

from collections import deque
from pathlib import Path
from typing import List, Dict, Deque
import json
import time
import logging
import numpy as np

from emx_mcp.storage.vector_store import VectorStore
from emx_mcp.storage.graph_store import GraphStore
from emx_mcp.storage.disk_manager import DiskManager
from emx_mcp.models.events import EpisodicEvent

logger = logging.getLogger(__name__)


class HierarchicalMemoryStore:
    """
    3-tier memory with full disk offloading integration.

    - Tier 1: Initial tokens (attention sinks)
    - Tier 2: Local context (working memory)
    - Tier 3: Episodic events (IVF + Graph + Disk)
    """

    def __init__(self, storage_path: str, config: dict):
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)
        self.config = config

        # Tier 1: Initial tokens
        self.initial_tokens: List[str] = []
        self.n_init = config["memory"]["n_init"]

        # Tier 2: Local context
        self.n_local = config["memory"]["n_local"]
        self.local_context: Deque[str] = deque(maxlen=self.n_local)

        # Tier 3: Episodic memory (FULLY INTEGRATED)
        vector_path = self.storage_path / "vector_index"
        graph_path = self.storage_path / "graph_db"
        disk_path = self.storage_path / "disk_offload"
        events_path = self.storage_path / "events"

        # Initialize all storage backends
        self.vector_store = VectorStore(
            str(vector_path),
            dimension=config["storage"]["vector_dim"],
            nprobe=config["storage"].get("nprobe", 8),
        )
        self.graph_store = GraphStore(str(graph_path))
        self.disk_manager = DiskManager(
            str(disk_path),
            offload_threshold=config["storage"].get("disk_offload_threshold", 300000),
        )

        # Events directory for JSON storage
        self.events_path = events_path
        self.events_path.mkdir(exist_ok=True)

        # Event cache (in-memory for fast access)
        self.event_cache: Dict[str, EpisodicEvent] = {}  # event_id -> EpisodicEvent
        self.max_cache_size = 1000

        # Metadata
        self.metadata_path = self.storage_path / "metadata.json"
        self._load_metadata()

        logger.info(f"HierarchicalMemoryStore initialized at {storage_path}")
        logger.info(
            f"Disk offloading enabled for events >{self.disk_manager.offload_threshold} tokens"
        )

    def _load_metadata(self):
        """Load or initialize metadata."""
        if self.metadata_path.exists():
            with open(self.metadata_path, "r") as f:
                self.metadata = json.load(f)
        else:
            self.metadata = {
                "created_at": time.time(),
                "last_modified": None,
                "event_count": 0,
                "total_tokens": 0,
                "offloaded_events": 0,
            }

    def _save_metadata(self):
        """Save metadata to disk."""
        self.metadata["last_modified"] = time.time()
        with open(self.metadata_path, "w") as f:
            json.dump(self.metadata, f, indent=2)

    def add_event(
        self,
        event_id: str,
        tokens: list,
        embeddings: list,
        metadata: dict,
        boundaries: list | None = None,
        surprise_scores: list | None = None,
    ) -> dict:
        """
        Add episodic event with FULL INTEGRATION.

        Flow:
        1. Create EpisodicEvent object
        2. Check if needs disk offloading
        3. Add embeddings to vector store (IVF)
        4. Add to graph store for temporal relationships
        5. Save event JSON (or offload to disk)
        6. Update local context
        7. Link to previous event in graph
        """
        timestamp = time.time()
        token_count = len(tokens)

        # 1. Create event object
        event = EpisodicEvent(
            event_id=event_id,
            tokens=tokens,
            embeddings=embeddings,
            boundaries=boundaries or [0, token_count],
            timestamp=timestamp,
            metadata=metadata,
            surprise_scores=surprise_scores,
        )

        # 2. Check disk offloading
        should_offload = self.disk_manager.should_offload(token_count)

        if should_offload:
            # Offload full event to disk with mmap support
            self.disk_manager.offload_event(event_id, event.to_dict())
            logger.info(f"Event {event_id} offloaded to disk ({token_count} tokens)")
            self.metadata["offloaded_events"] += 1
        else:
            # Save to JSON (fast access)
            event_file = self.events_path / f"{event_id}.json"
            with open(event_file, "w") as f:
                json.dump(event.to_dict(), f)

            # Keep in cache
            self.event_cache[event_id] = event

            # Limit cache size
            if len(self.event_cache) > self.max_cache_size:
                # Remove oldest
                oldest = min(
                    self.event_cache.keys(), key=lambda k: self.event_cache[k].timestamp
                )
                del self.event_cache[oldest]

        # 3. Add to vector store (IVF)
        vector_result = self.vector_store.add_vectors(
            vectors=np.array(embeddings, dtype=np.float32),
            event_ids=[event_id] * len(embeddings),
            metadata=[metadata or {}] * len(embeddings),
        )

        # 4. Add to graph store
        self.graph_store.add_event(
            event_id=event_id,
            timestamp=timestamp,
            token_count=token_count,
            metadata=json.dumps(metadata) if metadata else None,
        )

        # 5. Link to previous event (temporal relationship)
        if self.metadata["event_count"] > 0:
            # Get previous event ID
            prev_event_id = f"event_{self.metadata['event_count'] - 1}"
            try:
                self.graph_store.link_events(
                    from_id=prev_event_id,
                    to_id=event_id,
                    relationship="PRECEDES",
                    lag=1,
                )
            except Exception as e:
                logger.debug(f"Could not link to previous event: {e}")

        # 6. Update local context
        self.local_context.extend(tokens)

        # 7. Update metadata
        self.metadata["event_count"] += 1
        self.metadata["total_tokens"] += token_count
        self._save_metadata()

        return {
            "event_id": event_id,
            "status": "added",
            "offloaded": should_offload,
            "vector_result": vector_result,
            "token_count": token_count,
        }

    def get_event(self, event_id: str) -> EpisodicEvent:
        """
        Retrieve event with automatic cache/disk handling.

        Priority:
        1. Check in-memory cache
        2. Load from JSON file
        3. Load from disk offload
        """
        # Check cache
        if event_id in self.event_cache:
            return self.event_cache[event_id]

        # Check JSON file
        event_file = self.events_path / f"{event_id}.json"
        if event_file.exists():
            with open(event_file, "r") as f:
                event_data = json.load(f)
            event = EpisodicEvent.from_dict(event_data)
            self.event_cache[event_id] = event
            return event

        # Check disk offload
        event_data = self.disk_manager.load_event(event_id)
        if event_data:
            event = EpisodicEvent.from_dict(event_data)
            return event

        raise ValueError(f"Event {event_id} not found")

    def remove_events(self, event_ids: list[str]) -> dict:
        """
        Remove events from ALL storage backends.

        Removes from:
        1. Vector store (IVF)
        2. Graph store (SQLite)
        3. Disk offload (if offloaded)
        4. JSON files
        5. Cache
        """
        removed_count = 0

        for event_id in event_ids:
            # 1. Remove from vector store
            try:
                self.vector_store.remove_vectors([event_id])
                removed_count += 1
            except Exception as e:
                logger.warning(f"Could not remove {event_id} from vector store: {e}")

            # 2. Remove from graph store
            try:
                self.graph_store.remove_event(event_id)
            except Exception as e:
                logger.warning(f"Could not remove {event_id} from graph store: {e}")

            # 3. Remove from disk offload
            if self.disk_manager.remove_event(event_id):
                self.metadata["offloaded_events"] -= 1

            # 4. Remove JSON file
            event_file = self.events_path / f"{event_id}.json"
            if event_file.exists():
                event_file.unlink()

            # 5. Remove from cache
            if event_id in self.event_cache:
                del self.event_cache[event_id]

        # Update metadata
        self.metadata["event_count"] = self.graph_store.count_events()
        self._save_metadata()

        return {
            "status": "removed",
            "removed_count": removed_count,
            "total_events": self.metadata["event_count"],
        }

    def search_events(
        self, query_embedding: np.ndarray, k: int = 10
    ) -> list[EpisodicEvent]:
        """
        Search for similar events and retrieve full objects.

        Flow:
        1. Query vector store (IVF)
        2. Retrieve full event objects from storage
        3. Return list of EpisodicEvent objects
        """
        # Search vector store
        event_ids, distances, metadata = self.vector_store.search(query_embedding, k)

        # Retrieve full events
        events = []
        for event_id, distance in zip(event_ids, distances):
            try:
                event = self.get_event(event_id)
                events.append(event)
            except Exception as e:
                logger.warning(f"Could not retrieve event {event_id}: {e}")

        return events

    def get_temporal_neighbors(self, event_id: str, max_distance: int = 3) -> list[str]:
        """Get temporally adjacent events via graph store."""
        return self.graph_store.get_neighbors(
            event_id, max_distance=max_distance, bidirectional=True
        )

    def prune_least_accessed(self, limit: int = 1000) -> int:
        """Prune least accessed events from all backends."""
        to_prune = self.graph_store.get_least_accessed_events(limit)
        if to_prune:
            result = self.remove_events(to_prune)
            return result["removed_count"]
        return 0

    def retrain_index(self, force: bool = False) -> dict:
        """Retrain IVF index."""
        return self.vector_store.retrain(force)

    def get_index_info(self) -> dict:
        """Get IVF index statistics."""
        info = self.vector_store.get_info()
        info["disk_offloaded"] = self.metadata.get("offloaded_events", 0)
        info["disk_stats"] = self.disk_manager.get_stats()
        return info

    def event_count(self) -> int:
        """Get total event count."""
        return self.metadata["event_count"]

    def get_semantic_knowledge(self) -> dict:
        """Get semantic knowledge summary."""
        return {
            "event_count": self.event_count(),
            "vector_count": self.vector_store.count(),
            "total_tokens": self.metadata["total_tokens"],
            "offloaded_events": self.metadata.get("offloaded_events", 0),
        }

    def clear(self):
        """Clear all memory from all backends."""
        # Clear vector store
        self.vector_store.clear()

        # Clear graph store
        self.graph_store.clear()

        # Clear disk manager
        self.disk_manager.clear()

        # Clear JSON files
        for event_file in self.events_path.glob("*.json"):
            event_file.unlink()

        # Clear cache
        self.event_cache.clear()
        self.initial_tokens = []
        self.local_context.clear()

        # Reset metadata
        self.metadata["event_count"] = 0
        self.metadata["total_tokens"] = 0
        self.metadata["offloaded_events"] = 0
        self._save_metadata()

        logger.info("All memory cleared from all storage backends")
