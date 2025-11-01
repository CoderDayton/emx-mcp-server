#!/usr/bin/env python3
"""
FIXED storage.py - Corrected Event Addition

KEY FIX:
- CRITICAL BUG: Changed event_ids=[event_id] * len(embeddings)
  to event_ids=[event_id] (ONE event per embedding!)

This was collapsing 903 events into duplicates!

storage.py calls add_vectors with:
- vectors: (1, 384) - ONE embedding vector
- event_ids: [event_id] - ONE event_id
- metadata: [{...}] - ONE metadata dict
"""

from collections import deque
from pathlib import Path
from typing import List, Dict, Deque, Optional, TYPE_CHECKING
import json
import time
import logging
import numpy as np
import tempfile
import os

from emx_mcp.storage.vector_store import VectorStore
from emx_mcp.storage.graph_store import GraphStore
from emx_mcp.storage.disk_manager import DiskManager
from emx_mcp.models.events import EpisodicEvent

if TYPE_CHECKING:
    from emx_mcp.gpu.stream_manager import StreamManager

logger = logging.getLogger(__name__)


class HierarchicalMemoryStore:
    """3-tier memory with full disk offloading integration."""

    def __init__(
        self,
        storage_path: str,
        config: dict,
        stream_manager: Optional["StreamManager"] = None,
    ):
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
        # Calculate expected vector count if token budget is known
        expected_vectors = None
        min_training_size = None

        if "expected_total_tokens" in config["storage"]:
            # Estimate: ~27 vectors per event, ~30 tokens per event
            expected_tokens = config["storage"]["expected_total_tokens"]
            expected_events = expected_tokens // 30
            expected_vectors = expected_events * 27

            # Train at 90% of expected vectors to get very close to optimal nlist
            min_training_size = int(expected_vectors * 0.9)

            logger.info(
                f"Estimated {expected_vectors} vectors from {expected_tokens} tokens "
                f"({expected_events} events × ~27 vectors/event), "
                f"will train at {min_training_size} vectors (90%)"
            )

        self.vector_store = VectorStore(
            storage_path=str(vector_path),
            dimension=config["storage"]["vector_dim"],
            nprobe=config["storage"].get("nprobe", 16),
            use_gpu=config["storage"].get("use_gpu", True),
            use_sq=config["storage"].get("use_sq", True),
            sq_bits=config["storage"].get("sq_bits", 8),
            expected_vector_count=expected_vectors,
            min_training_size=min_training_size
            or config["storage"].get("min_training_size"),
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
        self.event_cache: Dict[str, EpisodicEvent] = {}
        self.max_cache_size = 1000

        # GPU stream manager (optional for pipelined operations)
        self.stream_manager = stream_manager

        # Metadata
        self.metadata_path = self.storage_path / "metadata.json"

        # Track previous event ID for temporal linking
        self.last_event_id: Optional[str] = None
        self._load_metadata()

        logger.info(f"HierarchicalMemoryStore initialized at {storage_path}")
        logger.info(
            f"Disk offloading enabled for events >{self.disk_manager.offload_threshold} tokens"
        )

        if self.stream_manager:
            logger.info("GPU stream pipelining enabled for event storage")

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
                "last_event_id": None,
            }

        self.last_event_id = self.metadata.get("last_event_id")

    def _save_metadata(self):
        """Save metadata to disk."""
        self.metadata["last_modified"] = time.time()
        self.metadata["last_event_id"] = self.last_event_id
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
        use_streams: bool = False,
    ) -> dict:
        """
        Add episodic event with FULL INTEGRATION and atomic rollback.

        CRITICAL FIX:
        - embeddings is a SINGLE vector (1, 384)
        - event_ids=[event_id] (NOT [event_id]*len(embeddings)!)
        - metadata=[metadata] (ONE metadata dict)
        """
        timestamp = time.time()
        token_count = len(tokens)
        temp_file_path = None
        offload_completed = False
        vector_added = False
        graph_added = False

        try:
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
                self.disk_manager.offload_event(event_id, event.to_dict())
                offload_completed = True
                logger.info(
                    f"Event {event_id} offloaded to disk ({token_count} tokens)"
                )
            else:
                # Atomic JSON write
                event_file = self.events_path / f"{event_id}.json"
                with tempfile.NamedTemporaryFile(
                    mode="w",
                    dir=str(self.events_path),
                    delete=False,
                    suffix=".tmp",
                ) as tmp:
                    json.dump(event.to_dict(), tmp)
                    tmp.flush()
                    os.fsync(tmp.fileno())
                    temp_file_path = tmp.name

                os.replace(temp_file_path, str(event_file))
                temp_file_path = None

                # Keep in cache
                self.event_cache[event_id] = event
                if len(self.event_cache) > self.max_cache_size:
                    oldest = min(
                        self.event_cache.keys(),
                        key=lambda k: self.event_cache[k].timestamp,
                    )
                    del self.event_cache[oldest]

            # 3. Add to vector store (IVF)
            # CRITICAL FIX: embeddings is (1, 384), event_ids is [event_id]
            if embeddings is not None and len(embeddings) > 0:
                embeddings_array = np.array(embeddings, dtype=np.float32)
                if embeddings_array.ndim == 1:
                    embeddings_array = embeddings_array.reshape(1, -1)

                vector_result = self.vector_store.add_vectors(
                    vectors=embeddings_array,
                    event_ids=[event_id],  # ← CRITICAL FIX: ONE event_id!
                    metadata=[metadata or {}],  # ← ONE metadata dict!
                )
                vector_added = True
            else:
                vector_result = {"status": "no_embeddings"}

            # 4. Add to graph store
            self.graph_store.add_event(
                event_id=event_id,
                timestamp=timestamp,
                token_count=token_count,
                metadata=json.dumps(metadata) if metadata else None,
            )
            graph_added = True

            # 5. Link to previous event
            if self.last_event_id is not None:
                try:
                    self.graph_store.link_events(
                        from_id=self.last_event_id,
                        to_id=event_id,
                        relationship="PRECEDES",
                        lag=1,
                    )
                    logger.debug(f"Linked {self.last_event_id} -> {event_id}")
                except Exception as e:
                    logger.debug(f"Could not link to previous event: {e}")

            # Update last_event_id
            self.last_event_id = event_id

            # 6. Update local context
            self.local_context.extend(tokens)

            # 7. Update metadata
            self.metadata["event_count"] += 1
            self.metadata["total_tokens"] += token_count
            if should_offload:
                self.metadata["offloaded_events"] += 1
            self._save_metadata()

            return {
                "event_id": event_id,
                "status": "added",
                "offloaded": should_offload,
                "vector_result": vector_result,
                "token_count": token_count,
            }

        except Exception as e:
            logger.error(f"Failed to add event {event_id}, rolling back: {e}")

            # Rollback in reverse order
            if graph_added:
                try:
                    self.graph_store.remove_event(event_id)
                    logger.debug(f"Rolled back graph entry for {event_id}")
                except Exception as rollback_err:
                    logger.warning(f"Graph rollback failed: {rollback_err}")

            if vector_added:
                try:
                    self.vector_store.remove_vectors([event_id])
                    logger.debug(f"Rolled back vector entries for {event_id}")
                except Exception as rollback_err:
                    logger.warning(f"Vector rollback failed: {rollback_err}")

            if offload_completed:
                try:
                    self.disk_manager.remove_event(event_id)
                    logger.debug(f"Rolled back disk offload for {event_id}")
                except Exception as rollback_err:
                    logger.warning(f"Disk offload rollback failed: {rollback_err}")
            else:
                if temp_file_path and Path(temp_file_path).exists():
                    try:
                        os.unlink(temp_file_path)
                    except Exception as cleanup_err:
                        logger.warning(f"Temp file cleanup failed: {cleanup_err}")

                event_file = self.events_path / f"{event_id}.json"
                if event_file.exists():
                    try:
                        event_file.unlink()
                    except Exception as rollback_err:
                        logger.warning(f"JSON rollback failed: {rollback_err}")

            if event_id in self.event_cache:
                del self.event_cache[event_id]

            raise

    def get_event(self, event_id: str) -> EpisodicEvent:
        """Retrieve event with automatic cache/disk handling."""
        if event_id in self.event_cache:
            return self.event_cache[event_id]

        event_file = self.events_path / f"{event_id}.json"
        if event_file.exists():
            with open(event_file, "r") as f:
                event_data = json.load(f)
            event = EpisodicEvent.from_dict(event_data)
            self.event_cache[event_id] = event
            return event

        event_data = self.disk_manager.load_event(event_id)
        if event_data:
            event = EpisodicEvent.from_dict(event_data)
            return event

        raise ValueError(f"Event {event_id} not found")

    def remove_events(self, event_ids: list[str]) -> dict:
        """Remove events from ALL storage backends."""
        removed_count = 0
        for event_id in event_ids:
            try:
                self.vector_store.remove_vectors([event_id])
                removed_count += 1
            except Exception as e:
                logger.warning(f"Could not remove {event_id} from vector store: {e}")

            try:
                self.graph_store.remove_event(event_id)
            except Exception as e:
                logger.warning(f"Could not remove {event_id} from graph store: {e}")

            if self.disk_manager.remove_event(event_id):
                self.metadata["offloaded_events"] -= 1

            event_file = self.events_path / f"{event_id}.json"
            if event_file.exists():
                event_file.unlink()

            if event_id in self.event_cache:
                del self.event_cache[event_id]

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
        """Search for similar events and retrieve full objects."""
        event_ids, distances, metadata = self.vector_store.search(query_embedding, k)

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

    def retrain_index(
        self, force: bool = False, expected_vector_count: Optional[int] = None
    ) -> dict:
        """Retrain IVF index with optional expected vector count for optimal nlist."""
        return self.vector_store.retrain(force, expected_vector_count)

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
        self.vector_store.clear()
        self.graph_store.clear()
        self.disk_manager.clear()

        for event_file in self.events_path.glob("*.json"):
            event_file.unlink()

        self.event_cache.clear()
        self.initial_tokens = []
        self.local_context.clear()

        self.metadata["event_count"] = 0
        self.metadata["total_tokens"] = 0
        self.metadata["offloaded_events"] = 0
        self.metadata["last_event_id"] = None
        self.last_event_id = None
        self._save_metadata()

        logger.info("All memory cleared from all storage backends")
