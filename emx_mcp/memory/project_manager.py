"""Project-based memory manager with IVF-indexed storage + O(n) linear segmentation only."""

import time
import tarfile
import numpy as np
import uuid
from pathlib import Path
from typing import Optional, Any, Dict, List
import logging
from numpy.typing import NDArray

from emx_mcp.memory.storage import HierarchicalMemoryStore
from emx_mcp.memory.segmentation import SurpriseSegmenter
from emx_mcp.memory.retrieval import TwoStageRetrieval
from emx_mcp.embeddings.encoder import EmbeddingEncoder
from emx_mcp.utils.hardware import enrich_config_with_hardware

logger = logging.getLogger(__name__)


class ProjectMemoryManager:
    """Manages per-project and global episodic memory with IVF indexing."""

    encoder: EmbeddingEncoder

    def __init__(self, project_path: str, global_path: str, config: dict):
        self.project_path = Path(project_path)
        self.global_path = Path(global_path)

        # Enrich config with hardware detection (device/batch_size)
        # This makes config the single source of truth for runtime values
        enriched_config = enrich_config_with_hardware(config)
        self.config = enriched_config

        # Initialize .memories folder in project
        self.memory_dir = self.project_path / ".memories"
        self.memory_dir.mkdir(parents=True, exist_ok=True)

        # Initialize global memories
        self.global_path.mkdir(parents=True, exist_ok=True)

        # Initialize embedding encoder FIRST (required for auto-detecting vector dimension)
        try:
            self.encoder = EmbeddingEncoder(
                model_name=enriched_config["model"]["name"],
                device=enriched_config["model"]["device"],
                batch_size=enriched_config["model"]["batch_size"],
                gpu_config=enriched_config.get("gpu"),
            )

            logger.info(f"Embedding encoder initialized (dim={self.encoder.dimension})")

            # Auto-detect vector dimension from model if not explicitly configured
            configured_dim = enriched_config["storage"]["vector_dim"]
            actual_dim = self.encoder.dimension

            if configured_dim is None:
                # Auto-detection: use model's native dimension
                enriched_config["storage"]["vector_dim"] = actual_dim
                logger.info(f"Auto-detected vector dimension: {actual_dim}")
            elif configured_dim != actual_dim:
                # Explicit dimension set but mismatches model
                error_msg = (
                    f"Vector dimension mismatch: EMX_STORAGE_VECTOR_DIM={configured_dim} "
                    f"but model '{enriched_config['model']['name']}' outputs {actual_dim}-dimensional vectors.\n"
                    f"\n"
                    f"Common model dimensions:\n"
                    f" - all-MiniLM-L6-v2: 384\n"
                    f" - all-mpnet-base-v2: 768\n"
                    f" - paraphrase-multilingual-MiniLM-L12-v2: 384\n"
                    f"\n"
                    f"Fix: Set EMX_STORAGE_VECTOR_DIM={actual_dim} or remove it to enable auto-detection.\n"
                    f"See: ENVIRONMENT_VARIABLES.md#emx_storage_vector_dim"
                )

                logger.error(error_msg)
                raise ValueError(error_msg)

        except ImportError as e:
            logger.error(
                "Embedding encoder not available (sentence-transformers not installed)"
            )

            raise ImportError(
                "sentence-transformers required. Install with: pip install sentence-transformers"
            ) from e

        # Create project and global stores (after encoder initialization for auto-detection)
        self.project_store = HierarchicalMemoryStore(
            str(self.memory_dir), enriched_config
        )
        self.global_store = HierarchicalMemoryStore(
            str(self.global_path), enriched_config
        )

        # Initialize segmentation and retrieval
        self.segmenter = SurpriseSegmenter(gamma=enriched_config["memory"]["gamma"])
        self.retrieval = TwoStageRetrieval(self.project_store, enriched_config)

        # Batch encoding buffer for event aggregation (2-3x faster ingestion)
        self.pending_events: List[Dict[str, Any]] = []
        self.batch_event_threshold = enriched_config.get("memory", {}).get(
            "batch_event_threshold", 10
        )
        logger.info(
            f"Batch event threshold: {self.batch_event_threshold} events per encoding batch"
        )

        logger.info(f"Project memory initialized at {self.memory_dir}")
        logger.info("Using O(n) linear segmentation for all document sizes")

    def get_local_context(self) -> list:
        """Get project's current local context."""
        return list(self.project_store.local_context)

    def get_global_context(self) -> list:
        """Get global shared context."""
        return list(self.global_store.local_context)

    def get_global_semantic(self) -> dict:
        """Get global semantic knowledge."""
        return self.global_store.get_semantic_knowledge()

    def segment_tokens(
        self,
        tokens: list,
        gamma: float,
        context_window: int = 10,
    ) -> dict:
        """
        Segment tokens into episodic events using embedding-based surprise.

        Uses embedding-based surprise calculation and O(n) linear segmentation.
        Pure O(n) complexity - no O(n³) refinement overhead.

        Args:
            tokens: List of token strings
            gamma: Surprise threshold sensitivity (higher = fewer boundaries)
            context_window: Context window size for embedding-based surprise calculation

        Returns:
            Dictionary containing segmentation results:
            - initial_boundaries: Surprise-based boundaries (O(n) method)
            - refined_boundaries: Same as initial (no refinement needed with O(n))
            - num_events: Number of events detected
            - method: "embedding-surprise-linear"
            - context_window: Context window used
            - embedding_model: Model name
            - success: True if segmentation succeeded
        """
        # Generate per-token embeddings with local context
        logger.info(f"Computing embeddings for {len(tokens)} tokens (O(n))...")
        token_embeddings = self.encoder.encode_tokens_with_context(
            tokens, context_window=context_window
        )

        # Use O(n) linear segmentation for all documents
        logger.info(f"Segmenting {len(tokens)} tokens using O(n) linear method...")
        boundaries = self.segmenter.segment_by_coherence_linear(
            token_embeddings,
            window_size=5,
            min_segment_length=20,
            surprise_threshold=None,
        )

        # Prepare results
        return {
            "initial_boundaries": boundaries,
            "refined_boundaries": boundaries,  # No refinement needed
            "num_events": len(boundaries) - 1,
            "method": "embedding-surprise-linear",
            "context_window": context_window,
            "embedding_model": self.encoder.model_name,
            "success": True,
            "complexity": "O(n)",
        }

    def retrieve_memories(
        self,
        query_embedding: list,
        k_similarity: int,
        k_contiguity: int,
        use_contiguity: bool,
    ) -> dict:
        """Retrieve relevant memories via two-stage retrieval."""
        return self.retrieval.retrieve(
            query_embedding, k_similarity, k_contiguity, use_contiguity
        )

    def add_event(
        self,
        tokens: list,
        embeddings: Optional[NDArray[np.float32]] = None,
        metadata: Optional[dict] = None,
        force_flush: bool = False,
    ) -> dict:
        """
        Add episodic event to project memory with batch encoding optimization.

        Events are buffered and encoded in batches to reduce model loading overhead.
        This provides 2-3x faster ingestion compared to per-event encoding.

        Args:
            tokens: List of token strings
            embeddings: Optional embeddings (if None, will compute in batch)
            metadata: Event metadata
            force_flush: If True, immediately flush pending events

        Returns:
            Dict with event_id and status, or buffer status if not flushed yet
        """
        # Generate UUID-based event ID (no collisions, works across sessions)
        event_id = f"event_{uuid.uuid4().hex}"

        # If embeddings provided, bypass batching (already computed)
        if embeddings is not None:
            result = self.project_store.add_event(
                event_id, tokens, embeddings.tolist(), metadata or {}
            )
            result["event_id"] = event_id
            return result

        # Buffer event for batch encoding
        self.pending_events.append(
            {
                "event_id": event_id,
                "tokens": tokens,
                "metadata": metadata or {},
            }
        )

        # Check if we should flush the buffer
        should_flush = (
            len(self.pending_events) >= self.batch_event_threshold or force_flush
        )

        if should_flush:
            flush_result = self._flush_pending_events()
            # Return result for the event we just added
            for result in flush_result["events_added"]:
                if result["event_id"] == event_id:
                    return result

        # Event buffered but not yet added
        return {
            "event_id": event_id,
            "status": "buffered",
            "buffered_count": len(self.pending_events),
            "threshold": self.batch_event_threshold,
        }

    def _flush_pending_events(self) -> dict:
        """
        Flush buffered events by batch-encoding all tokens together.

        This is the core optimization: instead of encoding 26 tokens per event
        in separate model calls, we encode 260+ tokens in a single batch pass.

        Returns:
            Dict with flush statistics and results for each event
        """
        if not self.pending_events:
            return {"status": "no_events", "events_added": []}

        start_time = time.time()
        num_events = len(self.pending_events)

        # Flatten all tokens from all pending events
        all_tokens = []
        event_boundaries = [0]  # Track where each event's tokens start/end

        for event in self.pending_events:
            all_tokens.extend(event["tokens"])
            event_boundaries.append(len(all_tokens))

        # Single batch encoding pass (KEY OPTIMIZATION)
        logger.info(
            f"Batch encoding {num_events} events ({len(all_tokens)} tokens) in single pass"
        )
        all_embeddings = self.encoder.encode_tokens_with_context(
            all_tokens, context_window=self.config["memory"]["context_window"]
        )

        # Split embeddings back to per-event chunks
        results = []
        for i, event in enumerate(self.pending_events):
            start_idx = event_boundaries[i]
            end_idx = event_boundaries[i + 1]
            event_embeddings = all_embeddings[start_idx:end_idx]

            # Add to storage
            result = self.project_store.add_event(
                event["event_id"],
                event["tokens"],
                event_embeddings.tolist(),
                event["metadata"],
            )
            result["event_id"] = event["event_id"]
            results.append(result)

        # Clear buffer
        self.pending_events.clear()

        elapsed = time.time() - start_time
        tokens_per_sec = len(all_tokens) / elapsed if elapsed > 0 else 0

        logger.info(
            f"Flushed {num_events} events ({len(all_tokens)} tokens) in {elapsed:.2f}s "
            f"({tokens_per_sec:.0f} tokens/sec)"
        )

        return {
            "status": "flushed",
            "events_added": results,
            "num_events": num_events,
            "total_tokens": len(all_tokens),
            "elapsed_seconds": elapsed,
            "tokens_per_second": tokens_per_sec,
        }

    def flush_events(self) -> dict:
        """
        Manually flush any buffered events.

        Useful at the end of a batch operation to ensure all events are persisted.
        """
        return self._flush_pending_events()

    def remove_events(self, event_ids: list[str]) -> dict:
        """Remove events from project memory."""
        return self.project_store.remove_events(event_ids)

    def retrain_index(
        self, force: bool = False, expected_vector_count: Optional[int] = None
    ) -> dict:
        """Retrain IVF index with optional expected vector count for optimal nlist."""
        return self.project_store.retrain_index(force, expected_vector_count)

    def optimize_memory(
        self, prune_old_events: bool, compress_embeddings: bool
    ) -> dict:
        """Optimize memory storage."""
        results: Dict[str, List[Dict[str, Any]]] = {"optimizations": []}

        if prune_old_events:
            pruned = self.project_store.prune_least_accessed(limit=1000)

            results["optimizations"].append(
                {"type": "pruning", "events_removed": pruned}
            )

        if compress_embeddings:
            # Future: implement PQ compression
            results["optimizations"].append(
                {"type": "compression", "status": "not_implemented"}
            )

        return results

    def get_stats(self) -> dict:
        """Get memory statistics."""
        return {
            "project_events": self.project_store.event_count(),
            "global_events": self.global_store.event_count(),
            "local_context_size": len(self.project_store.local_context),
            "memory_path": str(self.memory_dir),
        }

    def get_index_info(self) -> dict:
        """Get IVF index information."""
        return self.project_store.get_index_info()

    def clear_memory(self):
        """Clear project memory (keeps global)."""
        self.project_store.clear()

    def export_memory(self, output_path: str | Path) -> dict:
        """Export project memory to tar.gz."""
        output_path = Path(output_path)

        with tarfile.open(output_path, "w:gz") as tar:
            tar.add(self.memory_dir, arcname=".memories")

        return {
            "status": "exported",
            "path": str(output_path),
            "size_bytes": output_path.stat().st_size,
        }

    def import_memory(self, input_path: str, merge: bool) -> dict:
        """Import memory from tar.gz."""
        if not merge:
            self.clear_memory()

        with tarfile.open(input_path, "r:gz") as tar:
            # Use 'data' filter (PEP 706) to prevent path traversal attacks
            # Python 3.12+ only: blocks absolute paths, symlinks outside dest, devices
            tar.extractall(self.project_path, filter="data")

        # Reload store
        self.project_store = HierarchicalMemoryStore(str(self.memory_dir), self.config)

        return {
            "status": "imported",
            "merge": merge,
            "events": self.project_store.event_count(),
        }

    def encode_query(self, query: str) -> NDArray[np.float32]:
        """
        Encode query string for retrieval.

        Args:
            query: Query text

        Returns:
            Query embedding vector
        """
        return self.encoder.get_query_embedding(query)

    def encode_tokens(self, tokens: list[str]) -> NDArray[np.float32]:
        """
        Encode token list into embeddings.

        Args:
            tokens: List of token strings

        Returns:
            Embeddings array
        """
        return self.encoder.encode_tokens_with_context(tokens)

    def has_encoder(self) -> bool:
        """Check if embedding encoder is available."""
        return True
