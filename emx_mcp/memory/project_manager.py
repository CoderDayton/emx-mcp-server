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
        self.project_store = HierarchicalMemoryStore(str(self.memory_dir), enriched_config)
        self.global_store = HierarchicalMemoryStore(str(self.global_path), enriched_config)

        # Initialize segmentation and retrieval
        self.segmenter = SurpriseSegmenter(gamma=enriched_config["memory"]["gamma"])
        self.retrieval = TwoStageRetrieval(self.project_store, enriched_config)

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
    ) -> dict:
        """
        Add episodic event to project memory.

        Args:
            tokens: List of token strings
            embeddings: Optional embeddings (if None, will compute locally)
            metadata: Event metadata
        """
        # Compute embeddings if not provided
        if embeddings is None:
            embeddings = self.encoder.encode_individual_tokens(tokens)
            logger.info(f"Generated embeddings locally for {len(tokens)} tokens")

        # Generate UUID-based event ID (no collisions, works across sessions)
        event_id = f"event_{uuid.uuid4().hex}"

        result = self.project_store.add_event(
            event_id, tokens, embeddings.tolist(), metadata or {}
        )

        result["event_id"] = event_id

        return result

    def remove_events(self, event_ids: list[str]) -> dict:
        """Remove events from project memory."""
        return self.project_store.remove_events(event_ids)

    def retrain_index(self, force: bool = False) -> dict:
        """Retrain IVF index."""
        return self.project_store.retrain_index(force)

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
        return self.encoder.encode_individual_tokens(tokens)

    def has_encoder(self) -> bool:
        """Check if embedding encoder is available."""
        return True

    def get_performance_metrics(self) -> dict:
        """
        Get real-time indexing performance metrics for production monitoring.
        
        Returns:
            Dictionary containing:
            - embedding: throughput, batch size, device
            - indexing: vectors/sec, training time, IVF parameters
            - memory: index size, metadata size, total footprint
        """
        import os
        
        index_info = self.get_index_info()
        
        # Calculate memory footprint
        vector_store_path = self.memory_dir / "vector_store"
        graph_store_path = self.memory_dir / "graph_store.db"
        
        index_size_mb = 0.0
        metadata_size_mb = 0.0
        
        if vector_store_path.exists():
            for file_path in vector_store_path.rglob("*"):
                if file_path.is_file():
                    size_bytes = file_path.stat().st_size
                    if file_path.suffix == ".pkl":
                        metadata_size_mb += size_bytes / (1024 * 1024)
                    else:
                        index_size_mb += size_bytes / (1024 * 1024)
        
        if graph_store_path.exists():
            metadata_size_mb += graph_store_path.stat().st_size / (1024 * 1024)
        
        # Embedding throughput estimation (based on configured batch size)
        # Actual throughput will vary; this provides configured capacity
        embedding_batch_size = self.config["model"]["batch_size"]
        embedding_device = self.config["model"]["device"]
        
        return {
            "embedding": {
                "model": self.encoder.model_name,
                "dimension": self.encoder.dimension,
                "device": embedding_device,
                "configured_batch_size": embedding_batch_size,
                "note": "Use OTel emx.embedding.duration histogram for actual throughput",
            },
            "indexing": {
                "total_vectors": index_info["total_vectors"],
                "is_trained": index_info["is_trained"],
                "nlist": index_info["nlist"],
                "optimal_nlist": index_info["optimal_nlist"],
                "nlist_drift": index_info["nlist_drift"],
                "nprobe": index_info["nprobe"],
                "buffered_vectors": index_info["buffered_vectors"],
                "gpu_enabled": index_info["gpu_enabled"],
                "gpu_device_id": index_info.get("gpu_device_id"),
                "auto_retrain": index_info["auto_retrain"],
                "optimization_count": index_info["optimization_count"],
            },
            "memory": {
                "index_size_mb": round(index_size_mb, 2),
                "metadata_size_mb": round(metadata_size_mb, 2),
                "total_size_mb": round(index_size_mb + metadata_size_mb, 2),
                "vector_count": index_info["total_vectors"],
                "avg_bytes_per_vector": (
                    round((index_size_mb * 1024 * 1024) / index_info["total_vectors"], 2)
                    if index_info["total_vectors"] > 0
                    else 0
                ),
            },
            "storage_path": str(self.memory_dir),
        }
