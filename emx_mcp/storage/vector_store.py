#!/usr/bin/env python3
"""
CORRECTED vector_store.py - Supports Multiple Vectors Per Event

THE REAL ARCHITECTURE:
- Each event can have MULTIPLE embedding vectors (one per token)
- storage.add_event() passes: event_id, [27 embeddings], [metadata]
- We need to create N vectors for 1 event_id (not 1 vector per event!)

This is different from the EM-LLM paper (which uses event-level embeddings).
Your implementation uses TOKEN-level embeddings within each event.

The solution:
- Store mapping: event_id → list of vector_ids
- When searching, return event_id (not individual vectors)
- Aggregate results at event level
"""

import json
import logging
import pickle
import threading
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import faiss  # type: ignore[import-untyped]
import numpy as np

logger = logging.getLogger(__name__)


class GPUResourceManager:
    """Manages FAISS GPU resources with automatic fallback to CPU."""

    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return

        self.gpu_available = False
        self.gpu_resources = None

        try:
            num_gpus = faiss.get_num_gpus()
            if num_gpus > 0:
                self.gpu_available = True
                self.gpu_resources = faiss.StandardGpuResources()
                logger.info(f"GPU acceleration available: {num_gpus} GPU(s) detected")
            else:
                logger.warning("No GPUs detected, using CPU-only mode")
        except Exception as e:
            logger.warning(f"GPU initialization failed: {e}, falling back to CPU")
            self.gpu_available = False

        self._initialized = True

    def get_resources(self):
        return self.gpu_resources if self.gpu_available else None

    def is_available(self) -> bool:
        return self.gpu_available


class VectorStore:
    """
    IVF+SQ vector database with support for multiple vectors per event.

    KEY DESIGN:
    - Events have MULTIPLE vectors (one per token in the event)
    - event_id → [vector_id_0, vector_id_1, ...]
    - Search returns event_ids (aggregated from vector results)
    - One training pass with optimal nlist (stays FIXED)
    """

    def __init__(
        self,
        storage_path: str,
        dimension: int = 384,
        nlist: Optional[int] = None,
        nprobe: int = 16,
        use_gpu: bool = True,
        gpu_device_id: int = 0,
        use_sq: bool = True,
        sq_bits: int = 8,
        expected_vector_count: Optional[int] = None,
        min_training_size: Optional[int] = None,
    ):
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)
        self.dimension = dimension
        self.nprobe = nprobe
        self.use_gpu = use_gpu
        self.gpu_device_id = gpu_device_id
        self.use_sq = use_sq
        self.sq_bits = sq_bits

        self.gpu_manager = GPUResourceManager()
        self.gpu_enabled = self.use_gpu and self.gpu_manager.is_available()

        self.index_path = self.storage_path / "faiss_ivf_index.bin"
        self.metadata_path = self.storage_path / "metadata.pkl"
        self.id_map_path = self.storage_path / "id_map.json"

        # CRITICAL: event_id → list of vector_ids (MULTIPLE vectors per event!)
        self.event_id_to_vector_ids: Dict[str, List[int]] = {}
        self.vector_id_to_event_id: Dict[int, str] = {}
        self.next_vector_id = 0
        self._id_map_lock = threading.Lock()

        self.is_trained = False
        self.metadata: List[dict] = []

        # Training strategy configuration
        self.expected_vector_count = expected_vector_count

        # If expected count provided, use optimal nlist and train at 10% of expected
        if expected_vector_count:
            self.nlist = nlist or self._calculate_optimal_nlist(expected_vector_count)
            self.min_training_size = min_training_size or max(
                100, expected_vector_count // 10
            )
            logger.info(
                f"Expected {expected_vector_count} vectors → "
                f"nlist={self.nlist}, min_training_size={self.min_training_size}"
            )
        else:
            # Fallback: train early, accept suboptimal nlist
            self.nlist = nlist or 128
            self.min_training_size = min_training_size or 100
            logger.info(
                f"No expected count → fallback mode: "
                f"nlist={self.nlist}, min_training_size={self.min_training_size}"
            )

        # Training buffer
        self.training_vectors: List[np.ndarray] = []
        self.training_event_ids: List[List[str]] = []  # One entry per batch
        self.training_metadata: List[List[dict]] = []

        if self.index_path.exists():
            self._load_index()
        else:
            self._create_index()

    def _normalize_vectors(self, vectors: np.ndarray) -> np.ndarray:
        """L2 normalize for cosine similarity."""
        vectors = np.array(vectors, dtype=np.float32)
        faiss.normalize_L2(vectors)
        return vectors

    def _calculate_optimal_nlist(self, n_vectors: int) -> int:
        """Calculate optimal nlist: 4 * sqrt(N)"""
        return max(1, int(4 * np.sqrt(n_vectors)))

    def _create_index(self):
        """Create new IVF+SQ index."""
        logger.info(
            f"Creating 8-bit SQ index (dim={self.dimension}, nlist={self.nlist})"
        )

        if self.use_sq:
            quantizer = faiss.IndexFlatIP(self.dimension)
            cpu_index = faiss.IndexIVFScalarQuantizer(
                quantizer,
                self.dimension,
                self.nlist,
                faiss.ScalarQuantizer.QT_8bit,
                faiss.METRIC_INNER_PRODUCT,
            )
        else:
            quantizer = faiss.IndexFlatIP(self.dimension)
            cpu_index = faiss.IndexIVFFlat(
                quantizer, self.dimension, self.nlist, faiss.METRIC_INNER_PRODUCT
            )

        if self.gpu_enabled:
            try:
                gpu_resources = self.gpu_manager.get_resources()
                if gpu_resources is None:
                    raise RuntimeError("GPU resources not available")
                self.index = faiss.index_cpu_to_gpu(
                    gpu_resources, self.gpu_device_id, cpu_index
                )
                logger.info(f"Index transferred to GPU {self.gpu_device_id}")
            except Exception as e:
                logger.warning(f"GPU transfer failed: {e}, using CPU")
                self.index = cpu_index
                self.gpu_enabled = False
        else:
            self.index = cpu_index

        self.is_trained = False

    def _train_index(self):
        """Train index ONCE on initial batch."""
        if not self.training_vectors:
            logger.warning("No training vectors available")
            return

        all_training = np.vstack(self.training_vectors)
        n_vectors = all_training.shape[0]

        optimal_nlist = self._calculate_optimal_nlist(n_vectors)
        old_nlist = self.nlist

        logger.info(
            f"Training index on {n_vectors} vectors "
            f"(optimal_nlist={optimal_nlist}, use_sq={self.use_sq})"
        )

        # Cap to Faiss requirement: n_training >= nlist
        if n_vectors < optimal_nlist:
            fg = 39  # Faiss guideline
            logger.warning(
                f"Training vectors ({n_vectors}) less than optimal nlist ({optimal_nlist}), "
                f"capping nlist to {n_vectors // fg}"
            )
            optimal_nlist = max(1, n_vectors // fg)

        if self.nlist != optimal_nlist:
            self.nlist = optimal_nlist
            logger.info(f"Set nlist: {old_nlist} → {optimal_nlist} (FIXED)")

        # Create new index with optimal nlist
        if self.use_sq:
            quantizer = faiss.IndexFlatIP(self.dimension)
            cpu_index = faiss.IndexIVFScalarQuantizer(
                quantizer,
                self.dimension,
                self.nlist,
                faiss.ScalarQuantizer.QT_8bit,
                faiss.METRIC_INNER_PRODUCT,
            )
        else:
            quantizer = faiss.IndexFlatIP(self.dimension)
            cpu_index = faiss.IndexIVFFlat(
                quantizer, self.dimension, self.nlist, faiss.METRIC_INNER_PRODUCT
            )

        all_training = self._normalize_vectors(all_training)
        logger.info(f"Training quantizer on {n_vectors} vectors...")
        cpu_index.train(all_training)  # type: ignore
        logger.info("Training complete")

        if self.gpu_enabled:
            try:
                gpu_resources = self.gpu_manager.get_resources()
                if gpu_resources is None:
                    raise RuntimeError("GPU resources not available")
                self.index = faiss.index_cpu_to_gpu(
                    gpu_resources, self.gpu_device_id, cpu_index
                )
                logger.info(f"Trained index moved to GPU {self.gpu_device_id}")
            except Exception as e:
                logger.warning(f"GPU transfer failed: {e}, using CPU")
                self.index = cpu_index
                self.gpu_enabled = False
        else:
            self.index = cpu_index

        self.is_trained = True
        self._set_nprobe()

        # Add all buffered training vectors
        logger.info(f"Adding {n_vectors} buffered vectors to trained index")
        vectors_added = 0

        for batch_vectors, batch_event_ids, batch_metadata in zip(
            self.training_vectors, self.training_event_ids, self.training_metadata
        ):
            # FIX: batch_vectors is shape (n,384) not (batch_size, n, 384)
            # batch_event_ids is FLAT list of event_ids for each vector

            vector_ids = []
            with self._id_map_lock:
                for i, event_id in enumerate(batch_event_ids):
                    vid = self.next_vector_id
                    self.vector_id_to_event_id[vid] = event_id

                    # Add to event_id_to_vector_ids mapping
                    if event_id not in self.event_id_to_vector_ids:
                        self.event_id_to_vector_ids[event_id] = []
                    self.event_id_to_vector_ids[event_id].append(vid)

                    vector_ids.append(vid)
                    self.next_vector_id += 1

            vector_ids_array = np.array(vector_ids, dtype=np.int64)

            if len(vector_ids_array) != batch_vectors.shape[0]:
                raise AssertionError(
                    f"Shape mismatch: {len(vector_ids_array)} vector_ids vs "
                    f"{batch_vectors.shape[0]} vectors in batch. "
                    f"batch_event_ids length: {len(batch_event_ids)}"
                )

            self.index.add_with_ids(batch_vectors, vector_ids_array)  # type: ignore
            self.metadata.extend(batch_metadata)
            vectors_added += batch_vectors.shape[0]

        self.training_vectors.clear()
        self.training_event_ids.clear()
        self.training_metadata.clear()

        logger.info(f"Successfully trained and added {vectors_added} vectors")
        self._save()

    def _set_nprobe(self):
        """Set nprobe on IVF index."""
        try:
            index_type = type(self.index).__name__
            if "IVF" in index_type:
                self.index.nprobe = self.nprobe
                logger.debug(f"Set nprobe={self.nprobe} on {index_type}")
        except Exception as e:
            logger.warning(f"Could not set nprobe: {e}")

    def add_vectors(
        self, vectors: np.ndarray, event_ids: List[str], metadata: List[dict]
    ) -> dict:  # sourcery skip: extract-method
        """
        Add vectors to index - retrain if optimal nlist changes significantly.

        Args:
            vectors: Embedding vectors to add
            event_ids: Event IDs for the vectors
            metadata: Metadata for the vectors
        """
        vectors = self._normalize_vectors(vectors)
        n_vectors = vectors.shape[0]

        # Handle both cases:
        # 1. event_ids=[event_id] (ONE for all vectors in batch)
        # 2. event_ids=[event_id]*n_vectors (REPEATED)
        if len(event_ids) == 1:
            # Case 1: Single event_id for all vectors
            event_id = event_ids[0]
            event_ids_expanded = [event_id] * n_vectors
            metadata_expanded = metadata * n_vectors if len(metadata) == 1 else metadata
        elif len(event_ids) == n_vectors:
            # Case 2: One event_id per vector (or all same)
            event_ids_expanded = event_ids
            metadata_expanded = metadata
        else:
            raise ValueError(
                f"Mismatch: {n_vectors} vectors but {len(event_ids)} event_ids! "
                f"Should be 1 event_id for all vectors or one per vector."
            )

        if not self.is_trained:
            self.training_vectors.append(vectors)
            self.training_event_ids.append(event_ids_expanded)
            self.training_metadata.append(metadata_expanded)

            total_training = sum(v.shape[0] for v in self.training_vectors)

            if total_training >= self.min_training_size:
                self._train_index()
                # Training succeeded, vectors were added during training
                return {"status": "added", "vectors_added": n_vectors}

            # Still buffering
            return {
                "status": "buffered",
                "vectors_added": n_vectors,
                "awaiting_training": True,
                "buffered_count": total_training,
                "min_training_size": self.min_training_size,
            }

        # Note: IVF+SQ nlist cannot be changed after training without rebuilding
        # the entire index from scratch. The index is trained once with optimal
        # nlist for the initial batch size. To use a different nlist, clear and
        # retrain from scratch.

        # Normal add path: DIRECT to trained index
        vector_ids = []
        with self._id_map_lock:
            for event_id in event_ids_expanded:
                vid = self.next_vector_id
                self.vector_id_to_event_id[vid] = event_id

                # Add to event_id_to_vector_ids mapping
                if event_id not in self.event_id_to_vector_ids:
                    self.event_id_to_vector_ids[event_id] = []
                self.event_id_to_vector_ids[event_id].append(vid)

                vector_ids.append(vid)
                self.next_vector_id += 1

        vector_ids_array = np.array(vector_ids, dtype=np.int64)
        self.index.add_with_ids(vectors, vector_ids_array)  # type: ignore
        self.metadata.extend(metadata_expanded)
        self._save()

        return {"status": "added", "vectors_added": n_vectors}

    def remove_vectors(self, event_ids: List[str]) -> dict:
        """Remove all vectors associated with event_ids."""
        if not self.is_trained:
            logger.warning("Index not trained, cannot remove vectors")
            return {"removed": 0}

        removed_count = 0

        with self._id_map_lock:
            for event_id in event_ids:
                if event_id in self.event_id_to_vector_ids:
                    vector_ids = self.event_id_to_vector_ids[event_id]
                    for vid in vector_ids:
                        if vid in self.vector_id_to_event_id:
                            del self.vector_id_to_event_id[vid]
                        if vid < len(self.metadata):
                            self.metadata[vid]["deleted"] = True
                    del self.event_id_to_vector_ids[event_id]
                    removed_count += len(vector_ids)

        if removed_count > 0:
            logger.info(
                f"Marked {removed_count} vectors as deleted (from {len(event_ids)} events)"
            )
            self._save()

        return {"removed": removed_count}

    def search(
        self, query: np.ndarray, k: int = 10
    ) -> Tuple[List[str], List[float], List[dict]]:
        """
        Search for k nearest neighbors and aggregate by event_id.

        Returns:
        - event_ids: Unique event IDs (not individual vectors)
        - distances: Min distance per event
        - metadata: First metadata dict per event
        """
        if not self.is_trained:
            logger.warning("Index not trained yet, returning empty results")
            return [], [], []

        query = self._normalize_vectors(query.reshape(1, -1))
        distances, vector_ids = self.index.search(query, k * 5)  # type: ignore # Get more to aggregate

        event_results: Dict[str, Tuple[float, dict]] = {}

        for i, vid in enumerate(vector_ids[0]):
            if vid == -1:
                continue

            event_id = self.vector_id_to_event_id.get(int(vid))
            if (
                event_id
                and int(vid) < len(self.metadata)
                and not self.metadata[int(vid)].get("deleted", False)
            ):
                distance = float(distances[0][i])
                # Keep best distance for each event
                if (
                    event_id not in event_results
                    or distance > event_results[event_id][0]
                ):
                    event_results[event_id] = (distance, self.metadata[int(vid)])

        # Sort by distance (descending = higher similarity for inner product) and take top k
        sorted_events = sorted(
            event_results.items(), key=lambda x: x[1][0], reverse=True
        )[:k]

        event_ids = [eid for eid, _ in sorted_events]
        distances_list = [dist for _, (dist, _) in sorted_events]
        metadata_list = [meta for _, (_, meta) in sorted_events]

        return event_ids, distances_list, metadata_list

    def _should_use_batch(self, num_queries: int) -> bool:
        """
        Determine if batch search should be used based on hardware and query count.

        GPU: Always batch (kernel fusion benefits)
        CPU: Batch only for >=100 queries (overhead otherwise)
        """
        return True if self.gpu_enabled else num_queries >= 100

    def search_batch(
        self, queries: np.ndarray, k: int = 10, force_batch: bool = False
    ) -> List[Tuple[List[str], List[float], List[dict]]]:
        """
        Batch search for multiple queries.

        Args:
            queries: Array of shape (n_queries, dimension)
            k: Number of results per query
            force_batch: Override adaptive routing

        Returns:
            List of (event_ids, distances, metadata) tuples, one per query
        """
        if not self.is_trained:
            logger.warning("Index not trained yet, returning empty results")
            return [([], [], []) for _ in range(len(queries))]

        queries = self._normalize_vectors(queries)

        # Perform batch search
        distances, vector_ids = self.index.search(queries, k * 5)  # type: ignore # Get more to aggregate

        results = []
        for query_idx in range(len(queries)):
            event_results: Dict[str, Tuple[float, dict]] = {}

            for i, vid in enumerate(vector_ids[query_idx]):
                if vid == -1:
                    continue

                event_id = self.vector_id_to_event_id.get(int(vid))
                if (
                    event_id
                    and int(vid) < len(self.metadata)
                    and not self.metadata[int(vid)].get("deleted", False)
                ):
                    distance = float(distances[query_idx][i])
                    # Keep best distance for each event
                    if (
                        event_id not in event_results
                        or distance > event_results[event_id][0]
                    ):
                        event_results[event_id] = (
                            distance,
                            self.metadata[int(vid)],
                        )

            # Sort by distance (descending = higher similarity) and take top k
            sorted_events = sorted(
                event_results.items(), key=lambda x: x[1][0], reverse=True
            )[:k]

            event_ids = [eid for eid, _ in sorted_events]
            distances_list = [dist for _, (dist, _) in sorted_events]
            metadata_list = [meta for _, (_, meta) in sorted_events]

            results.append((event_ids, distances_list, metadata_list))

        return results

    def get_recommended_batch_size(self) -> int:
        """
        Get recommended batch size based on index size.

        Returns appropriate batch size for efficient processing.
        """
        n_vectors = self.index.ntotal if self.is_trained else 0

        if n_vectors < 10000:
            return 32
        elif n_vectors < 100000:
            return 64
        else:
            return 128

    def count(self) -> int:
        """Get total number of vectors in index."""
        return self.index.ntotal if self.is_trained else 0

    def count_events(self) -> int:
        """Get number of unique events."""
        return len(self.event_id_to_vector_ids)

    def get_info(self) -> dict:
        """Get index statistics."""
        n_vectors = self.index.ntotal if self.is_trained else 0
        optimal_nlist = (
            self._calculate_optimal_nlist(n_vectors) if n_vectors > 0 else self.nlist
        )

        # Calculate nlist_ratio, handling None values
        nlist_ratio = 1.0
        if self.nlist is not None and optimal_nlist is not None and optimal_nlist > 0:
            nlist_ratio = self.nlist / optimal_nlist

        return {
            "total_vectors": n_vectors,
            "total_events": len(self.event_id_to_vector_ids),
            "dimension": self.dimension,
            "nlist": self.nlist,
            "optimal_nlist": optimal_nlist,
            "nlist_ratio": nlist_ratio,
            "nlist_formula": "4 * sqrt(n)",  # Document the formula used
            "nprobe": self.nprobe,
            "is_trained": self.is_trained,
            "index_type": type(self.index).__name__,
            "use_sq": self.use_sq,
            "sq_bits": self.sq_bits,
            "gpu_enabled": self.gpu_enabled,
            "gpu_device_id": self.gpu_device_id,
            "recommended_batch_size": self.get_recommended_batch_size(),
        }

    def retrain(
        self, force: bool = False, expected_vector_count: Optional[int] = None
    ) -> dict:
        """
        Check index health and optionally update nlist for expected vector count.

        FAISS IVF nlist is a constructor parameter that cannot be changed
        after index creation without rebuilding from scratch. If expected_vector_count
        is provided and force=True, will clear and rebuild index with optimal nlist.
        """
        if not self.is_trained:
            logger.warning("Index not trained yet")
            return {"status": "not_trained"}

        n_vectors = self.index.ntotal

        # Use expected count if provided, otherwise use current count
        target_count = expected_vector_count or n_vectors
        optimal_nlist = self._calculate_optimal_nlist(target_count)

        drift_ratio = (
            abs(optimal_nlist - self.nlist) / optimal_nlist
            if optimal_nlist > 0 and self.nlist
            else 0
        )

        # If force=True and expected_vector_count provided, rebuild with new nlist
        if force and expected_vector_count and drift_ratio > 0.1:
            logger.info(
                f"Force rebuild: current nlist={self.nlist}, "
                f"optimal={optimal_nlist} for {target_count} vectors"
            )

            # Save all existing vectors and metadata
            all_vectors = []
            all_ids = []
            all_metadata = []

            for vec_id in range(self.index.ntotal):
                vec = np.zeros(self.dimension, dtype="float32")
                self.index.reconstruct(vec_id, vec)
                all_vectors.append(vec)
                all_ids.append(vec_id)
                all_metadata.append(
                    self.metadata[vec_id] if vec_id < len(self.metadata) else {}
                )

            # Update nlist and recreate index
            self.nlist = optimal_nlist
            self.expected_vector_count = expected_vector_count
            self._create_index()

            # Re-add all vectors
            if all_vectors:
                vectors_array = np.array(all_vectors, dtype="float32")
                if not self.is_trained and len(vectors_array) >= self.min_training_size:
                    self.index.train(vectors_array)  # type: ignore
                    self.is_trained = True

                if self.is_trained:
                    ids_array = np.array(all_ids, dtype="int64")
                    self.index.add_with_ids(vectors_array, ids_array)  # type: ignore
                    self.metadata = all_metadata
                    self.next_vector_id = max(all_ids) + 1

            self._save()

            return {
                "status": "rebuilt",
                "success": True,
                "nlist": self.nlist,
                "optimal": optimal_nlist,
                "vectors_restored": len(all_vectors),
            }

        if drift_ratio < 0.1:
            return {
                "status": "optimal",
                "success": True,
                "nlist": self.nlist,
                "optimal": optimal_nlist,
                "drift_ratio": drift_ratio,
            }

        logger.warning(
            f"nlist drift detected: current={self.nlist}, optimal={optimal_nlist} "
            f"(drift={drift_ratio:.1%}). Use force=True with expected_vector_count to rebuild."
        )

        return {
            "status": "drift_detected",
            "success": False,
            "nlist": self.nlist,
            "optimal": optimal_nlist,
            "drift_ratio": drift_ratio,
            "note": "IVF+SQ nlist cannot be changed after training. Use force=True with expected_vector_count to rebuild.",
        }

    def clear(self):
        """Clear all vectors and reset index."""
        self._create_index()
        self.metadata = []
        self.event_id_to_vector_ids = {}
        self.vector_id_to_event_id = {}
        self.next_vector_id = 0
        self.training_vectors.clear()
        self.training_event_ids.clear()
        self.training_metadata.clear()
        self._save()
        logger.info("Index cleared")

    def _load_index(self):
        """Load existing index from disk."""
        try:
            cpu_index = faiss.read_index(str(self.index_path))

            self.nlist = cpu_index.nlist if hasattr(cpu_index, "nlist") else None

            if self.gpu_enabled:
                try:
                    gpu_resources = self.gpu_manager.get_resources()
                    if gpu_resources is None:
                        raise RuntimeError("GPU resources not available")
                    self.index = faiss.index_cpu_to_gpu(
                        gpu_resources, self.gpu_device_id, cpu_index
                    )
                    logger.info(f"Loaded index transferred to GPU {self.gpu_device_id}")
                except Exception as e:
                    logger.warning(f"GPU transfer failed: {e}, using CPU")
                    self.index = cpu_index
                    self.gpu_enabled = False
            else:
                self.index = cpu_index

            self.is_trained = self.index.is_trained

            with open(self.metadata_path, "rb") as f:
                self.metadata = pickle.load(f)

            with open(self.id_map_path, "r") as f:
                id_map_data = json.load(f)

            self.event_id_to_vector_ids = dict(id_map_data["event_to_vectors"].items())
            self.vector_id_to_event_id = {
                int(k): v for k, v in id_map_data["vector_to_event"].items()
            }
            self.next_vector_id = id_map_data["next_id"]

            self._set_nprobe()

            logger.info(
                f"Loaded index: {type(self.index).__name__}, "
                f"{self.index.ntotal} vectors, {len(self.event_id_to_vector_ids)} events, trained={self.is_trained}"
            )

        except Exception as e:
            logger.error(f"Failed to load index: {e}")
            self._create_index()

    def _save(self):
        """Persist index, metadata, and ID mappings to disk."""
        if self.is_trained:
            index_to_save = self.index
            if self.gpu_enabled:
                try:
                    index_to_save = faiss.index_gpu_to_cpu(self.index)
                except Exception as e:
                    logger.error(f"GPU→CPU conversion failed: {e}")
                    return

            faiss.write_index(index_to_save, str(self.index_path))

            with open(self.metadata_path, "wb") as f:
                pickle.dump(self.metadata, f)

            id_map_data = {
                "event_to_vectors": self.event_id_to_vector_ids,
                "vector_to_event": {
                    str(k): v for k, v in self.vector_id_to_event_id.items()
                },
                "next_id": self.next_vector_id,
            }
            with open(self.id_map_path, "w") as f:
                json.dump(id_map_data, f)

            logger.debug("Index saved to disk")
