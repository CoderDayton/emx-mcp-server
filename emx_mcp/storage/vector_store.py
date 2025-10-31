"""FAISS IVF-based vector storage for high-performance episodic memory."""

import faiss
import numpy as np
import pickle
import json
import time
import threading
from collections import deque
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Deque
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


class GPUResourceManager:
    """
    Manages FAISS GPU resources with automatic fallback to CPU.
    
    Provides centralized GPU resource allocation, error handling,
    and graceful degradation when GPU is unavailable.
    """
    
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
        self.gpu_count = 0
        
        try:
            self.gpu_count = faiss.get_num_gpus()
            if self.gpu_count > 0:
                # Allocate GPU resources (shared across all indices)
                self.gpu_resources = faiss.StandardGpuResources()
                self.gpu_available = True
                logger.info(f"GPU acceleration enabled: {self.gpu_count} GPU(s) detected")
            else:
                logger.warning("No GPUs detected, using CPU-only mode")
        except Exception as e:
            logger.warning(f"GPU initialization failed: {e}, falling back to CPU")
            self.gpu_available = False
        
        self._initialized = True
    
    def get_resources(self) -> Optional[faiss.StandardGpuResources]:
        """Get GPU resources if available."""
        return self.gpu_resources if self.gpu_available else None
    
    def is_available(self) -> bool:
        """Check if GPU acceleration is available."""
        return self.gpu_available


class VectorStore:
    """
    IVF-indexed vector database for similarity search with GPU acceleration.
    Optimized for 1M–10M+ vectors with <500ms search time.
    
    GPU Acceleration:
    - Automatic GPU detection and resource management
    - Graceful fallback to CPU when GPU unavailable
    - 40-50x speedup at 1M+ vectors (GPU vs CPU)
    
    Adaptive nlist tuning:
    - Continuously calculates optimal nlist = sqrt(n_vectors)
    - Auto-retrains when drift > 2x from optimal
    - Tracks optimization history for observability
    """

    def __init__(
        self,
        storage_path: str,
        dimension: int = 768,
        nlist: Optional[int] = None,
        nprobe: int = 8,
        auto_retrain: bool = True,
        nlist_drift_threshold: float = 2.0,
        nlist_formula: str = "sqrt",
        use_gpu: bool = True,
        gpu_device_id: int = 0,
    ):
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)

        self.dimension = dimension
        self.nprobe = nprobe
        self.auto_retrain = auto_retrain
        self.nlist_drift_threshold = nlist_drift_threshold
        self.nlist_formula = nlist_formula
        self.use_gpu = use_gpu
        self.gpu_device_id = gpu_device_id

        # GPU resource management
        self.gpu_manager = GPUResourceManager()
        self.gpu_enabled = self.use_gpu and self.gpu_manager.is_available()

        # Paths
        self.index_path = self.storage_path / "faiss_ivf_index.bin"
        self.metadata_path = self.storage_path / "metadata.pkl"
        self.id_map_path = self.storage_path / "id_map.json"
        self.optimization_history_path = self.storage_path / "nlist_history.json"

        # ID mapping for event removal (protected by lock for thread safety)
        self.event_id_to_vector_id: Dict[str, int] = {}
        self.vector_id_to_event_id: Dict[int, str] = {}
        self.next_vector_id = 0
        self._id_map_lock = threading.Lock()  # Protects ID mapping mutations

        # Training state
        self.is_trained = False
        self.training_vectors: Deque[np.ndarray] = deque(maxlen=10000)
        self.training_event_ids: Deque[List[str]] = deque(maxlen=10000)
        self.training_metadata: Deque[List[dict]] = deque(maxlen=10000)
        self.min_training_size = 1000  # Minimum vectors before training

        # Optimization tracking
        self.optimization_history: List[Dict] = []
        self._load_optimization_history()

        # Initialize or load index
        if self.index_path.exists():
            self._load_index()
        else:
            # Calculate optimal nlist based on expected size
            # Rule of thumb: nlist = sqrt(n_vectors)
            # For 10M vectors: nlist ~ 3162
            # For 1M vectors: nlist ~ 1000
            self.nlist = nlist or 128  # Start small, will adjust
            self._create_index()

    def _create_index(self):
        """Create new IVF index with optional GPU acceleration."""
        logger.info(f"Creating IVF index (dim={self.dimension}, nlist={self.nlist}, gpu={self.gpu_enabled})")

        # Create index based on GPU availability
        if self.gpu_enabled:
            try:
                gpu_resources = self.gpu_manager.get_resources()
                config = faiss.GpuIndexIVFFlatConfig()
                config.device = self.gpu_device_id
                
                # Create GPU index directly for optimal performance
                self.index = faiss.GpuIndexIVFFlat(
                    gpu_resources,
                    self.dimension,
                    self.nlist,
                    faiss.METRIC_L2,
                    config
                )
                logger.info(f"GPU IVF index created on device {self.gpu_device_id}")
            except Exception as e:
                logger.warning(f"GPU index creation failed: {e}, using CPU")
                quantizer = faiss.IndexFlatL2(self.dimension)
                self.index = faiss.IndexIVFFlat(
                    quantizer, self.dimension, self.nlist, faiss.METRIC_L2
                )
                self.gpu_enabled = False
        else:
            # Create CPU IVF index
            quantizer = faiss.IndexFlatL2(self.dimension)
            self.index = faiss.IndexIVFFlat(
                quantizer, self.dimension, self.nlist, faiss.METRIC_L2
            )
            
        self.metadata = []
        self.is_trained = False

    def _load_index(self):
        """Load existing index from disk and optionally transfer to GPU."""
        logger.info(f"Loading IVF index from {self.index_path}")
        cpu_index = faiss.read_index(str(self.index_path))
        
        # Transfer to GPU using copyFrom for optimal performance
        if self.gpu_enabled:
            try:
                gpu_resources = self.gpu_manager.get_resources()
                config = faiss.GpuIndexIVFFlatConfig()
                config.device = self.gpu_device_id
                
                # Get nlist from loaded CPU index
                loaded_nlist = cpu_index.nlist
                
                # Create GPU index with matching configuration
                gpu_index = faiss.GpuIndexIVFFlat(
                    gpu_resources,
                    self.dimension,
                    loaded_nlist,
                    faiss.METRIC_L2,
                    config
                )
                
                # Copy trained index from CPU to GPU
                gpu_index.copyFrom(cpu_index)
                self.index = gpu_index
                self.nlist = loaded_nlist
                
                logger.info(f"Index transferred to GPU {self.gpu_device_id} via copyFrom()")
            except Exception as e:
                logger.warning(f"GPU transfer failed during load: {e}, using CPU")
                self.index = cpu_index
                self.gpu_enabled = False
        else:
            self.index = cpu_index
        
        self.is_trained = self.index.is_trained

        with open(self.metadata_path, "rb") as f:
            self.metadata = pickle.load(f)
        with open(self.id_map_path, "r") as f:
            id_map_data = json.load(f)

        self.event_id_to_vector_id = id_map_data["event_to_vector"]
        self.vector_id_to_event_id = {
            int(k): v for k, v in id_map_data["vector_to_event"].items()
        }
        self.next_vector_id = id_map_data["next_id"]

        # Set search parameters
        self.index.nprobe = self.nprobe
        logger.info(f"Loaded {self.index.ntotal} vectors (trained={self.is_trained}, gpu={self.gpu_enabled})")
    
    def _load_optimization_history(self):
        """Load optimization history from disk."""
        if self.optimization_history_path.exists():
            with open(self.optimization_history_path, "r") as f:
                self.optimization_history = json.load(f)
        else:
            self.optimization_history = []
    
    def _record_optimization(
        self,
        old_nlist: int,
        new_nlist: int,
        n_vectors: int,
        elapsed_time: float,
        trigger: str,
    ):
        """
        Record nlist optimization event for observability.
        
        Args:
            old_nlist: Previous nlist value
            new_nlist: New nlist value
            n_vectors: Number of vectors at optimization time
            elapsed_time: Time taken for retraining (seconds)
            trigger: What triggered the optimization (auto/manual/initial_training)
        """
        event = {
            "timestamp": datetime.now().isoformat(),
            "old_nlist": old_nlist,
            "new_nlist": new_nlist,
            "n_vectors": n_vectors,
            "elapsed_time": elapsed_time,
            "trigger": trigger,
            "drift_ratio": old_nlist / new_nlist if new_nlist > 0 else None,
        }
        self.optimization_history.append(event)
        
        # Keep only last 100 events to prevent unbounded growth
        if len(self.optimization_history) > 100:
            self.optimization_history = self.optimization_history[-100:]
        
        logger.info(f"Recorded optimization event: {trigger}, {old_nlist}→{new_nlist}")

    def add_vectors(
        self, vectors: np.ndarray, event_ids: List[str], metadata: List[dict]
    ) -> dict:
        """
        Add vectors to index with event ID mapping.

        Args:
            vectors: Numpy array of shape (n, dimension)
            event_ids: List of event IDs corresponding to vectors
            metadata: List of metadata dicts for each vector

        Returns:
            Dictionary with add status and retraining recommendation
        """
        vectors = np.array(vectors, dtype=np.float32)
        n_vectors = vectors.shape[0]

        # Train index if not trained and have enough data
        if not self.is_trained:
            self.training_vectors.append(vectors)
            self.training_event_ids.append(event_ids)
            self.training_metadata.append(metadata)
            total_training = sum(v.shape[0] for v in self.training_vectors)
            if total_training >= self.min_training_size:
                self._train_index()

            if not self.is_trained:
                # Store for later training
                logger.warning(
                    f"Index not trained yet ({len(self.training_vectors)} batches buffered)"
                )
                return {
                    "status": "buffered",
                    "vectors_added": n_vectors,
                    "retrain_recommended": False,
                    "awaiting_training": True,
                }

        # Assign vector IDs (thread-safe)
        vector_ids = []
        with self._id_map_lock:
            for event_id in event_ids:
                vid = self.next_vector_id
                self.event_id_to_vector_id[event_id] = vid
                self.vector_id_to_event_id[vid] = event_id
                vector_ids.append(vid)
                self.next_vector_id += 1

        # Add to index with explicit IDs (FAISS Python API differs from type stubs)
        vector_ids_array = np.array(vector_ids, dtype=np.int64)
        self.index.add_with_ids(vectors, vector_ids_array)  # type: ignore[call-arg]

        # Store metadata
        self.metadata.extend(metadata)

        # Save updated index
        self._save()

        # Check if retraining recommended and auto-trigger if enabled
        retrain_recommended = self._should_retrain()
        auto_retrained = False
        
        if retrain_recommended and self.auto_retrain:
            logger.info("Auto-retraining triggered due to nlist drift")
            retrain_result = self.retrain(force=False)
            auto_retrained = retrain_result.get("status") == "retrained"
        
        return {
            "status": "added",
            "vectors_added": n_vectors,
            "total_vectors": self.index.ntotal,
            "retrain_recommended": retrain_recommended,
            "auto_retrained": auto_retrained,
            "current_nlist": self.nlist,
            "optimal_nlist": self._calculate_optimal_nlist(self.index.ntotal, formula=self.nlist_formula),
        }

    def search(
        self, query: np.ndarray, k: int = 10
    ) -> Tuple[List[str], List[float], List[dict]]:
        """
        Search for k nearest neighbors.

        Args:
            query: Query vector of shape (dimension,)
            k: Number of neighbors to return

        Returns:
            Tuple of (event_ids, distances, metadata)
        """
        if not self.is_trained:
            logger.warning("Index not trained yet, returning empty results")
            return [], [], []

        query = np.array(query, dtype=np.float32).reshape(1, -1)
        distances, vector_ids = self.index.search(query, k)  # type: ignore[call-arg]

        # Convert vector IDs back to event IDs
        event_ids, valid_metadata, valid_distances = [], [], []
        for i, vid in enumerate(vector_ids[0]):
            if vid == -1:  # FAISS returns -1 for empty slots
                continue
            event_id = self.vector_id_to_event_id.get(int(vid))
            if event_id and int(vid) < len(self.metadata):
                event_ids.append(event_id)
                valid_metadata.append(self.metadata[int(vid)])
                valid_distances.append(float(distances[0][i]))

        return event_ids, valid_distances, valid_metadata
    
    def search_batch(
        self, queries: np.ndarray, k: int = 10
    ) -> List[Tuple[List[str], List[float], List[dict]]]:
        """
        Batch search for k nearest neighbors (GPU-optimized).
        
        Processes multiple queries in a single GPU kernel call, amortizing
        kernel launch overhead (~10-20µs per call). Essential for GPU efficiency.

        Args:
            queries: Query vectors of shape (n_queries, dimension)
            k: Number of neighbors to return per query

        Returns:
            List of (event_ids, distances, metadata) tuples, one per query
        """
        if not self.is_trained:
            logger.warning("Index not trained yet, returning empty results")
            return [([],  [], []) for _ in range(len(queries))]
        
        # Ensure correct shape and dtype
        queries = np.asarray(queries, dtype=np.float32)
        if queries.ndim == 1:
            queries = queries.reshape(1, -1)
        
        # Single FAISS call for all queries (GPU-efficient)
        distances, vector_ids = self.index.search(queries, k)  # type: ignore[call-arg]
        
        # Convert results for each query
        results = []
        for query_idx in range(len(queries)):
            event_ids, valid_metadata, valid_distances = [], [], []
            
            for i, vid in enumerate(vector_ids[query_idx]):
                if vid == -1:  # FAISS returns -1 for empty slots
                    continue
                event_id = self.vector_id_to_event_id.get(int(vid))
                if event_id and int(vid) < len(self.metadata):
                    event_ids.append(event_id)
                    valid_metadata.append(self.metadata[int(vid)])
                    valid_distances.append(float(distances[query_idx][i]))
            
            results.append((event_ids, valid_distances, valid_metadata))
        
        return results

    def remove_vectors(self, event_ids: List[str]) -> dict:
        """
        Remove vectors by event IDs.

        Args:
            event_ids: List of event IDs to remove

        Returns:
            Dictionary with removal status
        """
        if not self.is_trained:
            return {"error": "Index not trained yet"}

        # Get vector IDs to remove
        vector_ids_to_remove = [
            self.event_id_to_vector_id[event_id]
            for event_id in event_ids
            if event_id in self.event_id_to_vector_id
        ]

        if not vector_ids_to_remove:
            return {"status": "no_vectors_found", "removed": 0}

        # Remove from index
        vector_ids_array = np.array(vector_ids_to_remove, dtype=np.int64)
        n_removed = self.index.remove_ids(vector_ids_array)

        # Update ID mappings (thread-safe)
        with self._id_map_lock:
            for event_id in event_ids:
                vid = self.event_id_to_vector_id.pop(event_id, None)
                if vid is not None:
                    self.vector_id_to_event_id.pop(vid, None)

        self._save()

        retrain_recommended = self._should_retrain()
        return {
            "status": "removed",
            "removed": int(n_removed),
            "total_vectors": self.index.ntotal,
            "retrain_recommended": retrain_recommended,
        }

    def retrain(self, force: bool = False) -> dict:
        """
        Retrain IVF index for optimal clustering with GPU acceleration.

        Args:
            force: Force retraining even if not recommended

        Returns:
            Retraining status
        """
        if not force and not self._should_retrain():
            return {"status": "skipped", "reason": "Retraining not needed"}

        if self.index.ntotal < self.min_training_size:
            return {
                "status": "insufficient_data",
                "vectors": self.index.ntotal,
                "required": self.min_training_size,
            }

        logger.info("Retraining IVF index...")
        start_time = time.time()

        # Reconstruct all vectors (works on both CPU and GPU indices)
        n_vectors = self.index.ntotal
        all_vectors = np.zeros((n_vectors, self.dimension), dtype=np.float32)
        for i in range(n_vectors):
            try:
                vec = np.zeros(self.dimension, dtype=np.float32)
                self.index.reconstruct(i, vec)
                all_vectors[i] = vec
            except Exception:
                pass

        # Calculate optimal nlist
        old_nlist = self.nlist
        optimal_nlist = self._calculate_optimal_nlist(n_vectors, formula=self.nlist_formula)

        # Create new CPU index with optimal nlist
        self.nlist = optimal_nlist
        quantizer = faiss.IndexFlatL2(self.dimension)
        cpu_index = faiss.IndexIVFFlat(
            quantizer, self.dimension, self.nlist, faiss.METRIC_L2
        )

        # Train on all vectors (FAISS Python API differs from type stubs)
        cpu_index.train(all_vectors)  # type: ignore[call-arg]
        
        # Re-add all vectors
        vector_ids = np.arange(n_vectors, dtype=np.int64)
        cpu_index.add_with_ids(all_vectors, vector_ids)  # type: ignore[call-arg]
        
        # Transfer to GPU if enabled using copyFrom for optimal performance
        if self.gpu_enabled:
            try:
                gpu_resources = self.gpu_manager.get_resources()
                config = faiss.GpuIndexIVFFlatConfig()
                config.device = self.gpu_device_id
                
                # Create GPU index with matching configuration
                gpu_index = faiss.GpuIndexIVFFlat(
                    gpu_resources,
                    self.dimension,
                    self.nlist,
                    faiss.METRIC_L2,
                    config
                )
                
                # Copy trained index from CPU to GPU
                gpu_index.copyFrom(cpu_index)
                self.index = gpu_index
                logger.info(f"Retrained index transferred to GPU {self.gpu_device_id} via copyFrom()")
            except Exception as e:
                logger.warning(f"GPU transfer after retraining failed: {e}, using CPU")
                self.index = cpu_index
                self.gpu_enabled = False
        else:
            self.index = cpu_index
        
        self.is_trained = True
        self._save()

        # Record optimization event
        elapsed_time = time.time() - start_time
        self._record_optimization(
            old_nlist=old_nlist,
            new_nlist=self.nlist,
            n_vectors=n_vectors,
            elapsed_time=elapsed_time,
            trigger="manual" if force else "auto",
        )

        logger.info(
            f"Retraining complete: nlist {old_nlist}→{self.nlist}, "
            f"{n_vectors} vectors, {elapsed_time:.2f}s, gpu={self.gpu_enabled}"
        )
        return {
            "status": "retrained",
            "old_nlist": old_nlist,
            "new_nlist": self.nlist,
            "vectors": n_vectors,
            "elapsed_time": elapsed_time,
            "gpu_enabled": self.gpu_enabled,
        }

    def _train_index(self):
        """Train index on accumulated training vectors with GPU acceleration."""
        all_training = np.vstack(list(self.training_vectors))
        n_vectors = all_training.shape[0]

        # Calculate optimal nlist for initial training
        optimal_nlist = self._calculate_optimal_nlist(n_vectors, formula=self.nlist_formula)
        old_nlist = self.nlist
        self.nlist = optimal_nlist

        # Create CPU index first
        quantizer = faiss.IndexFlatL2(self.dimension)
        cpu_index = faiss.IndexIVFFlat(
            quantizer, self.dimension, self.nlist, faiss.METRIC_L2
        )

        logger.info(f"Training IVF index on {n_vectors} vectors (nlist={self.nlist}, gpu={self.gpu_enabled})")
        cpu_index.train(all_training)  # type: ignore[call-arg]
        
        # Transfer to GPU after training using copyFrom for optimal performance
        if self.gpu_enabled:
            try:
                gpu_resources = self.gpu_manager.get_resources()
                config = faiss.GpuIndexIVFFlatConfig()
                config.device = self.gpu_device_id
                
                # Create GPU index with matching configuration
                gpu_index = faiss.GpuIndexIVFFlat(
                    gpu_resources,
                    self.dimension,
                    self.nlist,
                    faiss.METRIC_L2,
                    config
                )
                
                # Copy trained index from CPU to GPU
                gpu_index.copyFrom(cpu_index)
                self.index = gpu_index
                logger.info(f"Trained index transferred to GPU {self.gpu_device_id} via copyFrom()")
            except Exception as e:
                logger.warning(f"GPU transfer after training failed: {e}, using CPU")
                self.index = cpu_index
                self.gpu_enabled = False
        else:
            self.index = cpu_index
        
        self.is_trained = True

        # Bulk-add all buffered vectors to the now-trained index
        logger.info(f"Adding {n_vectors} buffered training vectors to trained index")
        vectors_added = 0
        
        for batch_vectors, batch_event_ids, batch_metadata in zip(
            self.training_vectors, self.training_event_ids, self.training_metadata
        ):
            # Assign vector IDs for this batch
            batch_vector_ids = []
            with self._id_map_lock:
                for event_id in batch_event_ids:
                    vid = self.next_vector_id
                    self.event_id_to_vector_id[event_id] = vid
                    self.vector_id_to_event_id[vid] = event_id
                    batch_vector_ids.append(vid)
                    self.next_vector_id += 1
            
            # Add to index
            vector_ids_array = np.array(batch_vector_ids, dtype=np.int64)
            self.index.add_with_ids(batch_vectors, vector_ids_array)  # type: ignore[call-arg]
            
            # Store metadata
            self.metadata.extend(batch_metadata)
            vectors_added += len(batch_vector_ids)
        
        logger.info(f"Successfully added {vectors_added} vectors to trained index")

        # Clear training buffers
        self.training_vectors.clear()
        self.training_event_ids.clear()
        self.training_metadata.clear()
        
        # Save index with buffered data
        self._save()
        
        # Record initial training optimization
        self._record_optimization(
            old_nlist=old_nlist,
            new_nlist=self.nlist,
            n_vectors=n_vectors,
            elapsed_time=0,  # Not tracking time for initial training
            trigger="initial_training",
        )
        
        logger.info("IVF index trained successfully")

    def _calculate_optimal_nlist(self, n_vectors: int, formula: str = "sqrt") -> int:
        """
        Calculate optimal nlist for current vector count using FAISS research formulas.
        
        Formulas (from FAISS wiki):
        - "sqrt": nlist = sqrt(n_vectors) - balanced speed/accuracy (default)
        - "2sqrt": nlist = 2*sqrt(n_vectors) - better recall, slight speed penalty
        - "4sqrt": nlist = 4*sqrt(n_vectors) - optimal recall, 2x slower search
        
        Bounds:
        - Lower: 128 (minimum for reasonable clustering)
        - Upper: n_vectors // 39 (ensures ~39 vectors per cluster minimum)
        
        Args:
            n_vectors: Current number of vectors in index
            formula: Nlist calculation formula ("sqrt", "2sqrt", "4sqrt")
            
        Returns:
            Optimal nlist value
        """
        if n_vectors < self.min_training_size:
            return 128
        
        # Base calculation using selected formula
        base_sqrt = int(np.sqrt(n_vectors))
        
        if formula == "sqrt":
            optimal = base_sqrt
        elif formula == "2sqrt":
            optimal = 2 * base_sqrt
        elif formula == "4sqrt":
            optimal = 4 * base_sqrt
        else:
            logger.warning(f"Unknown formula '{formula}', using 'sqrt'")
            optimal = base_sqrt
        
        # Apply bounds
        min_nlist = 128
        max_nlist = max(min_nlist, n_vectors // 39)  # ~39 vectors per cluster
        
        return max(min_nlist, min(optimal, max_nlist))
    
    def _should_retrain(self) -> bool:
        """
        Check if retraining is recommended based on nlist drift.
        
        Retraining triggers:
        1. nlist drift > threshold (default 2x) from optimal
        2. Sufficient vectors for meaningful retraining (>= min_training_size)
        
        Returns:
            True if retraining recommended
        """
        if not self.is_trained:
            return False

        n_vectors = self.index.ntotal
        if n_vectors < self.min_training_size:
            return False

        optimal_nlist = self._calculate_optimal_nlist(n_vectors, formula=self.nlist_formula)
        nlist_ratio = self.nlist / optimal_nlist if optimal_nlist > 0 else 1

        # Check drift threshold
        drift_exceeded = (
            nlist_ratio < (1.0 / self.nlist_drift_threshold) or 
            nlist_ratio > self.nlist_drift_threshold
        )
        
        if drift_exceeded:
            logger.info(
                f"nlist drift detected: current={self.nlist}, "
                f"optimal={optimal_nlist}, ratio={nlist_ratio:.2f}"
            )
        
        return drift_exceeded

    def count(self) -> int:
        """Get number of vectors in index."""
        return self.index.ntotal if self.is_trained else 0

    def get_info(self) -> dict:
        """Get index statistics with adaptive nlist monitoring and GPU status."""
        n_vectors = self.index.ntotal if self.is_trained else 0
        optimal_nlist = self._calculate_optimal_nlist(n_vectors, formula=self.nlist_formula) if n_vectors > 0 else self.nlist
        
        nlist_drift = None
        if self.is_trained and n_vectors > 0 and optimal_nlist > 0:
            nlist_drift = self.nlist / optimal_nlist
        
        return {
            "total_vectors": n_vectors,
            "dimension": self.dimension,
            "nlist": self.nlist,
            "optimal_nlist": optimal_nlist,
            "nlist_drift": nlist_drift,
            "nprobe": self.nprobe,
            "is_trained": self.is_trained,
            "index_type": "IVFFlat",
            "gpu_enabled": self.gpu_enabled,
            "gpu_device_id": self.gpu_device_id if self.gpu_enabled else None,
            "gpu_count": self.gpu_manager.gpu_count,
            "awaiting_training": len(self.training_vectors) > 0,
            "buffered_vectors": sum(v.shape[0] for v in self.training_vectors),
            "auto_retrain": self.auto_retrain,
            "nlist_drift_threshold": self.nlist_drift_threshold,
            "nlist_formula": self.nlist_formula,
            "recommended_batch_size": self.get_recommended_batch_size(),
            "optimization_count": len(self.optimization_history),
        }

    def clear(self):
        """Clear all vectors and reset index."""
        self._create_index()
        self.metadata = []
        self.event_id_to_vector_id = {}
        self.vector_id_to_event_id = {}
        self.next_vector_id = 0
        self.training_vectors.clear()
        self._save()

    def _save(self):
        """Persist index, metadata, and ID mappings to disk."""
        if self.is_trained:
            # For GPU indices, convert to CPU for saving
            index_to_save = self.index
            if self.gpu_enabled:
                try:
                    # Use faiss.index_gpu_to_cpu for GPU→CPU conversion
                    index_to_save = faiss.index_gpu_to_cpu(self.index)
                except Exception as e:
                    logger.error(f"GPU→CPU conversion failed: {e}")
                    return
            
            faiss.write_index(index_to_save, str(self.index_path))
            
        with open(self.metadata_path, "wb") as f:
            pickle.dump(self.metadata, f)
        with open(self.id_map_path, "w") as f:
            json.dump(
                {
                    "event_to_vector": self.event_id_to_vector_id,
                    "vector_to_event": {
                        str(k): v for k, v in self.vector_id_to_event_id.items()
                    },
                    "next_id": self.next_vector_id,
                },
                f,
            )
        
        # Save optimization history
        if self.optimization_history:
            with open(self.optimization_history_path, "w") as f:
                json.dump(self.optimization_history, f, indent=2)
    
    def get_optimization_history(self, limit: Optional[int] = None) -> List[Dict]:
        """
        Get nlist optimization history for observability.
        
        Args:
            limit: Maximum number of recent events to return (default: all)
            
        Returns:
            List of optimization events in chronological order
        """
        if limit is None:
            return self.optimization_history.copy()
        return self.optimization_history[-limit:].copy()
    
    def get_recommended_batch_size(self) -> int:
        """
        Calculate recommended batch size for search operations based on index size.
        
        Balances GPU kernel overhead amortization with memory pressure.
        Research-backed sizing from FAISS GPU performance analysis.
        
        Returns:
            Recommended batch size for current index
        """
        n_vectors = self.index.ntotal if self.is_trained else 0
        
        if n_vectors < 50_000:
            # Small indices: smaller batches (20-50)
            return 32
        elif n_vectors < 500_000:
            # Medium indices: moderate batches (50-100)
            return 64
        else:
            # Large indices: larger batches for better GPU utilization (100-200)
            return 128

