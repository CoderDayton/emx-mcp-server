"""FAISS IVF-based vector storage for high-performance episodic memory."""

import faiss
import numpy as np
import pickle
import json
from pathlib import Path
from typing import List, Tuple, Optional, Dict
import logging

logger = logging.getLogger(__name__)


class VectorStore:
    """
    IVF-indexed vector database for similarity search.
    Optimized for 1M–10M+ vectors with <500ms search time.
    """

    def __init__(
        self,
        storage_path: str,
        dimension: int = 768,
        nlist: Optional[int] = None,
        nprobe: int = 8,
    ):
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)

        self.dimension = dimension
        self.nprobe = nprobe

        # Paths
        self.index_path = self.storage_path / "faiss_ivf_index.bin"
        self.metadata_path = self.storage_path / "metadata.pkl"
        self.id_map_path = self.storage_path / "id_map.json"

        # ID mapping for event removal
        self.event_id_to_vector_id: Dict[str, int] = {}
        self.vector_id_to_event_id: Dict[int, str] = {}
        self.next_vector_id = 0

        # Training state
        self.is_trained = False
        self.training_vectors: List[np.ndarray] = []
        self.min_training_size = 1000  # Minimum vectors before training

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
        """Create new IVF index."""
        logger.info(f"Creating IVF index (dim={self.dimension}, nlist={self.nlist})")

        # Create quantizer (coarse quantizer for IVF)
        quantizer = faiss.IndexFlatL2(self.dimension)

        # Create IVF index
        self.index = faiss.IndexIVFFlat(
            quantizer, self.dimension, self.nlist, faiss.METRIC_L2
        )
        self.metadata = []
        self.is_trained = False

    def _load_index(self):
        """Load existing index from disk."""
        logger.info(f"Loading IVF index from {self.index_path}")
        self.index = faiss.read_index(str(self.index_path))
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
        logger.info(f"Loaded {self.index.ntotal} vectors (trained={self.is_trained})")

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

        # Assign vector IDs
        vector_ids = []
        for event_id in event_ids:
            vid = self.next_vector_id
            self.event_id_to_vector_id[event_id] = vid
            self.vector_id_to_event_id[vid] = event_id
            vector_ids.append(vid)
            self.next_vector_id += 1

        # Add to index with explicit IDs
        vector_ids_array = np.array(vector_ids, dtype=np.int64)
        self.index.add_with_ids(
            vectors, vector_ids_array
        )  # pyright: ignore[reportCallIssue]

        # Store metadata
        self.metadata.extend(metadata)

        # Save updated index
        self._save()

        # Check if retraining recommended
        retrain_recommended = self._should_retrain()
        return {
            "status": "added",
            "vectors_added": n_vectors,
            "total_vectors": self.index.ntotal,
            "retrain_recommended": retrain_recommended,
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
        distances, vector_ids = self.index.search(
            query, k
        )  # pyright: ignore[reportCallIssue]

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

        # Update ID mappings
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
        Retrain IVF index for optimal clustering.

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

        # Reconstruct all vectors
        n_vectors = self.index.ntotal
        all_vectors = np.zeros((n_vectors, self.dimension), dtype=np.float32)
        for i in range(n_vectors):
            try:
                vec = np.zeros(self.dimension, dtype=np.float32)
                self.index.reconstruct(i, vec)
                all_vectors[i] = vec
            except Exception:
                pass

        # Adjust nlist based on current size
        optimal_nlist = max(128, int(np.sqrt(n_vectors)))

        # Create new index
        old_nlist = self.nlist
        self.nlist = optimal_nlist
        self._create_index()

        # Train on all vectors
        self.index.train(all_vectors)  # pyright: ignore[reportCallIssue]
        self.is_trained = True

        # Re-add all vectors
        vector_ids = np.arange(n_vectors, dtype=np.int64)
        self.index.add_with_ids(
            all_vectors, vector_ids
        )  # pyright: ignore[reportCallIssue]
        self._save()

        logger.info(f"Retraining complete (nlist: {old_nlist}->{self.nlist})")
        return {
            "status": "retrained",
            "old_nlist": old_nlist,
            "new_nlist": self.nlist,
            "vectors": self.index.ntotal,
        }

    def _train_index(self):
        """Train index on accumulated training vectors."""
        all_training = np.vstack(self.training_vectors)
        n_vectors = all_training.shape[0]

        # Adjust nlist based on training size
        optimal_nlist = max(128, int(np.sqrt(n_vectors)))
        self.nlist = optimal_nlist

        # Recreate index with optimal nlist
        quantizer = faiss.IndexFlatL2(self.dimension)
        self.index = faiss.IndexIVFFlat(
            quantizer, self.dimension, self.nlist, faiss.METRIC_L2
        )

        logger.info(f"Training IVF index on {n_vectors} vectors (nlist={self.nlist})")
        self.index.train(all_training)  # pyright: ignore[reportCallIssue]
        self.is_trained = True

        # Clear training buffer
        self.training_vectors = []
        logger.info("IVF index trained successfully")

    def _should_retrain(self) -> bool:
        """Check if retraining is recommended."""
        if not self.is_trained:
            return False

        n_vectors = self.index.ntotal

        # Retrain if nlist is significantly off from optimal
        optimal_nlist = max(128, int(np.sqrt(n_vectors)))
        nlist_ratio = self.nlist / optimal_nlist if optimal_nlist > 0 else 1

        # Retrain if nlist is 2x off in either direction
        if nlist_ratio < 0.5 or nlist_ratio > 2.0:
            return True

        # Could add additional criteria (e.g. sparsity)
        return False

    def count(self) -> int:
        """Get number of vectors in index."""
        return self.index.ntotal if self.is_trained else 0

    def get_info(self) -> dict:
        """Get index statistics."""
        return {
            "total_vectors": self.index.ntotal if self.is_trained else 0,
            "dimension": self.dimension,
            "nlist": self.nlist,
            "nprobe": self.nprobe,
            "is_trained": self.is_trained,
            "index_type": "IVFFlat",
            "awaiting_training": len(self.training_vectors) > 0,
            "buffered_vectors": sum(v.shape[0] for v in self.training_vectors),
        }

    def clear(self):
        """Clear all vectors and reset index."""
        self._create_index()
        self.metadata = []
        self.event_id_to_vector_id = {}
        self.vector_id_to_event_id = {}
        self.next_vector_id = 0
        self.training_vectors = []
        self._save()

    def _save(self):
        """Persist index, metadata, and ID mappings to disk."""
        if self.is_trained:
            faiss.write_index(self.index, str(self.index_path))
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
