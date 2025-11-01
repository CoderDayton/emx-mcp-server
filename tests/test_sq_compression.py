"""Tests for 8-bit Scalar Quantization (SQ) compression."""

import pytest
import tempfile
import shutil
from pathlib import Path
import numpy as np

from emx_mcp.storage.vector_store import VectorStore


class TestSQCompression:
    """Test 8-bit Scalar Quantization compression functionality."""

    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for test storage."""
        temp_path = tempfile.mkdtemp()
        yield temp_path
        shutil.rmtree(temp_path, ignore_errors=True)

    def test_sq_config_validation(self):
        """Test SQ configuration validation."""
        from emx_mcp.utils.config_validator import StorageConfig

        # Valid SQ config
        config = StorageConfig(
            vector_dim=384,
            use_sq=True,
            sq_bits=8,
        )
        assert config.use_sq is True
        assert config.sq_bits == 8

        # Test default sq_bits
        config_default = StorageConfig(
            vector_dim=384,
            use_sq=True,
        )
        assert config_default.sq_bits == 8  # Default value

    def test_sq_index_creation_and_training(self, temp_dir):
        """Test creating and training SQ index."""
        dimension = 384
        num_vectors = 2000  # Reduced for faster test (still enough for IVF training)

        # Create VectorStore with SQ
        store = VectorStore(
            storage_path=str(Path(temp_dir) / "vector_sq"),
            dimension=dimension,
            use_sq=True,
            sq_bits=8,
        )

        # Generate random vectors
        vectors = np.random.randn(num_vectors, dimension).astype(np.float32)
        event_ids = [f"event_{i}" for i in range(num_vectors)]
        metadata = [{"idx": i} for i in range(num_vectors)]

        # Add vectors (triggers training)
        result = store.add_vectors(vectors, event_ids, metadata)
        assert result["status"] == "added"

        # Verify index is SQ
        info = store.get_info()
        assert info["use_sq"] is True
        assert info["sq_bits"] == 8
        assert info["is_trained"] is True
        # Accept both CPU and GPU index types
        assert "ScalarQuantizer" in info["index_type"] or info["index_type"] in [
            "IVFSQ",
            "SQ",
        ]

    def test_sq_search_recall(self, temp_dir):
        """Test SQ search maintains high recall (97-99%)."""
        dimension = 384
        num_vectors = 10000  # Increased from 5000 to avoid FAISS clustering warnings (needs 39*nlist)
        num_queries = 100
        k = 10

        # SQ with proper normalization and nprobe should achieve 97-99% recall
        # Use higher nprobe for better recall
        nprobe = 16  # Increased from 8 for better SQ recall

        # Create two stores: one with SQ, one without
        store_sq = VectorStore(
            storage_path=str(Path(temp_dir) / "vector_sq"),
            dimension=dimension,
            use_sq=True,
            sq_bits=8,
            nprobe=nprobe,
        )

        store_flat = VectorStore(
            storage_path=str(Path(temp_dir) / "vector_flat"),
            dimension=dimension,
            use_sq=False,
            nprobe=nprobe,
        )

        # Generate and add same vectors to both
        vectors = np.random.randn(num_vectors, dimension).astype(np.float32)
        event_ids = [f"event_{i}" for i in range(num_vectors)]
        metadata = [{"idx": i} for i in range(num_vectors)]

        store_sq.add_vectors(vectors, event_ids, metadata)
        store_flat.add_vectors(vectors, event_ids, metadata)

        # Generate queries
        queries = np.random.randn(num_queries, dimension).astype(np.float32)

        # Compare recall with debug info
        recall_scores = []
        print(f"\nSQ index info: {store_sq.get_info()}")
        print(f"Flat index info: {store_flat.get_info()}")

        for i, query in enumerate(queries[:5]):  # Debug first 5 queries
            # Search both indices
            ids_sq, dists_sq, _ = store_sq.search(query, k)
            ids_flat, dists_flat, _ = store_flat.search(query, k)

            # Calculate recall (overlap)
            overlap = len(set(ids_sq) & set(ids_flat))
            recall = overlap / k
            recall_scores.append(recall)

            if i < 2:  # Print first 2 for debugging
                print(f"\nQuery {i}:")
                print(f"  SQ IDs:   {ids_sq[:5]}")
                print(f"  Flat IDs: {ids_flat[:5]}")
                print(f"  SQ dists:   {dists_sq[:5]}")
                print(f"  Flat dists: {dists_flat[:5]}")
                print(f"  Recall: {recall:.2%}")

        # Process remaining queries without debug
        for query in queries[5:]:
            ids_sq, _, _ = store_sq.search(query, k)
            ids_flat, _, _ = store_flat.search(query, k)
            overlap = len(set(ids_sq) & set(ids_flat))
            recall_scores.append(overlap / k)

        avg_recall = np.mean(recall_scores)
        print(f"\nOverall SQ Search Recall: {avg_recall:.2%}")

        # SQ should maintain very high recall (97-99% with proper normalization and nprobe)
        # 8-bit Scalar Quantization with L2 normalization and Inner Product metric
        assert avg_recall > 0.97, f"SQ recall too low: {avg_recall:.2%} (expected >97%)"

    def test_sq_index_persistence(self, temp_dir):
        """Test SQ index can be saved and loaded."""
        dimension = 384
        num_vectors = 2000

        storage_path = str(Path(temp_dir) / "vector_sq_persist")

        # Create and populate SQ index
        store1 = VectorStore(
            storage_path=storage_path,
            dimension=dimension,
            use_sq=True,
            sq_bits=8,
        )

        vectors = np.random.randn(num_vectors, dimension).astype(np.float32)
        event_ids = [f"event_{i}" for i in range(num_vectors)]
        metadata = [{"idx": i} for i in range(num_vectors)]

        store1.add_vectors(vectors, event_ids, metadata)
        original_count = store1.count()

        # Load index in new store
        store2 = VectorStore(
            storage_path=storage_path,
            dimension=dimension,
            use_sq=True,
            sq_bits=8,
        )

        # Verify loaded correctly
        assert store2.count() == original_count
        info = store2.get_info()
        assert info["use_sq"] is True
        assert info["sq_bits"] == 8

        # Verify search works
        query = np.random.randn(dimension).astype(np.float32)
        ids, distances, meta = store2.search(query, k=10)
        assert len(ids) > 0

    def test_sq_memory_reduction(self, temp_dir):
        """Test SQ reduces memory footprint by 4x (float32 → 8-bit)."""
        dimension = 384
        num_vectors = 10000

        # Create SQ index
        store_sq = VectorStore(
            storage_path=str(Path(temp_dir) / "vector_sq"),
            dimension=dimension,
            use_sq=True,
            sq_bits=8,
        )

        # Create Flat index
        store_flat = VectorStore(
            storage_path=str(Path(temp_dir) / "vector_flat"),
            dimension=dimension,
            use_sq=False,
        )

        # Add same vectors
        vectors = np.random.randn(num_vectors, dimension).astype(np.float32)
        event_ids = [f"event_{i}" for i in range(num_vectors)]
        metadata = [{"idx": i} for i in range(num_vectors)]

        store_sq.add_vectors(vectors, event_ids, metadata)
        store_flat.add_vectors(vectors, event_ids, metadata)

        # Check file sizes
        sq_path = Path(temp_dir) / "vector_sq" / "faiss_ivf_index.bin"
        flat_path = Path(temp_dir) / "vector_flat" / "faiss_ivf_index.bin"

        sq_size = sq_path.stat().st_size
        flat_size = flat_path.stat().st_size

        compression_ratio = flat_size / sq_size
        print(f"\nMemory compression: {compression_ratio:.1f}x")
        print(f"Flat index: {flat_size / 1024 / 1024:.1f} MB")
        print(f"SQ index: {sq_size / 1024 / 1024:.1f} MB")

        # SQ should be approximately 4x smaller (float32 → 8-bit)
        assert compression_ratio > 3.5, (
            f"Insufficient compression: {compression_ratio:.1f}x (expected ~4x)"
        )

    def test_hierarchical_memory_with_sq(self, temp_dir):
        """Test HierarchicalMemoryStore with SQ enabled."""
        from emx_mcp.memory.storage import HierarchicalMemoryStore

        config = {
            "memory": {
                "n_init": 128,
                "n_local": 4096,
            },
            "storage": {
                "vector_dim": 384,
                "nprobe": 16,  # Increased for better SQ recall
                "auto_retrain": True,
                "nlist_drift_threshold": 2.0,
                "use_sq": True,
                "sq_bits": 8,
            },
        }

        store = HierarchicalMemoryStore(
            storage_path=str(Path(temp_dir) / "hierarchical"),
            config=config,
        )

        # Add test event
        tokens = ["hello", "world", "this", "is", "a", "test"]
        embeddings = np.random.randn(len(tokens), 384).astype(np.float32).tolist()

        result = store.add_event(
            event_id="test_event_1",
            tokens=tokens,
            embeddings=embeddings,
            metadata={"source": "test"},
        )

        assert result["status"] == "added"

        # Verify SQ is enabled in vector store
        info = store.get_index_info()
        assert info["use_sq"] is True
        assert info["sq_bits"] == 8

    def test_sq_flat_vs_ivf_selection(self, temp_dir):
        """Test that SQ automatically selects flat vs IVF based on vector count."""
        dimension = 384

        # Test small dataset (<100k) should use flat SQ
        store_small = VectorStore(
            storage_path=str(Path(temp_dir) / "vector_sq_small"),
            dimension=dimension,
            use_sq=True,
            nlist=None,  # Force flat index
        )

        vectors_small = np.random.randn(1000, dimension).astype(np.float32)
        event_ids_small = [f"event_{i}" for i in range(1000)]
        metadata_small = [{"idx": i} for i in range(1000)]

        store_small.add_vectors(vectors_small, event_ids_small, metadata_small)
        info_small = store_small.get_info()

        # Should be SQ (accept GPU or CPU variants)
        assert (
            "ScalarQuantizer" in info_small["index_type"]
            or info_small["index_type"] == "SQ"
        )

        # Test large dataset (≥100k) should use IVF+SQ
        store_large = VectorStore(
            storage_path=str(Path(temp_dir) / "vector_sq_large"),
            dimension=dimension,
            use_sq=True,
            nlist=1000,  # Force IVF index
        )

        vectors_large = np.random.randn(10000, dimension).astype(np.float32)
        event_ids_large = [f"event_{i}" for i in range(10000)]
        metadata_large = [{"idx": i} for i in range(10000)]

        store_large.add_vectors(vectors_large, event_ids_large, metadata_large)
        info_large = store_large.get_info()

        # Should be IVF+SQ (accept GPU or CPU variants)
        assert (
            "IVF" in info_large["index_type"]
            and "ScalarQuantizer" in info_large["index_type"]
        )

    def test_sq_cosine_similarity(self, temp_dir):
        """Test that SQ uses cosine similarity with proper normalization."""
        dimension = 384
        num_vectors = 1000

        store = VectorStore(
            storage_path=str(Path(temp_dir) / "vector_sq_cosine"),
            dimension=dimension,
            use_sq=True,
            sq_bits=8,
        )

        # Create vectors with known cosine similarity
        base_vector = np.random.randn(dimension).astype(np.float32)
        base_vector = base_vector / np.linalg.norm(base_vector)  # L2 normalize

        # Create similar vectors (small perturbations)
        similar_vectors = []
        for i in range(num_vectors):
            noise = np.random.randn(dimension) * 0.1  # Small noise
            vec = base_vector + noise
            vec = vec / np.linalg.norm(vec)  # L2 normalize
            similar_vectors.append(vec)

        similar_vectors = np.array(similar_vectors)

        # Add to store
        event_ids = [f"event_{i}" for i in range(num_vectors)]
        metadata = [{"idx": i} for i in range(num_vectors)]

        store.add_vectors(similar_vectors, event_ids, metadata)

        # Search with the base vector
        query = base_vector
        ids, distances, meta = store.search(query, k=10)

        # Should find similar vectors with reasonable cosine similarity
        # Since we're using Inner Product with normalized vectors, high similarity = high distance
        # With noise added, we expect lower similarity
        assert len(ids) > 0
        assert all(d > 0.3 for d in distances), f"Low cosine similarity: {distances}"

        # The top result should be reasonably similar (noise reduces perfect similarity)
        assert distances[0] > 0.5, f"Top result not similar enough: {distances[0]:.3f}"
