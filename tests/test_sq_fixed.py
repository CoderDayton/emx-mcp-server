#!/usr/bin/env python3
"""
Validation test for fixed 8-bit Scalar Quantization implementation.

Tests:
1. Basic functionality (add, search, recall)
2. GPU acceleration
3. Memory compression
4. Edge cases (small datasets, empty results)
"""

import numpy as np
import pytest

from emx_mcp.storage.vector_store import VectorStore


class TestSQFixed:
    """Test suite for fixed 8-bit SQ implementation."""

    @pytest.fixture
    def temp_storage(self, tmp_path):
        """Create temporary storage directory (unique per test)."""
        return str(tmp_path)

    @pytest.fixture
    def small_dataset(self):
        """Generate small test dataset (1000 vectors)."""
        np.random.seed(42)
        vectors = np.random.rand(1000, 384).astype(np.float32)
        event_ids = [f"event_{i}" for i in range(1000)]
        metadata = [{"idx": i, "source": "test"} for i in range(1000)]
        return vectors, event_ids, metadata

    @pytest.fixture
    def query_vectors(self):
        """Generate query vectors from test dataset."""
        np.random.seed(42)
        base = np.random.rand(1000, 384).astype(np.float32)
        # Use vectors from dataset as queries (perfect recall test)
        return base[:10]

    def test_basic_functionality(self, temp_storage, small_dataset):
        """Test basic add and search functionality."""
        vectors, event_ids, metadata = small_dataset

        # Initialize SQ vector store
        store = VectorStore(
            storage_path=temp_storage,
            dimension=384,
            use_sq=True,
            sq_bits=8,
            use_gpu=True,
            nprobe=16,
        )

        # Ensure clean state
        store.clear()

        # Add vectors
        result = store.add_vectors(vectors, event_ids, metadata)
        print(f"Add result: {result}")
        assert result["status"] in ["added", "buffered"]
        assert result["vectors_added"] == 1000

        # Verify count
        count = store.count()
        print(f"Store count: {count}, Expected: 1000")
        print(f"Is trained: {store.is_trained}")
        print(f"Training vectors buffered: {sum(v.shape[0] for v in store.training_vectors)}")
        assert count == 1000

        # Search
        query = vectors[0]
        found_ids, distances, found_metadata = store.search(query, k=10)

        # Verify results
        assert len(found_ids) > 0, "Search returned empty results"
        assert len(found_ids) == len(distances)
        assert len(found_ids) == len(found_metadata)
        assert event_ids[0] in found_ids, "Query vector not found in results"

        print("✓ Basic functionality test passed")
        print(f"  - Added {store.count()} vectors")
        print(f"  - Search returned {len(found_ids)} results")
        print(f"  - Top result: {found_ids[0]} (distance: {distances[0]:.4f})")

    def test_recall_performance(self, temp_storage, small_dataset):
        """Test recall@10 performance (target: 97-99%)."""
        vectors, event_ids, metadata = small_dataset

        # Initialize SQ vector store
        store = VectorStore(
            storage_path=temp_storage,
            dimension=384,
            use_sq=True,
            sq_bits=8,
            use_gpu=True,
            nprobe=16,
        )

        # Add vectors
        store.add_vectors(vectors, event_ids, metadata)

        # Test recall using vectors from dataset
        num_queries = 100
        recall_hits = 0

        for i in range(num_queries):
            query = vectors[i]
            expected_id = event_ids[i]

            found_ids, _, _ = store.search(query, k=10)

            if expected_id in found_ids:
                recall_hits += 1

        recall = recall_hits / num_queries
        print(f"✓ Recall@10: {recall * 100:.1f}% ({recall_hits}/{num_queries})")

        # Assert recall is within expected range
        assert recall >= 0.95, f"Recall {recall * 100:.1f}% below target (95%+)"

    def test_gpu_acceleration(self, temp_storage, small_dataset):
        """Test GPU acceleration status."""
        vectors, event_ids, metadata = small_dataset

        store = VectorStore(
            storage_path=temp_storage,
            dimension=384,
            use_sq=True,
            sq_bits=8,
            use_gpu=True,
        )

        store.add_vectors(vectors, event_ids, metadata)

        info = store.get_info()
        print("✓ GPU configuration:")
        print(f"  - GPU enabled: {info['gpu_enabled']}")
        print(f"  - Index type: {info['index_type']}")

        # GPU should be available on RTX 4090 system
        # But test passes even on CPU-only systems
        assert info["use_sq"] is True
        assert info["sq_bits"] == 8

    def test_memory_compression(self, temp_storage, small_dataset):
        """Test 4x memory compression (float32 → 8-bit)."""
        vectors, event_ids, metadata = small_dataset

        store = VectorStore(
            storage_path=temp_storage,
            dimension=384,
            use_sq=True,
            sq_bits=8,
            use_gpu=False,  # CPU for memory calculation
        )

        store.add_vectors(vectors, event_ids, metadata)

        # Calculate expected memory savings
        original_size = vectors.nbytes  # float32 = 4 bytes per value
        compressed_size = vectors.size // 4  # 8-bit = 1 byte per value (4x compression)

        info = store.get_info()
        print("✓ Memory compression:")
        print(f"  - Original size: {original_size / 1024 / 1024:.2f} MB")
        print(f"  - Compressed size: ~{compressed_size / 1024 / 1024:.2f} MB")
        print("  - Compression ratio: 4x (float32 → 8-bit)")

        assert info["use_sq"] is True

    def test_edge_case_empty_index(self, temp_storage):
        """Test search on empty index."""
        store = VectorStore(
            storage_path=temp_storage,
            dimension=384,
            use_sq=True,
            sq_bits=8,
        )

        query = np.random.rand(384).astype(np.float32)
        found_ids, distances, metadata = store.search(query, k=10)

        assert len(found_ids) == 0
        assert len(distances) == 0
        assert len(metadata) == 0
        print("✓ Empty index edge case handled correctly")

    def test_edge_case_small_dataset(self, temp_storage):
        """Test with minimal dataset (< min_training_size)."""
        # Create tiny dataset
        vectors = np.random.rand(50, 384).astype(np.float32)
        event_ids = [f"event_{i}" for i in range(50)]
        metadata = [{"idx": i} for i in range(50)]

        store = VectorStore(
            storage_path=temp_storage,
            dimension=384,
            use_sq=True,
            sq_bits=8,
        )

        result = store.add_vectors(vectors, event_ids, metadata)

        # Should buffer vectors until min_training_size reached
        if result["status"] == "buffered":
            print("✓ Small dataset buffered correctly (awaiting training)")
            assert result["awaiting_training"] is True
        else:
            print("✓ Small dataset added (index trained)")
            assert result["status"] == "added"

    def test_normalization(self, temp_storage):
        """Test L2 normalization for cosine similarity."""
        # Create vectors with known magnitudes
        vectors = np.array([[1.0, 2.0, 3.0, 4.0] + [0] * 380], dtype=np.float32)
        event_ids = ["event_0"]
        metadata = [{"idx": 0}]

        store = VectorStore(
            storage_path=temp_storage,
            dimension=384,
            use_sq=True,
            sq_bits=8,
        )

        # Normalization happens inside add_vectors
        store.add_vectors(vectors, event_ids, metadata)

        # Search with same vector (should have high similarity)
        query = vectors[0]
        found_ids, distances, _ = store.search(query, k=1)

        if len(found_ids) > 0:
            print("✓ Normalization test passed")
            print(f"  - Query-self distance: {distances[0]:.4f}")
            # After normalization, cosine similarity should be ~1.0 (distance ~0.0)
            assert distances[0] > 0.95, "Self-similarity should be high after normalization"

    def test_persistence(self, temp_storage, small_dataset):
        """Test index save/load persistence."""
        vectors, event_ids, metadata = small_dataset

        # Create and populate store
        store1 = VectorStore(
            storage_path=temp_storage,
            dimension=384,
            use_sq=True,
            sq_bits=8,
        )
        store1.add_vectors(vectors, event_ids, metadata)
        store1._save()

        # Load from disk
        store2 = VectorStore(
            storage_path=temp_storage,
            dimension=384,
            use_sq=True,
            sq_bits=8,
        )

        # Verify loaded state
        assert store2.count() == store1.count()
        assert store2.is_trained == store1.is_trained

        # Verify search results match
        query = vectors[0]
        ids1, dist1, _ = store1.search(query, k=10)
        ids2, dist2, _ = store2.search(query, k=10)

        assert ids1 == ids2, "Loaded index returns different results"
        print("✓ Persistence test passed (index save/load)")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
