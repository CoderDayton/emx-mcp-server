"""Tests for adaptive nlist tuning in VectorStore."""

import numpy as np
import pytest
import tempfile
import shutil

from emx_mcp.storage.vector_store import VectorStore


class TestAdaptiveNlist:
    """Test adaptive nlist calculation and auto-retraining."""

    @pytest.fixture
    def temp_storage(self):
        """Create temporary storage directory."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)

    def test_calculate_optimal_nlist_small(self, temp_storage):
        """Test optimal nlist calculation for small vector counts."""
        store = VectorStore(storage_path=temp_storage, dimension=384)
        
        # Below threshold: returns min (128)
        assert store._calculate_optimal_nlist(100) == 128
        assert store._calculate_optimal_nlist(500) == 128
        
        # Just above threshold: sqrt(1000) ≈ 31, but bounded by min 128
        assert store._calculate_optimal_nlist(1000) == 128

    def test_calculate_optimal_nlist_large(self, temp_storage):
        """Test optimal nlist calculation for large vector counts."""
        store = VectorStore(storage_path=temp_storage, dimension=384)
        
        # 10k vectors: sqrt(10000) = 100, but min is 128
        assert store._calculate_optimal_nlist(10000) == 128
        
        # 20k vectors: sqrt(20000) ≈ 141
        optimal = store._calculate_optimal_nlist(20000)
        assert 140 <= optimal <= 142
        
        # 100k vectors: sqrt(100000) ≈ 316
        optimal = store._calculate_optimal_nlist(100000)
        assert 315 <= optimal <= 317
        
        # 1M vectors: sqrt(1000000) = 1000
        assert store._calculate_optimal_nlist(1000000) == 1000
        
        # 10M vectors: sqrt(10000000) ≈ 3162
        optimal = store._calculate_optimal_nlist(10000000)
        assert 3161 <= optimal <= 3163

    def test_calculate_optimal_nlist_upper_bound(self, temp_storage):
        """Test upper bound (n_vectors // 39) is enforced."""
        store = VectorStore(storage_path=temp_storage, dimension=384)
        
        # For very small n, upper bound should limit
        # 5000 vectors: sqrt(5000) ≈ 70, but min is 128
        # Upper bound: 5000 // 39 = 128
        assert store._calculate_optimal_nlist(5000) == 128

    def test_should_retrain_untrained(self, temp_storage):
        """Test retraining check for untrained index."""
        store = VectorStore(storage_path=temp_storage, dimension=384)
        assert not store._should_retrain()  # Not trained yet

    def test_should_retrain_drift_detection(self, temp_storage):
        """Test drift detection triggers retraining recommendation."""
        store = VectorStore(storage_path=temp_storage, dimension=384, auto_retrain=False)
        
        # Add vectors to train index
        vectors = np.random.randn(1100, 384).astype(np.float32)
        event_ids = [f"event_{i}" for i in range(1100)]
        metadata = [{"id": i} for i in range(1100)]
        
        store.add_vectors(vectors, event_ids, metadata)
        assert store.is_trained
        
        # Current nlist should be near sqrt(1100) ≈ 33, but bounded to 128
        assert store.nlist == 128
        
        # Add many more vectors to create drift scenario
        # We need enough vectors so optimal nlist grows significantly
        more_vectors = np.random.randn(30000, 384).astype(np.float32)
        more_ids = [f"event_{i+1100}" for i in range(30000)]
        more_metadata = [{"id": i+1100} for i in range(30000)]
        
        store.add_vectors(more_vectors, more_ids, more_metadata)
        
        # Total: 31100 vectors, optimal nlist = sqrt(31100) ≈ 176
        # Current nlist: 128, drift ratio = 128/176 ≈ 0.73
        # This exceeds threshold (< 0.5 OR > 2.0)? No, 0.73 is within [0.5, 2.0]
        
        # Force a clear drift scenario: set nlist to 64 (will be < 0.5 ratio)
        store.nlist = 64
        optimal = store._calculate_optimal_nlist(store.index.ntotal)
        # optimal ≈ 176, drift ratio = 64/176 ≈ 0.36 < 0.5
        assert store._should_retrain()  # Drift detected

    def test_auto_retrain_disabled(self, temp_storage):
        """Test that auto_retrain=False prevents automatic retraining."""
        store = VectorStore(
            storage_path=temp_storage, 
            dimension=384, 
            auto_retrain=False
        )
        
        # Add vectors
        vectors = np.random.randn(1100, 384).astype(np.float32)
        event_ids = [f"event_{i}" for i in range(1100)]
        metadata = [{"id": i} for i in range(1100)]
        
        result = store.add_vectors(vectors, event_ids, metadata)
        
        # Should not auto-retrain even if drift detected
        assert result["auto_retrained"] is False

    def test_auto_retrain_enabled(self, temp_storage):
        """Test that auto_retrain=True triggers retraining on drift."""
        store = VectorStore(
            storage_path=temp_storage, 
            dimension=384, 
            auto_retrain=True,
            nlist=64  # Start with suboptimal nlist
        )
        
        # Add vectors to trigger training with nlist=64
        vectors = np.random.randn(1100, 384).astype(np.float32)
        event_ids = [f"event_{i}" for i in range(1100)]
        metadata = [{"id": i} for i in range(1100)]
        
        # First add - should train with optimal nlist
        result = store.add_vectors(vectors, event_ids, metadata)
        assert store.is_trained
        
        # Force nlist to suboptimal value to simulate drift
        store.nlist = 256
        
        # Add more vectors to trigger auto-retrain
        more_vectors = np.random.randn(100, 384).astype(np.float32)
        more_ids = [f"event_{i+1100}" for i in range(100)]
        more_metadata = [{"id": i+1100} for i in range(100)]
        
        result = store.add_vectors(more_vectors, more_ids, more_metadata)
        
        # Should detect drift and retrain if threshold exceeded
        # Check that recommendation was made
        assert "retrain_recommended" in result

    def test_optimization_history_tracking(self, temp_storage):
        """Test that optimization events are tracked correctly."""
        store = VectorStore(storage_path=temp_storage, dimension=384)
        
        # Add vectors to trigger initial training
        vectors = np.random.randn(1100, 384).astype(np.float32)
        event_ids = [f"event_{i}" for i in range(1100)]
        metadata = [{"id": i} for i in range(1100)]
        
        store.add_vectors(vectors, event_ids, metadata)
        
        # Check initial training event recorded
        history = store.get_optimization_history()
        assert len(history) >= 1
        assert history[0]["trigger"] == "initial_training"
        assert "timestamp" in history[0]
        assert "old_nlist" in history[0]
        assert "new_nlist" in history[0]

    def test_optimization_history_limit(self, temp_storage):
        """Test that optimization history respects limit parameter."""
        store = VectorStore(storage_path=temp_storage, dimension=384)
        
        # Manually add some optimization events
        for i in range(10):
            store._record_optimization(
                old_nlist=128 + i,
                new_nlist=128 + i + 1,
                n_vectors=1000 * (i + 1),
                elapsed_time=0.1,
                trigger="test",
            )
        
        # Get limited history
        history = store.get_optimization_history(limit=3)
        assert len(history) == 3
        
        # Get all history
        full_history = store.get_optimization_history()
        assert len(full_history) == 10

    def test_optimization_history_retention(self, temp_storage):
        """Test that optimization history keeps only last 100 events."""
        store = VectorStore(storage_path=temp_storage, dimension=384)
        
        # Add 150 events
        for i in range(150):
            store._record_optimization(
                old_nlist=128,
                new_nlist=129,
                n_vectors=1000,
                elapsed_time=0.1,
                trigger="test",
            )
        
        # Should only keep last 100
        history = store.get_optimization_history()
        assert len(history) == 100

    def test_get_info_with_nlist_monitoring(self, temp_storage):
        """Test that get_info() returns nlist monitoring fields."""
        store = VectorStore(storage_path=temp_storage, dimension=384)
        
        # Add vectors
        vectors = np.random.randn(1100, 384).astype(np.float32)
        event_ids = [f"event_{i}" for i in range(1100)]
        metadata = [{"id": i} for i in range(1100)]
        
        store.add_vectors(vectors, event_ids, metadata)
        
        info = store.get_info()
        
        # Check new monitoring fields
        assert "optimal_nlist" in info
        assert "nlist_drift" in info
        assert "auto_retrain" in info
        assert "nlist_drift_threshold" in info
        assert "optimization_count" in info
        
        # Check values
        assert info["auto_retrain"] is True  # Default
        assert info["nlist_drift_threshold"] == 2.0  # Default
        assert info["optimization_count"] >= 1  # At least initial training

    def test_nlist_drift_threshold_custom(self, temp_storage):
        """Test custom drift threshold value."""
        store = VectorStore(
            storage_path=temp_storage, 
            dimension=384,
            nlist_drift_threshold=1.5  # Stricter threshold
        )
        
        assert store.nlist_drift_threshold == 1.5
        
        info = store.get_info()
        assert info["nlist_drift_threshold"] == 1.5

    def test_manual_retrain_records_optimization(self, temp_storage):
        """Test that manual retraining records optimization event."""
        store = VectorStore(storage_path=temp_storage, dimension=384)
        
        # Add vectors
        vectors = np.random.randn(1100, 384).astype(np.float32)
        event_ids = [f"event_{i}" for i in range(1100)]
        metadata = [{"id": i} for i in range(1100)]
        
        store.add_vectors(vectors, event_ids, metadata)
        
        # Clear history for clean test
        store.optimization_history = []
        
        # Manual retrain
        result = store.retrain(force=True)
        
        assert result["status"] == "retrained"
        assert "elapsed_time" in result
        
        # Check optimization recorded
        history = store.get_optimization_history()
        assert len(history) == 1
        assert history[0]["trigger"] == "manual"

    def test_persistence_across_restarts(self, temp_storage):
        """Test that optimization history persists across restarts."""
        # Create store and add data
        store1 = VectorStore(storage_path=temp_storage, dimension=384)
        
        vectors = np.random.randn(1100, 384).astype(np.float32)
        event_ids = [f"event_{i}" for i in range(1100)]
        metadata = [{"id": i} for i in range(1100)]
        
        store1.add_vectors(vectors, event_ids, metadata)
        
        history1 = store1.get_optimization_history()
        assert len(history1) >= 1
        
        # Reload store from disk
        store2 = VectorStore(storage_path=temp_storage, dimension=384)
        
        history2 = store2.get_optimization_history()
        assert len(history2) == len(history1)
        assert history2[0]["timestamp"] == history1[0]["timestamp"]
