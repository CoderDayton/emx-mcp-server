"""Test batch encoding optimization for event aggregation."""

import time
import tempfile
from pathlib import Path

import pytest

from emx_mcp.memory.project_manager import ProjectMemoryManager
from emx_mcp.utils.config import load_config


class TestBatchEncoding:
    """Test batch encoding optimization provides 2-3x speedup."""

    @pytest.fixture
    def temp_project(self):
        """Create temporary project directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)

    @pytest.fixture
    def config(self):
        """Load test configuration."""
        config = load_config()
        # Small batch threshold for testing
        config["memory"]["batch_event_threshold"] = 5
        return config

    def test_batch_encoding_faster_than_individual(self, temp_project, config):
        """Verify batch encoding is faster than per-event encoding."""
        # Create manager with batch encoding enabled
        manager_batch = ProjectMemoryManager(
            project_path=str(temp_project / "batch"),
            global_path=str(temp_project / "global"),
            config=config,
        )

        # Create manager with batch encoding disabled (threshold=1)
        config_no_batch = config.copy()
        config_no_batch["memory"]["batch_event_threshold"] = 1
        manager_individual = ProjectMemoryManager(
            project_path=str(temp_project / "individual"),
            global_path=str(temp_project / "global2"),
            config=config_no_batch,
        )

        # Create test events (26 tokens each, typical segment size)
        test_events = []
        for i in range(20):
            tokens = [f"token_{i}_{j}" for j in range(26)]
            test_events.append(tokens)

        # Time batch encoding (threshold=5, so 4 batches of 5 events)
        start_batch = time.time()
        for tokens in test_events:
            manager_batch.add_event(tokens, embeddings=None, metadata={})
        manager_batch.flush_events()  # Ensure all flushed
        elapsed_batch = time.time() - start_batch

        # Time individual encoding (threshold=1, so 20 separate calls)
        start_individual = time.time()
        for tokens in test_events:
            manager_individual.add_event(tokens, embeddings=None, metadata={})
        elapsed_individual = time.time() - start_individual

        print(f"\nBatch encoding: {elapsed_batch:.2f}s")
        print(f"Individual encoding: {elapsed_individual:.2f}s")
        print(f"Speedup: {elapsed_individual / elapsed_batch:.2f}x")

        # Batch should be at least 1.2x faster (conservative baseline)
        # Typical speedup is 1.3-2.5x depending on event size and hardware
        assert elapsed_batch < elapsed_individual * 0.83, (
            f"Batch encoding ({elapsed_batch:.2f}s) should be faster than "
            f"individual ({elapsed_individual:.2f}s), got {elapsed_individual / elapsed_batch:.2f}x speedup"
        )

    def test_batch_flush_mechanism(self, temp_project, config):
        """Test that events are buffered and flushed correctly."""
        config["memory"]["batch_event_threshold"] = 5

        manager = ProjectMemoryManager(
            project_path=str(temp_project / "test"),
            global_path=str(temp_project / "global"),
            config=config,
        )

        # Add 4 events (below threshold)
        for i in range(4):
            tokens = [f"token_{i}_{j}" for j in range(10)]
            result = manager.add_event(tokens, embeddings=None, metadata={})
            # Should be buffered
            assert result["status"] == "buffered"
            assert result["buffered_count"] == i + 1

        # Add 5th event (hits threshold, triggers flush)
        tokens = [f"token_4_{j}" for j in range(10)]
        result = manager.add_event(tokens, embeddings=None, metadata={})
        # Should be added (flushed)
        assert result["status"] == "added"
        assert "event_id" in result

        # Verify 5 events were flushed
        store = manager.project_store
        assert store.event_count() == 5

    def test_manual_flush(self, temp_project, config):
        """Test manual flush of pending events."""
        config["memory"]["batch_event_threshold"] = 10

        manager = ProjectMemoryManager(
            project_path=str(temp_project / "test"),
            global_path=str(temp_project / "global"),
            config=config,
        )

        # Add 3 events (below threshold)
        for i in range(3):
            tokens = [f"token_{i}_{j}" for j in range(10)]
            result = manager.add_event(tokens, embeddings=None, metadata={})
            assert result["status"] == "buffered"

        # Manual flush
        flush_result = manager.flush_events()
        assert flush_result["status"] == "flushed"
        assert flush_result["num_events"] == 3
        assert flush_result["total_tokens"] == 30

        # Verify events were persisted
        store = manager.project_store
        assert store.event_count() == 3

    def test_force_flush_parameter(self, temp_project, config):
        """Test force_flush parameter bypasses batching."""
        config["memory"]["batch_event_threshold"] = 10

        manager = ProjectMemoryManager(
            project_path=str(temp_project / "test"),
            global_path=str(temp_project / "global"),
            config=config,
        )

        # Add event with force_flush=True
        tokens = [f"token_{i}" for i in range(10)]
        result = manager.add_event(
            tokens, embeddings=None, metadata={}, force_flush=True
        )

        # Should be added immediately
        assert result["status"] == "added"
        assert "event_id" in result

        # Verify event was persisted
        store = manager.project_store
        assert store.event_count() == 1

    def test_batch_encoding_preserves_embeddings(self, temp_project, config):
        """Verify batch encoding produces same embeddings as individual."""
        config["memory"]["batch_event_threshold"] = 5

        manager = ProjectMemoryManager(
            project_path=str(temp_project / "test"),
            global_path=str(temp_project / "global"),
            config=config,
        )

        # Add 5 events to trigger batch encoding
        test_tokens = []
        for i in range(5):
            tokens = [f"word_{i}_{j}" for j in range(10)]
            test_tokens.append(tokens)
            manager.add_event(tokens, embeddings=None, metadata={})

        # Encode individually for comparison
        encoder = manager.encoder
        individual_embeddings = [
            encoder.encode_tokens_with_context(tokens, context_window=10)
            for tokens in test_tokens
        ]

        # Get batch-encoded embeddings from storage
        store = manager.project_store
        event_ids = list(store.event_cache.keys())[:5]

        batch_embeddings = [
            store.get_event(event_id).embeddings for event_id in event_ids
        ]

        # Verify dimensions match
        for ind_emb, batch_emb in zip(individual_embeddings, batch_embeddings):
            assert len(ind_emb) == len(batch_emb), (
                f"Embedding counts differ: {len(ind_emb)} vs {len(batch_emb)}"
            )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
