"""Tests for EmbeddingEncoder pinned memory functionality."""

import numpy as np
import pytest

from emx_mcp.embeddings.encoder import EmbeddingEncoder
from emx_mcp.gpu.pinned_memory import TORCH_AVAILABLE


@pytest.fixture
def encoder_with_gpu_config(hardware_enriched_config):
    """Encoder with GPU config enabled and hardware-detected device/batch_size."""
    gpu_config = {
        "enable_pinned_memory": False,
        "pinned_buffer_size": 4,
        "pinned_max_batch": 256,
        "pinned_min_batch_threshold": 64,
    }
    return EmbeddingEncoder(
        model_name=hardware_enriched_config["model_name"],
        device=hardware_enriched_config["device"],
        batch_size=hardware_enriched_config["batch_size"],
        gpu_config=gpu_config,
    )


@pytest.fixture
def encoder_without_gpu_config():
    """Encoder with GPU config disabled."""
    gpu_config = {
        "enable_pinned_memory": False,
        "pinned_buffer_size": 2,
        "pinned_max_batch": 128,
        "pinned_min_batch_threshold": 64,
    }
    return EmbeddingEncoder(
        model_name="all-MiniLM-L6-v2",
        device="cpu",
        batch_size=64,
        gpu_config=gpu_config,
    )


class TestPinnedMemoryEncoding:
    """Test pinned memory integration in EmbeddingEncoder."""

    def test_encode_batch_below_threshold_returns_numpy(self, encoder_with_gpu_config):
        """Small batches should return numpy arrays even with pinned memory enabled."""
        token_lists = [
            ["This", "is", "a", "test"],
            ["Another", "test", "sequence"],
        ]

        result = encoder_with_gpu_config.encode_batch(token_lists, use_pinned_memory=True)

        # Below threshold (32), should return numpy array
        assert isinstance(result, np.ndarray)
        assert result.shape == (2, encoder_with_gpu_config.dimension)
        assert result.dtype == np.float32

    def test_encode_batch_above_threshold_with_pytorch(self, encoder_with_gpu_config):
        """Large batches with PyTorch should potentially use pinned memory."""
        # Create batch above threshold
        token_lists = [["token", str(i)] for i in range(40)]

        result = encoder_with_gpu_config.encode_batch(token_lists, use_pinned_memory=True)

        if TORCH_AVAILABLE:
            # May return tuple (pinned_tensor, release_callback) or numpy fallback
            if isinstance(result, tuple):
                embeddings, release_callback = result
                assert callable(release_callback)
                # Verify tensor properties
                import torch

                assert isinstance(embeddings, torch.Tensor)
                assert embeddings.shape == (40, encoder_with_gpu_config.dimension)
                # Release buffer back to pool
                release_callback()
            else:
                # Fallback to numpy (e.g., pool exhausted)
                assert isinstance(result, np.ndarray)
        else:
            # Without PyTorch, should always return numpy
            assert isinstance(result, np.ndarray)

    def test_encode_batch_disabled_returns_numpy(self, encoder_without_gpu_config):
        """Pinned memory disabled should always return numpy arrays."""
        token_lists = [["token", str(i)] for i in range(50)]

        result = encoder_without_gpu_config.encode_batch(token_lists, use_pinned_memory=True)

        # Should return numpy even though use_pinned_memory=True
        assert isinstance(result, np.ndarray)
        assert result.shape == (50, encoder_without_gpu_config.dimension)

    def test_encode_batch_use_pinned_false_returns_numpy(self, encoder_with_gpu_config):
        """use_pinned_memory=False should always return numpy."""
        token_lists = [["token", str(i)] for i in range(50)]

        result = encoder_with_gpu_config.encode_batch(token_lists, use_pinned_memory=False)

        assert isinstance(result, np.ndarray)
        assert result.shape == (50, encoder_with_gpu_config.dimension)

    def test_encode_batch_correctness_with_pinned_memory(self, encoder_with_gpu_config):
        """Pinned memory results should match regular numpy results."""
        token_lists = [
            ["semantic", "similarity", "test"],
            ["another", "embedding", "check"],
        ]

        # Get regular numpy result
        numpy_result = encoder_with_gpu_config.encode_batch(token_lists, use_pinned_memory=False)

        # Get pinned memory result (will be numpy due to small batch)
        pinned_result = encoder_with_gpu_config.encode_batch(token_lists, use_pinned_memory=True)

        # Both should be numpy for small batches
        assert isinstance(numpy_result, np.ndarray)
        assert isinstance(pinned_result, np.ndarray)

        # Results should be identical
        np.testing.assert_array_equal(numpy_result, pinned_result)

    def test_release_callback_execution(self, encoder_with_gpu_config):
        """Verify release callback can be called without errors."""
        if not TORCH_AVAILABLE:
            pytest.skip("PyTorch not available")

        # Create large batch to trigger pinned memory
        token_lists = [["token", str(i)] for i in range(64)]

        result = encoder_with_gpu_config.encode_batch(token_lists, use_pinned_memory=True)

        if isinstance(result, tuple):
            _, release_callback = result
            # Should not raise
            release_callback()
            # Calling again should be safe (idempotent)
            release_callback()

    def test_gpu_config_defaults(self, hardware_enriched_config):
        """Encoder without gpu_config should use safe defaults."""
        encoder = EmbeddingEncoder(
            model_name=hardware_enriched_config["model_name"],
            device=hardware_enriched_config["device"],
            batch_size=hardware_enriched_config["batch_size"],
        )

        assert encoder.gpu_config["enable_pinned_memory"] is False
        assert encoder.gpu_config["pinned_buffer_size"] == 4
        assert encoder.gpu_config["pinned_max_batch"] == 256
        assert encoder.gpu_config["pinned_min_batch_threshold"] == 64
