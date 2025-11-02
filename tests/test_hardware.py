"""Tests for hardware detection utilities."""

import pytest

from emx_mcp.utils.hardware import (
    detect_batch_size,
    detect_device,
    enrich_config_with_hardware,
)


class TestHardwareDetection:
    """Test hardware detection and configuration."""

    def test_detect_device_returns_valid_value(self):
        """Test that device detection returns valid device string."""
        device = detect_device()
        assert device in ["cpu", "cuda"]

    def test_detect_device_consistency(self):
        """Test that device detection is consistent across calls."""
        device1 = detect_device()
        device2 = detect_device()
        assert device1 == device2

    def test_detect_batch_size_cpu(self):
        """Test batch size detection for CPU."""
        batch_size = detect_batch_size("cpu")
        assert isinstance(batch_size, int)
        assert batch_size > 0
        assert batch_size <= 512

    def test_detect_batch_size_cuda(self):
        """Test batch size detection for CUDA."""
        batch_size = detect_batch_size("cuda")
        assert isinstance(batch_size, int)
        assert batch_size > 0
        # CUDA can handle larger batches, up to 1024
        assert batch_size <= 1024

    def test_detect_batch_size_with_model_name(self):
        """Test batch size detection with different model names."""
        batch_size_small = detect_batch_size("cpu", "all-MiniLM-L6-v2")
        batch_size_large = detect_batch_size("cpu", "all-mpnet-base-v2")

        assert isinstance(batch_size_small, int)
        assert isinstance(batch_size_large, int)
        assert batch_size_small > 0
        assert batch_size_large > 0

    def test_enrich_config_basic(self):
        """Test basic config enrichment with hardware detection."""
        config = {"model": {"name": "all-MiniLM-L6-v2", "device": None, "batch_size": None}}

        enriched = enrich_config_with_hardware(config)

        assert enriched["model"]["device"] in ["cpu", "cuda"]
        assert isinstance(enriched["model"]["batch_size"], int)
        assert enriched["model"]["batch_size"] > 0

    def test_enrich_config_preserves_explicit_values(self):
        """Test that explicit config values are preserved."""
        config = {"model": {"name": "all-MiniLM-L6-v2", "device": "cpu", "batch_size": 64}}

        enriched = enrich_config_with_hardware(config)

        assert enriched["model"]["device"] == "cpu"
        assert enriched["model"]["batch_size"] == 64

    def test_enrich_config_with_missing_keys(self):
        """Test config enrichment with minimal input."""
        config = {"model": {}}

        enriched = enrich_config_with_hardware(config)

        # Should add device and batch_size
        assert "device" in enriched["model"]
        assert "batch_size" in enriched["model"]
        assert enriched["model"]["device"] in ["cpu", "cuda"]
        assert isinstance(enriched["model"]["batch_size"], int)

    def test_device_detection_never_fails(self):
        """Test that device detection always returns a valid value."""
        try:
            device = detect_device()
            assert device in ["cpu", "cuda"]
        except Exception as e:
            pytest.fail(f"detect_device should not raise: {e}")

    def test_batch_size_ranges(self):
        """Test that batch sizes are within reasonable ranges."""
        cpu_batch = detect_batch_size("cpu")
        cuda_batch = detect_batch_size("cuda")

        # CPU should typically have smaller batch sizes
        assert 8 <= cpu_batch <= 128

        # CUDA can handle larger batches, up to 1024
        assert 16 <= cuda_batch <= 1024

    def test_enrich_config_idempotent(self):
        """Test that enriching config multiple times is safe."""
        config = {"model": {"name": "all-MiniLM-L6-v2"}}

        enriched1 = enrich_config_with_hardware(config)
        enriched2 = enrich_config_with_hardware(enriched1)

        assert enriched1["model"]["device"] == enriched2["model"]["device"]
        assert enriched1["model"]["batch_size"] == enriched2["model"]["batch_size"]

    def test_batch_size_scales_with_device(self):
        """Test that batch size is appropriate for device type."""
        cpu_batch = detect_batch_size("cpu")
        cuda_batch = detect_batch_size("cuda")

        # CUDA should generally support larger or equal batch sizes
        assert cuda_batch >= cpu_batch or cuda_batch > 32
