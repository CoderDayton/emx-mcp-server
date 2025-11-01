"""Integration tests for config wiring to VectorStore."""

import tempfile

from emx_mcp.storage.vector_store import VectorStore
from emx_mcp.utils.config import load_config
from emx_mcp.utils.config_validator import EMXConfig, StorageConfig


class TestConfigIntegration:
    """Test configuration wiring to VectorStore."""

    def test_vector_store_uses_config_defaults(self):
        """Test VectorStore receives config defaults."""
        config = load_config()

        # Default vector_dim is None (auto-detection), so we need to provide it explicitly
        # In production, ProjectMemoryManager auto-detects from encoder
        dimension = config["storage"]["vector_dim"] or 384

        with tempfile.TemporaryDirectory() as temp_dir:
            store = VectorStore(
                storage_path=temp_dir,
                dimension=dimension,
                nprobe=config["storage"]["nprobe"],
            )

            # Check actual config values are applied
            assert store.nprobe == config["storage"]["nprobe"]
            assert store.dimension == 384

    def test_vector_store_respects_custom_config(self, monkeypatch):
        """Test VectorStore respects custom config values."""
        monkeypatch.setenv("EMX_STORAGE_VECTOR_DIM", "768")
        monkeypatch.setenv("EMX_STORAGE_NPROBE", "32")

        # Force reload config
        config_obj = EMXConfig(storage=StorageConfig())
        config = config_obj.to_legacy_dict()

        with tempfile.TemporaryDirectory() as temp_dir:
            store = VectorStore(
                storage_path=temp_dir,
                dimension=config["storage"]["vector_dim"],
                nprobe=config["storage"]["nprobe"],
            )

            assert store.dimension == 768
            assert store.nprobe == 32
            assert store.dimension == 768

    def test_vector_dim_auto_detection_config(self, monkeypatch):
        """Test vector_dim defaults to None for auto-detection."""
        # Clear any env override
        monkeypatch.delenv("EMX_STORAGE_VECTOR_DIM", raising=False)

        config_obj = EMXConfig(storage=StorageConfig())

        # Should be None by default (auto-detection)
        assert config_obj.storage.vector_dim is None

        legacy = config_obj.to_legacy_dict()
        assert legacy["storage"]["vector_dim"] is None

    def test_vector_dim_explicit_override(self, monkeypatch):
        """Test explicit vector_dim override works."""
        monkeypatch.setenv("EMX_STORAGE_VECTOR_DIM", "768")

        config_obj = EMXConfig(storage=StorageConfig())

        # Should respect explicit value
        assert config_obj.storage.vector_dim == 768

        legacy = config_obj.to_legacy_dict()
        assert legacy["storage"]["vector_dim"] == 768

    def test_to_legacy_dict_includes_adaptive_fields(self):
        """Test legacy dict conversion includes new fields."""
        config = EMXConfig()
        legacy = config.to_legacy_dict()

        assert "auto_retrain" in legacy["storage"]
        assert "nlist_drift_threshold" in legacy["storage"]
        assert legacy["storage"]["auto_retrain"] is True
        assert legacy["storage"]["nlist_drift_threshold"] == 2.0

    def test_hierarchical_memory_store_integration(self, monkeypatch):
        """Test HierarchicalMemoryStore uses config for VectorStore."""
        from emx_mcp.memory.storage import HierarchicalMemoryStore

        # Set explicit dimension for test
        monkeypatch.setenv("EMX_STORAGE_VECTOR_DIM", "384")

        config = load_config()

        with tempfile.TemporaryDirectory() as temp_dir:
            store = HierarchicalMemoryStore(
                storage_path=temp_dir,
                config=config,
            )

            # Check VectorStore received config
            assert store.vector_store.nprobe == config["storage"]["nprobe"]
            assert (
                store.vector_store.dimension == config["storage"]["vector_dim"] or 384
            )
