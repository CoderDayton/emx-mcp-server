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
        
        with tempfile.TemporaryDirectory() as temp_dir:
            store = VectorStore(
                storage_path=temp_dir,
                dimension=config["storage"]["vector_dim"],
                nprobe=config["storage"]["nprobe"],
                auto_retrain=config["storage"]["auto_retrain"],
                nlist_drift_threshold=config["storage"]["nlist_drift_threshold"],
            )
            
            assert store.auto_retrain is True
            assert store.nlist_drift_threshold == 2.0
            assert store.dimension == 384

    def test_vector_store_respects_custom_config(self, monkeypatch):
        """Test VectorStore respects custom config values."""
        monkeypatch.setenv("EMX_STORAGE_AUTO_RETRAIN", "false")
        monkeypatch.setenv("EMX_STORAGE_NLIST_DRIFT_THRESHOLD", "3.5")
        
        # Force reload config
        config_obj = EMXConfig(
            storage=StorageConfig()
        )
        config = config_obj.to_legacy_dict()
        
        with tempfile.TemporaryDirectory() as temp_dir:
            store = VectorStore(
                storage_path=temp_dir,
                dimension=config["storage"]["vector_dim"],
                auto_retrain=config["storage"]["auto_retrain"],
                nlist_drift_threshold=config["storage"]["nlist_drift_threshold"],
            )
            
            assert store.auto_retrain is False
            assert store.nlist_drift_threshold == 3.5

    def test_to_legacy_dict_includes_adaptive_fields(self):
        """Test legacy dict conversion includes new fields."""
        config = EMXConfig()
        legacy = config.to_legacy_dict()
        
        assert "auto_retrain" in legacy["storage"]
        assert "nlist_drift_threshold" in legacy["storage"]
        assert legacy["storage"]["auto_retrain"] is True
        assert legacy["storage"]["nlist_drift_threshold"] == 2.0

    def test_hierarchical_memory_store_integration(self):
        """Test HierarchicalMemoryStore uses config for VectorStore."""
        from emx_mcp.memory.storage import HierarchicalMemoryStore
        
        config = load_config()
        
        with tempfile.TemporaryDirectory() as temp_dir:
            store = HierarchicalMemoryStore(
                storage_path=temp_dir,
                config=config,
            )
            
            # Check VectorStore received config
            assert store.vector_store.auto_retrain == config["storage"]["auto_retrain"]
            assert store.vector_store.nlist_drift_threshold == config["storage"]["nlist_drift_threshold"]
