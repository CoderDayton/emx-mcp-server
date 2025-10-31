"""Comprehensive configuration validation tests."""

import os
from pathlib import Path

import pytest
from pydantic import ValidationError

from emx_mcp.utils.config import load_config
from emx_mcp.utils.config_validator import (
    EMXConfig,
    LoggingConfig,
    MemoryConfig,
    ModelConfig,
    StorageConfig,
    load_validated_config,
)


class TestModelConfig:
    """Test ModelConfig validation."""

    def test_default_values(self):
        """Test default configuration values."""
        config = ModelConfig()
        assert config.name == "all-MiniLM-L6-v2"
        assert config.device == "cpu"
        assert config.batch_size == 32

    def test_valid_cuda_device(self, monkeypatch):
        """Test CUDA device configuration."""
        monkeypatch.setenv("EMX_MODEL_DEVICE", "cuda")
        config = ModelConfig()
        assert config.device == "cuda"

    def test_invalid_device(self, monkeypatch):
        """Test invalid device raises error."""
        monkeypatch.setenv("EMX_MODEL_DEVICE", "gpu")
        with pytest.raises(ValidationError) as exc:
            ModelConfig()
        assert "device" in str(exc.value).lower()

    def test_batch_size_validation(self, monkeypatch):
        """Test batch size range validation."""
        # Valid range
        monkeypatch.setenv("EMX_MODEL_BATCH_SIZE", "16")
        config = ModelConfig()
        assert config.batch_size == 16

        # Too small
        monkeypatch.setenv("EMX_MODEL_BATCH_SIZE", "0")
        with pytest.raises(ValidationError):
            ModelConfig()

        # Too large
        monkeypatch.setenv("EMX_MODEL_BATCH_SIZE", "1000")
        with pytest.raises(ValidationError):
            ModelConfig()

    def test_empty_model_name(self, monkeypatch):
        """Test empty model name raises error."""
        monkeypatch.setenv("EMX_MODEL_NAME", "")
        with pytest.raises(ValidationError) as exc:
            ModelConfig()
        assert "empty" in str(exc.value).lower()

    def test_whitespace_model_name(self, monkeypatch):
        """Test whitespace-only model name raises error."""
        monkeypatch.setenv("EMX_MODEL_NAME", "   ")
        with pytest.raises(ValidationError):
            ModelConfig()


class TestMemoryConfig:
    """Test MemoryConfig validation."""

    def test_default_values(self):
        """Test default memory configuration."""
        config = MemoryConfig()
        assert config.gamma == 1.0
        assert config.context_window == 10
        assert config.window_offset == 128
        assert config.min_block_size == 8
        assert config.max_block_size == 128

    def test_gamma_range_validation(self, monkeypatch):
        """Test gamma range validation."""
        # Valid
        monkeypatch.setenv("EMX_MEMORY_GAMMA", "2.5")
        config = MemoryConfig()
        assert config.gamma == 2.5

        # Too small
        monkeypatch.setenv("EMX_MEMORY_GAMMA", "0.05")
        with pytest.raises(ValidationError):
            MemoryConfig()

        # Too large
        monkeypatch.setenv("EMX_MEMORY_GAMMA", "15.0")
        with pytest.raises(ValidationError):
            MemoryConfig()

    def test_block_size_validation(self, monkeypatch):
        """Test min_block_size <= max_block_size constraint."""
        # Valid
        monkeypatch.setenv("EMX_MEMORY_MIN_BLOCK_SIZE", "10")
        monkeypatch.setenv("EMX_MEMORY_MAX_BLOCK_SIZE", "100")
        config = MemoryConfig()
        assert config.min_block_size == 10
        assert config.max_block_size == 100

        # Invalid: min > max
        monkeypatch.setenv("EMX_MEMORY_MIN_BLOCK_SIZE", "200")
        monkeypatch.setenv("EMX_MEMORY_MAX_BLOCK_SIZE", "100")
        with pytest.raises(ValidationError) as exc:
            MemoryConfig()
        assert "min_block_size" in str(exc.value).lower()
        assert "max_block_size" in str(exc.value).lower()

    def test_memory_size_validation(self, monkeypatch):
        """Test memory size relationships."""
        # Valid: n_mem <= n_local
        monkeypatch.setenv("EMX_MEMORY_N_MEM", "1000")
        monkeypatch.setenv("EMX_MEMORY_N_LOCAL", "2000")
        config = MemoryConfig()
        assert config.n_mem <= config.n_local

        # Invalid: n_mem > n_local
        monkeypatch.setenv("EMX_MEMORY_N_MEM", "5000")
        monkeypatch.setenv("EMX_MEMORY_N_LOCAL", "2000")
        with pytest.raises(ValidationError) as exc:
            MemoryConfig()
        assert "n_mem" in str(exc.value).lower()

    def test_invalid_refinement_metric(self, monkeypatch):
        """Test invalid refinement metric."""
        monkeypatch.setenv("EMX_MEMORY_REFINEMENT_METRIC", "invalid_metric")
        with pytest.raises(ValidationError) as exc:
            MemoryConfig()
        assert "refinement_metric" in str(exc.value).lower()

    def test_context_window_range(self, monkeypatch):
        """Test context window validation."""
        # Valid
        monkeypatch.setenv("EMX_MEMORY_CONTEXT_WINDOW", "20")
        config = MemoryConfig()
        assert config.context_window == 20

        # Too small
        monkeypatch.setenv("EMX_MEMORY_CONTEXT_WINDOW", "0")
        with pytest.raises(ValidationError):
            MemoryConfig()

        # Too large
        monkeypatch.setenv("EMX_MEMORY_CONTEXT_WINDOW", "200")
        with pytest.raises(ValidationError):
            MemoryConfig()


class TestStorageConfig:
    """Test StorageConfig validation."""

    def test_default_values(self):
        """Test default storage configuration."""
        config = StorageConfig()
        assert config.vector_dim == 384
        assert config.nprobe == 8
        assert config.index_type == "IVF"
        assert config.metric == "cosine"

    def test_vector_dim_validation(self, monkeypatch):
        """Test vector dimension validation."""
        # Valid standard dimension
        monkeypatch.setenv("EMX_STORAGE_VECTOR_DIM", "768")
        config = StorageConfig()
        assert config.vector_dim == 768

        # Non-standard but valid
        monkeypatch.setenv("EMX_STORAGE_VECTOR_DIM", "500")
        config = StorageConfig()
        assert config.vector_dim == 500

    def test_invalid_index_type(self, monkeypatch):
        """Test invalid index type."""
        monkeypatch.setenv("EMX_STORAGE_INDEX_TYPE", "INVALID")
        with pytest.raises(ValidationError) as exc:
            StorageConfig()
        assert "index_type" in str(exc.value).lower()

    def test_invalid_metric(self, monkeypatch):
        """Test invalid distance metric."""
        monkeypatch.setenv("EMX_STORAGE_METRIC", "manhattan")
        with pytest.raises(ValidationError) as exc:
            StorageConfig()
        assert "metric" in str(exc.value).lower()

    def test_nprobe_range(self, monkeypatch):
        """Test nprobe range validation."""
        # Valid
        monkeypatch.setenv("EMX_STORAGE_NPROBE", "16")
        config = StorageConfig()
        assert config.nprobe == 16

        # Too small
        monkeypatch.setenv("EMX_STORAGE_NPROBE", "0")
        with pytest.raises(ValidationError):
            StorageConfig()


class TestLoggingConfig:
    """Test LoggingConfig validation."""

    def test_default_values(self):
        """Test default logging configuration."""
        config = LoggingConfig()
        assert config.level == "INFO"
        assert "asctime" in config.format

    def test_valid_log_levels(self, monkeypatch):
        """Test all valid log levels."""
        for level in ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]:
            monkeypatch.setenv("EMX_LOGGING_LEVEL", level)
            config = LoggingConfig()
            assert config.level == level

    def test_invalid_log_level(self, monkeypatch):
        """Test invalid log level."""
        monkeypatch.setenv("EMX_LOGGING_LEVEL", "TRACE")
        with pytest.raises(ValidationError) as exc:
            LoggingConfig()
        assert "level" in str(exc.value).lower()

    def test_empty_format(self, monkeypatch):
        """Test empty log format raises error."""
        monkeypatch.setenv("EMX_LOGGING_FORMAT", "")
        with pytest.raises(ValidationError) as exc:
            LoggingConfig()
        assert "format" in str(exc.value).lower()


class TestEMXConfig:
    """Test root EMXConfig validation."""

    def test_default_configuration(self):
        """Test full default configuration."""
        config = EMXConfig(
            model=ModelConfig(),
            memory=MemoryConfig(),
            storage=StorageConfig(),
            logging=LoggingConfig(),
        )
        assert config.model.name == "all-MiniLM-L6-v2"
        assert config.memory.gamma == 1.0
        assert config.storage.vector_dim == 384
        assert config.logging.level == "INFO"

    def test_path_resolution(self, monkeypatch):
        """Test automatic path resolution."""
        config = EMXConfig(
            model=ModelConfig(),
            memory=MemoryConfig(),
            storage=StorageConfig(),
            logging=LoggingConfig(),
        )
        # Project path defaults to cwd
        assert config.project_path == str(Path.cwd())

        # Global path defaults to ~/.emx-mcp/global_memories
        expected_global = str(Path.home() / ".emx-mcp" / "global_memories")
        assert config.global_path == expected_global

    def test_custom_paths(self, monkeypatch):
        """Test custom path configuration."""
        monkeypatch.setenv("EMX_PROJECT_PATH", "/custom/project")
        monkeypatch.setenv("EMX_GLOBAL_PATH", "/custom/global")

        config = EMXConfig(
            model=ModelConfig(),
            memory=MemoryConfig(),
            storage=StorageConfig(),
            logging=LoggingConfig(),
        )
        assert config.project_path == "/custom/project"
        assert config.global_path == "/custom/global"

    def test_to_legacy_dict(self):
        """Test conversion to legacy dictionary format."""
        config = EMXConfig(
            model=ModelConfig(),
            memory=MemoryConfig(),
            storage=StorageConfig(),
            logging=LoggingConfig(),
        )
        legacy = config.to_legacy_dict()

        # Check structure
        assert "model" in legacy
        assert "memory" in legacy
        assert "storage" in legacy
        assert "logging" in legacy

        # Check values
        assert legacy["model"]["name"] == "all-MiniLM-L6-v2"
        assert legacy["memory"]["gamma"] == 1.0
        assert legacy["storage"]["vector_dim"] == 384
        assert legacy["logging"]["level"] == "INFO"


class TestLoadValidatedConfig:
    """Test load_validated_config function."""

    def test_load_default_config(self):
        """Test loading default configuration."""
        config = load_validated_config()
        assert isinstance(config, EMXConfig)
        assert config.model.name == "all-MiniLM-L6-v2"

    def test_load_with_invalid_values(self, monkeypatch):
        """Test loading with invalid configuration raises error."""
        monkeypatch.setenv("EMX_MEMORY_GAMMA", "20.0")  # Out of range
        with pytest.raises(ValueError) as exc:
            load_validated_config()
        assert "validation failed" in str(exc.value).lower()


class TestLoadConfig:
    """Test backward-compatible load_config function."""

    def test_load_config_returns_dict(self):
        """Test load_config returns dictionary."""
        config = load_config()
        assert isinstance(config, dict)
        assert "model" in config
        assert "memory" in config
        assert "storage" in config
        assert "logging" in config

    def test_load_config_with_env_vars(self, monkeypatch):
        """Test load_config respects environment variables."""
        monkeypatch.setenv("EMX_MODEL_BATCH_SIZE", "64")
        monkeypatch.setenv("EMX_MEMORY_GAMMA", "1.5")

        config = load_config()
        assert config["model"]["batch_size"] == 64
        assert config["memory"]["gamma"] == 1.5

    def test_load_config_validation_failure(self, monkeypatch):
        """Test load_config raises on validation failure."""
        monkeypatch.setenv("EMX_MODEL_BATCH_SIZE", "9999")  # Out of range
        with pytest.raises(ValueError):
            load_config()


class TestEdgeCases:
    """Test edge cases and error conditions."""

    def test_type_coercion(self, monkeypatch):
        """Test type coercion from string env vars."""
        monkeypatch.setenv("EMX_MEMORY_GAMMA", "2")  # String int
        config = MemoryConfig()
        assert config.gamma == 2.0
        assert isinstance(config.gamma, float)

    def test_invalid_type(self, monkeypatch):
        """Test invalid type conversion."""
        monkeypatch.setenv("EMX_MEMORY_GAMMA", "not_a_number")
        with pytest.raises(ValidationError):
            MemoryConfig()

    def test_negative_values(self, monkeypatch):
        """Test negative values are rejected."""
        monkeypatch.setenv("EMX_STORAGE_VECTOR_DIM", "-100")
        with pytest.raises(ValidationError):
            StorageConfig()

    def test_zero_values(self, monkeypatch):
        """Test zero values where invalid."""
        monkeypatch.setenv("EMX_MEMORY_CONTEXT_WINDOW", "0")
        with pytest.raises(ValidationError):
            MemoryConfig()

    def test_case_insensitive_enum(self, monkeypatch):
        """Test case-insensitive enum values."""
        # Lowercase should work due to case_sensitive=False
        monkeypatch.setenv("EMX_MODEL_DEVICE", "cpu")
        config = ModelConfig()
        assert config.device == "cpu"

    def test_extra_env_vars_ignored(self, monkeypatch):
        """Test extra environment variables are ignored."""
        monkeypatch.setenv("EMX_MODEL_UNKNOWN_PARAM", "value")
        config = ModelConfig()  # Should not raise error
        assert not hasattr(config, "unknown_param")


class TestVectorDimensionValidation:
    """Test vector dimension mismatch validation."""

    def test_dimension_mismatch_384_to_768(self, tmp_path, monkeypatch):
        """Test dimension mismatch: config expects 768 but model outputs 384."""
        from emx_mcp.memory.project_manager import ProjectMemoryManager
        
        # Configure for 768-dim (wrong for all-MiniLM-L6-v2 which is 384-dim)
        monkeypatch.setenv("EMX_STORAGE_VECTOR_DIM", "768")
        config = load_config()
        
        project_path = tmp_path / "project"
        project_path.mkdir()
        global_path = tmp_path / "global"
        
        with pytest.raises(ValueError) as exc:
            ProjectMemoryManager(str(project_path), str(global_path), config)
        
        error_msg = str(exc.value)
        assert "Vector dimension mismatch" in error_msg
        assert "EMX_STORAGE_VECTOR_DIM=768" in error_msg
        assert "384-dimensional vectors" in error_msg
        assert "all-MiniLM-L6-v2" in error_msg
        assert "EMX_STORAGE_VECTOR_DIM=384" in error_msg
        assert "ENVIRONMENT_VARIABLES.md" in error_msg

    def test_dimension_mismatch_768_to_384(self, tmp_path, monkeypatch):
        """Test dimension mismatch: config expects 384 but model outputs 768."""
        from emx_mcp.memory.project_manager import ProjectMemoryManager
        
        # Use mpnet model (768-dim) but configure for 384-dim
        monkeypatch.setenv("EMX_MODEL_NAME", "all-mpnet-base-v2")
        monkeypatch.setenv("EMX_STORAGE_VECTOR_DIM", "384")
        config = load_config()
        
        project_path = tmp_path / "project"
        project_path.mkdir()
        global_path = tmp_path / "global"
        
        with pytest.raises(ValueError) as exc:
            ProjectMemoryManager(str(project_path), str(global_path), config)
        
        error_msg = str(exc.value)
        assert "Vector dimension mismatch" in error_msg
        assert "EMX_STORAGE_VECTOR_DIM=384" in error_msg
        assert "768-dimensional vectors" in error_msg
        assert "all-mpnet-base-v2" in error_msg
        assert "EMX_STORAGE_VECTOR_DIM=768" in error_msg

    def test_dimension_match_384(self, tmp_path, monkeypatch):
        """Test correct dimension configuration for 384-dim model."""
        from emx_mcp.memory.project_manager import ProjectMemoryManager
        
        # Correct configuration for all-MiniLM-L6-v2 (384-dim)
        monkeypatch.setenv("EMX_STORAGE_VECTOR_DIM", "384")
        config = load_config()
        
        project_path = tmp_path / "project"
        project_path.mkdir()
        global_path = tmp_path / "global"
        
        # Should succeed without error
        manager = ProjectMemoryManager(str(project_path), str(global_path), config)
        assert manager.encoder.dimension == 384
        assert config["storage"]["vector_dim"] == 384

    def test_dimension_match_768(self, tmp_path, monkeypatch):
        """Test correct dimension configuration for 768-dim model."""
        from emx_mcp.memory.project_manager import ProjectMemoryManager
        
        # Correct configuration for all-mpnet-base-v2 (768-dim)
        monkeypatch.setenv("EMX_MODEL_NAME", "all-mpnet-base-v2")
        monkeypatch.setenv("EMX_STORAGE_VECTOR_DIM", "768")
        config = load_config()
        
        project_path = tmp_path / "project"
        project_path.mkdir()
        global_path = tmp_path / "global"
        
        # Should succeed without error
        manager = ProjectMemoryManager(str(project_path), str(global_path), config)
        assert manager.encoder.dimension == 768
        assert config["storage"]["vector_dim"] == 768

    def test_error_message_actionable(self, tmp_path, monkeypatch):
        """Test error message provides actionable guidance."""
        from emx_mcp.memory.project_manager import ProjectMemoryManager
        
        monkeypatch.setenv("EMX_STORAGE_VECTOR_DIM", "512")
        config = load_config()
        
        project_path = tmp_path / "project"
        project_path.mkdir()
        global_path = tmp_path / "global"
        
        with pytest.raises(ValueError) as exc:
            ProjectMemoryManager(str(project_path), str(global_path), config)
        
        error_msg = str(exc.value)
        # Check for actionable components
        assert "Common model dimensions:" in error_msg
        assert "all-MiniLM-L6-v2: 384" in error_msg
        assert "all-mpnet-base-v2: 768" in error_msg
        assert "Fix:" in error_msg
        assert "EMX_STORAGE_VECTOR_DIM=" in error_msg

