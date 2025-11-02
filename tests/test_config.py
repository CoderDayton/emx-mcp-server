"""Tests for configuration module."""

import pytest

from emx_mcp.utils.config import (
    get_bool_env_var,
    get_env_var,
    get_float_env_var,
    get_int_env_var,
    load_config,
)


class TestConfig:
    """Test configuration loading and defaults."""

    def test_load_config_returns_dict(self):
        """Test that load_config returns a dictionary."""
        config = load_config()
        assert isinstance(config, dict)

    def test_config_has_required_keys(self):
        """Test that config has all required keys."""
        config = load_config()

        # Model configuration
        assert "model" in config
        assert "name" in config["model"]
        assert "device" in config["model"]
        assert "batch_size" in config["model"]

        # Memory configuration
        assert "memory" in config
        assert "gamma" in config["memory"]
        assert "context_window" in config["memory"]

        # Paths
        assert "project_path" in config
        assert "global_path" in config

    def test_get_env_var_default(self, monkeypatch):
        """Test getting environment variable with default."""
        monkeypatch.delenv("TEST_VAR", raising=False)

        value = get_env_var("TEST_VAR", "default")
        assert value == "default"

    def test_get_env_var_override(self, monkeypatch):
        """Test getting environment variable with override."""
        monkeypatch.setenv("TEST_VAR", "custom")

        value = get_env_var("TEST_VAR", "default")
        assert value == "custom"

    def test_get_bool_env_var_true_values(self, monkeypatch):
        """Test boolean environment variable parsing for true values."""
        for true_value in ["true", "True", "TRUE", "1", "yes", "YES", "on", "ON"]:
            monkeypatch.setenv("TEST_BOOL", true_value)
            assert get_bool_env_var("TEST_BOOL", False) is True

    def test_get_bool_env_var_false_values(self, monkeypatch):
        """Test boolean environment variable parsing for false values."""
        for false_value in ["false", "False", "FALSE", "0", "no", "NO", "off", "OFF"]:
            monkeypatch.setenv("TEST_BOOL", false_value)
            assert get_bool_env_var("TEST_BOOL", True) is False

    def test_get_bool_env_var_default(self, monkeypatch):
        """Test boolean environment variable default."""
        monkeypatch.delenv("TEST_BOOL", raising=False)

        assert get_bool_env_var("TEST_BOOL", True) is True
        assert get_bool_env_var("TEST_BOOL", False) is False

    def test_get_int_env_var(self, monkeypatch):
        """Test integer environment variable parsing."""
        monkeypatch.setenv("TEST_INT", "42")
        assert get_int_env_var("TEST_INT", 0) == 42

    def test_get_int_env_var_default(self, monkeypatch):
        """Test integer environment variable default."""
        monkeypatch.delenv("TEST_INT", raising=False)
        assert get_int_env_var("TEST_INT", 100) == 100

    def test_get_float_env_var(self, monkeypatch):
        """Test float environment variable parsing."""
        monkeypatch.setenv("TEST_FLOAT", "3.14")
        assert get_float_env_var("TEST_FLOAT", 0.0) == 3.14

    def test_get_float_env_var_default(self, monkeypatch):
        """Test float environment variable default."""
        monkeypatch.delenv("TEST_FLOAT", raising=False)
        assert get_float_env_var("TEST_FLOAT", 2.5) == 2.5

    def test_config_with_env_overrides(self, monkeypatch):
        """Test that environment variables override defaults."""
        monkeypatch.setenv("EMX_MEMORY_GAMMA", "2.5")

        config = load_config()
        assert config["memory"]["gamma"] == 2.5

    def test_config_device_is_valid(self):
        """Test that device configuration is valid."""
        config = load_config()
        # Device can be None (auto-detect) or set to cpu/cuda
        device = config["model"]["device"]
        assert device is None or device in ["cpu", "cuda"]

    def test_config_batch_size_is_positive(self):
        """Test that batch size is positive or None (auto-detect)."""
        config = load_config()
        batch_size = config["model"]["batch_size"]
        assert batch_size is None or batch_size > 0

    def test_config_gamma_is_positive(self):
        """Test that gamma is positive."""
        config = load_config()
        assert config["memory"]["gamma"] > 0

    def test_config_paths_are_set(self):
        """Test that paths are properly set."""
        config = load_config()

        assert config["project_path"] is not None
        assert config["global_path"] is not None

    def test_int_env_var_invalid_raises(self, monkeypatch):
        """Test that invalid integer raises error."""
        monkeypatch.setenv("TEST_INT", "not_a_number")

        with pytest.raises(ValueError):
            get_int_env_var("TEST_INT", 0)

    def test_float_env_var_invalid_raises(self, monkeypatch):
        """Test that invalid float raises error."""
        monkeypatch.setenv("TEST_FLOAT", "not_a_number")

        with pytest.raises(ValueError):
            get_float_env_var("TEST_FLOAT", 0.0)

    def test_config_consistency(self):
        """Test that config values are consistent across calls."""
        config1 = load_config()
        config2 = load_config()

        assert config1["model"]["name"] == config2["model"]["name"]
        assert config1["memory"]["gamma"] == config2["memory"]["gamma"]

    def test_empty_string_env_var_uses_default(self, monkeypatch):
        """Test that empty string uses default value."""
        monkeypatch.setenv("TEST_VAR", "")

        value = get_env_var("TEST_VAR", "default")
        assert value == "default"
