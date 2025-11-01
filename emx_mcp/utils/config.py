"""Configuration management with environment variable support and validation."""

import os
from pathlib import Path

from dotenv import load_dotenv

from emx_mcp.utils.config_validator import load_validated_config

# Auto-load .env file if present
env_file = Path.cwd() / ".env"
if env_file.exists():
    load_dotenv(dotenv_path=env_file)


def load_config() -> dict:
    """
    Load and validate configuration from environment variables.

    Automatically loads .env file from current directory if present.
    Uses Pydantic validation for type safety and range checking.

    Returns:
        dict: Validated configuration in legacy format with project_path and global_path

    Raises:
        ValueError: If configuration validation fails
    """
    config_obj = load_validated_config()
    legacy_dict = config_obj.to_legacy_dict()

    # Add project and global paths to top level
    legacy_dict["project_path"] = config_obj.project_path
    legacy_dict["global_path"] = config_obj.global_path

    return legacy_dict


# Legacy helper functions (maintained for backward compatibility)
def _get_env_bool(name: str, default: bool) -> bool:
    """Get boolean environment variable."""
    value = os.getenv(name)
    if value is None:
        return default

    if value.lower() in ("true", "1", "yes", "on"):
        return True
    elif value.lower() in ("false", "0", "no", "off"):
        return False
    else:
        return bool(value)


def _get_env_int(name: str, default: int) -> int:
    """Get integer environment variable."""
    value = os.getenv(name)
    if value is None:
        return default
    return int(value)


def _get_env_float(name: str, default: float) -> float:
    """Get float environment variable."""
    value = os.getenv(name)
    if value is None:
        return default
    return float(value)


def _get_env_str(name: str, default: str) -> str:
    """Get string environment variable."""
    value = os.getenv(name)
    if value is None:
        return default
    return str(value)


def get_env_var(name: str, default: str = "") -> str:
    """Get environment variable with optional default."""
    return os.getenv(name, default) or default


def get_bool_env_var(name: str, default: bool = False) -> bool:
    """Get boolean environment variable."""
    return _get_env_bool(name, default)


def get_int_env_var(name: str, default: int = 0) -> int:
    """Get integer environment variable."""
    return _get_env_int(name, default)


def get_float_env_var(name: str, default: float = 0.0) -> float:
    """Get float environment variable."""
    return _get_env_float(name, default)
