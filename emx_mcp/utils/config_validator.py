"""Type-safe configuration validation using Pydantic Settings."""

from pathlib import Path
from typing import Literal

from pydantic import Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class ModelConfig(BaseSettings):
    """Model configuration with validation."""

    model_config = SettingsConfigDict(env_prefix="EMX_MODEL_")

    name: str = Field(
        default="all-MiniLM-L6-v2",
        description="Sentence-transformers model name",
    )
    device: Literal["cpu", "cuda"] = Field(
        default="cpu",
        description="Device to run model on",
    )
    batch_size: int = Field(
        default=32,
        ge=1,
        le=512,
        description="Inference batch size",
    )

    @field_validator("name")
    @classmethod
    def validate_model_name(cls, v: str) -> str:
        """Validate model name is not empty."""
        if not v or not v.strip():
            raise ValueError("Model name cannot be empty")
        return v.strip()


class MemoryConfig(BaseSettings):
    """Memory configuration with validation."""

    model_config = SettingsConfigDict(env_prefix="EMX_MEMORY_")

    gamma: float = Field(
        default=1.0,
        ge=0.1,
        le=10.0,
        description="Surprise threshold sensitivity",
    )
    context_window: int = Field(
        default=10,
        ge=1,
        le=100,
        description="Context window size for embeddings",
    )
    window_offset: int = Field(
        default=128,
        ge=1,
        le=2048,
        description="Window offset for surprise calculation",
    )
    min_block_size: int = Field(
        default=8,
        ge=1,
        le=1024,
        description="Minimum block size for segmentation",
    )
    max_block_size: int = Field(
        default=128,
        ge=1,
        le=4096,
        description="Maximum block size for segmentation",
    )
    n_init: int = Field(
        default=128,
        ge=1,
        le=10000,
        description="Initial tokens (attention sinks)",
    )
    n_local: int = Field(
        default=4096,
        ge=1,
        le=100000,
        description="Local context window size",
    )
    n_mem: int = Field(
        default=2048,
        ge=1,
        le=50000,
        description="Retrieved memory size",
    )
    repr_topk: int = Field(
        default=4,
        ge=1,
        le=100,
        description="Representative tokens per event",
    )
    refinement_metric: Literal["modularity", "conductance", "coverage"] = Field(
        default="modularity",
        description="Boundary refinement metric",
    )

    @model_validator(mode="after")
    def validate_block_sizes(self) -> "MemoryConfig":
        """Validate min_block_size <= max_block_size."""
        if self.min_block_size > self.max_block_size:
            raise ValueError(
                f"min_block_size ({self.min_block_size}) must be <= "
                f"max_block_size ({self.max_block_size})"
            )
        return self

    @model_validator(mode="after")
    def validate_memory_sizes(self) -> "MemoryConfig":
        """Validate memory size relationships."""
        if self.n_mem > self.n_local:
            raise ValueError(
                f"n_mem ({self.n_mem}) should be <= n_local ({self.n_local})"
            )
        if self.n_init > self.n_local:
            raise ValueError(
                f"n_init ({self.n_init}) should be <= n_local ({self.n_local})"
            )
        return self


class StorageConfig(BaseSettings):
    """Storage configuration with validation."""

    model_config = SettingsConfigDict(env_prefix="EMX_STORAGE_")

    vector_dim: int = Field(
        default=384,
        ge=1,
        le=4096,
        description="Embedding dimension (must match model)",
    )
    nprobe: int = Field(
        default=8,
        ge=1,
        le=1024,
        description="IVF search parameter",
    )
    disk_offload_threshold: int = Field(
        default=300000,
        ge=1000,
        le=10_000_000,
        description="Tokens before disk offloading",
    )
    min_training_size: int = Field(
        default=1000,
        ge=10,
        le=1_000_000,
        description="Minimum vectors before IVF training",
    )
    index_type: Literal["IVF", "Flat", "HNSW"] = Field(
        default="IVF",
        description="FAISS index type",
    )
    metric: Literal["cosine", "euclidean", "dot"] = Field(
        default="cosine",
        description="Distance metric",
    )

    @field_validator("vector_dim")
    @classmethod
    def validate_vector_dim(cls, v: int) -> int:
        """Validate vector dimension is reasonable."""
        common_dims = {128, 256, 384, 512, 768, 1024, 1536, 2048}
        if v not in common_dims:
            # Warning: non-standard dimension (still valid)
            pass
        return v


class LoggingConfig(BaseSettings):
    """Logging configuration with validation."""

    model_config = SettingsConfigDict(env_prefix="EMX_LOGGING_")

    level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = Field(
        default="INFO",
        description="Logging level",
    )
    format: str = Field(
        default="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        description="Log format string",
    )

    @field_validator("format")
    @classmethod
    def validate_format(cls, v: str) -> str:
        """Validate format string is not empty."""
        if not v or not v.strip():
            raise ValueError("Logging format cannot be empty")
        return v


class EMXConfig(BaseSettings):
    """Root EMX configuration with auto-loading from .env."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # Nested configuration sections
    model: ModelConfig = Field(default_factory=ModelConfig)
    memory: MemoryConfig = Field(default_factory=MemoryConfig)
    storage: StorageConfig = Field(default_factory=StorageConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)

    # Runtime paths (not prefixed)
    project_path: str = Field(
        default="",
        alias="EMX_PROJECT_PATH",
        description="Project path override",
    )
    global_path: str = Field(
        default="",
        alias="EMX_GLOBAL_PATH",
        description="Global memory path override",
    )

    @model_validator(mode="after")
    def resolve_paths(self) -> "EMXConfig":
        """Resolve and validate paths."""
        # Project path defaults to cwd if not set
        if not self.project_path:
            self.project_path = str(Path.cwd())

        # Global path defaults to ~/.emx-mcp/global_memories
        if not self.global_path:
            self.global_path = str(Path.home() / ".emx-mcp" / "global_memories")

        return self

    def to_legacy_dict(self) -> dict:
        """Convert to legacy dictionary format for backward compatibility."""
        return {
            "model": {
                "name": self.model.name,
                "device": self.model.device,
                "batch_size": self.model.batch_size,
            },
            "memory": {
                "gamma": self.memory.gamma,
                "context_window": self.memory.context_window,
                "window_offset": self.memory.window_offset,
                "min_block_size": self.memory.min_block_size,
                "max_block_size": self.memory.max_block_size,
                "n_init": self.memory.n_init,
                "n_local": self.memory.n_local,
                "n_mem": self.memory.n_mem,
                "repr_topk": self.memory.repr_topk,
                "refinement_metric": self.memory.refinement_metric,
            },
            "storage": {
                "vector_dim": self.storage.vector_dim,
                "nprobe": self.storage.nprobe,
                "disk_offload_threshold": self.storage.disk_offload_threshold,
                "min_training_size": self.storage.min_training_size,
                "index_type": self.storage.index_type,
                "metric": self.storage.metric,
            },
            "logging": {
                "level": self.logging.level,
                "format": self.logging.format,
            },
        }


def load_validated_config() -> EMXConfig:
    """
    Load and validate configuration from environment variables and .env file.

    Returns:
        EMXConfig: Validated configuration object

    Raises:
        ValueError: If configuration validation fails with detailed error messages
    """
    try:
        return EMXConfig(
            model=ModelConfig(),
            memory=MemoryConfig(),
            storage=StorageConfig(),
            logging=LoggingConfig(),
        )
    except Exception as e:
        raise ValueError(f"Configuration validation failed: {e}") from e
