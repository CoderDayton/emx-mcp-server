"""Type-safe configuration validation using Pydantic Settings."""

from pathlib import Path
from typing import Literal, Optional

from pydantic import Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class ModelConfig(BaseSettings):
    """Model configuration with validation."""

    model_config = SettingsConfigDict(env_prefix="EMX_MODEL_")

    name: str = Field(
        default="all-MiniLM-L6-v2",
        description="Sentence-transformers model name",
    )
    device: Optional[Literal["cpu", "cuda"]] = Field(
        default=None,
        description="Device to run model on (None = auto-detect based on CUDA availability)",
    )
    batch_size: Optional[int] = Field(
        default=None,
        ge=1,
        le=512,
        description="Inference batch size (None = auto-scale: GPU 64-512 based on VRAM, CPU 64)",
    )

    @field_validator("name")
    @classmethod
    def validate_model_name(cls, v: str) -> str:
        """Validate model name is not empty."""
        if not v or not v.strip():
            raise ValueError("Model name cannot be empty")
        return v.strip()

    @field_validator("batch_size")
    @classmethod
    def validate_batch_size(cls, v: Optional[int]) -> Optional[int]:
        """Validate batch_size when provided (None allowed for auto-scaling)."""
        if v is not None and (v < 1 or v > 512):
            raise ValueError(f"batch_size must be between 1 and 512, got {v}")
        return v


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

    vector_dim: Optional[int] = Field(
        default=None,
        ge=1,
        le=4096,
        description="Embedding dimension (None = auto-detect from model, recommended)",
    )
    nprobe: int = Field(
        default=16,  # Increased from 8 for better SQ recall
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
    auto_retrain: bool = Field(
        default=True,
        description="Automatically retrain index when nlist drift detected",
    )
    nlist_drift_threshold: float = Field(
        default=2.0,
        ge=1.1,
        le=10.0,
        description="Trigger retraining when nlist drift exceeds this ratio",
    )
    use_sq: bool = Field(
        default=True,
        description="Enable 8-bit Scalar Quantization (SQ) compression for 4x memory reduction and 97-99% recall",
    )
    sq_bits: int = Field(
        default=8,
        ge=8,
        le=8,
        description="Bits per SQ code (only 8-bit supported for optimal recall)",
    )
    use_gpu: bool = Field(
        default=True,
        description="Enable GPU acceleration for vector operations (auto-detects availability)",
    )
    adaptive_nprobe: bool = Field(
        default=True,
        description="Automatically adjust nprobe to maintain target recall as nlist grows",
    )
    target_recall: float = Field(
        default=0.95,
        ge=0.80,
        le=0.99,
        description="Target recall threshold for adaptive nprobe tuning (0.80-0.99)",
    )
    nprobe_min: int = Field(
        default=8,
        ge=1,
        le=512,
        description="Minimum nprobe value (adaptive tuning lower bound)",
    )
    nprobe_max: int = Field(
        default=128,
        ge=1,
        le=1024,
        description="Maximum nprobe value (adaptive tuning upper bound)",
    )

    @field_validator("vector_dim")
    @classmethod
    def validate_vector_dim(cls, v: Optional[int]) -> Optional[int]:
        """Validate vector dimension is reasonable (None allowed for auto-detection)."""
        if v is None:
            return None  # Auto-detect from model
        common_dims = {128, 256, 384, 512, 768, 1024, 1536, 2048}
        if v not in common_dims:
            # Warning: non-standard dimension (still valid)
            pass
        return v
    
    @model_validator(mode="after")
    def validate_sq_config(self) -> "StorageConfig":
        """Validate SQ configuration compatibility."""
        if self.use_sq:
            # SQ works with both Flat and IVF indices
            if self.index_type not in ("IVF", "Flat"):
                raise ValueError(
                    f"SQ compression requires index_type='IVF' or 'Flat', got '{self.index_type}'"
                )
            
            # Validate sq_bits (only 8-bit supported)
            if self.sq_bits != 8:
                raise ValueError(
                    f"SQ compression only supports 8-bit quantization, got {self.sq_bits}-bit"
                )
        
        return self


class GPUConfig(BaseSettings):
    """GPU optimization configuration with validation."""

    model_config = SettingsConfigDict(env_prefix="EMX_GPU_")

    enable_pinned_memory: bool = Field(
        default=True,
        description="Enable pinned memory pool for async GPU transfers",
    )
    pinned_buffer_size: int = Field(
        default=4,
        ge=1,
        le=32,
        description="Number of pinned memory buffers in pool",
    )
    pinned_max_batch: int = Field(
        default=128,
        ge=32,
        le=512,
        description="Maximum batch size per pinned buffer",
    )
    pinned_min_batch_threshold: int = Field(
        default=64,
        ge=1,
        le=256,
        description="Minimum batch size to use pinned memory (overhead below this)",
    )

    @model_validator(mode="after")
    def validate_batch_sizes(self) -> "GPUConfig":
        """Validate pinned memory batch thresholds."""
        if self.pinned_min_batch_threshold > self.pinned_max_batch:
            raise ValueError(
                f"pinned_min_batch_threshold ({self.pinned_min_batch_threshold}) "
                f"must be <= pinned_max_batch ({self.pinned_max_batch})"
            )
        return self


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
    gpu: GPUConfig = Field(default_factory=GPUConfig)
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
                "auto_retrain": self.storage.auto_retrain,
                "nlist_drift_threshold": self.storage.nlist_drift_threshold,
                "use_sq": self.storage.use_sq,
                "sq_bits": self.storage.sq_bits,
                "use_gpu": self.storage.use_gpu,
            },
            "gpu": {
                "enable_pinned_memory": self.gpu.enable_pinned_memory,
                "pinned_buffer_size": self.gpu.pinned_buffer_size,
                "pinned_max_batch": self.gpu.pinned_max_batch,
                "pinned_min_batch_threshold": self.gpu.pinned_min_batch_threshold,
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
            gpu=GPUConfig(),
            logging=LoggingConfig(),
        )
    except Exception as e:
        raise ValueError(f"Configuration validation failed: {e}") from e
