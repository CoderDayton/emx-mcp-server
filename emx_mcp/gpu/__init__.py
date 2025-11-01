"""GPU optimization utilities for FAISS vector search."""

from emx_mcp.gpu.pinned_memory import (
    TORCH_AVAILABLE,
    PinnedMemoryPool,
    get_global_pool,
)

__all__ = ["PinnedMemoryPool", "get_global_pool", "TORCH_AVAILABLE"]
