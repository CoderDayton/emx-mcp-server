"""GPU optimization utilities for FAISS vector search."""

from emx_mcp.gpu.pinned_memory import (
    PinnedMemoryPool,
    get_global_pool,
    TORCH_AVAILABLE,
)

__all__ = ["PinnedMemoryPool", "get_global_pool", "TORCH_AVAILABLE"]
