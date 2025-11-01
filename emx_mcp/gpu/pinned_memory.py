"""Pinned memory pool for async GPU transfers with PyTorch/FAISS interop.

Based on EM-LLM reference implementation pattern:
https://github.com/em-llm/EM-LLM-model/blob/main/em_llm/attention/context_manager.py

Key patterns:
- Pre-allocate pinned CPU tensors (lines 80-81)
- Async transfer with non_blocking=True (line 155)
- Event-based synchronization (lines 89-91, 156-158)
"""

import logging
from collections.abc import Callable, Generator
from contextlib import contextmanager
from threading import Lock
from typing import TYPE_CHECKING, Any, Optional

import numpy as np

if TYPE_CHECKING:
    import torch

logger = logging.getLogger(__name__)

# Check for PyTorch availability at runtime
try:
    import torch

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    logger.warning("PyTorch not available - pinned memory disabled")


class PinnedMemoryPool:
    """
    Pool of reusable pinned CPU memory buffers for async GPU transfers.

    Pinned (page-locked) memory enables non-blocking CPU→GPU transfers
    via DMA, overlapping computation with data movement. Only beneficial
    for batch_size ≥ 32 due to allocation overhead.

    Architecture:
        1. Pre-allocate fixed-size pinned tensors at init
        2. acquire() returns (buffer, release_callback)
        3. Caller fills buffer with embeddings
        4. GPU transfer: buffer.to(device, non_blocking=True)
        5. Record CUDA event, call release_callback after event completes

    Thread-safety:
        All operations protected by internal lock. Acquire blocks if pool
        exhausted until buffer released.

    Example usage:
        >>> pool = PinnedMemoryPool(buffer_size=4, dimension=384)
        >>> buffer, release = pool.acquire(batch_size=32)
        >>> buffer[:32] = embeddings  # Fill with data
        >>> gpu_tensor = buffer[:32].to('cuda', non_blocking=True)
        >>> event = torch.cuda.Event()
        >>> event.record()
        >>> # Later, after event completes:
        >>> event.synchronize()
        >>> release()
    """

    def __init__(
        self,
        buffer_size: int = 4,
        dimension: int = 384,
        max_batch_size: int = 128,
        dtype: Any | None = None,  # torch.dtype at runtime
    ):
        """
        Initialize pool with pre-allocated pinned buffers.

        Args:
            buffer_size: Number of buffers in pool (default: 4)
            dimension: Embedding dimension (default: 384 for all-MiniLM-L6-v2)
            max_batch_size: Maximum batch size per buffer (default: 128)
            dtype: PyTorch dtype (default: torch.float32)

        Raises:
            RuntimeError: If PyTorch not available or CUDA not initialized
        """
        if not TORCH_AVAILABLE:
            raise RuntimeError("PyTorch required for pinned memory pool")

        if dtype is None:
            dtype = torch.float32

        self.buffer_size = buffer_size
        self.dimension = dimension
        self.max_batch_size = max_batch_size
        self.dtype = dtype

        # Pre-allocate pinned buffers
        self._buffers = []
        self._available = []  # Stack of available buffer indices
        self._lock = Lock()

        logger.info(
            f"Initializing PinnedMemoryPool: {buffer_size} buffers, "
            f"dim={dimension}, max_batch={max_batch_size}"
        )

        # Allocate all buffers upfront
        for i in range(buffer_size):
            try:
                # Allocate on CPU and pin memory
                buffer = torch.empty(
                    (max_batch_size, dimension), dtype=dtype, device="cpu"
                ).pin_memory()
                self._buffers.append(buffer)
                self._available.append(i)
            except Exception as e:
                logger.error(f"Failed to allocate pinned buffer {i}: {e}")
                # Clean up partially allocated buffers
                self._buffers.clear()
                self._available.clear()
                raise RuntimeError(f"Pinned memory allocation failed: {e}") from e

        logger.info(f"Successfully allocated {len(self._buffers)} pinned buffers")

    def acquire(
        self, batch_size: int, timeout: float | None = None
    ) -> tuple[Any, Callable[[], None]]:  # Returns (torch.Tensor, release_callback)
        """
        Acquire a pinned buffer from pool.

        Blocks if no buffers available until one is released. Returns a
        slice of the buffer sized for the requested batch and a release
        callback to return it to the pool.

        Args:
            batch_size: Number of vectors needed (≤ max_batch_size)
            timeout: Maximum wait time in seconds (None = infinite)

        Returns:
            Tuple of (pinned_tensor_slice, release_callback)

        Raises:
            ValueError: If batch_size > max_batch_size
            RuntimeError: If timeout expires with no buffer available
        """
        if batch_size > self.max_batch_size:
            raise ValueError(
                f"Requested batch_size {batch_size} exceeds max_batch_size {self.max_batch_size}"
            )

        # Acquire buffer with lock
        with self._lock:
            if not self._available:
                # TODO: Implement timeout with condition variable if needed
                # For now, fail fast rather than blocking indefinitely
                raise RuntimeError(
                    f"No buffers available in pool (size={self.buffer_size}). "
                    "Consider increasing buffer_size or reducing concurrency."
                )

            buffer_idx = self._available.pop()
            buffer = self._buffers[buffer_idx]

        # Return slice sized for batch + release callback
        buffer_slice = buffer[:batch_size]

        def release() -> None:
            """Return buffer to pool."""
            with self._lock:
                self._available.append(buffer_idx)

        return buffer_slice, release

    @contextmanager
    def acquire_context(self, batch_size: int) -> Generator[Any, None, None]:  # Yields torch.Tensor
        """
        Context manager for automatic buffer release.

        Example:
            >>> with pool.acquire_context(32) as buffer:
            ...     buffer[:] = embeddings
            ...     gpu_tensor = buffer.to('cuda', non_blocking=True)

        Args:
            batch_size: Number of vectors needed

        Yields:
            Pinned tensor slice of shape (batch_size, dimension)
        """
        buffer, release = self.acquire(batch_size)
        try:
            yield buffer
        finally:
            release()

    def available_buffers(self) -> int:
        """Get number of available buffers in pool."""
        with self._lock:
            return len(self._available)

    def total_buffers(self) -> int:
        """Get total buffer capacity."""
        return self.buffer_size

    @staticmethod
    def numpy_to_pinned(
        array: np.ndarray, pool: Optional["PinnedMemoryPool"] = None
    ) -> tuple[Any, Callable[[], None] | None]:  # Returns (torch.Tensor, release)
        """
        Convert numpy array to pinned PyTorch tensor.

        If pool provided, acquires from pool. Otherwise allocates temporary
        pinned tensor (slower, no reuse).

        Args:
            array: Numpy array to convert (shape: [batch_size, dimension])
            pool: Optional pool to acquire from

        Returns:
            Tuple of (pinned_tensor, release_callback or None)
        """
        if not TORCH_AVAILABLE:
            raise RuntimeError("PyTorch required for pinned memory")

        array = np.asarray(array, dtype=np.float32)
        if array.ndim == 1:
            array = array.reshape(1, -1)

        batch_size = array.shape[0]

        if pool is not None:
            # Acquire from pool (reusable)
            buffer, release = pool.acquire(batch_size)
            buffer[:] = torch.from_numpy(array)
            return buffer, release
        else:
            # Temporary allocation (slower)
            tensor = torch.from_numpy(array).pin_memory()
            return tensor, None

    def __repr__(self) -> str:
        return (
            f"PinnedMemoryPool(buffers={self.buffer_size}, "
            f"available={self.available_buffers()}, "
            f"dim={self.dimension}, max_batch={self.max_batch_size})"
        )


# Global singleton pool (lazy initialization)
_global_pool: PinnedMemoryPool | None = None
_global_pool_lock = Lock()


def get_global_pool(
    buffer_size: int = 4,
    dimension: int = 384,
    max_batch_size: int = 128,
) -> PinnedMemoryPool | None:
    """
    Get or create global pinned memory pool singleton.

    Returns None if PyTorch unavailable or initialization fails.
    Thread-safe lazy initialization.

    Args:
        buffer_size: Number of buffers (only used on first call)
        dimension: Embedding dimension (only used on first call)
        max_batch_size: Max batch per buffer (only used on first call)

    Returns:
        Global PinnedMemoryPool instance or None if unavailable
    """
    global _global_pool

    if not TORCH_AVAILABLE:
        return None

    if _global_pool is None:
        with _global_pool_lock:
            # Double-check after acquiring lock
            if _global_pool is None:
                try:
                    _global_pool = PinnedMemoryPool(
                        buffer_size=buffer_size,
                        dimension=dimension,
                        max_batch_size=max_batch_size,
                    )
                    logger.info("Initialized global PinnedMemoryPool")
                except Exception as e:
                    logger.warning(f"Failed to initialize global pool: {e}")
                    return None

    return _global_pool
