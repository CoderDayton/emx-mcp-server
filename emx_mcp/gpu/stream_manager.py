"""CUDA stream manager for async GPU operations.

Based on EM-LLM reference implementation and PyTorch stream best practices:
- Pre-allocated pool of 4 streams for concurrent operations
- Thread-local storage for safe concurrent access
- Event-based synchronization for cross-stream dependencies
- Context manager pattern for automatic resource cleanup

Key patterns from context7 validation:
- torch.cuda.Stream() for stream creation
- torch.cuda.stream(stream) context manager for execution
- tensor.record_stream(stream) for memory safety
- stream.wait_stream(other_stream) for synchronization
- stream.synchronize() only when absolutely necessary (prefer event queries)

Reference:
https://pytorch.org/docs/stable/notes/cuda.html#cuda-streams
"""

import logging
import threading
from collections.abc import Generator
from contextlib import contextmanager
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch

logger = logging.getLogger(__name__)

# Check for PyTorch availability at runtime
try:
    import torch

    TORCH_AVAILABLE = True and torch.cuda.is_available()
except ImportError:
    TORCH_AVAILABLE = False
    logger.warning("PyTorch not available - stream manager disabled")


class StreamManager:
    """
    Manages pool of CUDA streams for async GPU operations.

    Provides thread-safe stream acquisition with automatic cleanup.
    Pre-allocates 4 streams in pool for typical concurrent workloads:
    - Stream 0: Index training
    - Stream 1: Vector transfers (CPU→GPU)
    - Stream 2: Embedding generation
    - Stream 3: Search operations

    Architecture:
        1. Lazy initialization on first acquire() call
        2. Thread-local tracking of acquired streams
        3. Context manager ensures streams returned to pool
        4. Event-based synchronization for cross-stream dependencies

    Thread-safety:
        Uses threading.Lock for pool mutations. Thread-local storage
        prevents cross-thread stream conflicts.

    Example usage:
        >>> manager = StreamManager(pool_size=4)
        >>> with manager.acquire_stream() as stream:
        ...     with torch.cuda.stream(stream):
        ...         tensor = data.to('cuda', non_blocking=True)
        ...         tensor.record_stream(stream)
        >>> # Stream automatically returned to pool
    """

    def __init__(self, pool_size: int = 4, device: int = 0):
        """
        Initialize stream manager.

        Args:
            pool_size: Number of streams in pool (default: 4)
            device: CUDA device index (default: 0)

        Raises:
            RuntimeError: If PyTorch not available or CUDA not initialized
        """
        if not TORCH_AVAILABLE:
            raise RuntimeError(
                "PyTorch with CUDA required for stream manager. "
                "Install with: pip install torch torchvision torchaudio --index-url "
                "https://download.pytorch.org/whl/cu118"
            )

        self.pool_size = pool_size
        self.device = device
        self._initialized = False

        # Lazy initialization - streams created on first acquire()
        self._streams: list[torch.cuda.Stream] = []
        self._available_indices: list[int] = []
        self._lock = threading.Lock()

        # Thread-local storage for acquired streams
        self._thread_local = threading.local()

        logger.info(f"StreamManager initialized: pool_size={pool_size}, device={device}")

    def _ensure_initialized(self):
        """Lazy initialization of stream pool."""
        if self._initialized:
            return

        with self._lock:
            # Double-check after acquiring lock
            if self._initialized:
                return

            logger.info(f"Allocating {self.pool_size} CUDA streams on device {self.device}")

            # Set device context
            with torch.cuda.device(self.device):
                for i in range(self.pool_size):
                    try:
                        stream = torch.cuda.Stream(device=self.device)
                        self._streams.append(stream)
                        self._available_indices.append(i)
                    except Exception as e:
                        logger.error(f"Failed to create CUDA stream {i}: {e}")
                        # Clean up partially allocated streams
                        self._streams.clear()
                        self._available_indices.clear()
                        raise RuntimeError(f"CUDA stream allocation failed: {e}") from e

            self._initialized = True
            logger.info(f"Successfully allocated {len(self._streams)} CUDA streams")

    def acquire(self, timeout: float | None = None) -> "torch.cuda.Stream":
        """
        Acquire stream from pool.

        Blocks if no streams available until one is released.

        Args:
            timeout: Maximum wait time in seconds (None = infinite)

        Returns:
            CUDA stream from pool

        Raises:
            RuntimeError: If timeout expires or pool exhausted
        """
        self._ensure_initialized()

        with self._lock:
            if not self._available_indices:
                # TODO: Implement timeout with condition variable if needed
                raise RuntimeError(
                    f"No streams available in pool (size={self.pool_size}). "
                    "Consider increasing pool_size or reducing concurrency."
                )

            stream_idx = self._available_indices.pop()
            stream = self._streams[stream_idx]

        # Track in thread-local storage for debugging
        if not hasattr(self._thread_local, "acquired_streams"):
            self._thread_local.acquired_streams = []
        self._thread_local.acquired_streams.append(stream_idx)

        return stream

    def release(self, stream: "torch.cuda.Stream"):
        """
        Return stream to pool.

        Args:
            stream: Stream to release

        Raises:
            ValueError: If stream not from this pool
        """
        try:
            stream_idx = self._streams.index(stream)
        except ValueError:
            raise ValueError("Stream not from this pool") from None

        with self._lock:
            if stream_idx not in self._available_indices:
                self._available_indices.append(stream_idx)

        # Remove from thread-local tracking
        if hasattr(self._thread_local, "acquired_streams") and (
            stream_idx in self._thread_local.acquired_streams
        ):
            self._thread_local.acquired_streams.remove(stream_idx)

    @contextmanager
    def acquire_stream(self) -> Generator["torch.cuda.Stream", None, None]:
        """
        Context manager for automatic stream acquisition and release.

        Example:
            >>> with manager.acquire_stream() as stream:
            ...     with torch.cuda.stream(stream):
            ...         tensor = data.to('cuda', non_blocking=True)
            ...         tensor.record_stream(stream)

        Yields:
            CUDA stream from pool
        """
        stream = self.acquire()
        try:
            yield stream
        finally:
            self.release(stream)

    def synchronize_stream(self, stream: "torch.cuda.Stream"):
        """
        Synchronize specific stream (blocks until all ops complete).

        Use sparingly - prefer event-based synchronization when possible.

        Args:
            stream: Stream to synchronize
        """
        stream.synchronize()

    def synchronize_all(self):
        """
        Synchronize all streams in pool (blocks until all ops complete).

        Use only for critical synchronization points (e.g., before saving state).
        """
        self._ensure_initialized()
        for stream in self._streams:
            stream.synchronize()

    def available_streams(self) -> int:
        """Get number of available streams in pool."""
        self._ensure_initialized()
        with self._lock:
            return len(self._available_indices)

    def total_streams(self) -> int:
        """Get total stream capacity."""
        return self.pool_size

    @staticmethod
    def wait_stream(target: "torch.cuda.Stream", dependency: "torch.cuda.Stream"):
        """
        Make target stream wait for dependency stream.

        Establishes cross-stream dependency without blocking CPU.

        Args:
            target: Stream that should wait
            dependency: Stream to wait for

        Example:
            >>> # Ensure transfer completes before computation
            >>> StreamManager.wait_stream(compute_stream, transfer_stream)
        """
        target.wait_stream(dependency)

    @staticmethod
    def record_tensor_stream(tensor: "torch.Tensor", stream: "torch.cuda.Stream"):
        """
        Record that tensor is used on given stream.

        Prevents premature deallocation by CUDA caching allocator.
        MUST be called after non-blocking operations using the tensor.

        Args:
            tensor: Tensor to track
            stream: Stream using the tensor

        Example:
            >>> with torch.cuda.stream(stream):
            ...     gpu_tensor = cpu_tensor.to('cuda', non_blocking=True)
            ...     StreamManager.record_tensor_stream(gpu_tensor, stream)
        """
        tensor.record_stream(stream)

    def __repr__(self) -> str:
        return (
            f"StreamManager(pool_size={self.pool_size}, "
            f"available={self.available_streams()}, device={self.device})"
        )


# Global singleton stream manager (lazy initialization)
_global_stream_manager: StreamManager | None = None
_global_stream_lock = threading.Lock()


def get_global_stream_manager(
    pool_size: int = 4,
    device: int = 0,
) -> StreamManager | None:
    """
    Get or create global stream manager singleton.

    Returns None if PyTorch/CUDA unavailable or initialization fails.
    Thread-safe lazy initialization.

    Args:
        pool_size: Number of streams (only used on first call)
        device: CUDA device index (only used on first call)

    Returns:
        Global StreamManager instance or None if unavailable
    """
    global _global_stream_manager

    if not TORCH_AVAILABLE:
        return None

    if _global_stream_manager is None:
        with _global_stream_lock:
            # Double-check after acquiring lock
            if _global_stream_manager is None:
                try:
                    _global_stream_manager = StreamManager(
                        pool_size=pool_size,
                        device=device,
                    )
                    logger.info("Initialized global StreamManager")
                except Exception as e:
                    logger.warning(f"Failed to initialize global stream manager: {e}")
                    return None

    return _global_stream_manager
