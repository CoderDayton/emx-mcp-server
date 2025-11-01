"""
Embedding encoder for converting tokens to vector representations.
Uses sentence-transformers for efficient, high-quality embeddings.
"""

import numpy as np
from typing import Any, Callable, List, Optional, Tuple, Union, TYPE_CHECKING
import logging

from emx_mcp.gpu.pinned_memory import get_global_pool, TORCH_AVAILABLE

if TYPE_CHECKING:
    import torch

logger = logging.getLogger(__name__)

# Lazy-loaded metrics instruments
_instruments = None


def _get_instruments():
    """Lazy-load metrics instruments to avoid circular imports."""
    global _instruments
    if _instruments is None:
        try:
            from emx_mcp.metrics.instruments import get_instruments

            _instruments = get_instruments()
        except (ImportError, RuntimeError):
            # Metrics not initialized or unavailable
            _instruments = False
    return _instruments if _instruments is not False else None


class EmbeddingEncoder:
    """
    Encodes token sequences into dense vector embeddings.

    Uses sentence-transformers models for semantic encoding.
    Default: all-MiniLM-L6-v2 (384-dim, fast, good quality)

    OPTIMIZATIONS:
    - Automatic GPU detection for WSL2/CUDA environments
    - Adaptive batch size: GPU 64-512 based on VRAM, CPU fixed at 32
    - Stream-based async execution with proper context handling
    - Efficient tensor transfers with pinned memory
    """

    def __init__(
        self,
        model_name: str,
        device: Optional[str] = None,
        batch_size: Optional[int] = None,
        gpu_config: Optional[dict] = None,
        enable_cuda_graphs: bool = False,
    ):
        """
        Initialize embedding encoder with GPU optimization.

        Args:
            model_name: HuggingFace model identifier
            device: "cpu" or "cuda" (must be enriched by hardware.py, not None at runtime)
            batch_size: Batch size for encoding (must be enriched by hardware.py, not None at runtime)
            gpu_config: Optional GPU configuration dict with pinned memory settings
            enable_cuda_graphs: Enable CUDA graph capture for inference speedup (requires PyTorch 2.0+)

        Raises:
            AssertionError: If device or batch_size is None (config not enriched)
        """
        # Runtime assertions: config must be enriched before passing to encoder
        assert device is not None, (
            "device must not be None. Config should be enriched via "
            "enrich_config_with_hardware() before initializing encoder."
        )
        assert batch_size is not None, (
            "batch_size must not be None. Config should be enriched via "
            "enrich_config_with_hardware() before initializing encoder."
        )

        try:
            import torch
            from sentence_transformers import SentenceTransformer

            # Validate CUDA availability if requested
            if device == "cuda" and not torch.cuda.is_available():
                raise RuntimeError(
                    "CUDA device specified but not available. "
                    "Hardware detection should have caught this."
                )

            self.device = device
            self.torch_available = True
            self.torch = torch

            # Load model
            self.model = SentenceTransformer(model_name, device=device)
            self.dimension = self.model.get_sentence_embedding_dimension()
            self.model_name = model_name
            self.batch_size = batch_size

            logger.info(f"Using provided batch_size={batch_size} for {device.upper()}")

            # Store GPU config for pinned memory decisions
            # WSL2-safe: pinned memory disabled by default
            self.gpu_config = gpu_config or {
                "enable_pinned_memory": False,  # Disabled for WSL2 stability
                "pinned_buffer_size": 4,
                "pinned_max_batch": 256,
                "pinned_min_batch_threshold": 64,
            }

            # WSL2-safe pinned memory auto-detection
            if device == "cuda" and self.gpu_config.get("enable_pinned_memory", False):
                try:
                    # Test if pinned memory works on this system
                    # Create pinned tensor on CPU, then transfer to GPU
                    test_tensor = torch.zeros(10, 384, pin_memory=True)
                    test_tensor_gpu = test_tensor.to(device)
                    del test_tensor, test_tensor_gpu
                    logger.debug("Pinned memory available on this system")
                except RuntimeError as e:
                    logger.warning(f"Pinned memory not available (WSL2 issue?): {e}")
                    self.gpu_config["enable_pinned_memory"] = False

            # CUDA graph support (PyTorch 2.0+)
            self.enable_cuda_graphs = enable_cuda_graphs and device == "cuda"
            if self.enable_cuda_graphs and torch.__version__ >= "2.0.0":
                logger.info("CUDA graphs enabled for inference speedup")

            logger.info(
                f"EmbeddingEncoder initialized: {model_name} (dim={self.dimension}, "
                f"device={device}, batch_size={self.batch_size})"
            )

        except ImportError as e:
            logger.error(
                "sentence-transformers not installed. Run: pip install sentence-transformers"
            )
            raise ImportError(
                "sentence-transformers required for embedding generation. "
                "Install with: pip install sentence-transformers"
            ) from e

    def encode_tokens(self, tokens: List[str]) -> np.ndarray:
        """
        Encode token sequence into single embedding.

        Args:
            tokens: List of token strings

        Returns:
            Embedding vector of shape (dimension,)
        """
        if not tokens:
            raise ValueError("Token list cannot be empty")

        text = " ".join(tokens)
        embedding = self.model.encode(
            text,
            batch_size=self.batch_size,
            convert_to_numpy=True,
            show_progress_bar=False,
            device=self.device,
        )
        return embedding.astype(np.float32)

    def encode_batch(
        self,
        token_lists: List[List[str]],
        use_pinned_memory: bool = False,
        stream: Optional["torch.cuda.Stream"] = None,
    ) -> Union[np.ndarray, Tuple[Any, Callable[[], None]]]:
        """
        Batch encode multiple token sequences with GPU optimization.

        Args:
            token_lists: List of token lists
            use_pinned_memory: If True, attempt to use pinned memory buffers
            stream: Optional CUDA stream for async execution

        Returns:
            If pinned memory used: Tuple of (pinned_tensor, release_callback)
            Otherwise: Embeddings array of shape (n_sequences, dimension)
        """
        if not token_lists:
            raise ValueError("Token lists cannot be empty")

        texts = [" ".join(tokens) for tokens in token_lists]
        batch_size = len(texts)

        # Track embedding generation metrics
        instruments = _get_instruments()
        ctx = (
            instruments.track_embedding(batch_size, self.device)
            if instruments
            else None
        )

        try:
            if ctx:
                ctx.__enter__()

            # Fixed stream context to properly wrap entire encoding operation
            if stream is not None and self.torch_available and self.device == "cuda":
                import torch

                with torch.cuda.stream(stream):
                    embeddings = self.model.encode(
                        texts,
                        batch_size=self.batch_size,
                        convert_to_numpy=False,
                        show_progress_bar=False,
                        device=self.device,
                    )
                    # model.encode with convert_to_numpy=False returns list of tensors
                    if isinstance(embeddings, list):
                        embeddings = torch.stack(embeddings)

                    if embeddings.dtype != torch.float32:
                        embeddings = embeddings.to(torch.float32)
                    embeddings = embeddings.cpu().numpy().astype(np.float32)
            else:
                embeddings = self.model.encode(
                    texts,
                    batch_size=self.batch_size,
                    convert_to_numpy=True,
                    show_progress_bar=False,
                    device=self.device,
                )
                embeddings = embeddings.astype(np.float32)

            # Check if we should use pinned memory
            should_pin = (
                use_pinned_memory
                and self.torch_available
                and self.device == "cuda"
                and self.gpu_config.get("enable_pinned_memory", False)
                and batch_size >= self.gpu_config.get("pinned_min_batch_threshold", 64)
            )

            if not should_pin:
                if ctx:
                    ctx.__exit__(None, None, None)
                return embeddings

            # Efficient pinned memory transfer
            try:
                dimension = int(self.dimension) if self.dimension else 384
                pool = get_global_pool(
                    dimension=dimension,
                    max_batch_size=self.gpu_config.get("pinned_max_batch", 256),
                    buffer_size=self.gpu_config.get("pinned_buffer_size", 4),
                )

                if pool is None:
                    logger.debug("Pinned memory pool unavailable, using regular numpy")
                    if ctx:
                        ctx.__exit__(None, None, None)
                    return embeddings

                buffer, release_callback = pool.acquire(batch_size)
                buffer[:batch_size] = embeddings

                if stream is not None:
                    try:
                        from emx_mcp.gpu.stream_manager import StreamManager

                        StreamManager.record_tensor_stream(buffer[:batch_size], stream)
                    except ImportError:
                        logger.debug(
                            "StreamManager not available, skipping stream recording"
                        )

                logger.debug(
                    f"Using pinned memory for batch_size={batch_size} "
                    f"(threshold={self.gpu_config.get('pinned_min_batch_threshold', 64)})"
                )

                if ctx:
                    ctx.__exit__(None, None, None)
                return buffer[:batch_size], release_callback

            except Exception as e:
                logger.warning(
                    f"Failed to acquire pinned memory (batch_size={batch_size}): {e}. "
                    f"Falling back to regular numpy array"
                )
                if ctx:
                    ctx.__exit__(None, None, None)
                return embeddings

        except Exception as e:
            if ctx:
                ctx.__exit__(type(e), e, e.__traceback__)
            raise

    def encode_tokens_with_context(
        self, tokens: List[str], context_window: int = 10
    ) -> np.ndarray:
        """
        Encode tokens individually with local context for surprise calculation.

        PERFORMANCE: O(n) - Single pass encoding all tokens in one batch.

        Args:
            tokens: List of token strings
            context_window: Number of previous tokens to include as context

        Returns:
            Array of embeddings of shape (n_tokens, embedding_dim)
        """
        if not tokens:
            raise ValueError("Token list cannot be empty")

        n_tokens = len(tokens)
        logger.info(
            f"Building context strings for {n_tokens} tokens (window={context_window})"
        )

        # Build all context strings upfront - O(n) operation
        context_texts = []
        for i in range(n_tokens):
            start = max(0, i - context_window)
            context_tokens = tokens[start : i + 1]
            context_texts.append(" ".join(context_tokens))

        # CRITICAL: Encode ALL contexts in ONE batch call - O(n) not O(n*batch_size)
        # sentence-transformers handles internal batching automatically
        logger.info(f"Encoding {n_tokens} contexts in single batch operation...")

        embeddings = self.model.encode(
            context_texts,
            batch_size=self.batch_size,  # Internal batching, not repeated calls
            convert_to_numpy=True,
            show_progress_bar=False,
            device=self.device,
        )

        logger.info(f"Batch encoding complete: {embeddings.shape}")
        return embeddings.astype(np.float32)

    def get_query_embedding(self, query: str) -> np.ndarray:
        """
        Encode query string for retrieval.

        Args:
            query: Query text

        Returns:
            Query embedding of shape (dimension,)
        """
        if not query:
            raise ValueError("Query cannot be empty")

        embedding = self.model.encode(
            query,
            batch_size=self.batch_size,
            convert_to_numpy=True,
            show_progress_bar=False,
            device=self.device,
        )
        return embedding.astype(np.float32)

    def warmup_gpu_cache(self) -> None:
        """
        Pre-allocate GPU memory cache for faster first inference.
        """
        if self.device != "cuda":
            logger.debug("Warmup skipped (not using CUDA)")
            return

        try:
            logger.info("Warming up GPU cache...")
            dummy_texts = ["sample text for warmup"] * min(8, self.batch_size)
            _ = self.model.encode(
                dummy_texts,
                batch_size=self.batch_size,
                convert_to_numpy=True,
                show_progress_bar=False,
            )
            logger.info("GPU cache warmup complete")
        except Exception as e:
            logger.warning(f"GPU cache warmup failed: {e}")

    def get_device_info(self) -> dict:
        """
        Get detailed device and performance information.

        Returns:
            Dictionary with GPU/CPU specs and configuration
        """
        info = {
            "device": self.device,
            "batch_size": self.batch_size,
            "model": self.model_name,
            "embedding_dim": self.dimension,
        }

        if self.device == "cuda" and self.torch_available:
            props = self.torch.cuda.get_device_properties(0)
            info.update(
                {
                    "gpu_name": props.name,
                    "gpu_memory_gb": props.total_memory / 1e9,
                    "cuda_capability": f"{props.major}.{props.minor}",
                    "pinned_memory_enabled": self.gpu_config.get(
                        "enable_pinned_memory", False
                    ),
                }
            )

        return info
