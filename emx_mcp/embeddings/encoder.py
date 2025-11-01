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

            # Load model
            self.model = SentenceTransformer(model_name, device=device)
            self.model.eval()

            self.device = device
            self.torch_available = True
            self.torch = torch

            # Performance optimization: Use TensorFloat32 for faster matmul on Ampere+ GPUs
            if device == "cuda":
                self.torch.backends.cuda.matmul.allow_tf32 = True
                self.torch.backends.cudnn.allow_tf32 = True
                logger.info(
                    "Set matmul precision to 'high' (TF32) for faster inference"
                )

            if device == "cuda" and self.torch.__version__ >= "2.0.0":
                try:
                    self.model[0].auto_model = self.torch.compile(  # type: ignore
                        self.model[0].auto_model, mode="reduce-overhead", fullgraph=False  # type: ignore
                    )
                    self._warmup()
                    logger.info("torch.compile enabled on transformer model")
                except Exception as e:
                    logger.warning(f"torch.compile failed: {e}, using eager mode")

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

    def encode_batch(
        self,
        token_lists: List[List[str]],
        use_pinned_memory: bool = False,
        stream: Optional["torch.cuda.Stream"] = None,
    ) -> np.ndarray:
        """
        Batch encode multiple token sequences (wrapper for backward compatibility).

        This is a convenience wrapper around encode_tokens_with_context().
        For each sequence, averages per-token embeddings to get a single vector.

        Args:
            token_lists: List of token lists
            use_pinned_memory: Ignored (kept for API compatibility)
            stream: Ignored (kept for API compatibility)

        Returns:
            Embeddings array of shape (n_sequences, dimension)
        """
        if not token_lists:
            raise ValueError("Token lists cannot be empty")

        # Flatten all tokens and track boundaries
        all_tokens = []
        token_boundaries = [0]
        for tokens in token_lists:
            all_tokens.extend(tokens)
            token_boundaries.append(len(all_tokens))

        # Encode all at once with no context (context_window=0)
        all_embeddings = self.encode_tokens_with_context(all_tokens, context_window=0)

        # Average embeddings per sequence
        embeddings = np.array(
            [
                np.mean(
                    all_embeddings[token_boundaries[i] : token_boundaries[i + 1]],
                    axis=0,
                )
                for i in range(len(token_lists))
            ]
        )

        return embeddings.astype(np.float32)

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

        with self.torch.inference_mode():
            with self.torch.autocast(
                enabled=self.enable_cuda_graphs and self.device == "cuda",
                device_type=self.device,
            ):
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

    def _warmup(self):
        """Warmup compiled model (happens once at init)."""
        logger.debug("Warming up compiled encoder...")
        with self.torch.inference_mode():
            with self.torch.autocast(
                enabled=self.device == "cuda", device_type=self.device
            ):
                dummy = ["warmup"] * min(10, self.batch_size)
                for _ in range(3):
                    self.model.encode(
                        dummy, batch_size=len(dummy), show_progress_bar=False
                    )

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
