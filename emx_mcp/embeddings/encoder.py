"""
Embedding encoder for converting tokens to vector representations.
Uses sentence-transformers for efficient, high-quality embeddings.

PINNED MEMORY OPTIMIZATION:
Based on MemAscend research (arXiv:2505.23254), pinned (page-locked) memory
enables high-bandwidth GPU transfers via PCIe DMA without staging copies:
- Normal path: pageable DRAM → driver staging → GPU (2 copies)
- Pinned path: pinned DRAM → GPU (1 copy, ~2x faster)

Automatically disabled on WSL2 due to driver limitations:
- https://github.com/microsoft/WSL/issues/8447
- https://forums.developer.nvidia.com/t/what-are-the-pinned-memory-limitations-on-cuda-for-wsl2/255472

Enable via: gpu_config={'enable_pinned_memory': True, 'pinned_min_batch_threshold': 64}
"""

import logging
from typing import TYPE_CHECKING, Optional

import numpy as np

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
    - Pinned memory for faster CPU→GPU transfers (disabled on WSL2)
    """

    def __init__(
        self,
        model_name: str,
        device: str | None = None,
        batch_size: int | None = None,
        gpu_config: dict | None = None,
        enable_cuda_graphs: bool = False,
    ):  # sourcery skip: swap-if-else-branches, use-named-expression
        """
        Initialize embedding encoder with GPU optimization.

        Args:
            model_name: HuggingFace model identifier
            device: "cpu" or "cuda" (must be enriched by hardware.py, not None at runtime)
            batch_size: Batch size for encoding (must be enriched by
                hardware.py, not None at runtime)
            gpu_config: Optional GPU configuration dict with pinned memory
                settings
            enable_cuda_graphs: Enable CUDA graph capture for inference
                speedup (requires PyTorch 2.0+)

        Raises:
            ValueError: If device or batch_size is None (config not enriched)
        """
        # Runtime validation: config must be enriched before passing to encoder
        if device is None:
            raise ValueError(
                "device must not be None. Config should be enriched via "
                "enrich_config_with_hardware() before initializing encoder."
            )
        if batch_size is None:
            raise ValueError(
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

            # Set batch_size and model info early (needed by warmup)
            self.batch_size = batch_size
            self.dimension = self.model.get_sentence_embedding_dimension()
            self.model_name = model_name

            # Performance optimization: Use TensorFloat32 for faster matmul on Ampere+ GPUs
            if device == "cuda":
                self.torch.backends.cuda.matmul.allow_tf32 = True
                self.torch.backends.cudnn.allow_tf32 = True
                logger.info("Set matmul precision to 'high' (TF32) for faster inference")

            if device == "cuda" and self.torch.__version__ >= "2.0.0":
                try:
                    self.model[0].auto_model = self.torch.compile(  # type: ignore[attr-defined]
                        self.model[0].auto_model,  # type: ignore
                        mode="reduce-overhead",
                        fullgraph=False,
                    )
                    self._warmup()
                    logger.info("torch.compile enabled on transformer model")
                except PermissionError as e:
                    logger.warning(
                        f"torch.compile failed (nvcc permission denied): {e}, using eager mode"
                    )
                except FileNotFoundError as e:
                    logger.warning(f"torch.compile failed (nvcc not found): {e}, using eager mode")
                except Exception as e:
                    logger.warning(f"torch.compile failed: {e}, using eager mode")

            logger.info(f"Using provided batch_size={batch_size} for {device.upper()}")

            # Detect WSL2 environment for pinned memory safety
            is_wsl2 = self._detect_wsl2()
            if is_wsl2:
                logger.info("WSL2 environment detected - pinned memory disabled by default")

            # Store GPU config for pinned memory decisions
            # WSL2-safe: pinned memory disabled by default due to driver limitations
            # See: https://github.com/microsoft/WSL/issues/8447
            # See: https://forums.developer.nvidia.com/t/what-are-the-pinned-memory-limitations-on-cuda-for-wsl2/255472
            default_pinned_enabled = (
                False if is_wsl2 else (gpu_config or {}).get("enable_pinned_memory", False)
            )

            self.gpu_config = gpu_config or {}
            self.gpu_config.setdefault("enable_pinned_memory", default_pinned_enabled)
            self.gpu_config.setdefault("pinned_buffer_size", 4)
            self.gpu_config.setdefault("pinned_max_batch", 256)
            self.gpu_config.setdefault("pinned_min_batch_threshold", 64)

            # Test pinned memory availability if enabled
            if device == "cuda" and self.gpu_config.get("enable_pinned_memory", False):
                pinned_available = self._test_pinned_memory()
                if not pinned_available:
                    logger.warning(
                        "Pinned memory test failed - disabling pinned memory optimization. "
                        "This is expected on WSL2 environments."
                    )
                    self.gpu_config["enable_pinned_memory"] = False
                else:
                    logger.info(
                        f"Pinned memory available: min_batch_threshold="
                        f"{self.gpu_config['pinned_min_batch_threshold']}"
                    )

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
        token_lists: list[list[str]],
        use_pinned_memory: bool = False,
        stream: Optional["torch.cuda.Stream"] = None,
    ) -> np.ndarray:
        """
        Batch encode multiple token sequences with optional pinned memory optimization.

        This method supports pinned memory for faster CPU-GPU transfers when:
        1. CUDA is available
        2. Pinned memory is enabled in config (disabled by default on WSL2)
        3. Batch size meets the threshold (>= pinned_min_batch_threshold)

        For each sequence, averages per-token embeddings to get a single vector.

        Args:
            token_lists: List of token lists
            use_pinned_memory: Enable pinned memory for GPU transfers (requires CUDA)
            stream: Optional CUDA stream for async execution

        Returns:
            Embeddings array of shape (n_sequences, dimension)

        Note:
            Pinned memory optimization is automatically disabled on WSL2 due to
            driver limitations. See: https://github.com/microsoft/WSL/issues/8447
        """
        if not token_lists:
            raise ValueError("Token lists cannot be empty")

        # Flatten all tokens and track boundaries
        all_tokens: list[str] = []
        token_boundaries: list[int] = [0]
        for tokens in token_lists:
            all_tokens.extend(tokens)
            token_boundaries.append(len(all_tokens))

        # Check if we should use pinned memory optimization
        can_use_pinned = (
            use_pinned_memory
            and self.device == "cuda"
            and self.gpu_config.get("enable_pinned_memory", False)
            and len(token_lists) >= self.gpu_config.get("pinned_min_batch_threshold", 64)
        )

        if can_use_pinned:
            logger.debug(f"Using pinned memory for batch of {len(token_lists)} sequences")

        # Encode all at once with no context (context_window=0)
        # Execute within stream context if provided for GPU parallelism
        if stream is not None and self.device == "cuda":
            with self.torch.cuda.stream(stream):
                all_embeddings = self.encode_tokens_with_context(all_tokens, context_window=0)
        else:
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

        result = embeddings.astype(np.float32)

        # If pinned memory requested and available, convert to pinned tensor
        if can_use_pinned and self.torch_available:
            try:
                # Create pinned tensor for faster CPU->GPU transfers
                pinned_tensor = self.torch.from_numpy(result).pin_memory()

                # If stream provided, use non_blocking transfer with record_stream
                if stream is not None:
                    gpu_tensor = pinned_tensor.to(self.device, non_blocking=True)
                    # Record stream to prevent premature memory deallocation
                    gpu_tensor.record_stream(stream)
                    # Synchronize stream before CPU access
                    stream.synchronize()
                    return gpu_tensor.cpu().numpy()
                else:
                    # Synchronous transfer
                    gpu_tensor = pinned_tensor.to(self.device)
                    return gpu_tensor.cpu().numpy()

            except RuntimeError as e:
                logger.warning(
                    f"Pinned memory allocation failed (WSL2 issue?): {e}. "
                    "Falling back to regular memory."
                )
                # Disable pinned memory for future calls
                self.gpu_config["enable_pinned_memory"] = False

        return result

    def encode_tokens_with_context(self, tokens: list[str], context_window: int = 10) -> np.ndarray:
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
        logger.info(f"Building context strings for {n_tokens} tokens (window={context_window})")

        # Build all context strings upfront - O(n) operation
        context_texts = []
        for i in range(n_tokens):
            start = max(0, i - context_window)
            context_tokens = tokens[start : i + 1]
            context_texts.append(" ".join(context_tokens))

        # CRITICAL: Encode ALL contexts in ONE batch call - O(n) not O(n*batch_size)
        # sentence-transformers handles internal batching automatically
        logger.info(f"Encoding {n_tokens} contexts in single batch operation...")

        with (
            self.torch.inference_mode(),
            self.torch.autocast(
                enabled=self.enable_cuda_graphs and self.device == "cuda",
                device_type=self.device,
            ),
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

        with (
            self.torch.inference_mode(),
            self.torch.autocast(
                enabled=self.enable_cuda_graphs and self.device == "cuda",
                device_type=self.device,
            ),
        ):
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
        with (
            self.torch.inference_mode(),
            self.torch.autocast(enabled=self.device == "cuda", device_type=self.device),
        ):
            dummy = ["warmup"] * min(10, self.batch_size)
            for _ in range(3):
                self.model.encode(
                    dummy,
                    batch_size=len(dummy),
                    show_progress_bar=False,
                    convert_to_numpy=False,
                )
        logger.debug("Warmup complete.")

    def _detect_wsl2(self) -> bool:
        """
        Detect if running in WSL2 environment.

        WSL2 has known issues with CUDA pinned memory allocation.
        See: https://github.com/microsoft/WSL/issues/8447

        Returns:
            True if WSL2 detected, False otherwise
        """
        try:
            with open("/proc/version") as f:
                version_str = f.read().lower()
                # WSL2 kernel version contains "microsoft" and "wsl"
                return "microsoft" in version_str and "wsl" in version_str
        except (FileNotFoundError, PermissionError):
            # Not a Linux system or cannot read /proc/version
            return False

    def _test_pinned_memory(self) -> bool:
        """
        Test if pinned memory allocation works on this system.

        Pinned memory enables faster CPU->GPU transfers via DMA, but
        has limitations on WSL2 and some systems (~2GB per allocation).

        Based on research from MemAscend (arXiv:2505.23254):
        - Pinned memory enables PCIe DMA without staging copies
        - Large allocations (>2GB) may fail on some systems
        - WSL2 has driver-level limitations

        Returns:
            True if pinned memory is functional, False otherwise
        """
        if not self.torch_available or self.device != "cuda":
            return False

        try:
            # Test small allocation (10MB)
            test_size_mb = 10
            test_tensor = self.torch.zeros(
                test_size_mb * 1024 * 256,  # 10MB in float32
                dtype=self.torch.float32,
                pin_memory=True,
            )

            # Test GPU transfer
            gpu_tensor = test_tensor.to(self.device)

            # Cleanup
            del test_tensor, gpu_tensor
            self.torch.cuda.empty_cache()

            logger.debug("Pinned memory test passed")
            return True

        except (RuntimeError, AssertionError) as e:
            logger.debug(f"Pinned memory test failed: {e}")
            return False

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
            info |= {
                "gpu_name": props.name,
                "gpu_memory_gb": props.total_memory / 1e9,
                "cuda_capability": f"{props.major}.{props.minor}",
                "pinned_memory_enabled": self.gpu_config.get("enable_pinned_memory", False),
            }

        return info
