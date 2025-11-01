"""Hardware detection and config enrichment for runtime device/batch_size resolution."""

import logging
from typing import Literal

logger = logging.getLogger(__name__)


def detect_device() -> Literal["cpu", "cuda"]:
    """
    Detect available compute device.

    Returns:
        "cuda" if GPU available and CUDA installed, else "cpu"
    """
    try:
        import torch

        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            logger.info(f"GPU detected: {gpu_name}. Using CUDA acceleration.")
            return "cuda"
        else:
            logger.warning("No GPU detected. Using CPU (will be slow). Check CUDA/WSL2 setup.")
            return "cpu"
    except ImportError:
        logger.warning("PyTorch not available. Falling back to CPU.")
        return "cpu"


def detect_batch_size(device: str, model_name: str = "all-MiniLM-L6-v2") -> int:
    """
    Determine optimal batch size based on device and available memory.

    Args:
        device: Compute device ("cpu" or "cuda")
        model_name: Model name for memory estimation

    Returns:
        Batch size (64-512 for GPU based on VRAM, fixed 64 for CPU)
    """
    if device == "cuda":
        try:
            import torch

            if not torch.cuda.is_available():
                logger.warning("CUDA device specified but not available. Using CPU batch size.")
                return 64

            # GPU: Adaptive based on VRAM
            props = torch.cuda.get_device_properties(0)
            total_memory_gb = props.total_memory / 1e9

            # Scale batch size based on available memory
            model_overhead_gb = 3.0  # Model weights + buffers
            available_for_batch = max(2.0, total_memory_gb - model_overhead_gb)

            # Empirical scaling: all-MiniLM uses ~40MB per batch of 32
            # Increased cap to 1024 for high-end GPUs like RTX 4090
            batch_size = max(64, min(1024, int(available_for_batch * 50)))

            logger.info(
                f"Auto-scaled batch_size={batch_size} for GPU "
                f"({total_memory_gb:.1f}GB VRAM, model: {model_name})"
            )
            return batch_size

        except ImportError:
            logger.warning("PyTorch not available. Using CPU batch size.")
            return 64
    else:
        # CPU: Fixed conservative batch size
        logger.info("CPU mode: batch_size=64")
        return 64


def enrich_config_with_hardware(config: dict) -> dict:
    """
    Enrich config dict with concrete hardware values.

    Replaces None device/batch_size with detected values.
    Validates CUDA availability when device="cuda" is explicitly set.

    Args:
        config: Configuration dict from load_config()

    Returns:
        Updated config dict with concrete device/batch_size

    Raises:
        RuntimeError: If CUDA requested but not available
    """
    model_config = config.get("model", {})
    gpu_config = config.get("gpu", {})
    device = model_config.get("device")
    batch_size = model_config.get("batch_size")
    model_name = model_config.get("name", "all-MiniLM-L6-v2")

    # Device resolution
    if device is None:
        device = detect_device()
    elif device == "cuda":
        # Validate explicitly requested CUDA is available
        try:
            import torch

            if not torch.cuda.is_available():
                logger.error("CUDA requested but not available. Falling back to CPU.")
                device = "cpu"
        except ImportError:
            logger.error("CUDA requested but PyTorch not available. Falling back to CPU.")
            device = "cpu"

    # Batch size resolution
    if batch_size is None:
        batch_size = detect_batch_size(device, model_name)

    # Update config with enriched values
    enriched_config = config.copy()
    enriched_config["model"] = model_config.copy()
    enriched_config["model"]["device"] = device
    enriched_config["model"]["batch_size"] = batch_size
    enriched_config["gpu"] = gpu_config.copy()

    return enriched_config
