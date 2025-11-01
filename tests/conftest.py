"""
Pytest configuration and shared fixtures for EMX-MCP-Server tests.
"""

import shutil
import tempfile
from pathlib import Path

import numpy as np
import pytest


@pytest.fixture(scope="session")
def temp_dir():
    """Create a temporary directory for test artifacts."""
    temp_path = tempfile.mkdtemp(prefix="emx_mcp_test_")
    yield Path(temp_path)
    shutil.rmtree(temp_path, ignore_errors=True)


@pytest.fixture
def sample_tokens():
    """Provide realistic test token sequences."""
    return {
        "coding_session": [
            "I need to create a user authentication system",
            "First I'll design the database schema",
            "Users table with id email password_hash",
            "Login endpoint POST /api/auth/login",
            "Registration endpoint POST /api/auth/register",
            "JWT tokens for sessions",
            # Phase shift - implementation
            "Now let me start coding the User model",
            "class User extends Model {}",
            "add required fields to constructor",
            "implement hashPassword method",
            "create validatePassword static method",
            "add to database migrations",
            # Phase shift - debugging
            "The login isn't working properly",
            "Checking error logs first",
            "Token expiration seems wrong",
            "Need to increase expiration time",
            "Also need better error handling",
            "Let me add try catch blocks",
        ],
        "meeting_transcript": [
            # Status update
            "Good morning everyone",
            "Last week's sprint was successful",
            "We completed user authentication",
            "Payment integration is working",
            "Mobile app performance improved",
            "User feedback has been positive",
            # Planning
            "Next sprint goals",
            "Focus on API documentation",
            "Add automated testing coverage",
            "Improve CI/CD pipeline",
            "Set up monitoring dashboard",
            "Plan user onboarding flow",
            # Issues
            "However we have some challenges",
            "Database queries are slow",
            "Memory usage increasing over time",
            "Need to optimize caching",
        ],
        "research_presentation": [
            # Background
            "Machine learning models require large datasets",
            "Data preprocessing is crucial step",
            "Feature engineering affects model performance",
            "Traditional approaches use handcrafted features",
            "Deep learning automates feature extraction",
            "Neural networks learn hierarchical representations",
            # Methodology
            "Our approach uses attention mechanisms",
            "Self-attention captures long-range dependencies",
            "Multi-head attention provides diverse perspectives",
            "Transformer architecture enables parallel processing",
            "We train on domain-specific dataset",
            "Preprocessing includes normalization and tokenization",
            # Results
            "Results show significant improvement",
            "Accuracy increased by 15%",
            "Training time reduced by 30%",
            "Model generalizes better to unseen data",
        ],
        "short_sequence": [
            "This is a test sequence",
            "with some tokens",
            "for validation",
            "and checking functionality",
            "of the embedding approach",
        ],
    }


@pytest.fixture
def embedding_config():
    """Configuration for embedding models."""
    return {
        "model_name": "all-MiniLM-L6-v2",
        "device": "cpu",
        "batch_size": 32,
        "dimension": 384,  # Expected dimension for all-MiniLM-L6-v2
    }


@pytest.fixture
def hardware_enriched_config():
    """
    Embedding config with hardware detection applied.

    Returns concrete device/batch_size values via hardware.py detection.
    Use this fixture for tests that instantiate EmbeddingEncoder directly
    without going through ProjectMemoryManager.
    """
    from emx_mcp.utils.hardware import detect_batch_size, detect_device

    device = detect_device()
    batch_size = detect_batch_size(device)

    return {
        "model_name": "all-MiniLM-L6-v2",
        "device": device,
        "batch_size": batch_size,
        "dimension": 384,
    }


@pytest.fixture
def segmentation_config():
    """Configuration for segmentation."""
    return {
        "gamma": 1.0,
        "window_offset": 128,
        "refinement_metric": "modularity",
        "context_window": 5,
    }


@pytest.fixture
def expected_boundaries():
    """Expected boundary patterns for test sequences."""
    return {
        "coding_session": 3,  # planning -> implementation -> debugging
        "meeting_transcript": 3,  # status -> planning -> issues
        "research_presentation": 3,  # background -> methodology -> results
        "short_sequence": 2,  # simple split somewhere in middle
    }


@pytest.fixture
def sample_embeddings():
    """Generate sample embeddings for testing."""

    def _generate_embeddings(n_tokens: int, dimension: int = 384, seed: int = 42):
        np.random.seed(seed)
        # Create embeddings with some structure
        embeddings = np.random.randn(n_tokens, dimension).astype(np.float32)
        # Normalize to unit vectors (sentence-transformers does this)
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        embeddings = embeddings / (norms + 1e-8)
        return embeddings

    return _generate_embeddings


@pytest.fixture
def mock_project_config(temp_dir):
    """Mock configuration for project manager testing."""
    project_path = temp_dir / "test_project"
    global_path = temp_dir / "global_memories"

    # Create directories
    project_path.mkdir(parents=True, exist_ok=True)
    global_path.mkdir(parents=True, exist_ok=True)

    config = {
        "model": {"name": "all-MiniLM-L6-v2", "device": "cpu", "batch_size": 32},
        "storage": {"index_type": "IVF", "metric": "cosine", "vector_dim": 384},
        "memory": {
            "gamma": 1.0,
            "window_offset": 128,
            "refinement_metric": "modularity",
            "ivf_nlist": 100,
            "ivf_nprobe": 10,
            "n_init": 1000,
            "n_local": 500,
        },
    }

    return {
        "project_path": str(project_path),
        "global_path": str(global_path),
        "config": config,
    }


@pytest.fixture(scope="session")
def test_data_cache():
    """Cache for expensive test data generation."""
    cache = {}

    def get_or_compute(key, compute_func):
        if key not in cache:
            cache[key] = compute_func()
        return cache[key]

    return get_or_compute
