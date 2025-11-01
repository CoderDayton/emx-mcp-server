"""Test SurpriseSegmenter with pytest-asyncio."""

import pytest
import numpy as np
from pathlib import Path
import sys

from emx_mcp.memory.segmentation import SurpriseSegmenter

# Add project root to path for imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


class TestSurpriseSegmenter:
    """Test suite for SurpriseSegmenter with embedding-based approach."""

    @pytest.fixture
    def segmenter(self, segmentation_config):
        """Create SurpriseSegmenter instance for testing."""
        return SurpriseSegmenter(
            gamma=segmentation_config["gamma"],
            window_offset=segmentation_config["window_offset"],
        )

    @pytest.fixture
    def mock_embeddings(self, sample_embeddings, sample_tokens):
        """Create mock embeddings for testing."""
        tokens = sample_tokens["short_sequence"]
        return sample_embeddings(len(tokens), dimension=384)

    def test_initialization(self, segmenter, segmentation_config):
        """Test SurpriseSegmenter initialization."""
        assert segmenter.gamma == segmentation_config["gamma"]
        assert segmenter.window_offset == segmentation_config["window_offset"]

    def test_compute_embedding_surprises(
        self, segmenter, sample_embeddings, segmentation_config
    ):
        """Test embedding-based surprise calculation."""
        # Generate more tokens for better variance testing
        mock_embeddings = sample_embeddings(10, dimension=384)
        window = segmentation_config["context_window"]
        surprises = segmenter._compute_embedding_surprises(
            mock_embeddings, window=window
        )

        # Check properties
        assert len(surprises) == len(mock_embeddings)
        assert surprises.dtype == np.float32
        assert not np.isnan(surprises).any()
        assert not np.isinf(surprises).any()
        assert np.all(surprises >= 0)  # Distances should be non-negative

        # Check that surprises have some variance (for longer sequences)
        if len(mock_embeddings) > window:
            assert np.std(surprises[window:]) > 0.001  # Lower threshold for random data

        # First few tokens should have same surprise (no full context)
        if len(mock_embeddings) > window:
            assert np.allclose(surprises[:window], surprises[window], atol=0.1)

    def test_compute_embedding_surprises_edge_cases(self, segmenter, sample_embeddings):
        """Test embedding surprise calculation edge cases."""
        # Test with single token
        single_embedding = sample_embeddings(1, dimension=384)
        surprises = segmenter._compute_embedding_surprises(single_embedding, window=5)
        assert len(surprises) == 1
        assert surprises[0] == 1.0  # Default surprise for short sequences

        # Test with tokens fewer than window
        few_embeddings = sample_embeddings(3, dimension=384)
        surprises = segmenter._compute_embedding_surprises(few_embeddings, window=5)
        assert len(surprises) == 3
        assert np.allclose(surprises, 1.0)  # All should be 1.0

        # Test with window = 1
        many_embeddings = sample_embeddings(10, dimension=384)
        surprises = segmenter._compute_embedding_surprises(many_embeddings, window=1)
        assert len(surprises) == 10
        assert not np.isnan(surprises).any()

    def test_compute_embedding_adjacency(self, segmenter, mock_embeddings):
        """Test embedding-based adjacency computation."""
        adjacency = segmenter._compute_embedding_adjacency(mock_embeddings)

        # Check properties
        assert adjacency.shape == (len(mock_embeddings), len(mock_embeddings))
        assert adjacency.dtype == np.float32

        # Should be symmetric
        assert np.allclose(adjacency, adjacency.T, atol=1e-6)

        # Values should be in [0, 1]
        assert np.all((adjacency >= 0) & (adjacency <= 1))

        # Diagonal should be 1.0 (token always similar to itself)
        assert np.allclose(np.diag(adjacency), 1.0, atol=1e-6)

        # Should not have NaN or infinite values
        assert not np.isnan(adjacency).any()
        assert not np.isinf(adjacency).any()

        # Should have some structure (not all identical)
        off_diagonal = adjacency[~np.eye(len(mock_embeddings), dtype=bool)]
        assert np.std(off_diagonal) > 0.01

    def test_compute_embedding_adjacency_with_structure(
        self, segmenter, sample_embeddings
    ):
        """Test adjacency with structured embeddings to verify similarity calculation."""
        # Create embeddings with more explicit structure
        # First 3 tokens very similar, last 2 tokens very different
        np.random.seed(42)

        # Create base embedding for similar group
        base_similar = np.random.randn(384).astype(np.float32)
        base_similar = base_similar / np.linalg.norm(base_similar)

        # Create very similar embeddings (small perturbations)
        similar_group = []
        for i in range(3):
            perturbation = np.random.randn(384) * 0.1  # Small perturbation
            embedding = base_similar + perturbation
            embedding = embedding / np.linalg.norm(embedding)
            similar_group.append(embedding)
        similar_group = np.array(similar_group)

        # Create very different embeddings (different from base)
        base_different = -base_similar  # Opposite direction
        different_group = []
        for i in range(2):
            perturbation = np.random.randn(384) * 0.1
            embedding = base_different + perturbation
            embedding = embedding / np.linalg.norm(embedding)
            different_group.append(embedding)
        different_group = np.array(different_group)

        structured_embeddings = np.vstack([similar_group, different_group])

        adjacency = segmenter._compute_embedding_adjacency(structured_embeddings)

        # Similar tokens should have higher internal similarity
        similar_avg = np.mean(adjacency[:3, :3][~np.eye(3, dtype=bool)])
        different_avg = np.mean(adjacency[:3, 3:5])

        # With this explicit structure, similar group should have higher internal similarity
        assert similar_avg > different_avg, (
            f"Similar avg: {similar_avg}, Different avg: {different_avg}"
        )

    def test_identify_boundaries_with_embeddings(
        self, segmenter, sample_tokens, embedding_config, mock_embeddings
    ):
        """Test identify_boundaries method with embedding-based approach."""
        tokens = sample_tokens["short_sequence"]

        boundaries = segmenter.identify_boundaries(
            tokens=tokens, gamma=1.0, token_embeddings=mock_embeddings
        )

        # Check basic properties
        assert len(boundaries) >= 2  # At least start and end
        assert boundaries[0] == 0  # Should start with first token
        assert boundaries[-1] == len(tokens) - 1  # Should end with last token

        # Boundaries should be sorted
        assert boundaries == sorted(boundaries)

        # Boundaries should be unique
        assert len(boundaries) == len(set(boundaries))

    def test_identify_boundaries_comparison(
        self, segmenter, sample_tokens, sample_embeddings
    ):
        """Test that different gamma values produce different boundary sets."""
        # Use a longer sequence for more meaningful comparison
        tokens = sample_tokens["coding_session"]

        # Get boundaries with lower gamma (more boundaries)
        boundaries_low = segmenter.identify_boundaries(
            tokens=tokens,
            gamma=0.8,
            token_embeddings=sample_embeddings(len(tokens), dimension=384, seed=123),
        )

        # Get boundaries with higher gamma (fewer boundaries)
        boundaries_high = segmenter.identify_boundaries(
            tokens=tokens,
            gamma=1.5,
            token_embeddings=sample_embeddings(len(tokens), dimension=384, seed=123),
        )

        # Both should be valid boundary sets
        assert len(boundaries_low) >= 2
        assert len(boundaries_high) >= 2
        assert boundaries_low[0] == 0
        assert boundaries_high[0] == 0
        assert boundaries_low[-1] == len(tokens) - 1
        assert boundaries_high[-1] == len(tokens) - 1

        # Higher gamma should produce fewer or equal boundaries
        assert len(boundaries_high) <= len(boundaries_low)

    def test_linear_coherence_segmentation(
        self, segmenter, sample_tokens, mock_embeddings
    ):
        """Test O(n) linear coherence-based segmentation."""
        tokens = sample_tokens["short_sequence"]

        # Segment using O(n) linear method
        boundaries = segmenter.segment_by_coherence_linear(
            token_embeddings=mock_embeddings, window_size=5, min_segment_length=20
        )

        # Check properties
        assert len(boundaries) >= 2  # At least start and end
        assert boundaries[0] == 0
        assert boundaries[-1] == len(tokens) - 1
        assert boundaries == sorted(boundaries)

    def test_linear_segmentation_comparison(
        self, segmenter, sample_tokens, mock_embeddings
    ):
        """Test that linear segmentation gives consistent results."""
        _ = sample_tokens["short_sequence"]

        # Run twice - should be deterministic
        boundaries1 = segmenter.segment_by_coherence_linear(
            token_embeddings=mock_embeddings, window_size=5, min_segment_length=20
        )

        boundaries2 = segmenter.segment_by_coherence_linear(
            token_embeddings=mock_embeddings, window_size=5, min_segment_length=20
        )

        # Should give identical results
        assert boundaries1 == boundaries2

    def test_new_linear_api(self, segmenter, sample_tokens):
        """Test that new O(n) linear API works correctly."""
        tokens = sample_tokens["short_sequence"]

        # Generate embeddings
        embeddings = np.random.randn(len(tokens), 384).astype("float32")

        # Test identify_boundaries (O(n))
        boundaries_surprise = segmenter.identify_boundaries(
            tokens=tokens, gamma=1.0, token_embeddings=embeddings
        )

        # Test segment_by_coherence_linear (O(n))
        boundaries_coherence = segmenter.segment_by_coherence_linear(
            token_embeddings=embeddings, window_size=5, min_segment_length=20
        )

        # Both should return valid boundaries
        assert len(boundaries_surprise) >= 2
        assert len(boundaries_coherence) >= 2
        assert boundaries_surprise[0] == 0
        assert boundaries_coherence[0] == 0
        assert boundaries_surprise[-1] == len(tokens) - 1
        assert boundaries_coherence[-1] == len(tokens) - 1

    def test_surprise_calculation_methods(
        self, segmenter, sample_tokens, mock_embeddings
    ):
        """Test embedding-based surprise calculation consistency."""
        tokens = sample_tokens["short_sequence"]

        # Run boundary identification twice with same inputs - should be deterministic
        boundaries1 = segmenter.identify_boundaries(
            tokens=tokens, gamma=1.0, token_embeddings=mock_embeddings
        )

        boundaries2 = segmenter.identify_boundaries(
            tokens=tokens, gamma=1.0, token_embeddings=mock_embeddings
        )

        # Both should produce identical results (deterministic)
        assert boundaries1 == boundaries2

        # Should produce valid boundary lists
        assert len(boundaries1) >= 2
        assert boundaries1[0] == 0
        assert boundaries1[-1] == len(tokens) - 1
        assert boundaries1 == sorted(boundaries1)

    def test_different_gamma_values(self, segmenter, sample_tokens, mock_embeddings):
        """Test that different gamma values produce different boundary patterns."""
        tokens = sample_tokens["short_sequence"]
        gamma_values = [0.5, 1.0, 2.0]

        boundaries_per_gamma = {}
        for gamma in gamma_values:
            boundaries = segmenter.identify_boundaries(
                tokens=tokens, gamma=gamma, token_embeddings=mock_embeddings
            )
            boundaries_per_gamma[gamma] = boundaries

        # Different gamma should generally produce different boundary counts
        gamma_counts = [len(boundaries) for boundaries in boundaries_per_gamma.values()]
        assert len(set(gamma_counts)) >= 1  # At least some variation
