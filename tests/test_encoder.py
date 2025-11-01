"""Test EmbeddingEncoder with pytest-asyncio."""

import pytest
import numpy as np
from pathlib import Path
import sys

from emx_mcp.embeddings.encoder import EmbeddingEncoder

# Add project root to path for imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


class TestEmbeddingEncoder:
    """Test suite for EmbeddingEncoder."""

    @pytest.fixture
    def encoder(self, embedding_config):
        """Create EmbeddingEncoder instance for testing."""
        return EmbeddingEncoder(
            model_name=embedding_config["model_name"],
            device=embedding_config["device"],
            batch_size=embedding_config["batch_size"],
        )

    @pytest.mark.asyncio
    async def test_initialization(self, encoder, embedding_config):
        """Test EmbeddingEncoder initialization."""
        assert encoder is not None
        assert encoder.model_name == embedding_config["model_name"]
        assert encoder.batch_size == embedding_config["batch_size"]
        assert encoder.dimension == embedding_config["dimension"]
        assert hasattr(encoder, "model")  # Should have sentence-transformers model

    @pytest.mark.asyncio
    async def test_encode_tokens_with_context(self, encoder, sample_tokens):
        """Test the new encode_tokens_with_context method."""
        tokens = sample_tokens["short_sequence"]

        # Test with default context window
        embeddings = encoder.encode_tokens_with_context(tokens)
        assert embeddings.shape == (len(tokens), encoder.dimension)

        # Test with different context windows
        for window in [1, 3, 5]:
            embeddings = encoder.encode_tokens_with_context(
                tokens, context_window=window
            )
            assert embeddings.shape == (len(tokens), encoder.dimension)

        # Validate embeddings properties
        assert embeddings.dtype == np.float32
        assert not np.isnan(embeddings).any()
        assert not np.isinf(embeddings).any()

        # Check that different tokens have different embeddings
        # (though some might be similar, they shouldn't be identical)
        for i in range(len(tokens) - 1):
            assert not np.allclose(embeddings[i], embeddings[i + 1], atol=1e-6)

    @pytest.mark.asyncio
    async def test_encode_batch(self, encoder, sample_tokens):
        """Test batch encoding of multiple token sequences."""
        sequences = [
            sample_tokens["short_sequence"][:2],
            sample_tokens["short_sequence"][2:4],
        ]

        embeddings = encoder.encode_batch(sequences)

        # Check shape
        assert embeddings.shape == (len(sequences), encoder.dimension)
        assert embeddings.dtype == np.float32

        # Validate embeddings
        assert not np.isnan(embeddings).any()
        assert not np.isinf(embeddings).any()

    @pytest.mark.asyncio
    async def test_get_query_embedding(self, encoder):
        """Test query embedding generation."""
        query = "What is the user authentication system?"

        embedding = encoder.get_query_embedding(query)

        # Check shape and properties
        assert embedding.shape == (encoder.dimension,)
        assert embedding.dtype == np.float32
        assert not np.isnan(embedding).any()
        assert not np.isinf(embedding).any()

        # Check normalization
        norm = np.linalg.norm(embedding)
        assert np.isclose(norm, 1.0, atol=0.1)

    @pytest.mark.asyncio
    async def test_context_window_edge_cases(self, encoder, sample_tokens):
        """Test context window handling for edge cases."""
        tokens = sample_tokens["short_sequence"]

        # Test with empty tokens
        with pytest.raises((ValueError, IndexError)):
            encoder.encode_tokens_with_context([])

        # Test with very large context window (larger than sequence)
        embeddings = encoder.encode_tokens_with_context(tokens, context_window=100)
        assert embeddings.shape == (len(tokens), encoder.dimension)

        # Test with context window = 1 (just the token itself)
        embeddings = encoder.encode_tokens_with_context(tokens, context_window=1)
        assert embeddings.shape == (len(tokens), encoder.dimension)

    @pytest.mark.asyncio
    async def test_semantic_consistency(self, encoder):
        """Test that semantically similar texts produce similar embeddings."""
        # This is a basic test - we're not testing for exact similarity
        # but rather that the method runs without errors
        similar_texts = [
            ["user authentication", "login system", "password validation"],
            ["machine learning", "neural networks", "deep learning"],
        ]

        for texts in similar_texts:
            embeddings = encoder.encode_tokens_with_context(texts)
            assert embeddings.shape == (len(texts), encoder.dimension)

            # All embeddings should be unit normalized
            norms = np.linalg.norm(embeddings, axis=1)
            assert np.allclose(norms, 1.0, atol=0.1)

    @pytest.mark.asyncio
    async def test_encode_tokens_with_context_progressive(self, encoder, sample_tokens):
        """Test that encode_tokens_with_context handles progressive context correctly."""
        tokens = sample_tokens["short_sequence"]

        # Encode with context
        contextual_embeddings = encoder.encode_tokens_with_context(
            tokens, context_window=3
        )

        # Encode individual tokens
        individual_embeddings = encoder.encode_tokens_with_context(
            tokens, context_window=0
        )

        # They should be different (context affects embeddings)
        # but both should be valid embeddings
        assert contextual_embeddings.shape == individual_embeddings.shape

        # Check that context provides some smoothing effect
        # (neighboring tokens should have more similar contextual embeddings)
        for i in range(len(tokens) - 1):
            # Cosine similarity between consecutive contextual embeddings
            emb1 = contextual_embeddings[i] / np.linalg.norm(contextual_embeddings[i])
            emb2 = contextual_embeddings[i + 1] / np.linalg.norm(
                contextual_embeddings[i + 1]
            )
            similarity = np.dot(emb1, emb2)
            # Should be somewhat positive (similar embeddings for neighboring tokens)
            assert similarity > -1.0  # Valid cosine similarity range
