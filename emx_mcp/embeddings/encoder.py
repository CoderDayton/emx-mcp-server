"""
Embedding encoder for converting tokens to vector representations.
Uses sentence-transformers for efficient, high-quality embeddings.
"""

import numpy as np
from typing import List
import logging

logger = logging.getLogger(__name__)


class EmbeddingEncoder:
    """
    Encodes token sequences into dense vector embeddings.

    Uses sentence-transformers models for semantic encoding.
    Default: all-MiniLM-L6-v2 (384-dim, fast, good quality)
    """

    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        device: str = "cpu",
        batch_size: int = 1,
    ):
        """
        Initialize embedding encoder.

        Args:
            model_name: HuggingFace model identifier
            device: "cpu" or "cuda"
        """
        try:
            from sentence_transformers import SentenceTransformer

            self.model = SentenceTransformer(model_name, device=device)
            self.dimension = self.model.get_sentence_embedding_dimension()
            self.model_name = model_name
            self.batch_size = batch_size

            logger.info(
                f"EmbeddingEncoder initialized: {model_name} (dim={self.dimension})"
            )
        except ImportError:
            logger.error(
                "sentence-transformers not installed. Run: pip install sentence-transformers"
            )
            raise ImportError(
                "sentence-transformers required for embedding generation. "
                "Install with: pip install sentence-transformers"
            )

    def encode_tokens(self, tokens: List[str]) -> np.ndarray:
        """
        Encode token sequence into single embedding.

        Args:
            tokens: List of token strings

        Returns:
            Embedding vector of shape (dimension,)
        """
        # Join tokens into text
        text = " ".join(tokens)

        # Encode
        embedding = self.model.encode(
            text, convert_to_numpy=True, show_progress_bar=False
        )

        return embedding.astype(np.float32)

    def encode_batch(self, token_lists: List[List[str]]) -> np.ndarray:
        """
        Batch encode multiple token sequences.

        Args:
            token_lists: List of token lists

        Returns:
            Embeddings array of shape (n_sequences, dimension)
        """
        # Join each token list into text
        texts = [" ".join(tokens) for tokens in token_lists]

        # Batch encode
        embeddings = self.model.encode(
            texts, batch_size=self.batch_size, convert_to_numpy=True, show_progress_bar=False
        )

        return embeddings.astype(np.float32)

    def encode_individual_tokens(self, tokens: List[str]) -> np.ndarray:
        """
        Encode each token separately (for per-token embeddings).

        Args:
            tokens: List of token strings

        Returns:
            Embeddings array of shape (n_tokens, dimension)
        """
        embeddings = self.model.encode(
            tokens,
            batch_size=self.batch_size,
            convert_to_numpy=True,
            show_progress_bar=False,
        )

        return embeddings.astype(np.float32)

    def encode_tokens_with_context(self, tokens: List[str], context_window: int = 10) -> np.ndarray:
        """
        Encode tokens individually with local context for surprise calculation.

        This method computes embeddings for each token while considering its local
        context window. This provides better semantic representation for detecting
        semantic shifts and computing surprise values.

        Args:
            tokens: List of token strings
            context_window: Number of previous tokens to include as context

        Returns:
            Array of embeddings of shape (n_tokens, embedding_dim)
        """
        embeddings = []
        
        if not tokens:
            raise ValueError("Token list cannot be empty")
        
        for i, token in enumerate(tokens):
            # Get context window
            start = max(0, i - context_window)
            context_tokens = tokens[start:i+1]
            
            # Encode token with context - join as text for sentence-transformers
            text = " ".join(context_tokens)
            
            # Encode using sentence-transformers
            embedding = self.model.encode(
                text, 
                convert_to_numpy=True, 
                show_progress_bar=False
            )
            embeddings.append(embedding)
        
        return np.array(embeddings, dtype=np.float32)

    def get_query_embedding(self, query: str) -> np.ndarray:
        """
        Encode query string for retrieval.

        Args:
            query: Query text

        Returns:
            Query embedding of shape (dimension,)
        """
        embedding = self.model.encode(
            query, convert_to_numpy=True, show_progress_bar=False
        )
        return embedding.astype(np.float32)
