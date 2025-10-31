"""
Surprise-based event segmentation with boundary refinement.
Implements Algorithm 1 from EM-LLM paper (ICLR 2025).
"""

import numpy as np
from typing import List, Optional
import logging

logger = logging.getLogger(__name__)


class SurpriseSegmenter:
    """
    Implements surprise-based event segmentation with boundary refinement.

    Uses Bayesian surprise (negative log-likelihood) to identify event
    boundaries, then refines them using graph-theoretic metrics.
    """

    def __init__(
        self,
        gamma: float = 1.0,
        window_offset: int = 128,
        refinement_metric: str = "modularity",
    ):
        """
        Initialize surprise segmenter.

        Args:
            gamma: Surprise threshold sensitivity (higher = fewer boundaries)
            window_offset: Window size for adaptive threshold calculation
            refinement_metric: "modularity" or "conductance"
        """
        self.gamma = gamma
        self.window_offset = window_offset
        self.refinement_metric = refinement_metric

        logger.info(f"SurpriseSegmenter initialized (gamma={gamma})")

    def compute_surprise(self, token_probs: np.ndarray) -> np.ndarray:
        """
        Compute Bayesian surprise: -log(P(token|context)).

        Args:
            token_probs: Array of token probabilities from LLM

        Returns:
            Array of surprise values
        """
        # Clip to avoid log(0)
        probs_clipped = np.clip(token_probs, 1e-10, 1.0)
        return -np.log(probs_clipped)

    def identify_boundaries(
        self,
        tokens: list,
        gamma: Optional[float] = None,
        token_probs: Optional[np.ndarray] = None,
        token_embeddings: Optional[np.ndarray] = None,
    ) -> List[int]:
        """
        Identify event boundaries using adaptive surprise threshold.

        Implements Equation 1 from EM-LLM paper:
        -log P(x_t | x_1,...,x_{t-1}) > T
        where T = μ_{t-τ:t} + γσ_{t-τ:t}

        Supports multiple surprise calculation methods:
        1. Embedding-based (recommended): Uses semantic distances from local context
        2. LLM-based: Uses token probabilities from language model
        3. Placeholder: Random surprises for testing

        Args:
            tokens: List of tokens
            gamma: Override default gamma
            token_probs: Token probabilities (alternative method)
            token_embeddings: Pre-computed embeddings for embedding-based surprise

        Returns:
            List of boundary positions
        """
        if gamma is None:
            gamma = self.gamma

        # Get surprise values using preferred method
        if token_embeddings is not None:
            # Primary method: embedding-based surprise calculation
            surprises = self._compute_embedding_surprises(token_embeddings)
            logger.debug("Using embedding-based surprise calculation")
        elif token_probs is not None:
            # Legacy method: LLM token probabilities
            surprises = self.compute_surprise(token_probs)
            logger.debug("Using LLM token probabilities for surprise calculation")
        else:
            # Fallback: placeholder surprises for testing
            surprises = self._get_placeholder_surprises(len(tokens))
            logger.debug("Using placeholder surprises (testing mode)")

        boundaries = [0]  # Start with first token

        for t in range(self.window_offset, len(surprises)):
            # Calculate adaptive threshold from local window
            window = surprises[t - self.window_offset : t]
            mu = np.mean(window)
            sigma = np.std(window)
            threshold = mu + gamma * sigma

            if surprises[t] > threshold:
                boundaries.append(t)
                logger.debug(
                    f"Boundary at token {t} (surprise={surprises[t]:.2f}, threshold={threshold:.2f})"
                )

        # Add final boundary
        if boundaries[-1] != len(tokens) - 1:
            boundaries.append(len(tokens) - 1)

        logger.info(
            f"Identified {len(boundaries)} boundaries from {len(tokens)} tokens"
        )
        return boundaries

    def refine_boundaries(
        self,
        initial_boundaries: List[int],
        tokens: list,
        attention_keys: Optional[np.ndarray] = None,
        token_embeddings: Optional[np.ndarray] = None,
        max_search_range: int = 512,
    ) -> List[int]:
        """
        Refine boundaries using modularity or conductance maximization.

        Implements boundary refinement step from EM-LLM Algorithm 1.

        Supports multiple adjacency computation methods:
        1. Embedding-based (recommended): Uses cosine similarity of embeddings
        2. LLM-based: Uses attention keys from transformer layers
        3. Placeholder: Random adjacency for testing

        Args:
            initial_boundaries: Initial surprise-based boundaries
            tokens: Token list
            attention_keys: Key vectors for computing similarity (alternative method)
            token_embeddings: Pre-computed embeddings for embedding-based adjacency
            max_search_range: Maximum tokens to search per boundary

        Returns:
            Refined boundary positions
        """
        if len(initial_boundaries) < 2:
            return initial_boundaries

        # Get adjacency matrix using preferred method
        if token_embeddings is not None:
            # Primary method: embedding-based adjacency computation
            adjacency = self._compute_embedding_adjacency(token_embeddings)
            logger.debug("Using embedding-based adjacency computation")
        elif attention_keys is not None:
            # Legacy method: LLM attention keys
            adjacency = self._compute_adjacency(attention_keys)
            logger.debug("Using LLM attention keys for adjacency computation")
        else:
            # Fallback: placeholder adjacency for testing
            adjacency = self._get_placeholder_adjacency(len(tokens))
            logger.debug("Using placeholder adjacency (testing mode)")

        refined = [initial_boundaries[0]]  # Keep first boundary

        for i in range(1, len(initial_boundaries) - 1):
            alpha = initial_boundaries[i - 1]
            beta = initial_boundaries[i]
            next_boundary = initial_boundaries[i + 1]

            # Limit search range
            search_start = max(alpha + 1, beta - max_search_range // 2)
            search_end = min(next_boundary - 1, beta + max_search_range // 2)

            best_score = (
                -float("inf")
                if self.refinement_metric == "modularity"
                else float("inf")
            )
            best_pos = beta

            # Search for optimal position
            for pos in range(search_start, search_end + 1):
                test_boundaries = refined + [pos]

                if self.refinement_metric == "modularity":
                    score = self._compute_modularity(
                        adjacency, test_boundaries, len(tokens)
                    )
                    if score > best_score:
                        best_score = score
                        best_pos = pos
                else:  # conductance
                    score = self._compute_conductance(adjacency, test_boundaries)
                    if score < best_score:  # Lower is better for conductance
                        best_score = score
                        best_pos = pos

            refined.append(best_pos)
            logger.debug(
                f"Refined boundary {i}: {beta} -> {best_pos} (score={best_score:.4f})"
            )

        # Keep last boundary
        refined.append(initial_boundaries[-1])

        logger.info(f"Boundary refinement complete ({len(refined)} boundaries)")
        return refined

    def _compute_adjacency(self, key_vectors: np.ndarray) -> np.ndarray:
        """
        Compute similarity-based adjacency matrix from attention keys.

        Args:
            key_vectors: Attention key vectors (n_tokens, key_dim)

        Returns:
            Adjacency matrix (n_tokens, n_tokens)
        """
        # Normalize keys
        norms = np.linalg.norm(key_vectors, axis=1, keepdims=True)
        normalized_keys = key_vectors / (norms + 1e-8)

        # Compute dot product similarity
        adjacency = np.dot(normalized_keys, normalized_keys.T)

        # Ensure positive values (similarity in [0, 1])
        adjacency = (adjacency + 1) / 2

        return adjacency

    def _compute_modularity(
        self, adjacency: np.ndarray, boundaries: List[int], n_tokens: int
    ) -> float:
        """
        Compute modularity score (Equation 3 from EM-LLM paper).

        Q = 1/(4m) * Σ[A_ij - (k_i * k_j)/(2m)] * δ(c_i, c_j)

        Args:
            adjacency: Adjacency matrix
            boundaries: Current boundary positions
            n_tokens: Total number of tokens

        Returns:
            Modularity score (higher is better)
        """
        m = np.sum(adjacency) / 2  # Total edge weight
        if m == 0:
            return 0.0

        communities = self._boundaries_to_communities(boundaries, n_tokens)
        Q = 0.0

        for community in communities:
            if len(community) == 0:
                continue

            for i in community:
                k_i = np.sum(adjacency[i, :])
                for j in community:
                    k_j = np.sum(adjacency[j, :])
                    Q += adjacency[i, j] - (k_i * k_j) / (2 * m)

        return Q / (4 * m)

    def _compute_conductance(
        self, adjacency: np.ndarray, boundaries: List[int]
    ) -> float:
        """
        Compute conductance score (Equation 4 from EM-LLM paper).

        Lower conductance = better community structure.

        Args:
            adjacency: Adjacency matrix
            boundaries: Current boundary positions

        Returns:
            Conductance score (lower is better)
        """
        communities = self._boundaries_to_communities(boundaries, len(adjacency))
        min_conductance = float("inf")

        for community in communities:
            if len(community) == 0:
                continue

            # Internal volume
            vol_S = np.sum(adjacency[np.ix_(community, community)])

            # Cut (edges leaving community)
            all_indices = set(range(len(adjacency)))
            outside = list(all_indices - set(community))

            if len(outside) == 0:
                continue

            cut = np.sum(adjacency[np.ix_(community, outside)])

            # Conductance = cut / min(vol(S), vol(V\S))
            vol_complement = np.sum(adjacency[np.ix_(outside, outside)])

            if vol_S > 0 and vol_complement > 0:
                conductance = cut / min(vol_S, vol_complement)
                min_conductance = min(min_conductance, conductance)

        return min_conductance if min_conductance != float("inf") else 0.0

    def _boundaries_to_communities(
        self, boundaries: List[int], n_tokens: int
    ) -> List[List[int]]:
        """
        Convert boundary list to community assignments.

        Args:
            boundaries: Boundary positions
            n_tokens: Total number of tokens

        Returns:
            List of communities (each community is a list of token indices)
        """
        communities = []
        for i in range(len(boundaries) - 1):
            start = boundaries[i]
            end = boundaries[i + 1]
            communities.append(list(range(start, end)))

        # Ensure the last community extends to n_tokens if needed
        if boundaries and boundaries[-1] < n_tokens:
            communities.append(list(range(boundaries[-1], n_tokens)))

        return communities

    def _get_placeholder_surprises(self, n_tokens: int) -> np.ndarray:
        """
        Generate placeholder surprise values for testing.

        In production, replace with actual LLM token probabilities.
        """
        # Exponential distribution simulates realistic surprise
        return np.random.exponential(2.0, n_tokens)

    def _get_placeholder_adjacency(self, n_tokens: int) -> np.ndarray:
        """
        Generate placeholder adjacency matrix for testing.

        In production, compute from actual attention keys.
        """
        # Create block-diagonal-ish structure (similar tokens cluster)
        adjacency = np.random.rand(n_tokens, n_tokens) * 0.3

        # Add block structure
        block_size = 50
        for i in range(0, n_tokens, block_size):
            end = min(i + block_size, n_tokens)
            adjacency[i:end, i:end] += 0.5

        # Make symmetric
        adjacency = (adjacency + adjacency.T) / 2

        return adjacency

    def _compute_embedding_surprises(self, token_embeddings: np.ndarray, window: int = 10) -> np.ndarray:
        """
        Compute surprises from embedding distances to local context centroids.

        This method implements embedding-based surprise calculation as an alternative
        to LLM token probabilities. Surprise is computed as the distance between
        a token's embedding and the centroid of its local context window.

        Args:
            token_embeddings: Array of shape (n_tokens, embedding_dim)
            window: Size of context window for computing surprise

        Returns:
            Array of surprise values of shape (n_tokens,)
        """
        n_tokens = len(token_embeddings)
        surprises = np.zeros(n_tokens, dtype=np.float32)
        
        for t in range(window, n_tokens):
            # Get context window (previous 'window' tokens)
            context = token_embeddings[t - window:t]
            context_centroid = np.mean(context, axis=0)
            
            # Compute distance from current token to context centroid
            current_embedding = token_embeddings[t]
            distance = np.linalg.norm(current_embedding - context_centroid)
            
            # Store as surprise value
            surprises[t] = distance
        
        # Handle tokens at the beginning (no full context)
        if n_tokens > window:
            # Use mean surprise for first 'window' tokens
            mean_surprise = np.mean(surprises[window:])
            surprises[:window] = mean_surprise
        else:
            # For very short sequences, use a default surprise
            surprises[:] = 1.0
        
        return surprises

    def _compute_embedding_adjacency(self, token_embeddings: np.ndarray) -> np.ndarray:
        """
        Compute adjacency matrix from embedding cosine similarities.

        This method creates an adjacency matrix where edge weights represent
        the semantic similarity between tokens, computed using cosine similarity
        of their embeddings. This approximates attention-based similarity.

        Args:
            token_embeddings: Array of shape (n_tokens, embedding_dim)

        Returns:
            Adjacency matrix of shape (n_tokens, n_tokens) with values in [0, 1]
        """
        # Normalize embeddings to unit vectors for cosine similarity
        norms = np.linalg.norm(token_embeddings, axis=1, keepdims=True)
        normalized_embeddings = token_embeddings / (norms + 1e-8)
        
        # Compute cosine similarity matrix
        adjacency = np.dot(normalized_embeddings, normalized_embeddings.T)
        
        # Map from [-1, 1] range to [0, 1] range for graph algorithms
        adjacency = (adjacency + 1) / 2
        
        # Ensure diagonal is 1 (token always "similar" to itself)
        np.fill_diagonal(adjacency, 1.0)
        
        return adjacency
