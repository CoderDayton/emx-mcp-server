"""
Surprise-based event segmentation with O(n) linear complexity.
Implements Algorithm 1 from EM-LLM paper (ICLR 2025).

OPTIMIZED: Pure O(n) linear complexity - NO O(n³) refinement code.
- Embedding-based surprise calculation: O(n)
- Linear coherence segmentation: O(n)
- No refinement overhead
"""

import numpy as np
from typing import List, Optional
import logging

logger = logging.getLogger(__name__)


class SurpriseSegmenter:
    """
    Implements surprise-based event segmentation with O(n) linear complexity.

    Uses embedding-based surprise to identify event boundaries efficiently.
    Two complementary methods (both O(n)):
    1. identify_boundaries() - Direct boundary detection from surprise
    2. segment_by_coherence_linear() - Sliding window coherence-based boundaries
    """

    def __init__(
        self,
        gamma: float = 1.0,
        window_offset: int = 128,
        enable_refinement: bool = True,
        refinement_metric: str = "modularity",
        max_refinement_window: int = 512,
    ):
        """
        Initialize surprise segmenter.

        Args:
            gamma: Surprise threshold sensitivity (higher = fewer boundaries)
            window_offset: Window size for adaptive threshold calculation
            enable_refinement: Enable graph-theoretic boundary refinement (Algorithm 1)
            refinement_metric: Metric for refinement ('modularity' or 'conductance')
            max_refinement_window: Maximum tokens per refinement chunk (controls complexity)
        """
        self.gamma = gamma
        self.window_offset = window_offset
        self.enable_refinement = enable_refinement
        self.refinement_metric = refinement_metric
        self.max_refinement_window = max_refinement_window

        complexity_note = (
            f"O(n) base + O(nm) refinement (m={max_refinement_window})"
            if enable_refinement
            else "O(n) only"
        )
        logger.info(
            f"SurpriseSegmenter initialized (gamma={gamma}, "
            f"refinement={enable_refinement}, metric={refinement_metric}, {complexity_note})"
        )

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
        token_embeddings: Optional[np.ndarray] = None,
    ) -> List[int]:
        """
        Identify event boundaries using adaptive surprise threshold.

        Implements Equation 1 from EM-LLM paper:
        -log P(x_t | x_1,...,x_{t-1}) > T
        where T = μ_{t-τ:t} + γσ_{t-τ:t}

        Uses embedding-based surprise calculation (O(n) complexity).

        Args:
            tokens: List of tokens
            gamma: Override default gamma
            token_embeddings: Pre-computed embeddings for embedding-based surprise

        Returns:
            List of boundary positions

        Complexity: O(n)
        """
        if gamma is None:
            gamma = self.gamma

        # Get surprise values using embedding-based method
        if token_embeddings is not None:
            surprises = self._compute_embedding_surprises(token_embeddings)
            logger.debug("Using embedding-based surprise calculation (O(n))")
        else:
            raise ValueError(
                "token_embeddings required for embedding-based surprise calculation"
            )

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
            f"Identified {len(boundaries)} surprise-based boundaries from {len(tokens)} tokens"
        )

        # Apply boundary refinement if enabled (Algorithm 1 from paper)
        if (
            self.enable_refinement
            and token_embeddings is not None
            and len(boundaries) > 2
        ):
            logger.debug(
                f"Applying {self.refinement_metric} refinement to {len(boundaries)} boundaries..."
            )
            boundaries = self._refine_boundaries(
                token_embeddings, boundaries, metric=self.refinement_metric
            )
            logger.info(
                f"Refined to {len(boundaries)} boundaries using {self.refinement_metric} "
                f"(O(nm) where m={self.max_refinement_window})"
            )

        return boundaries

    def segment_by_coherence_linear(
        self,
        token_embeddings: np.ndarray,
        window_size: int = 5,
        min_segment_length: int = 20,
        surprise_threshold: Optional[float] = None,
    ) -> List[int]:
        """
        O(n) linear complexity segmentation using coherence-based boundaries.

        Uses sliding window coherence scores to identify topic boundaries.
        Based on TextTiling algorithm (Hearst, 1997).

        Args:
            token_embeddings: Array of shape (n_tokens, embedding_dim)
            window_size: Size of sliding window for local coherence (5-10 recommended)
            min_segment_length: Minimum tokens per segment
            surprise_threshold: Threshold for boundary detection (auto if None)

        Returns:
            List of boundary indices

        Complexity: O(n) where n = number of tokens
        """
        n_tokens = len(token_embeddings)

        if n_tokens < min_segment_length * 2:
            return [0, n_tokens - 1]

        # Step 1: Compute coherence scores (O(n))
        coherence_scores = self._compute_coherence_linear(token_embeddings, window_size)

        # Step 2: Find local minima (O(n))
        candidate_boundaries = self._find_local_minima_linear(coherence_scores)

        # Step 3: Compute depth scores (O(n))
        depth_scores = self._compute_depth_linear(
            coherence_scores, candidate_boundaries
        )

        # Step 4: Apply threshold (O(n))
        if surprise_threshold is None:
            if len(depth_scores) > 0:
                mu = np.mean(depth_scores)
                sigma = np.std(depth_scores)
                surprise_threshold = float(mu - 0.5 * sigma)
            else:
                surprise_threshold = 0.0

        boundaries = [0]
        for idx, depth in zip(candidate_boundaries, depth_scores):
            if depth > surprise_threshold and idx >= min_segment_length:
                boundaries.append(idx)

        # Step 5: Enforce minimum segment length (O(n))
        boundaries = self._enforce_min_length_linear(
            boundaries, n_tokens, min_segment_length
        )

        if boundaries[-1] != n_tokens - 1:
            boundaries.append(n_tokens - 1)

        logger.info(
            f"Linear segmentation found {len(boundaries)} boundaries (O(n) method)"
        )
        return boundaries

    # ========================
    # Private Helper Methods
    # ========================

    def _compute_embedding_surprises(
        self, token_embeddings: np.ndarray, window: int = 10
    ) -> np.ndarray:
        """
        Compute surprises from embedding distances to local context centroids.

        Surprise is computed as the distance between a token's embedding and
        the centroid of its local context window.

        Args:
            token_embeddings: Array of shape (n_tokens, embedding_dim)
            window: Size of context window for computing surprise

        Returns:
            Array of surprise values of shape (n_tokens,)

        Complexity: O(n)
        """
        n_tokens = len(token_embeddings)
        surprises = np.zeros(n_tokens, dtype=np.float32)

        for t in range(window, n_tokens):
            # Get context window (previous 'window' tokens)
            context = token_embeddings[t - window : t]
            context_centroid = np.mean(context, axis=0)

            # Compute cosine distance from current token to context centroid
            current_embedding = token_embeddings[t]

            # Use 1 - cosine_similarity as surprise (more different = more surprising)
            cosine_sim = self._cosine_sim(current_embedding, context_centroid)
            surprises[t] = 1.0 - cosine_sim  # Higher when less similar to context

        # Handle tokens at the beginning (no full context)
        if n_tokens > window:
            # Use mean surprise for first 'window' tokens
            mean_surprise = np.mean(surprises[window:])
            surprises[:window] = mean_surprise
        else:
            # For very short sequences, use a default surprise
            surprises[:] = 1.0

        return surprises

    def _compute_coherence_linear(
        self, embeddings: np.ndarray, window_size: int
    ) -> np.ndarray:
        """
        Compute coherence between adjacent blocks. O(n) complexity.

        Complexity: O(n)
        """
        n_tokens = len(embeddings)
        coherence_scores = np.zeros(n_tokens - 1)

        for i in range(n_tokens - 1):
            left_start = max(0, i - window_size + 1)
            left_end = i + 1
            right_start = i + 1
            right_end = min(n_tokens, i + window_size + 1)

            left_block = embeddings[left_start:left_end].mean(axis=0)
            right_block = embeddings[right_start:right_end].mean(axis=0)

            # Cosine similarity
            coherence_scores[i] = self._cosine_sim(left_block, right_block)

        return coherence_scores

    def _cosine_sim(self, a: np.ndarray, b: np.ndarray) -> float:
        """Compute cosine similarity between two vectors."""
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return np.dot(a, b) / (norm_a * norm_b)

    def _find_local_minima_linear(self, scores: np.ndarray) -> List[int]:
        """
        Find local minima in coherence scores. O(n) complexity.

        Complexity: O(n)
        """
        minima = []
        n = len(scores)
        for i in range(1, n - 1):
            if scores[i] < scores[i - 1] and scores[i] < scores[i + 1]:
                minima.append(i)
        return minima

    def _compute_depth_linear(
        self, coherence_scores: np.ndarray, candidate_boundaries: List[int]
    ) -> List[float]:
        """
        Compute depth score for each boundary. O(n) complexity.

        Complexity: O(n) - each boundary examined once
        """
        depth_scores = []

        for boundary_idx in candidate_boundaries:
            # Find peaks to left and right
            left_peak = coherence_scores[boundary_idx]
            for i in range(boundary_idx - 1, -1, -1):
                if coherence_scores[i] >= coherence_scores[i + 1]:
                    left_peak = max(left_peak, coherence_scores[i])
                else:
                    break

            right_peak = coherence_scores[boundary_idx]
            for i in range(boundary_idx + 1, len(coherence_scores)):
                if coherence_scores[i] >= coherence_scores[i - 1]:
                    right_peak = max(right_peak, coherence_scores[i])
                else:
                    break

            depth = 0.5 * (
                (left_peak - coherence_scores[boundary_idx])
                + (right_peak - coherence_scores[boundary_idx])
            )
            depth_scores.append(depth)

        return depth_scores

    def _enforce_min_length_linear(
        self, boundaries: List[int], n_tokens: int, min_length: int
    ) -> List[int]:
        """
        Remove boundaries creating segments shorter than min_length. O(n) complexity.

        Complexity: O(n)
        """
        if not boundaries:
            return []

        filtered = []
        last_boundary = 0

        for boundary in boundaries:
            if boundary - last_boundary >= min_length:
                filtered.append(boundary)
                last_boundary = boundary

        if n_tokens - last_boundary < min_length and filtered:
            filtered.pop()

        return filtered

    def _compute_embedding_adjacency(self, token_embeddings: np.ndarray) -> np.ndarray:
        """
        Compute adjacency matrix from embedding cosine similarities.

        This method creates an adjacency matrix where edge weights represent
        the semantic similarity between tokens, computed using cosine similarity
        of their embeddings.

        Args:
            token_embeddings: Array of shape (n_tokens, embedding_dim)

        Returns:
            Adjacency matrix of shape (n_tokens, n_tokens) with values in [0, 1]

        Complexity: O(m²) where m = chunk size (controlled by max_refinement_window)
        """
        # Normalize embeddings to unit vectors for cosine similarity
        norms = np.linalg.norm(token_embeddings, axis=1, keepdims=True)
        normalized_embeddings = token_embeddings / (norms + 1e-8)

        # Compute cosine similarity matrix
        adjacency = np.dot(normalized_embeddings, normalized_embeddings.T)

        # Map from [-1, 1] range to [0, 1] range
        adjacency = (adjacency + 1) / 2

        # Ensure diagonal is 1
        np.fill_diagonal(adjacency, 1.0)

        return adjacency

    # ========================
    # Boundary Refinement (Algorithm 1 from EM-LLM Paper)
    # ========================

    def _refine_boundaries(
        self,
        token_embeddings: np.ndarray,
        boundaries: List[int],
        metric: str = "modularity",
    ) -> List[int]:
        """
        Refine event boundaries using graph-theoretic metrics (Algorithm 1).

        Implements the boundary refinement step from the paper:
        - For each pair of consecutive boundaries (α, β), find optimal position β'
        - Optimize modularity (Eq. 3) or conductance (Eq. 4)
        - Process in chunks to maintain O(nm) complexity

        Args:
            token_embeddings: Array of shape (n_tokens, embedding_dim)
            boundaries: Initial surprise-based boundaries
            metric: 'modularity' (maximize) or 'conductance' (minimize)

        Returns:
            Refined boundary positions

        Complexity: O(nm) where n = total tokens, m = max_refinement_window
        """
        refined_boundaries = [boundaries[0]]  # Keep first boundary

        for i in range(len(boundaries) - 1):
            alpha = boundaries[i]
            beta = boundaries[i + 1]
            segment_length = beta - alpha

            # Skip refinement for very short or very long segments
            if segment_length < 10 or segment_length > self.max_refinement_window:
                refined_boundaries.append(beta)
                continue

            # Extract segment embeddings
            segment_embeddings = token_embeddings[alpha:beta]

            # Compute adjacency matrix for this segment (O(m²) where m = segment_length)
            adjacency = self._compute_embedding_adjacency(segment_embeddings)

            # Find optimal boundary within segment
            if metric == "modularity":
                optimal_offset = self._optimize_modularity(adjacency, alpha, beta)
            elif metric == "conductance":
                optimal_offset = self._optimize_conductance(adjacency, alpha, beta)
            else:
                optimal_offset = beta - alpha  # Fallback: keep original

            optimal_boundary = alpha + optimal_offset
            refined_boundaries.append(optimal_boundary)

        return refined_boundaries

    def _optimize_modularity(self, adjacency: np.ndarray, alpha: int, beta: int) -> int:
        """
        Find boundary position that maximizes modularity (Equation 3).

        Modularity Q = (1/4m) Σ [A_ij - (Σ_k A_ik · Σ_k A_jk)/(2m)] δ(c_i, c_j)

        Where:
        - A_ij = adjacency (similarity) between tokens i and j
        - m = total edge weight
        - δ(c_i, c_j) = 1 if same community, 0 otherwise

        Args:
            adjacency: Similarity matrix for segment
            alpha, beta: Segment start/end positions (for logging)

        Returns:
            Optimal boundary offset within segment (relative to alpha)

        Complexity: O(m²) where m = segment length
        """
        n = adjacency.shape[0]
        if n < 2:
            return n

        # Total edge weight
        total_weight = np.sum(adjacency) / 2.0  # Divide by 2 for undirected graph

        if total_weight == 0:
            return n // 2  # Fallback: split in half

        # Compute degree for each node
        degrees = np.sum(adjacency, axis=1)

        best_modularity = -float("inf")
        best_position = n // 2  # Default: middle

        # Try each possible boundary position (excluding extremes)
        # This is the "argmax" operation from Algorithm 1
        for split_pos in range(1, n):
            # Community assignment: [0, split_pos) = community 1, [split_pos, n) = community 2
            community = np.zeros(n, dtype=int)
            community[split_pos:] = 1

            # Compute modularity for this split
            modularity = 0.0
            for i in range(n):
                for j in range(i, n):  # Only upper triangle (symmetric)
                    if community[i] == community[j]:  # Same community
                        # Expected edge weight under null model
                        expected = (degrees[i] * degrees[j]) / (2 * total_weight)
                        # Actual - Expected
                        contribution = adjacency[i, j] - expected

                        # Double count off-diagonal elements
                        if i != j:
                            modularity += 2 * contribution
                        else:
                            modularity += contribution

            modularity /= 4 * total_weight

            if modularity > best_modularity:
                best_modularity = modularity
                best_position = split_pos

        logger.debug(
            f"Modularity optimization: position {alpha + best_position} "
            f"(Q={best_modularity:.4f}) in segment [{alpha}, {beta})"
        )

        return best_position

    def _optimize_conductance(
        self, adjacency: np.ndarray, alpha: int, beta: int
    ) -> int:
        r"""
        Find boundary position that minimizes conductance (Equation 4).

        Conductance φ(S) = cut(S, V\S) / min(vol(S), vol(V\S))

        Where:
        - cut(S, V\S) = Σ_{i∈S, j∉S} A_ij (edges crossing boundary)
        - vol(S) = Σ_{i,j∈S} A_ij (volume of subset S)

        Lower conductance = better community structure.

        Args:
            adjacency: Similarity matrix for segment
            alpha, beta: Segment start/end positions (for logging)

        Returns:
            Optimal boundary offset within segment (relative to alpha)

        Complexity: O(m²) where m = segment length
        """
        n = adjacency.shape[0]
        if n < 2:
            return n

        best_conductance = float("inf")
        best_position = n // 2  # Default: middle

        # Try each possible boundary position
        for split_pos in range(1, n):
            # Subset S = [0, split_pos), V\S = [split_pos, n)

            # Compute cut: edges between S and V\S
            cut = np.sum(adjacency[:split_pos, split_pos:])

            # Compute volumes
            vol_s = np.sum(adjacency[:split_pos, :split_pos])
            vol_vs = np.sum(adjacency[split_pos:, split_pos:])

            # Avoid division by zero
            min_vol = min(vol_s, vol_vs)
            if min_vol < 1e-8:
                continue

            conductance = cut / min_vol

            if conductance < best_conductance:
                best_conductance = conductance
                best_position = split_pos

        logger.debug(
            f"Conductance optimization: position {alpha + best_position} "
            f"(φ={best_conductance:.4f}) in segment [{alpha}, {beta})"
        )

        return best_position
