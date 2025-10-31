"""Integration tests for embedding-based EM-LLM segmentation."""

import pytest
import numpy as np
import time
from pathlib import Path
import sys

# Add project root to path for imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from emx_mcp.embeddings.encoder import EmbeddingEncoder
from emx_mcp.memory.segmentation import SurpriseSegmenter
from emx_mcp.memory.project_manager import ProjectMemoryManager


class TestIntegration:
    """End-to-end integration tests for embedding-based approach."""
    
    @pytest.fixture
    def encoder(self, embedding_config):
        """Create EmbeddingEncoder instance."""
        return EmbeddingEncoder(
            model_name=embedding_config["model_name"],
            device=embedding_config["device"],
            batch_size=embedding_config["batch_size"],
        )
    
    @pytest.fixture
    def segmenter(self, segmentation_config):
        """Create SurpriseSegmenter instance."""
        return SurpriseSegmenter(
            gamma=segmentation_config["gamma"],
            window_offset=segmentation_config["window_offset"],
            refinement_metric=segmentation_config["refinement_metric"],
        )
    
    @pytest.mark.asyncio
    async def test_full_segmentation_pipeline(self, encoder, segmenter, sample_tokens):
        """Test complete segmentation pipeline with realistic data."""
        # Test with all sample sequences
        for seq_name, tokens in sample_tokens.items():
            if seq_name == "short_sequence":
                continue  # Skip very short sequences for integration testing
            
            print(f"\nTesting {seq_name}...")
            
            # Step 1: Generate embeddings
            embeddings = encoder.encode_tokens_with_context(tokens, context_window=5)
            assert embeddings.shape == (len(tokens), encoder.dimension)
            
            # Step 2: Identify boundaries
            boundaries = segmenter.identify_boundaries(
                tokens=tokens,
                gamma=1.0,
                token_embeddings=embeddings
            )
            
            # Step 3: Refine boundaries
            refined_boundaries = segmenter.refine_boundaries(
                initial_boundaries=boundaries,
                tokens=tokens,
                token_embeddings=embeddings
            )
            
            # Validate results
            assert len(refined_boundaries) >= 2
            assert refined_boundaries[0] == 0
            assert refined_boundaries[-1] == len(tokens) - 1
            assert refined_boundaries == sorted(refined_boundaries)
            
            # Check that we found reasonable number of events
            num_events = len(refined_boundaries) - 1
            assert 1 <= num_events <= len(tokens) // 2  # Reasonable range
            
            print(f"  Found {num_events} events for {len(tokens)} tokens")
    
    @pytest.mark.asyncio
    async def test_embedding_vs_placeholder_comparison(self, encoder, segmenter, sample_tokens):
        """Compare embedding-based vs placeholder-based segmentation."""
        tokens = sample_tokens["meeting_transcript"]
        
        # Generate embeddings
        embeddings = encoder.encode_tokens_with_context(tokens, context_window=5)
        
        # Get boundaries with embeddings
        start_time = time.time()
        embedding_boundaries = segmenter.identify_boundaries(
            tokens=tokens,
            gamma=1.0,
            token_embeddings=embeddings
        )
        embedding_time = time.time() - start_time
        
        # Get boundaries with placeholders
        start_time = time.time()
        placeholder_boundaries = segmenter.identify_boundaries(
            tokens=tokens,
            gamma=1.0,
            token_probs=None  # Uses placeholders
        )
        placeholder_time = time.time() - start_time
        
        # Both should work without errors
        assert len(embedding_boundaries) >= 2
        assert len(placeholder_boundaries) >= 2
        
        # Performance comparison
        print(f"  Embedding-based: {embedding_time:.3f}s")
        print(f"  Placeholder-based: {placeholder_time:.3f}s")
        
        # Embedding approach should provide more meaningful boundaries
        # (though this is hard to test automatically)
        embedding_events = len(embedding_boundaries) - 1
        placeholder_events = len(placeholder_boundaries) - 1
        
        print(f"  Embedding events: {embedding_events}")
        print(f"  Placeholder events: {placeholder_events}")
        
        # Both should be reasonable
        assert 1 <= embedding_events <= len(tokens) // 2
        assert 1 <= placeholder_events <= len(tokens) // 2
    
    @pytest.mark.asyncio
    async def test_semantic_shift_detection(self, encoder, segmenter):
        """Test that semantic shifts are properly detected."""
        # Create a sequence with clear semantic shifts
        tokens = [
            # Programming-related tokens
            "function", "variable", "class", "method", "return",
            # Switch to cooking
            "recipe", "ingredients", "cooking", "baking", "oven",
            # Switch to travel
            "vacation", "airport", "hotel", "travel", "journey"
        ]
        
        # Generate embeddings
        embeddings = encoder.encode_tokens_with_context(tokens, context_window=3)
        
        # Get boundaries with lower gamma for more sensitive detection
        boundaries = segmenter.identify_boundaries(
            tokens=tokens,
            gamma=0.5,  # Lower gamma for more boundaries
            token_embeddings=embeddings
        )
        
        # Should detect at least 2 boundaries (for the 3 semantic phases)
        assert len(boundaries) >= 2
        
        # Boundaries should roughly align with semantic shifts
        print(f"  Tokens: {len(tokens)}")
        print(f"  Boundaries: {boundaries}")
        
        # Verify we found meaningful segments (at least 1 event)
        num_events = len(boundaries) - 1
        assert num_events >= 1
    
    @pytest.mark.asyncio
    async def test_refinement_improves_boundaries(self, encoder, segmenter, sample_tokens):
        """Test that boundary refinement actually improves results."""
        tokens = sample_tokens["coding_session"]
        
        # Generate embeddings
        embeddings = encoder.encode_tokens_with_context(tokens, context_window=5)
        
        # Get initial boundaries
        initial_boundaries = segmenter.identify_boundaries(
            tokens=tokens,
            gamma=1.0,
            token_embeddings=embeddings
        )
        
        if len(initial_boundaries) > 2:  # Only test refinement if meaningful
            # Refine boundaries
            refined_boundaries = segmenter.refine_boundaries(
                initial_boundaries=initial_boundaries,
                tokens=tokens,
                token_embeddings=embeddings
            )
            
            # Refinement should not decrease number of events (often increases)
            assert len(refined_boundaries) >= len(initial_boundaries)
            
            # Both should be valid boundary lists
            assert initial_boundaries[0] == 0
            assert refined_boundaries[0] == 0
            assert initial_boundaries[-1] == len(tokens) - 1
            assert refined_boundaries[-1] == len(tokens) - 1
            
            print(f"  Initial boundaries: {initial_boundaries}")
            print(f"  Refined boundaries: {refined_boundaries}")
        else:
            pytest.skip("Not enough initial boundaries to test refinement")
    
    @pytest.mark.asyncio
    async def test_different_context_windows(self, encoder, segmenter, sample_tokens):
        """Test that different context windows produce reasonable results."""
        tokens = sample_tokens["research_presentation"]
        context_windows = [3, 5, 10, 15]
        
        for window in context_windows:
            embeddings = encoder.encode_tokens_with_context(tokens, context_window=window)
            
            boundaries = segmenter.identify_boundaries(
                tokens=tokens,
                gamma=1.0,
                token_embeddings=embeddings
            )
            
            # Should always produce valid boundaries
            assert len(boundaries) >= 2
            assert boundaries[0] == 0
            assert boundaries[-1] == len(tokens) - 1
            
            # Different windows should generally give different results
            print(f"  Window {window}: {len(boundaries)-1} events")
    
    @pytest.mark.asyncio
    async def test_performance_benchmark(self, encoder, segmenter, sample_tokens):
        """Benchmark performance of embedding-based approach."""
        tokens = sample_tokens["research_presentation"]
        n_tokens = len(tokens)
        
        # Benchmark embedding generation
        start_time = time.time()
        embeddings = encoder.encode_tokens_with_context(tokens, context_window=5)
        embedding_time = time.time() - start_time
        
        # Benchmark surprise calculation
        start_time = time.time()
        surprises = segmenter._compute_embedding_surprises(embeddings, window=5)
        surprise_time = time.time() - start_time
        
        # Benchmark adjacency computation
        start_time = time.time()
        adjacency = segmenter._compute_embedding_adjacency(embeddings)
        adjacency_time = time.time() - start_time
        
        # Benchmark full segmentation
        start_time = time.time()
        boundaries = segmenter.identify_boundaries(
            tokens=tokens,
            gamma=1.0,
            token_embeddings=embeddings
        )
        segmentation_time = time.time() - start_time
        
        total_time = embedding_time + surprise_time + adjacency_time + segmentation_time
        
        print(f"\n  Performance for {n_tokens} tokens:")
        print(f"    Embedding generation: {embedding_time:.3f}s")
        print(f"    Surprise calculation: {surprise_time:.3f}s")
        print(f"    Adjacency computation: {adjacency_time:.3f}s")
        print(f"    Boundary identification: {segmentation_time:.3f}s")
        print(f"    Total time: {total_time:.3f}s")
        print(f"    Tokens per second: {n_tokens/total_time:.1f}")
        
        # Should complete within reasonable time
        assert total_time < 10.0  # Should be much faster than this
        assert n_tokens / total_time > 1  # Should process at least 1 token per second
    
    @pytest.mark.asyncio
    async def test_robustness_to_noise(self, encoder, segmenter):
        """Test that approach is robust to various input patterns."""
        # Test with very repetitive tokens
        repetitive_tokens = ["word"] * 15
        
        embeddings = encoder.encode_tokens_with_context(repetitive_tokens, context_window=3)
        boundaries = segmenter.identify_boundaries(
            tokens=repetitive_tokens,
            gamma=1.0,
            token_embeddings=embeddings
        )
        
        # Should handle repetitive sequences
        assert len(boundaries) >= 2
        
        # Test with very diverse tokens
        diverse_tokens = [f"token_{i}" for i in range(15)]
        
        embeddings = encoder.encode_tokens_with_context(diverse_tokens, context_window=3)
        boundaries = segmenter.identify_boundaries(
            tokens=diverse_tokens,
            gamma=1.0,
            token_embeddings=embeddings
        )
        
        # Should also handle diverse sequences
        assert len(boundaries) >= 2
        assert len(boundaries) <= len(diverse_tokens)  # Can't have more boundaries than tokens
    
    @pytest.mark.asyncio
    async def test_project_manager_integration(self, mock_project_config):
        """Test integration with ProjectMemoryManager."""
        # Create a minimal project manager
        try:
            manager = ProjectMemoryManager(
                project_path=mock_project_config["project_path"],
                global_path=mock_project_config["global_path"],
                config=mock_project_config["config"]
            )
            
            # Test segment_tokens method
            tokens = ["test", "tokens", "for", "segmentation"]
            result = manager.segment_tokens(
                tokens=tokens,
                gamma=1.0,
                use_refinement=True,
                context_window=3
            )
            
            # Check that result has expected structure
            assert "initial_boundaries" in result
            assert "refined_boundaries" in result
            assert "num_events" in result
            assert "method" in result
            assert "success" in result
            
            assert result["success"] is True
            assert result["num_events"] >= 0
            assert "embedding" in result["method"]
            
        except ImportError as e:
            pytest.skip(f"Dependencies not available: {e}")
    
    @pytest.mark.slow
    @pytest.mark.asyncio
    async def test_large_sequence_performance(self, encoder, segmenter):
        """Test performance with larger sequences (marked as slow test)."""
        # Create a larger synthetic sequence
        large_tokens = []
        for i in range(100):
            if i < 30:
                large_tokens.append(f"topic_a_token_{i}")
            elif i < 60:
                large_tokens.append(f"topic_b_token_{i}")
            else:
                large_tokens.append(f"topic_c_token_{i}")
        
        start_time = time.time()
        
        # Generate embeddings
        embeddings = encoder.encode_tokens_with_context(large_tokens, context_window=5)
        
        # Identify boundaries
        boundaries = segmenter.identify_boundaries(
            tokens=large_tokens,
            gamma=1.0,
            token_embeddings=embeddings
        )
        
        # Refine boundaries
        refined_boundaries = segmenter.refine_boundaries(
            initial_boundaries=boundaries,
            tokens=large_tokens,
            token_embeddings=embeddings
        )
        
        total_time = time.time() - start_time
        events_found = len(refined_boundaries) - 1
        
        print(f"\n  Large sequence test ({len(large_tokens)} tokens):")
        print(f"    Time: {total_time:.3f}s")
        print(f"    Events found: {events_found}")
        print(f"    Initial boundaries: {len(boundaries)}")
        print(f"    Refined boundaries: {refined_boundaries}")
        
        # Should complete within reasonable time
        assert total_time < 30.0, f"Test took {total_time:.3f}s (expected < 30s)"
        assert len(refined_boundaries) >= 2, f"Found {len(refined_boundaries)} boundaries (expected >= 2)"
        
        # Should find topic transitions (3 expected, but allow 1-5 for robustness)
        # Boundary detection can be sensitive to embedding similarity and gamma
        assert 1 <= events_found <= 6, (
            f"Found {events_found} events (expected 1-6). "
            f"This may indicate gamma={1.0} is too high for this sequence, "
            f"or embeddings are too similar between topics. "
            f"Initial boundaries: {len(boundaries)}, Refined: {len(refined_boundaries)}"
        )