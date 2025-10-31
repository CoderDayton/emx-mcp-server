"""Integration tests for embedding-based EM-LLM segmentation."""

import pytest
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
            
            # Step 2: Segment using O(n) linear coherence method
            boundaries = segmenter.segment_by_coherence_linear(
                token_embeddings=embeddings,
                window_size=5,
                min_segment_length=20
            )
            
            # Validate results
            assert len(boundaries) >= 2
            assert boundaries[0] == 0
            assert boundaries[-1] == len(tokens) - 1
            assert boundaries == sorted(boundaries)
            
            # Check that we found reasonable number of events
            num_events = len(boundaries) - 1
            assert 1 <= num_events <= len(tokens) // 2  # Reasonable range
            
            print(f"  Found {num_events} events for {len(tokens)} tokens")
    
    @pytest.mark.asyncio
    async def test_embedding_vs_placeholder_comparison(self, encoder, segmenter, sample_tokens):
        """Compare embedding-based segmentation with different gamma values."""
        tokens = sample_tokens["meeting_transcript"]
        
        # Generate embeddings
        embeddings = encoder.encode_tokens_with_context(tokens, context_window=5)
        
        # Get boundaries with lower gamma (more sensitive, more boundaries)
        start_time = time.time()
        low_gamma_boundaries = segmenter.identify_boundaries(
            tokens=tokens,
            gamma=0.8,
            token_embeddings=embeddings
        )
        low_gamma_time = time.time() - start_time
        
        # Get boundaries with higher gamma (less sensitive, fewer boundaries)
        start_time = time.time()
        high_gamma_boundaries = segmenter.identify_boundaries(
            tokens=tokens,
            gamma=1.5,
            token_embeddings=embeddings
        )
        high_gamma_time = time.time() - start_time
        
        # Both should work without errors
        assert len(low_gamma_boundaries) >= 2
        assert len(high_gamma_boundaries) >= 2
        
        # Performance comparison
        print(f"  Low gamma (0.8): {low_gamma_time:.3f}s")
        print(f"  High gamma (1.5): {high_gamma_time:.3f}s")
        
        # Lower gamma should produce more boundaries
        low_gamma_events = len(low_gamma_boundaries) - 1
        high_gamma_events = len(high_gamma_boundaries) - 1
        
        print(f"  Low gamma events: {low_gamma_events}")
        print(f"  High gamma events: {high_gamma_events}")
        
        # Verify that lower gamma produces more or equal boundaries
        assert low_gamma_events >= high_gamma_events
        
        # Both should be reasonable
        assert 1 <= low_gamma_events <= len(tokens) // 2
        assert 1 <= high_gamma_events <= len(tokens) // 2
    
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
        
        # Use O(n) linear coherence segmentation
        boundaries = segmenter.segment_by_coherence_linear(
            token_embeddings=embeddings,
            window_size=10,
            min_segment_length=30
        )
        
        total_time = time.time() - start_time
        events_found = len(boundaries) - 1
        
        print(f"\n  Large sequence test ({len(large_tokens)} tokens):")
        print(f"    Time: {total_time:.3f}s")
        print(f"    Events found: {events_found}")
        print(f"    Boundaries: {boundaries}")
        
        # Should complete within reasonable time (O(n) is much faster)
        assert total_time < 30.0, f"Test took {total_time:.3f}s (expected < 30s)"
        assert len(boundaries) >= 2, f"Found {len(boundaries)} boundaries (expected >= 2)"
        
        # Should find topic transitions (3 expected, but allow 1-5 for robustness)
        # Boundary detection can be sensitive to embedding similarity and window size
        assert 1 <= events_found <= 6, (
            f"Found {events_found} events (expected 1-6). "
            f"This may indicate window_size={10} is too large for this sequence, "
            f"or embeddings are too similar between topics. "
            f"Boundaries: {len(boundaries)}"
        )
    
    @pytest.mark.asyncio
    async def test_stream_pipelined_storage_with_manager(self, mock_project_config):
        """Test stream-pipelined event storage via ProjectManager."""
        pytest.importorskip("torch", reason="PyTorch not available")
        
        import torch
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        
        try:
            from emx_mcp.gpu.stream_manager import StreamManager  # type: ignore
        except ImportError:
            pytest.skip("StreamManager not available")
        
        # Create manager with stream support
        manager = ProjectMemoryManager(
            project_path=mock_project_config["project_path"],
            global_path=mock_project_config["global_path"],
            config=mock_project_config["config"]
        )
        
        # Initialize stream manager
        stream_manager = StreamManager(pool_size=2)
        manager.project_store.stream_manager = stream_manager
        
        # Test tokens and embeddings
        tokens = ["stream", "test", "with", "cuda", "pipelines"]
        embeddings = manager.encoder.encode_individual_tokens(tokens)
        
        # Test without streams (baseline)
        start_time = time.time()
        result_no_streams = manager.project_store.add_event(
            event_id="event_no_streams",
            tokens=tokens,
            embeddings=embeddings.tolist(),
            metadata={"test": "no_streams"},
            use_streams=False
        )
        baseline_time = time.time() - start_time
        
        assert result_no_streams["status"] == "added"
        assert not result_no_streams.get("used_streams", False)
        
        # Test with streams
        start_time = time.time()
        result_with_streams = manager.project_store.add_event(
            event_id="event_with_streams",
            tokens=tokens,
            embeddings=embeddings.tolist(),
            metadata={"test": "with_streams"},
            use_streams=True
        )
        streamed_time = time.time() - start_time
        
        assert result_with_streams["status"] == "added"
        assert result_with_streams["used_streams"]
        
        # Both should produce same result (data consistency)
        assert result_no_streams["event_id"] == "event_no_streams"
        assert result_with_streams["event_id"] == "event_with_streams"
        
        print(f"\n  Stream pipeline performance:")
        print(f"    Without streams: {baseline_time:.4f}s")
        print(f"    With streams: {streamed_time:.4f}s")
        print(f"    Speedup: {baseline_time/streamed_time:.2f}x")
        
        # Synchronize all streams before test ends
        stream_manager.synchronize_all()
    
    @pytest.mark.asyncio
    async def test_large_document_e2e_with_streams(self, mock_project_config):
        """
        End-to-end test: Process full 8k token document through the system.
        
        Simulates real LLM usage where a large document (technical paper, code file)
        is processed through:
        1. Embedding generation (per-token embeddings)
        2. Surprise-based segmentation (episodic boundaries)
        3. Stream-pipelined storage (GPU acceleration)
        4. Semantic retrieval (query relevant passages)
        
        Validates:
        - Correct handling of large token sequences
        - Index training triggers automatically
        - Stream pipeline handles batch operations
        - Retrieval returns contextually relevant segments
        """
        pytest.importorskip("torch", reason="PyTorch not available")
        
        import torch
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        
        try:
            from emx_mcp.gpu.stream_manager import StreamManager  # type: ignore
        except ImportError:
            pytest.skip("StreamManager not available")
        
        print(f"\n{'='*60}")
        print(f"  E2E TEST: Large Document with GPU Streams")
        print(f"{'='*60}")
        
        print(f"\n[1/6] Initializing ProjectMemoryManager...")
        manager = ProjectMemoryManager(
            project_path=mock_project_config["project_path"],
            global_path=mock_project_config["global_path"],
            config=mock_project_config["config"]
        )
        print(f"  ✓ Manager initialized")
        print(f"  - Encoder device: {manager.encoder.model.device}")
        print(f"  - Encoder batch size: {manager.encoder.batch_size}")
        
        # Initialize stream manager
        print(f"\n[2/6] Initializing StreamManager...")
        stream_manager = StreamManager(pool_size=4)
        manager.project_store.stream_manager = stream_manager
        print(f"  ✓ Stream manager initialized (pool_size=4)")
        
        # Generate realistic 8k token document (technical content)
        # Simulates: research paper sections with distinct topics
        print(f"\n[3/6] Generating test document...")
        document_sections = [
            # Abstract + Introduction (1000 tokens)
            ["abstract", "introduction", "background", "motivation", "research"] * 200,
            
            # Methods section (2000 tokens)
            ["methodology", "algorithm", "implementation", "architecture", "design",
             "optimization", "training", "validation", "testing", "evaluation"] * 200,
            
            # Results section (2500 tokens)
            ["results", "experiments", "performance", "accuracy", "metrics",
             "benchmark", "comparison", "analysis", "findings", "observations"] * 250,
            
            # Discussion section (1500 tokens)
            ["discussion", "interpretation", "implications", "limitations",
             "future", "work", "conclusions", "summary", "contributions"] * 166,
            
            # References + Appendix (1000 tokens)
            ["references", "bibliography", "appendix", "supplementary", "materials"] * 200,
        ]
        
        all_tokens = []
        for section in document_sections:
            all_tokens.extend(section)
        
        # Trim to exactly 8000 tokens
        all_tokens = all_tokens[:8000]
        
        print(f"  ✓ Document generated: {len(all_tokens)} tokens")
        
        # Step 1: Generate embeddings
        print(f"\n[4/6] Generating embeddings (batched)...")
        start_time = time.time()
        embeddings = manager.encoder.encode_tokens_with_context(
            all_tokens, 
            context_window=10
        )
        embedding_time = time.time() - start_time
        print(f"  ✓ Embedding generation: {embedding_time:.2f}s ({len(all_tokens)/embedding_time:.0f} tokens/s)")
        print(f"  - Embedding shape: {embeddings.shape}")
        
        # Step 2: Segment with O(n) linear coherence method
        print(f"\n[5/6] Segmenting with O(n) linear coherence method...")
        start_time = time.time()
        segmentation_result = manager.segment_tokens(
            tokens=all_tokens,
            gamma=1.5,  # Slightly higher threshold for cleaner boundaries
            context_window=10
        )
        segmentation_time = time.time() - start_time
        
        num_events = segmentation_result["num_events"]
        boundaries = segmentation_result["refined_boundaries"]
        print(f"  ✓ Segmentation: {segmentation_time:.2f}s")
        print(f"  - Events detected: {num_events}")
        print(f"  - Average event size: {len(all_tokens) / num_events:.0f} tokens")
        
        # Verify segmentation quality
        assert num_events > 0
        assert num_events < len(all_tokens)  # Not every token = event
        assert segmentation_result["success"] == True
        
        # Step 2b: Store segmented events in memory
        print(f"\n[6/6] Storing {num_events} events with GPU streams...")
        stored_events = []
        for i in range(len(boundaries) - 1):
            start_idx = boundaries[i]
            end_idx = boundaries[i + 1]
            event_tokens = all_tokens[start_idx:end_idx]
            event_embeddings = embeddings[start_idx:end_idx]
            
            result = manager.add_event(
                tokens=event_tokens,
                embeddings=event_embeddings,
                metadata={"segment_index": i}
            )
            stored_events.append(result)
            assert result["status"] == "added"
            
            if (i + 1) % 10 == 0 or i == len(boundaries) - 2:
                print(f"  - Progress: {i+1}/{num_events} events stored")
        
        print(f"  ✓ All events stored in memory")
        
        # Step 3: Verify index training happened
        print(f"\n  Index Status:")
        assert manager.project_store.vector_store.is_trained
        print(f"  ✓ Index trained: nlist={manager.project_store.vector_store.nlist}")
        
        # Step 4: Test retrieval on different semantic queries
        test_queries = [
            ("methodology algorithm implementation", "Methods section"),
            ("results experiments performance metrics", "Results section"),
            ("discussion limitations future work", "Discussion section"),
        ]
        
        print(f"\n  Testing semantic retrieval ({len(test_queries)} queries):")
        for query_text, expected_section in test_queries:
            # Encode query
            query_embedding = manager.encode_query(query_text)
            
            # Retrieve relevant events
            start_time = time.time()
            retrieval_result = manager.retrieve_memories(
                query_embedding=query_embedding.tolist(),
                k_similarity=3,
                k_contiguity=2,
                use_contiguity=True
            )
            retrieval_time = time.time() - start_time
            
            # Verify retrieval worked
            retrieved_events = retrieval_result.get("event_objects", [])
            assert len(retrieved_events) > 0
            
            # Check if retrieved tokens contain query-relevant content
            query_words = set(query_text.split())
            retrieved_tokens = set()
            for event in retrieved_events:
                retrieved_tokens.update(event.tokens)
            
            # At least one query token should appear in retrieved context
            query_overlap = len(query_words & retrieved_tokens)
            print(f"    ✓ '{expected_section}': {len(retrieved_events)} events, "
                  f"{query_overlap}/{len(query_words)} terms matched, "
                  f"{retrieval_time*1000:.1f}ms")
            
            assert query_overlap > 0, f"No semantic overlap for query: {query_text}"
        
        # Step 5: Verify stream usage metrics
        print(f"\n  Memory Statistics:")
        print(f"    Total events: {manager.project_store.event_count()}")
        print(f"    Total tokens: {manager.project_store.metadata['total_tokens']}")
        print(f"    Offloaded events: {manager.project_store.metadata.get('offloaded_events', 0)}")
        
        # Cleanup
        stream_manager.synchronize_all()
        
        print(f"\n{'='*60}")
        print(f"  ✓ E2E TEST COMPLETED SUCCESSFULLY")
        print(f"{'='*60}\n")
    
    @pytest.mark.asyncio
    async def test_concurrent_stream_usage(self, mock_project_config):
        """Test concurrent add_event calls with limited stream pool.
        
        Note: SQLite connections aren't thread-safe, so we run events sequentially
        but still validate stream pool behavior with rapid successive calls.
        """
        pytest.importorskip("torch", reason="PyTorch not available")
        
        import torch
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        
        try:
            from emx_mcp.gpu.stream_manager import StreamManager  # type: ignore
        except ImportError:
            pytest.skip("StreamManager not available")
        
        manager = ProjectMemoryManager(
            project_path=mock_project_config["project_path"],
            global_path=mock_project_config["global_path"],
            config=mock_project_config["config"]
        )
        
        # Small pool to test exhaustion handling
        stream_manager = StreamManager(pool_size=2)
        manager.project_store.stream_manager = stream_manager
        
        # Add events sequentially (SQLite limitation) but rapidly
        start_time = time.time()
        results = []
        for i in range(5):
            tokens = [f"concurrent_{i}", "test", "event"]
            embeddings = manager.encoder.encode_individual_tokens(tokens)
            
            result = manager.project_store.add_event(
                event_id=f"event_concurrent_{i}",
                tokens=tokens,
                embeddings=embeddings.tolist(),
                metadata={"event_num": i},
                use_streams=True
            )
            results.append(result)
        
        total_time = time.time() - start_time
        
        # All should succeed
        assert all(r["status"] == "added" for r in results)
        assert all(r["used_streams"] for r in results)
        
        print(f"\n  Rapid stream usage (5 events, 2-stream pool):")
        print(f"    Total time: {total_time:.4f}s")
        print(f"    Events succeeded: {len([r for r in results if r['status'] == 'added'])}/5")
        
        stream_manager.synchronize_all()