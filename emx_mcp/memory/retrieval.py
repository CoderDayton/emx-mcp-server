"""Two-stage retrieval with full storage integration."""

from collections import deque
from typing import List, Dict, Deque
import numpy as np
import logging

logger = logging.getLogger(__name__)


class TwoStageRetrieval:
    """
    Fully integrated two-stage retrieval.

    Uses:
    1. Vector store for similarity search
    2. Graph store for temporal neighbors
    3. Storage module for full event retrieval
    """

    def __init__(self, memory_store, config: dict):
        self.memory_store = memory_store
        self.config = config

        # Contiguity buffer
        self.contiguity_buffer: Deque[str] = deque(maxlen=100)
        self.contiguity_window = 3

        logger.info("TwoStageRetrieval initialized with full storage integration")

    def retrieve(
        self,
        query_embedding: List[float],
        k_similarity: int,
        k_contiguity: int,
        use_contiguity: bool,
    ) -> Dict:
        """
        Retrieve events with full object resolution.

        Returns actual EpisodicEvent objects, not just IDs.
        """
        query = np.array(query_embedding, dtype=np.float32)

        # Stage 1: Similarity search (returns full events)
        similarity_events = self.memory_store.search_events(query, k_similarity)

        # Stage 2: Temporal contiguity
        contiguity_events = []
        if use_contiguity and similarity_events:
            contiguity_event_ids = self._retrieve_contiguous_ids(
                [e.event_id for e in similarity_events], k_contiguity
            )

            # Resolve to full events
            for event_id in contiguity_event_ids:
                try:
                    event = self.memory_store.get_event(event_id)
                    contiguity_events.append(event)
                except Exception as e:
                    logger.warning(
                        f"Could not retrieve contiguous event {event_id}: {e}"
                    )

        # Combine (remove duplicates)
        all_events = self._deduplicate_events(similarity_events, contiguity_events)

        return {
            "events": [e.to_dict() for e in all_events],
            "event_objects": all_events,
            "similarity_count": len(similarity_events),
            "contiguity_count": len(contiguity_events),
            "total_count": len(all_events),
        }

    def _retrieve_contiguous_ids(
        self, anchor_event_ids: List[str], k_contiguity: int
    ) -> List[str]:
        """Get temporally adjacent event IDs."""
        contiguous: List[str] = []
        seen = set(anchor_event_ids)

        for event_id in anchor_event_ids:
            if len(contiguous) >= k_contiguity:
                break

            neighbors = self.memory_store.get_temporal_neighbors(
                event_id, max_distance=self.contiguity_window
            )

            for neighbor in neighbors:
                if neighbor not in seen and len(contiguous) < k_contiguity:
                    contiguous.append(neighbor)
                    seen.add(neighbor)

        return contiguous

    def _deduplicate_events(self, sim_events, cont_events):
        """Remove duplicates while preserving order."""
        seen_ids = set()
        result = []

        for event in sim_events + cont_events:
            if event.event_id not in seen_ids:
                result.append(event)
                seen_ids.add(event.event_id)

        return result
