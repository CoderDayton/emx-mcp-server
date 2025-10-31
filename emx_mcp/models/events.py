"""Data models for episodic events."""

from dataclasses import dataclass, field
from typing import List, Optional, Dict
import time


@dataclass
class EpisodicEvent:
    """
    Represents a single episodic event in memory.

    Corresponds to a coherent segment of experience bounded
    by surprise-based segmentation.
    """

    event_id: str
    tokens: List[str]
    embeddings: List[List[float]]
    boundaries: List[int]  # Token positions of event boundaries
    timestamp: float = field(default_factory=time.time)
    metadata: Optional[Dict] = None

    # Statistics
    surprise_scores: Optional[List[float]] = None
    access_count: int = 0

    def __post_init__(self):
        """Validate event data."""
        if len(self.tokens) != len(self.embeddings):
            raise ValueError("Token and embedding counts must match")

        if self.metadata is None:
            self.metadata = {}

    @property
    def token_count(self) -> int:
        """Get number of tokens in event."""
        return len(self.tokens)

    @property
    def duration(self) -> float:
        """Get time since event creation."""
        return time.time() - self.timestamp

    def increment_access(self):
        """Increment access counter."""
        self.access_count += 1

    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization."""
        return {
            "event_id": self.event_id,
            "tokens": self.tokens,
            "embeddings": self.embeddings,
            "boundaries": self.boundaries,
            "timestamp": self.timestamp,
            "metadata": self.metadata,
            "surprise_scores": self.surprise_scores,
            "access_count": self.access_count,
        }

    @classmethod
    def from_dict(cls, data: Dict) -> "EpisodicEvent":
        """Create event from dictionary."""
        return cls(
            event_id=data["event_id"],
            tokens=data["tokens"],
            embeddings=data["embeddings"],
            boundaries=data["boundaries"],
            timestamp=data.get("timestamp", time.time()),
            metadata=data.get("metadata"),
            surprise_scores=data.get("surprise_scores"),
            access_count=data.get("access_count", 0),
        )


@dataclass
class EventBoundary:
    """Represents a boundary between episodic events."""

    position: int  # Token position
    surprise_score: float
    refined_position: Optional[int] = None
    refinement_score: Optional[float] = None

    @property
    def is_refined(self) -> bool:
        """Check if boundary has been refined."""
        return self.refined_position is not None
