"""Data models for episodic events."""

import time
from dataclasses import dataclass, field


@dataclass
class EpisodicEvent:
    """
    Represents a single episodic event in memory.

    Corresponds to a coherent segment of experience bounded
    by surprise-based segmentation.
    """

    event_id: str
    tokens: list[str]
    embeddings: list[list[float]]
    boundaries: list[int]  # Token positions of event boundaries
    timestamp: float = field(default_factory=time.time)
    metadata: dict | None = None

    # Statistics
    surprise_scores: list[float] | None = None
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

    def to_dict(self) -> dict:
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
    def from_dict(cls, data: dict) -> "EpisodicEvent":
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
    refined_position: int | None = None
    refinement_score: float | None = None

    @property
    def is_refined(self) -> bool:
        """Check if boundary has been refined."""
        return self.refined_position is not None
