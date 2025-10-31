"""Data models for project contexts."""

from dataclasses import dataclass, field
from typing import Dict, List, Optional
from pathlib import Path
import time


@dataclass
class ProjectContext:
    """
    Represents a project's memory context.

    Tracks metadata, statistics, and configuration for
    a single project's .memories folder.
    """

    project_id: str
    project_path: Path
    memory_path: Path
    created_at: float = field(default_factory=time.time)
    last_modified: float = field(default_factory=time.time)

    # Statistics
    total_events: int = 0
    total_tokens: int = 0
    total_vectors: int = 0

    # Configuration overrides
    config_overrides: Optional[Dict] = None

    def update_stats(self, events: int = 0, tokens: int = 0, vectors: int = 0):
        """Update statistics."""
        self.total_events = events
        self.total_tokens = tokens
        self.total_vectors = vectors
        self.last_modified = time.time()

    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            "project_id": self.project_id,
            "project_path": str(self.project_path),
            "memory_path": str(self.memory_path),
            "created_at": self.created_at,
            "last_modified": self.last_modified,
            "total_events": self.total_events,
            "total_tokens": self.total_tokens,
            "total_vectors": self.total_vectors,
            "config_overrides": self.config_overrides,
        }

    @classmethod
    def from_dict(cls, data: Dict) -> "ProjectContext":
        """Create from dictionary."""
        return cls(
            project_id=data["project_id"],
            project_path=Path(data["project_path"]),
            memory_path=Path(data["memory_path"]),
            created_at=data.get("created_at", time.time()),
            last_modified=data.get("last_modified", time.time()),
            total_events=data.get("total_events", 0),
            total_tokens=data.get("total_tokens", 0),
            total_vectors=data.get("total_vectors", 0),
            config_overrides=data.get("config_overrides"),
        )


@dataclass
class GlobalContext:
    """
    Represents shared global memory context.

    Stores cross-project semantic knowledge and patterns.
    """

    created_at: float = field(default_factory=time.time)
    total_shared_events: int = 0
    contributing_projects: List[str] = field(default_factory=list)
    semantic_clusters: Dict[str, List[str]] = field(default_factory=dict)

    def add_project(self, project_id: str):
        """Add project to global context."""
        if project_id not in self.contributing_projects:
            self.contributing_projects.append(project_id)

    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            "created_at": self.created_at,
            "total_shared_events": self.total_shared_events,
            "contributing_projects": self.contributing_projects,
            "semantic_clusters": self.semantic_clusters,
        }
