"""Disk offloading with memory-mapped file support."""

import logging
import mmap
import pickle  # nosec B403 - Used only for internal data serialization, not untrusted input
import struct
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class DiskManager:
    """
    Manages disk offloading with memory-mapped files.

    Uses mmap for efficient random access to large events
    without loading entire files into memory.

    File Format:
    - Header: [version(4), data_offset(8), data_size(8)]
    - Data: pickled event dictionary
    """

    HEADER_SIZE = 20  # 4 + 8 + 8 bytes
    VERSION = 1

    def __init__(self, storage_path: str, offload_threshold: int = 300000):
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)
        self.offload_threshold = offload_threshold

        # Track offloaded events
        self.offloaded_events: dict[str, str] = {}  # event_id -> file_path
        self.mmap_cache: dict[str, tuple[mmap.mmap, Any]] = {}  # event_id -> (mmap, file_handle)
        self.max_mmap_cache = 50  # Max open mmap files

        # Load existing offloaded events index
        self.index_path = self.storage_path / "offload_index.json"
        self._load_index()

        logger.info(f"DiskManager initialized (threshold={offload_threshold} tokens)")
        logger.info("Memory-mapped I/O enabled")

    def _load_index(self):
        """Load index of offloaded events."""
        if self.index_path.exists():
            import json

            with open(self.index_path) as f:
                self.offloaded_events = json.load(f)

    def _save_index(self):
        """Save index of offloaded events."""
        import json

        with open(self.index_path, "w") as f:
            json.dump(self.offloaded_events, f, indent=2)

    def should_offload(self, token_count: int) -> bool:
        """Check if event should be offloaded to disk."""
        return token_count >= self.offload_threshold

    def offload_event(self, event_id: str, event_data: dict) -> str:
        """
        Offload event to disk with mmap support.

        Creates a memory-mapped file for efficient access.

        Args:
            event_id: Event identifier
            event_data: Event dictionary to offload

        Returns:
            Path to offloaded file
        """
        file_path = self.storage_path / f"{event_id}.mmap"

        # Serialize data
        data_bytes = pickle.dumps(event_data)
        data_size = len(data_bytes)

        # Create file and write header + data
        with open(file_path, "wb") as f:
            # Write header
            header = struct.pack(
                "IQQ",  # version (uint32), offset (uint64), size (uint64)
                self.VERSION,
                self.HEADER_SIZE,
                data_size,
            )
            f.write(header)

            # Write data
            f.write(data_bytes)

        # Store in index
        self.offloaded_events[event_id] = str(file_path)
        self._save_index()

        logger.info(f"Offloaded {event_id} to mmap file ({data_size} bytes)")
        return str(file_path)

    def load_event(self, event_id: str, use_mmap: bool = True) -> dict | None:
        """
        Load offloaded event from disk.

        Args:
            event_id: Event identifier
            use_mmap: Use memory-mapped I/O (faster for random access)

        Returns:
            Event data or None if not found
        """
        if event_id not in self.offloaded_events:
            return None

        file_path = Path(self.offloaded_events[event_id])

        if not file_path.exists():
            logger.warning(f"Offloaded file not found: {file_path}")
            return None

        if use_mmap:
            return self._load_with_mmap(event_id, file_path)
        else:
            return self._load_direct(file_path)

    def _load_with_mmap(self, event_id: str, file_path: Path) -> dict:
        """
        Load event using memory-mapped I/O.

        Caches mmap objects for faster repeated access.
        """
        # Check mmap cache
        if event_id in self.mmap_cache:
            mm, fh = self.mmap_cache[event_id]
            return self._read_event_from_file(mm, event_id, " from mmap cache")

        # Open new mmap with guaranteed cleanup on exception
        fh = open(file_path, "r+b")  # noqa: SIM115 - intentional for mmap caching
        try:
            mm = mmap.mmap(fh.fileno(), 0)
            try:
                # Cache mmap (limit cache size)
                if len(self.mmap_cache) >= self.max_mmap_cache:
                    # Close oldest mmap
                    oldest_id = next(iter(self.mmap_cache))
                    old_mm, old_fh = self.mmap_cache.pop(oldest_id)
                    old_mm.close()
                    old_fh.close()
                    logger.debug(f"Evicted {oldest_id} from mmap cache")

                self.mmap_cache[event_id] = (mm, fh)

                return self._read_event_from_file(mm, event_id, " from disk with mmap")
            except Exception:
                mm.close()
                raise
        except Exception:
            fh.close()
            raise

    def _read_event_from_file(self, file_obj, event_id: str, source_msg: str) -> dict:
        """Read event data from file object (mmap or regular file)."""
        file_obj.seek(0)
        event_data = self._deserialize_event_data(file_obj)
        logger.debug(f"Loaded {event_id}{source_msg}")
        return event_data

    def _load_direct(self, file_path: Path) -> dict:
        """Load event directly without mmap."""
        with open(file_path, "rb") as f:
            event_data = self._deserialize_event_data(f)
        logger.debug("Loaded event from disk (direct I/O)")
        return event_data

    def _deserialize_event_data(self, file_obj) -> dict:
        """Deserialize event data from file object (reads header and pickled data)."""
        header = struct.unpack("IQQ", file_obj.read(self.HEADER_SIZE))
        _version, data_offset, data_size = header
        file_obj.seek(data_offset)
        data_bytes = file_obj.read(data_size)
        return pickle.loads(data_bytes)  # nosec B301 - Internal data only

    def remove_event(self, event_id: str) -> bool:
        """
        Remove offloaded event from disk.

        Args:
            event_id: Event identifier

        Returns:
            True if removed, False otherwise
        """
        if event_id not in self.offloaded_events:
            return False

        file_path = Path(self.offloaded_events[event_id])

        # Close mmap if cached
        if event_id in self.mmap_cache:
            mm, fh = self.mmap_cache.pop(event_id)
            mm.close()
            fh.close()

        # Delete file
        if file_path.exists():
            file_path.unlink()

        # Remove from index
        del self.offloaded_events[event_id]
        self._save_index()

        logger.info(f"Removed offloaded event {event_id}")
        return True

    def get_stats(self) -> dict:
        """Get disk manager statistics."""
        total_size = sum(
            Path(path).stat().st_size
            for path in self.offloaded_events.values()
            if Path(path).exists()
        )

        return {
            "offloaded_count": len(self.offloaded_events),
            "total_size_bytes": total_size,
            "total_size_mb": round(total_size / (1024 * 1024), 2),
            "threshold_tokens": self.offload_threshold,
            "mmap_cache_size": len(self.mmap_cache),
            "mmap_enabled": True,
        }

    def clear(self):
        """Clear all offloaded events and close mmaps."""
        # Close all mmaps
        for event_id in list(self.mmap_cache.keys()):
            mm, fh = self.mmap_cache.pop(event_id)
            mm.close()
            fh.close()

        # Delete all files
        for event_id in list(self.offloaded_events.keys()):
            self.remove_event(event_id)

        logger.info("Disk manager cleared, all mmaps closed")

    def __del__(self):
        """Cleanup: close all mmap files."""
        for mm, fh in self.mmap_cache.values():
            try:
                mm.close()
                fh.close()
            except Exception as e:
                # Log cleanup failures during finalization (non-critical)
                logger.debug(f"Error closing mmap during cleanup: {e}")
