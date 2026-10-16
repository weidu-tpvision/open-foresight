"""Background filesystem watcher monitoring local folders for real-time RAG ingestion."""

from __future__ import annotations

import asyncio
import logging
import time
from pathlib import Path
from typing import Optional, Set
from watchdog.events import FileSystemEventHandler, FileSystemEvent
from watchdog.observers import Observer

from daemon.rag.vector_store import VectorStore

logger = logging.getLogger(__name__)

SUPPORTED_EXTENSIONS = {".md", ".markdown", ".pdf", ".txt"}


class KnowledgeFileHandler(FileSystemEventHandler):
    """Handles file system events and schedules vector store updates."""

    def __init__(
        self,
        vector_store: VectorStore,
        loop: asyncio.AbstractEventLoop,
        debounce_seconds: float = 1.0,
    ):
        super().__init__()
        self.vector_store = vector_store
        self.loop = loop
        self.debounce_seconds = debounce_seconds
        self._last_modified: dict[str, float] = {}

    def on_created(self, event: FileSystemEvent) -> None:
        if not event.is_directory:
            self._handle_change(event.src_path)

    def on_modified(self, event: FileSystemEvent) -> None:
        if not event.is_directory:
            self._handle_change(event.src_path)

    def on_deleted(self, event: FileSystemEvent) -> None:
        if not event.is_directory:
            self._handle_delete(event.src_path)

    def on_moved(self, event: FileSystemEvent) -> None:
        if not event.is_directory:
            if hasattr(event, "src_path"):
                self._handle_delete(event.src_path)
            if hasattr(event, "dest_path"):
                self._handle_change(event.dest_path)

    def _handle_delete(self, file_path_str: str) -> None:
        import hashlib
        path = Path(file_path_str).resolve()
        doc_id = hashlib.sha256(str(path).encode("utf-8")).hexdigest()[:16]
        logger.info(f"Knowledge file deletion detected: {path.name}. Removing doc_id {doc_id}...")
        future = asyncio.run_coroutine_threadsafe(
            self.vector_store.delete_document(doc_id), self.loop
        )
        future.add_done_callback(
            lambda f: logger.error(f"Error deleting {path.name}: {f.exception()}")
            if f.exception()
            else None
        )

    def _handle_change(self, file_path_str: str) -> None:
        path = Path(file_path_str)
        if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            return

        now = time.time()
        last = self._last_modified.get(file_path_str, 0)
        if now - last < self.debounce_seconds:
            return
        self._last_modified[file_path_str] = now

        logger.info(f"Knowledge file change detected: {path.name}. Triggering indexing...")
        # Schedule indexing in asyncio event loop and log any errors
        future = asyncio.run_coroutine_threadsafe(
            self.vector_store.index_file(path), self.loop
        )
        future.add_done_callback(
            lambda f: logger.error(f"Error indexing {path.name}: {f.exception()}")
            if f.exception()
            else None
        )


class FolderWatcherService:
    """Manages directory monitoring threads for autonomous knowledge ingestion."""

    def __init__(self, vector_store: VectorStore):
        self.vector_store = vector_store
        self.observer: Optional[Observer] = None
        self.watched_paths: Set[str] = set()

    def start_watching(self, path: str | Path) -> bool:
        """Start monitoring a local directory for changes."""
        dir_path = Path(path).resolve()
        if not dir_path.is_dir():
            logger.warning(f"Cannot watch non-existent directory: {dir_path}")
            return False

        path_str = str(dir_path)
        if path_str in self.watched_paths:
            return True

        loop = asyncio.get_running_loop()
        handler = KnowledgeFileHandler(vector_store=self.vector_store, loop=loop)

        if not self.observer:
            self.observer = Observer()
            self.observer.start()

        self.observer.schedule(handler, path=path_str, recursive=True)
        self.watched_paths.add(path_str)
        logger.info(f"FolderWatcher started for directory: {path_str}")
        return True

    def stop(self) -> None:
        """Stop watcher observer."""
        if self.observer:
            self.observer.stop()
            self.observer.join(timeout=2.0)
            self.observer = None
            self.watched_paths.clear()
            logger.info("FolderWatcher stopped.")
