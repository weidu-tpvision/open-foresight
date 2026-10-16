"""Multimodal Vector Store using SQLite and 768-D dense vectors from EmbeddingGemma 2."""

from __future__ import annotations

import asyncio
import hashlib
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence, AsyncIterator
from contextlib import asynccontextmanager
import aiosqlite
import numpy as np

from daemon.db.repository import DEFAULT_DB_PATH
from daemon.rag.chunker import DocumentChunker, DocumentChunk
from daemon.rag.embeddings import EmbeddingGemmaClient

logger = logging.getLogger(__name__)


@dataclass
class SearchResult:
    chunk_id: int
    document_id: str
    file_name: str
    file_path: str
    section_title: str
    page_number: int
    text: str
    score: float


class VectorStore:
    """Manages document chunk indexing, 768-D vector embedding, and similarity search."""

    def __init__(
        self,
        db_path: str | Path = DEFAULT_DB_PATH,
        embedding_client: Optional[EmbeddingGemmaClient] = None,
        chunker: Optional[DocumentChunker] = None,
    ):
        self.db_path = str(db_path)
        self.embed_client = embedding_client or EmbeddingGemmaClient()
        self.chunker = chunker or DocumentChunker()

    @asynccontextmanager
    async def connect(self) -> AsyncIterator[aiosqlite.Connection]:
        """Open an aiosqlite connection with foreign keys enabled."""
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute("PRAGMA foreign_keys = ON;")
            yield db

    async def index_file(self, file_path: str | Path, force: bool = False) -> dict:
        """Parse, chunk, embed, and store a document in the vector store."""
        path = Path(file_path).resolve()
        if not path.is_file():
            raise FileNotFoundError(f"Cannot index non-existent file: {path}")

        file_stat = path.stat()
        file_size = file_stat.st_size
        modified_at = int(file_stat.st_mtime * 1000)

        # Unique document ID derived from canonical path
        doc_id = hashlib.sha256(str(path).encode("utf-8")).hexdigest()[:16]

        # Incremental check: Skip re-indexing if file is unchanged
        if not force:
            async with self.connect() as db:
                db.row_factory = aiosqlite.Row
                async with db.execute(
                    "SELECT modified_at, file_size_bytes, chunk_count FROM documents WHERE id = ?",
                    (doc_id,),
                ) as cursor:
                    row = await cursor.fetchone()
                    if row and row["modified_at"] == modified_at and row["file_size_bytes"] == file_size:
                        logger.debug(f"Skipping unchanged document: {path.name}")
                        return {
                            "document_id": doc_id,
                            "file_name": path.name,
                            "file_path": str(path),
                            "chunks_indexed": row["chunk_count"],
                            "status": "skipped",
                        }

        # Chunk the document in a worker thread (avoids blocking event loop on PDF parsing)
        chunks = await asyncio.to_thread(self.chunker.chunk_file, path)
        if not chunks:
            logger.warning(f"No text extracted from {path}")
            return {"document_id": doc_id, "file_name": path.name, "chunks_indexed": 0}

        # Embed chunks in mini-batches (e.g. 32 at a time) to prevent timeouts/memory spikes
        texts_to_embed = [c.text for c in chunks]
        batch_size = 32
        embeddings: list[list[float]] = []
        for i in range(0, len(texts_to_embed), batch_size):
            batch = texts_to_embed[i : i + batch_size]
            batch_embs = await self.embed_client.embed_texts(batch)
            embeddings.extend(batch_embs)

        async with self.connect() as db:
            # Insert or replace document metadata
            await db.execute(
                """
                INSERT INTO documents (id, file_path, file_name, file_type, file_size_bytes, modified_at, chunk_count)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(id) DO UPDATE SET
                    file_size_bytes = excluded.file_size_bytes,
                    modified_at = excluded.modified_at,
                    chunk_count = excluded.chunk_count,
                    indexed_at = CURRENT_TIMESTAMP
                """,
                (
                    doc_id,
                    str(path),
                    path.name,
                    path.suffix.lstrip(".").lower() or "txt",
                    file_size,
                    modified_at,
                    len(chunks),
                ),
            )

            # Clear any previously indexed chunks for this document
            await db.execute("DELETE FROM document_chunks WHERE document_id = ?", (doc_id,))

            # Insert chunks with serialized 768-D binary embeddings
            for chunk, emb in zip(chunks, embeddings):
                emb_blob = np.array(emb, dtype=np.float32).tobytes()
                await db.execute(
                    """
                    INSERT INTO document_chunks (
                        document_id, chunk_index, text, page_number, section_title, embedding
                    ) VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (
                        doc_id,
                        chunk.chunk_index,
                        chunk.text,
                        chunk.page_number,
                        chunk.section_title,
                        emb_blob,
                    ),
                )

            await db.commit()

        logger.info(f"Indexed {len(chunks)} chunks for {path.name} into VectorStore.")
        return {
            "document_id": doc_id,
            "file_name": path.name,
            "file_path": str(path),
            "chunks_indexed": len(chunks),
        }

    async def index_directory(
        self,
        dir_path: str | Path,
        extensions: Sequence[str] = (".md", ".pdf", ".txt"),
    ) -> list[dict]:
        """Index all matching files in a local directory."""
        directory = Path(dir_path).resolve()
        if not directory.is_dir():
            raise NotADirectoryError(f"Directory not found: {directory}")

        indexed: list[dict] = []
        skip_dirs = {".git", ".venv", "node_modules", "build", "dist", "__pycache__", "target"}
        for root, dirs, files in os.walk(directory):
            # Prune skipped directories in-place
            dirs[:] = [d for d in dirs if not d.startswith(".") and d not in skip_dirs]
            for f in files:
                if f.startswith("."):
                    continue
                ext = Path(f).suffix.lower()
                if ext in extensions:
                    file_path = Path(root) / f
                    try:
                        res = await self.index_file(file_path)
                        indexed.append(res)
                    except Exception as e:
                        logger.error(f"Failed to index {file_path}: {e}")

        return indexed

    async def similarity_search(
        self,
        query: str,
        top_k: int = 5,
        min_score: float = 0.25,
    ) -> list[SearchResult]:
        """
        Compute EmbeddingGemma 2 query vector and perform cosine similarity search
        against all document chunks stored in SQLite.
        """
        if not query.strip():
            return []

        # 1. Embed query into 768-D vector
        query_vec = await self.embed_client.embed_single(query)
        if not query_vec:
            return []
        query_arr = np.array(query_vec, dtype=np.float32)
        norm_q = np.linalg.norm(query_arr)
        if norm_q == 0.0:
            return []

        # 2. Fetch all chunk vectors and metadata
        results: list[SearchResult] = []
        async with self.connect() as db:
            db.row_factory = aiosqlite.Row
            async with db.execute(
                """
                SELECT c.id, c.document_id, c.chunk_index, c.text, c.page_number, c.section_title, c.embedding,
                       d.file_name, d.file_path
                FROM document_chunks c
                JOIN documents d ON c.document_id = d.id
                """
            ) as cursor:
                rows = await cursor.fetchall()

        if not rows:
            return []

        # 3. Vectorized cosine similarity computation across candidate vectors
        valid_rows = [r for r in rows if r["embedding"]]
        if not valid_rows:
            return []

        # Stack chunk vectors into (N, 768) matrix
        matrix = np.frombuffer(
            b"".join(r["embedding"] for r in valid_rows), dtype=np.float32
        ).reshape(len(valid_rows), -1)

        # Vectorized dot products: (N,)
        dots = np.dot(matrix, query_arr)
        # Vectorized L2 norms: (N,)
        norms_c = np.linalg.norm(matrix, axis=1)
        # Avoid division by zero
        denom = norms_c * norm_q
        scores = np.where(denom > 0.0, dots / denom, 0.0)

        candidates: list[SearchResult] = []
        for idx, score in enumerate(scores):
            score_val = float(score)
            if score_val >= min_score:
                row = valid_rows[idx]
                candidates.append(
                    SearchResult(
                        chunk_id=row["id"],
                        document_id=row["document_id"],
                        file_name=row["file_name"],
                        file_path=row["file_path"],
                        section_title=row["section_title"],
                        page_number=row["page_number"],
                        text=row["text"],
                        score=score_val,
                    )
                )

        # 4. Sort descending by score
        candidates.sort(key=lambda x: x.score, reverse=True)
        return candidates[:top_k]

    async def list_documents(self) -> list[dict]:
        """List all indexed documents with statistics."""
        async with self.connect() as db:
            db.row_factory = aiosqlite.Row
            async with db.execute(
                "SELECT * FROM documents ORDER BY indexed_at DESC"
            ) as cursor:
                rows = await cursor.fetchall()
                return [dict(r) for r in rows]

    async def delete_document(self, doc_id: str) -> bool:
        """Remove a document and all its chunks from the vector store."""
        async with self.connect() as db:
            cursor = await db.execute("DELETE FROM documents WHERE id = ?", (doc_id,))
            await db.commit()
            return cursor.rowcount > 0
