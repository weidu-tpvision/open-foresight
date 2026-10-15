"""Integration tests for Phase 3: Document Chunking, VectorStore, and Live Assistant RAG."""

import pytest
import asyncio
import tempfile
from pathlib import Path
from starlette.testclient import TestClient

from daemon.app import app, embed_client, gemma_client
from daemon.rag.chunker import DocumentChunker
from daemon.rag.vector_store import VectorStore
from daemon.rag.assistant import LiveAssistant
from daemon.db.repository import DatabaseRepository


def test_markdown_chunker():
    """Verify Markdown chunking respects headers and sliding window."""
    chunker = DocumentChunker(target_chunk_size=200, chunk_overlap=30)
    with tempfile.NamedTemporaryFile("w+", suffix=".md", delete=False) as f:
        f.write("# Database Architecture\n\nWe utilize PostgreSQL with synchronous streaming replication.\n\n## Failover Policy\n\nIf the primary node fails, Patroni promotes the standby replica in under 5 seconds.")
        f.flush()
        temp_path = Path(f.name)

    try:
        chunks = chunker.chunk_file(temp_path)
        assert len(chunks) >= 2
        assert any(c.section_title == "Failover Policy" for c in chunks)
        assert any("PostgreSQL" in c.text for c in chunks)
    finally:
        temp_path.unlink()


@pytest.mark.asyncio
async def test_vector_store_indexing_and_search():
    """Verify document chunking, 768-D embedding, and similarity search."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        db_path = Path(tmp_dir) / "test_rag.db"
        repo = DatabaseRepository(db_path=db_path)
        await repo.initialize()

        sample_doc = Path(tmp_dir) / "architecture.md"
        sample_doc.write_text(
            "# System Architecture\n\n"
            "The authentication system uses JWT tokens with 15-minute expiration.\n"
            "The token refresh service runs on Port 8080 with a distributed Redis lock.\n\n"
            "## Storage Layer\n\n"
            "All relational data is stored in SQLite in WAL mode with immediate disk sync."
        )

        vs = VectorStore(db_path=db_path, embedding_client=embed_client)
        index_res = await vs.index_file(sample_doc)
        assert index_res["chunks_indexed"] >= 2

        # Verify documents list
        docs = await vs.list_documents()
        assert len(docs) == 1
        assert docs[0]["file_name"] == "architecture.md"

        # Search for authentication
        matches = await vs.similarity_search("How are tokens refreshed?", top_k=2)
        assert len(matches) > 0
        assert "JWT tokens" in matches[0].text or "refresh service" in matches[0].text
        assert matches[0].score > 0.3


@pytest.mark.asyncio
async def test_live_assistant_grounded_answer():
    """Verify Live Assistant provides grounded answers with citations."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        db_path = Path(tmp_dir) / "test_assistant.db"
        repo = DatabaseRepository(db_path=db_path)
        await repo.initialize()

        # Seed knowledge doc
        doc_path = Path(tmp_dir) / "limits.md"
        doc_path.write_text(
            "# Rate Limits\n\nThe public REST API is capped at 100 requests per second per IP.\n"
            "Enterprise customers have a dedicated quota of 1000 requests per second."
        )
        vs = VectorStore(db_path=db_path, embedding_client=embed_client)
        await vs.index_file(doc_path)

        assistant = LiveAssistant(vector_store=vs, repository=repo, llm_client=gemma_client)
        resp = await assistant.answer_query("What is the public API rate limit?")

        assert resp is not None
        assert len(resp.citations) > 0
        assert "100" in resp.answer_markdown
        assert "limits.md" in resp.citations[0].source_label


def test_knowledge_api_endpoints():
    """Verify FastAPI endpoints for knowledge indexing and assistant chat."""
    with TestClient(app) as client:
        with tempfile.NamedTemporaryFile("w+", suffix=".md", delete=False) as f:
            f.write("# Backup Procedures\n\nNightly snapshots are saved to local NVMe storage at 02:00 UTC.")
            f.flush()
            temp_file = Path(f.name)

        try:
            # 1. Index file
            idx_resp = client.post("/api/knowledge/index", json={"path": str(temp_file)})
            assert idx_resp.status_code == 200

            # 2. List documents
            list_resp = client.get("/api/knowledge/documents")
            assert list_resp.status_code == 200
            assert list_resp.json()["total_documents"] >= 1

            # 3. Vector search
            search_resp = client.post("/api/knowledge/search", json={"query": "nightly snapshots"})
            assert search_resp.status_code == 200
            assert len(search_resp.json()["matches"]) > 0

            # 4. Assistant chat
            chat_resp = client.post("/api/assistant/chat", json={"query": "When are backups taken?"})
            assert chat_resp.status_code == 200
            ans = chat_resp.json()
            assert "02:00" in ans["answer_markdown"] or "backup" in ans["answer_markdown"].lower()
            assert len(ans["citations"]) > 0
        finally:
            temp_file.unlink()
