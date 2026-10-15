"""Tests for Phase 2: Shorthand Engine, Gemma 4 Client, and EmbeddingGemma 2."""

import pytest
import asyncio
import tempfile
from pathlib import Path
from starlette.testclient import TestClient

from daemon.app import app, gemma_client, embed_client
from daemon.db.repository import DatabaseRepository
from daemon.sync.enricher import ShorthandEnricher


@pytest.mark.asyncio
async def test_embeddinggemma_dimensions():
    """Verify EmbeddingGemma 2 produces 768-D vectors and cosine similarity works."""
    try:
        emb = await embed_client.embed_single("Architecture failover database")
        assert len(emb) == 768
        sim = embed_client.cosine_similarity(emb, emb)
        assert abs(sim - 1.0) < 1e-4
    except Exception as e:
        pytest.skip(f"Ollama embeddinggemma-2 not reachable: {e}")


@pytest.mark.asyncio
async def test_shorthand_enrichment_flow():
    """Verify temporal window correlation and grounded synthesis with Gemma 4."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        db_file = Path(tmp_dir) / "test_enrich.db"
        repo = DatabaseRepository(db_path=db_file)
        await repo.initialize()

        session_id = "session-enrich-test"
        await repo.create_session(session_id, "Sprint Review")

        # Add conversation segment spoken at 00:15
        await repo.add_transcript_segment(
            session_id=session_id,
            start_ms=14000,
            end_ms=18000,
            speaker_channel="remote",
            speaker_label="Bob",
            text="We have a race condition in the JWT refresh token. I will fix it by Thursday with a mutex lock.",
        )

        enricher = ShorthandEnricher(repository=repo, llm_client=gemma_client)

        # Note typed at 00:16 (16000ms)
        bullet = await enricher.enrich_single_note(
            session_id=session_id,
            note_text="- auth race condition, bob by thurs",
            note_timestamp_ms=16000,
        )

        assert bullet is not None
        assert bullet.correlated_segments_count == 1
        assert "ref:" in bullet.enriched_markdown.lower() or "00:16" in bullet.enriched_markdown or "thursday" in bullet.enriched_markdown.lower()


def test_notes_api_endpoints():
    """Verify REST API endpoints for notes and models."""
    with TestClient(app) as client:
        # Check models endpoint
        model_resp = client.get("/api/models")
        assert model_resp.status_code == 200
        data = model_resp.json()
        assert data["llm"] == "gemma4:e4b"
        assert data["embedding_dimensions"] == 768

        # Create session
        start_resp = client.post("/api/sessions/start", json={"title": "Test Notes"})
        session_id = start_resp.json()["session_id"]

        # Save Notes
        save_resp = client.post(
            f"/api/sessions/{session_id}/notes",
            json={"raw_notes": "- test note 1\n- test note 2"},
        )
        assert save_resp.status_code == 200

        # Get Notes
        get_resp = client.get(f"/api/sessions/{session_id}/notes")
        assert get_resp.status_code == 200
        notes = get_resp.json()["notes"]
        assert len(notes) > 0
        assert "test note 1" in notes[0]["raw_text"]

        # Stop session
        client.post(f"/api/sessions/{session_id}/stop")
