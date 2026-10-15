"""Unit tests for SQLite schema and DatabaseRepository."""

import pytest
import asyncio
import tempfile
from pathlib import Path
from daemon.db.repository import DatabaseRepository


@pytest.mark.asyncio
async def test_database_initialization_and_operations():
    with tempfile.TemporaryDirectory() as tmp_dir:
        db_file = Path(tmp_dir) / "test_foresight.db"
        repo = DatabaseRepository(db_path=db_file)

        # 1. Initialize schema
        await repo.initialize()
        assert db_file.exists()

        # 2. Create Session
        session = await repo.create_session(
            session_id="session-test-1",
            title="Design Review",
            started_at=1000000,
        )
        assert session["id"] == "session-test-1"
        assert session["status"] == "recording"

        # 3. Add Transcript Segments
        seg1 = await repo.add_transcript_segment(
            session_id="session-test-1",
            start_ms=1000,
            end_ms=4500,
            speaker_channel="remote",
            speaker_label="Bob",
            text="Let us discuss the database failover strategy.",
        )
        seg2 = await repo.add_transcript_segment(
            session_id="session-test-1",
            start_ms=5000,
            end_ms=9000,
            speaker_channel="local",
            speaker_label="Me",
            text="Agreed, we need zero downtime across regions.",
        )
        seg3 = await repo.add_transcript_segment(
            session_id="session-test-1",
            start_ms=60000,
            end_ms=65000,
            speaker_channel="remote",
            speaker_label="Alice",
            text="I will write the migration script by Friday.",
        )
        assert seg1 > 0
        assert seg2 > 0
        assert seg3 > 0

        # 4. Query All Segments
        all_segs = await repo.get_transcript_segments("session-test-1")
        assert len(all_segs) == 3
        assert all_segs[0]["text"] == "Let us discuss the database failover strategy."

        # 5. Temporal Window Slicing [T_note - 45s, T_note + 15s]
        # Query at center = 62000 ms (around Alice's note)
        window = await repo.get_temporal_window(
            session_id="session-test-1",
            center_ms=62000,
            window_before_ms=45000,
            window_after_ms=15000,
        )
        # Segments in [17000, 77000] should include seg3 (60000-65000), but not seg1 or seg2 (1000-9000)
        assert len(window) == 1
        assert window[0]["speaker_label"] == "Alice"

        # 6. Stop Session
        stopped = await repo.stop_session("session-test-1", ended_at=1070000)
        assert stopped is True
        s = await repo.get_session("session-test-1")
        assert s is not None
        assert s["status"] == "stopped"


if __name__ == "__main__":
    asyncio.run(test_database_initialization_and_operations())
    print("Database test passed successfully!")
