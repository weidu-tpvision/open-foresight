"""Test for ASR engine and end-to-end VAD to DB transcription pipeline."""

import pytest
import asyncio
import numpy as np
import tempfile
from pathlib import Path
from daemon.asr.engine import ASREngine
from daemon.asr.timestamps import format_timestamp_ms, parse_timestamp_str
from daemon.db.repository import DatabaseRepository


def test_timestamp_formatting():
    assert format_timestamp_ms(0) == "00:00"
    assert format_timestamp_ms(65000) == "01:05"
    assert format_timestamp_ms(3665000) == "01:01:05"
    assert parse_timestamp_str("01:05") == 65000
    assert parse_timestamp_str("01:01:05") == 3665000


@pytest.mark.asyncio
async def test_asr_engine_transcription_and_db_persistence():
    # 1. Initialize ASR engine on GB10 CUDA
    engine = ASREngine(model_id="openai/whisper-tiny")
    assert engine is not None

    # 2. Synthesize audio buffer (1.5 seconds of 440Hz tone)
    sample_rate = 16000
    duration_s = 1.5
    t = np.linspace(0, duration_s, int(sample_rate * duration_s), endpoint=False)
    synthetic_pcm = (0.5 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)

    # 3. Transcribe audio
    base_start_ms = 12000  # Occurred 12 seconds into meeting
    chunks = engine.transcribe(synthetic_pcm, base_start_ms=base_start_ms)
    assert isinstance(chunks, list)
    # Even if speech model transcribes or emits background token, chunks should have valid bounds
    if chunks:
        assert chunks[0].start_ms >= base_start_ms
        assert chunks[0].end_ms > chunks[0].start_ms

    # 4. Verify DB insertion
    with tempfile.TemporaryDirectory() as tmp_dir:
        db_path = Path(tmp_dir) / "test_pipeline.db"
        repo = DatabaseRepository(db_path=db_path)
        await repo.initialize()

        session = await repo.create_session("session-asr-test", "Pipeline Test")
        for c in chunks:
            await repo.add_transcript_segment(
                session_id=session["id"],
                start_ms=c.start_ms,
                end_ms=c.end_ms,
                text=c.text,
                speaker_channel="local",
                speaker_label="Me",
            )

        persisted = await repo.get_transcript_segments("session-asr-test")
        assert len(persisted) == len(chunks)


if __name__ == "__main__":
    test_timestamp_formatting()
    asyncio.run(test_asr_engine_transcription_and_db_persistence())
    print("ASR tests passed successfully!")
