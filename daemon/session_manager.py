"""Live Session Manager coordinating PipeWire Capture, VAD, ASR, and WebSocket Broadcast."""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Any, Optional
from dataclasses import asdict

import numpy as np
from daemon.audio.capture import AudioPacket, DualChannelCaptureManager, discover_audio_nodes
from daemon.audio.vad import VADSegmenter, SpeechSegment
from daemon.asr.engine import ASREngine
from daemon.asr.timestamps import format_timestamp_ms
from daemon.db.repository import DatabaseRepository

logger = logging.getLogger(__name__)


class LiveSession:
    """Represents an active recording and transcription session."""

    def __init__(
        self,
        session_id: str,
        title: str,
        repository: DatabaseRepository,
        asr_engine: ASREngine,
        mic_node: Optional[str] = None,
        sink_node: Optional[str] = None,
    ):
        self.session_id = session_id
        self.title = title
        self.repository = repository
        self.asr_engine = asr_engine
        self.mic_node = mic_node
        self.sink_node = sink_node

        self.start_epoch_ms = int(time.time() * 1000)
        self.capture_manager = DualChannelCaptureManager(
            session_id=self.session_id,
            session_start_epoch_ms=self.start_epoch_ms,
            mic_node=self.mic_node,
            sink_node=self.sink_node,
        )

        # Independent VAD segmenters for local and remote streams
        self.vad_local = VADSegmenter(sample_rate=16000)
        self.vad_remote = VADSegmenter(sample_rate=16000)

        self._active_connections: set[Any] = set()
        self._worker_task: Optional[asyncio.Task] = None
        self._running = False

    def register_client(self, websocket: Any) -> None:
        self._active_connections.add(websocket)

    def unregister_client(self, websocket: Any) -> None:
        self._active_connections.discard(websocket)

    async def broadcast(self, message: dict[str, Any]) -> None:
        """Broadcast payload to all subscribed WebSocket clients."""
        dead_clients = []
        for ws in self._active_connections:
            try:
                await ws.send_json(message)
            except Exception:
                dead_clients.append(ws)
        for dead in dead_clients:
            self._active_connections.discard(dead)

    async def start(self) -> None:
        """Start DB session, PipeWire capture streams, and processing worker."""
        if self._running:
            return

        await self.repository.create_session(
            session_id=self.session_id,
            title=self.title,
            started_at=self.start_epoch_ms,
        )

        await self.capture_manager.start()
        self._running = True
        self._worker_task = asyncio.create_task(self._process_audio_loop())
        logger.info(f"LiveSession {self.session_id} started successfully.")

        # Audio level metering state
        self._last_level_broadcast_time = 0.0

    async def _process_audio_loop(self) -> None:
        """Consume raw audio packets from capture queue, run VAD and ASR, and emit level telemetry."""
        try:
            while self._running:
                packet = await self.capture_manager.audio_queue.get()
                channel = packet.channel
                vad = self.vad_local if channel == "local" else self.vad_remote
                speaker_label = "Me" if channel == "local" else "Remote"

                # Compute RMS audio amplitude level for waveform telemetry
                if self._active_connections and len(packet.pcm_bytes) >= 2:
                    now = time.time()
                    if now - self._last_level_broadcast_time >= 0.08:  # ~12.5 Hz update
                        self._last_level_broadcast_time = now
                        pcm_arr = np.frombuffer(packet.pcm_bytes, dtype=np.int16).astype(np.float32) / 32768.0
                        rms = float(np.sqrt(np.mean(pcm_arr**2)))
                        # Scale to approximate peak normalized range [0.0, 1.0]
                        norm_level = min(1.0, max(0.0, rms * 4.5))
                        asyncio.create_task(
                            self.broadcast(
                                {
                                    "type": "audio_level",
                                    "channel": channel,
                                    "level": norm_level,
                                }
                            )
                        )

                segments: list[SpeechSegment] = vad.process_pcm_bytes(
                    pcm_bytes=packet.pcm_bytes,
                    current_time_ms=packet.timestamp_ms,
                    session_id=self.session_id,
                    channel=channel,
                )

                for seg in segments:
                    await self._process_speech_segment(seg, speaker_label)

                self.capture_manager.audio_queue.task_done()
        except asyncio.CancelledError:
            pass
        except Exception as e:
            logger.error(f"Error in LiveSession audio processing loop: {e}")

    async def _process_speech_segment(
        self, seg: SpeechSegment, speaker_label: str
    ) -> None:
        """Transcribe speech burst and commit to DB and WebSocket clients."""
        chunks = await asyncio.to_thread(
            self.asr_engine.transcribe,
            audio_pcm=seg.audio_pcm,
            base_start_ms=seg.start_ms,
        )

        for chunk in chunks:
            if not chunk.text:
                continue

            # Persist to SQLite
            seg_id = await self.repository.add_transcript_segment(
                session_id=self.session_id,
                start_ms=chunk.start_ms,
                end_ms=chunk.end_ms,
                text=chunk.text,
                speaker_channel=seg.channel,
                speaker_label=speaker_label,
                confidence=chunk.confidence,
            )

            # Broadcast live segment to UI
            payload = {
                "type": "transcript_segment",
                "id": seg_id,
                "session_id": self.session_id,
                "start_ms": chunk.start_ms,
                "end_ms": chunk.end_ms,
                "speaker_channel": seg.channel,
                "speaker_label": speaker_label,
                "text": chunk.text,
                "timestamp_formatted": format_timestamp_ms(chunk.start_ms),
            }
            await self.broadcast(payload)

    async def stop(self) -> None:
        """Stop session, flush VAD buffers, and finalize."""
        if not self._running:
            return

        self._running = False
        await self.capture_manager.stop()

        if self._worker_task:
            self._worker_task.cancel()
            try:
                await self._worker_task
            except asyncio.CancelledError:
                pass
            self._worker_task = None

        # Flush any remaining speech from VAD
        now_offset_ms = int(time.time() * 1000) - self.start_epoch_ms
        for channel, vad, speaker_label in [
            ("local", self.vad_local, "Me"),
            ("remote", self.vad_remote, "Remote"),
        ]:
            flushed = vad.flush(now_offset_ms, self.session_id, channel)
            if flushed:
                await self._process_speech_segment(flushed, speaker_label)

        # Mark stopped in DB
        ended_at_epoch = int(time.time() * 1000)
        await self.repository.stop_session(self.session_id, ended_at=ended_at_epoch)

        await self.broadcast(
            {
                "type": "session_stopped",
                "session_id": self.session_id,
                "ended_at": ended_at_epoch,
            }
        )
        logger.info(f"LiveSession {self.session_id} finalized.")
