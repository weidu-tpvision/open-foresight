"""PipeWire loopback and microphone audio capture using native pw-record streaming."""

from __future__ import annotations

import asyncio
import logging
import re
import shutil
import subprocess
import time
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class AudioNode:
    id: Optional[int]
    name: str
    description: str
    media_class: str  # 'Audio/Source' or 'Audio/Sink'


@dataclass
class AudioPacket:
    session_id: str
    channel: str  # 'local', 'remote'
    timestamp_ms: int  # Offset from session start
    pcm_bytes: bytes


def discover_audio_nodes() -> list[AudioNode]:
    """Scan PipeWire nodes and return all available audio sources and sinks."""
    if not shutil.which("pw-cli"):
        logger.warning("pw-cli not found in system PATH.")
        return []

    try:
        output = subprocess.check_output(
            ["pw-cli", "list-objects", "Node"], text=True, stderr=subprocess.DEVNULL
        )
    except Exception as e:
        logger.error(f"Error querying PipeWire nodes: {e}")
        return []

    nodes: list[AudioNode] = []
    current_name = None
    current_desc = None
    current_class = None
    current_id = None

    for line in output.splitlines():
        line = line.strip()
        if line.startswith("id ") and "type PipeWire:Interface:Node" in line:
            if current_name and current_class in ["Audio/Source", "Audio/Sink"]:
                nodes.append(
                    AudioNode(
                        id=current_id,
                        name=current_name,
                        description=current_desc or current_name,
                        media_class=current_class,
                    )
                )
            # Reset
            current_name = None
            current_desc = None
            current_class = None
            id_match = re.search(r"id\s+(\d+)", line)
            current_id = int(id_match.group(1)) if id_match else None

        m_name = re.search(r'node\.name\s*=\s*"([^"]+)"', line)
        if m_name:
            current_name = m_name.group(1)

        m_desc = re.search(r'node\.description\s*=\s*"([^"]+)"', line)
        if m_desc:
            current_desc = m_desc.group(1)

        m_class = re.search(r'media\.class\s*=\s*"([^"]+)"', line)
        if m_class:
            current_class = m_class.group(1)

    if current_name and current_class in ["Audio/Source", "Audio/Sink"]:
        nodes.append(
            AudioNode(
                id=current_id,
                name=current_name,
                description=current_desc or current_name,
                media_class=current_class,
            )
        )

    return nodes


class PipeWireAudioStream:
    """
    Spawns and manages a pw-record subprocess streaming raw 16kHz 16-bit mono PCM to an asyncio Queue.
    """

    def __init__(
        self,
        session_id: str,
        channel: str,  # 'local' or 'remote'
        target_node: str,
        output_queue: asyncio.Queue[AudioPacket],
        session_start_epoch_ms: int,
        sample_rate: int = 16000,
        chunk_samples: int = 1024,  # 64ms per chunk
    ):
        self.session_id = session_id
        self.channel = channel
        self.target_node = target_node
        self.output_queue = output_queue
        self.session_start_epoch_ms = session_start_epoch_ms
        self.sample_rate = sample_rate
        self.chunk_bytes = chunk_samples * 2  # 16-bit PCM = 2 bytes per sample

        self.process: Optional[asyncio.subprocess.Process] = None
        self._reader_task: Optional[asyncio.Task] = None
        self._running = False

    async def start(self) -> None:
        """Start pw-record subprocess and background reader loop."""
        if self._running:
            return

        cmd = [
            "pw-record",
            "--target",
            self.target_node,
            "--rate",
            str(self.sample_rate),
            "--channels",
            "1",
            "--format",
            "s16",
            "-",
        ]

        logger.info(f"Starting audio stream on {self.target_node} ({self.channel})...")
        self.process = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.DEVNULL,
        )
        self._running = True
        self._reader_task = asyncio.create_task(self._read_loop())

    async def _read_loop(self) -> None:
        """Continuously reads PCM chunks from pw-record stdout."""
        try:
            while self._running and self.process and self.process.stdout:
                data = await self.process.stdout.read(self.chunk_bytes)
                if not data:
                    break

                now_ms = int(time.time() * 1000)
                offset_ms = max(0, now_ms - self.session_start_epoch_ms)

                packet = AudioPacket(
                    session_id=self.session_id,
                    channel=self.channel,
                    timestamp_ms=offset_ms,
                    pcm_bytes=data,
                )
                await self.output_queue.put(packet)
        except asyncio.CancelledError:
            pass
        except Exception as e:
            logger.error(f"Error in stream reader loop ({self.channel}): {e}")

    async def stop(self) -> None:
        """Terminate the pw-record process and reader task."""
        self._running = False
        if self._reader_task:
            self._reader_task.cancel()
            try:
                await self._reader_task
            except asyncio.CancelledError:
                pass
            self._reader_task = None

        if self.process:
            try:
                self.process.terminate()
                await self.process.wait()
            except Exception:
                pass
            self.process = None
        logger.info(f"Audio stream stopped for {self.channel}.")


class DualChannelCaptureManager:
    """
    Manages dual-channel audio capture (Local Mic + Remote Desktop Sink Monitor).
    """

    def __init__(
        self,
        session_id: str,
        session_start_epoch_ms: int,
        mic_node: Optional[str] = None,
        sink_node: Optional[str] = None,
    ):
        self.session_id = session_id
        self.session_start_epoch_ms = session_start_epoch_ms
        self.audio_queue: asyncio.Queue[AudioPacket] = asyncio.Queue()

        # If not explicitly provided, discover system defaults
        nodes = discover_audio_nodes()
        if not mic_node:
            sources = [n for n in nodes if n.media_class == "Audio/Source"]
            mic_node = sources[0].name if sources else None
        if not sink_node:
            sinks = [n for n in nodes if n.media_class == "Audio/Sink"]
            sink_node = sinks[0].name if sinks else None

        self.mic_node = mic_node
        self.sink_node = sink_node

        self.mic_stream: Optional[PipeWireAudioStream] = None
        self.sink_stream: Optional[PipeWireAudioStream] = None

    async def start(self) -> None:
        """Start both capture streams."""
        if self.mic_node:
            self.mic_stream = PipeWireAudioStream(
                session_id=self.session_id,
                channel="local",
                target_node=self.mic_node,
                output_queue=self.audio_queue,
                session_start_epoch_ms=self.session_start_epoch_ms,
            )
            await self.mic_stream.start()

        if self.sink_node:
            self.sink_stream = PipeWireAudioStream(
                session_id=self.session_id,
                channel="remote",
                target_node=self.sink_node,
                output_queue=self.audio_queue,
                session_start_epoch_ms=self.session_start_epoch_ms,
            )
            await self.sink_stream.start()

    async def stop(self) -> None:
        """Stop both streams."""
        if self.mic_stream:
            await self.mic_stream.stop()
            self.mic_stream = None
        if self.sink_stream:
            await self.sink_stream.stop()
            self.sink_stream = None
