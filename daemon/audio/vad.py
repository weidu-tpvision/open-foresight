"""Voice Activity Detection (VAD) segmenter using Silero VAD."""

from __future__ import annotations

import collections
import logging
from dataclasses import dataclass
from typing import Iterator, Optional
import numpy as np
import torch
from silero_vad import load_silero_vad

logger = logging.getLogger(__name__)


@dataclass
class SpeechSegment:
    session_id: str
    channel: str  # 'local', 'remote', 'mixed'
    start_ms: int
    end_ms: int
    audio_pcm: np.ndarray  # float32 array sampled at 16000Hz


class VADSegmenter:
    """
    Streaming speech segmenter using Silero VAD.
    Processes 16kHz audio in 512-sample (32ms) frames.
    Maintains pre-speech padding and hangover buffering to avoid clipping.
    """

    def __init__(
        self,
        threshold: float = 0.5,
        min_speech_duration_ms: int = 250,
        min_silence_duration_ms: int = 400,
        pre_speech_padding_ms: int = 200,
        sample_rate: int = 16000,
        device: str = "cpu",
    ):
        self.sample_rate = sample_rate
        self.frame_size = 512  # Silero optimal frame size at 16kHz
        self.frame_duration_ms = (self.frame_size / self.sample_rate) * 1000.0  # 32ms

        self.threshold = threshold
        self.min_speech_frames = int(min_speech_duration_ms / self.frame_duration_ms)
        self.min_silence_frames = int(min_silence_duration_ms / self.frame_duration_ms)
        self.pre_padding_frames = int(pre_speech_padding_ms / self.frame_duration_ms)

        self.device = device
        self.model = load_silero_vad()
        if device.startswith("cuda") and torch.cuda.is_available():
            self.model = self.model.to(device)

        self.reset()

    def reset(self) -> None:
        """Reset internal streaming state."""
        self.model.reset_states()
        self.pre_buffer = collections.deque(maxlen=max(1, self.pre_padding_frames))
        self.current_speech_frames: list[np.ndarray] = []
        self.is_speaking = False
        self.silence_counter = 0
        self.speech_start_ms: Optional[int] = None
        self.last_frame_ms = 0
        self.residual_bytes = b""

    def process_pcm_bytes(
        self,
        pcm_bytes: bytes,
        current_time_ms: int,
        session_id: str = "",
        channel: str = "mixed",
    ) -> list[SpeechSegment]:
        """
        Accepts raw 16-bit mono 16kHz PCM bytes.
        Returns any newly completed speech segments.
        """
        # Append to any residual bytes from previous call
        data = self.residual_bytes + pcm_bytes
        # 16-bit PCM = 2 bytes per sample
        bytes_per_frame = self.frame_size * 2
        num_frames = len(data) // bytes_per_frame

        if num_frames == 0:
            self.residual_bytes = data
            return []

        usable_bytes = num_frames * bytes_per_frame
        self.residual_bytes = data[usable_bytes:]

        # Convert to float32 numpy array normalized to [-1.0, 1.0]
        pcm_int16 = np.frombuffer(data[:usable_bytes], dtype=np.int16)
        pcm_float32 = pcm_int16.astype(np.float32) / 32768.0

        emitted_segments: list[SpeechSegment] = []

        # Calculate time step per frame
        for i in range(num_frames):
            frame_float = pcm_float32[i * self.frame_size : (i + 1) * self.frame_size]
            frame_start_ms = int(
                current_time_ms - (num_frames - i) * self.frame_duration_ms
            )

            tensor_frame = torch.from_numpy(frame_float)
            if self.device.startswith("cuda"):
                tensor_frame = tensor_frame.to(self.device)

            with torch.no_grad():
                prob = self.model(tensor_frame, self.sample_rate).item()

            if prob >= self.threshold:
                if not self.is_speaking:
                    # Speech onset detected
                    self.is_speaking = True
                    self.silence_counter = 0
                    self.speech_start_ms = max(
                        0,
                        frame_start_ms
                        - int(len(self.pre_buffer) * self.frame_duration_ms),
                    )
                    # Include pre-padding frames
                    self.current_speech_frames = list(self.pre_buffer)
                self.current_speech_frames.append(frame_float)
                self.silence_counter = 0
            else:
                if self.is_speaking:
                    self.current_speech_frames.append(frame_float)
                    self.silence_counter += 1
                    if self.silence_counter >= self.min_silence_frames:
                        # Speech pause / offset confirmed
                        end_ms = frame_start_ms
                        # Check minimum duration requirement
                        if len(self.current_speech_frames) >= self.min_speech_frames:
                            audio_concat = np.concatenate(self.current_speech_frames)
                            emitted_segments.append(
                                SpeechSegment(
                                    session_id=session_id,
                                    channel=channel,
                                    start_ms=self.speech_start_ms or 0,
                                    end_ms=end_ms,
                                    audio_pcm=audio_concat,
                                )
                            )
                        # Reset for next speech burst
                        self.is_speaking = False
                        self.current_speech_frames = []
                        self.silence_counter = 0
                        self.speech_start_ms = None
                else:
                    self.pre_buffer.append(frame_float)

        return emitted_segments

    def flush(
        self, current_time_ms: int, session_id: str = "", channel: str = "mixed"
    ) -> Optional[SpeechSegment]:
        """Flush any pending speech buffer upon meeting stop."""
        if self.is_speaking and len(self.current_speech_frames) >= self.min_speech_frames:
            audio_concat = np.concatenate(self.current_speech_frames)
            seg = SpeechSegment(
                session_id=session_id,
                channel=channel,
                start_ms=self.speech_start_ms or 0,
                end_ms=current_time_ms,
                audio_pcm=audio_concat,
            )
            self.reset()
            return seg
        self.reset()
        return None
