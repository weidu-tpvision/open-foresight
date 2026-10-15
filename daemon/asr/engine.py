"""Automated Speech Recognition (ASR) Engine powered by Whisper on GB10 CUDA."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional
import numpy as np
import torch
from transformers import pipeline

logger = logging.getLogger(__name__)


@dataclass
class ASRChunk:
    text: str
    start_ms: int
    end_ms: int
    confidence: float = 1.0


class ASREngine:
    """
    Speech recognition engine leveraging HuggingFace Whisper models
    accelerated by NVIDIA GB10 CUDA cores.
    """

    def __init__(
        self,
        model_id: str = "openai/whisper-tiny",
        device: Optional[str] = None,
        language: str = "en",
    ):
        if device is None:
            self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        self.model_id = model_id
        self.language = language

        logger.info(
            f"Initializing ASREngine with model={model_id} on device={self.device}..."
        )
        torch_dtype = torch.float16 if "cuda" in self.device else torch.float32

        self.pipe = pipeline(
            "automatic-speech-recognition",
            model=model_id,
            device=self.device,
            torch_dtype=torch_dtype,
        )
        logger.info("ASREngine initialized successfully.")

    def transcribe(
        self,
        audio_pcm: np.ndarray,
        base_start_ms: int = 0,
        sample_rate: int = 16000,
    ) -> list[ASRChunk]:
        """
        Transcribe a 1D float32 audio array and return timestamped text chunks
        offset by base_start_ms.
        """
        if len(audio_pcm) == 0:
            return []

        # Ensure float32 1D numpy array
        if audio_pcm.dtype != np.float32:
            audio_pcm = audio_pcm.astype(np.float32)

        try:
            result = self.pipe(
                {"raw": audio_pcm, "sampling_rate": sample_rate},
                return_timestamps=True,
                generate_kwargs={"language": self.language, "task": "transcribe"},
            )
        except Exception as e:
            logger.error(f"Error during ASR transcription: {e}")
            return []

        chunks: list[ASRChunk] = []

        # If chunks are present in output
        raw_chunks = result.get("chunks", [])
        if raw_chunks:
            for c in raw_chunks:
                text = c.get("text", "").strip()
                if not text:
                    continue

                ts = c.get("timestamp", (0.0, 0.0))
                start_sec = ts[0] if ts[0] is not None else 0.0
                end_sec = ts[1] if ts[1] is not None else (len(audio_pcm) / sample_rate)

                chunk_start_ms = base_start_ms + int(start_sec * 1000)
                chunk_end_ms = base_start_ms + int(end_sec * 1000)

                chunks.append(
                    ASRChunk(
                        text=text,
                        start_ms=chunk_start_ms,
                        end_ms=max(chunk_start_ms + 100, chunk_end_ms),
                    )
                )
        else:
            full_text = result.get("text", "").strip()
            if full_text:
                duration_ms = int((len(audio_pcm) / sample_rate) * 1000)
                chunks.append(
                    ASRChunk(
                        text=full_text,
                        start_ms=base_start_ms,
                        end_ms=base_start_ms + duration_ms,
                    )
                )

        return chunks
