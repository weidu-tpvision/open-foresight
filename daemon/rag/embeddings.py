"""Embedding pipeline powered by EmbeddingGemma 2 (768-D dense vectors)."""

from __future__ import annotations

import asyncio
import logging
import httpx
import numpy as np
from typing import Optional, Sequence

logger = logging.getLogger(__name__)

DEFAULT_OLLAMA_URL = "http://localhost:11434"
DEFAULT_EMBED_MODEL = "embeddinggemma-2:latest"


class EmbeddingGemmaClient:
    """Async client generating 768-dimensional dense vectors using EmbeddingGemma 2."""

    def __init__(
        self,
        base_url: str = DEFAULT_OLLAMA_URL,
        model: str = DEFAULT_EMBED_MODEL,
        timeout_seconds: float = 30.0,
    ):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout = timeout_seconds
        self._client: Optional[httpx.AsyncClient] = None
        self._loop: Optional[asyncio.AbstractEventLoop] = None

    def _get_client(self) -> httpx.AsyncClient:
        try:
            current_loop = asyncio.get_running_loop()
        except RuntimeError:
            current_loop = None

        if self._client is None or self._client.is_closed or self._loop != current_loop:
            self._client = httpx.AsyncClient(timeout=self.timeout)
            self._loop = current_loop
        return self._client

    async def close(self) -> None:
        """Close persistent HTTP client session."""
        if self._client and not self._client.is_closed:
            await self._client.aclose()
            self._client = None

    async def embed_texts(self, texts: Sequence[str]) -> list[list[float]]:
        """Compute 768-D dense embeddings for a batch of strings."""
        if not texts:
            return []

        url = f"{self.base_url}/api/embed"
        client = self._get_client()
        try:
            response = await client.post(
                url,
                json={
                    "model": self.model,
                    "input": list(texts),
                },
            )
            response.raise_for_status()
            data = response.json()
            embeddings = data.get("embeddings", [])
            return embeddings
        except Exception as e:
            logger.error(f"Error querying EmbeddingGemma 2: {e}")
            raise RuntimeError(f"EmbeddingGemma 2 failed: {e}") from e

    async def embed_single(self, text: str) -> list[float]:
        """Compute embedding for a single text chunk."""
        embs = await self.embed_texts([text])
        if embs:
            return embs[0]
        return []

    @staticmethod
    def cosine_similarity(v1: Sequence[float], v2: Sequence[float]) -> float:
        """Compute cosine similarity between two vector embeddings."""
        a = np.array(v1, dtype=np.float32)
        b = np.array(v2, dtype=np.float32)
        dot = np.dot(a, b)
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        if norm_a == 0.0 or norm_b == 0.0:
            return 0.0
        return float(dot / (norm_a * norm_b))
