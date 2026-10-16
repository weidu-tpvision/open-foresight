"""Client for local Gemma 4 and Ollama LLM inference on GB10."""

from __future__ import annotations

import asyncio
import logging
import httpx
from typing import Optional, Any

logger = logging.getLogger(__name__)

DEFAULT_OLLAMA_URL = "http://localhost:11434"
DEFAULT_MODEL = "gemma4:e4b"


class GemmaClient:
    """Async client communicating with local Ollama instance running Gemma 4."""

    def __init__(
        self,
        base_url: str = DEFAULT_OLLAMA_URL,
        model: str = DEFAULT_MODEL,
        timeout_seconds: float = 60.0,
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

    async def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.2,
        model: Optional[str] = None,
    ) -> str:
        """Generate text completion from local Gemma 4."""
        target_model = model or self.model
        payload: dict[str, Any] = {
            "model": target_model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": temperature,
            },
        }

        if system_prompt:
            payload["system"] = system_prompt

        url = f"{self.base_url}/api/generate"
        client = self._get_client()
        try:
            response = await client.post(url, json=payload)
            response.raise_for_status()
            data = response.json()
            return data.get("response", "").strip()
        except Exception as e:
            logger.error(f"Error querying local Gemma 4 at {url}: {e}")
            raise RuntimeError(f"Gemma 4 generation failed: {e}") from e

    async def is_ready(self) -> bool:
        """Verify Ollama server connection and model availability."""
        url = f"{self.base_url}/api/tags"
        client = self._get_client()
        try:
            res = await client.get(url, timeout=5.0)
            if res.status_code == 200:
                models = [m.get("name") for m in res.json().get("models", [])]
                return any(self.model in m for m in models)
            return False
        except Exception:
            return False
