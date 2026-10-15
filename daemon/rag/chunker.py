"""Multimodal and text document chunkers for local knowledge ingestion."""

from __future__ import annotations

import re
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional
import pypdf

logger = logging.getLogger(__name__)


@dataclass
class DocumentChunk:
    chunk_index: int
    text: str
    page_number: int = 1
    section_title: str = ""


class DocumentChunker:
    """Parses and chunks Markdown, PDF, and text documents for EmbeddingGemma 2 indexing."""

    def __init__(self, target_chunk_size: int = 600, chunk_overlap: int = 80):
        self.chunk_size = target_chunk_size
        self.overlap = chunk_overlap

    def chunk_file(self, file_path: str | Path) -> list[DocumentChunk]:
        """Inspect file extension and dispatch to the appropriate chunking strategy."""
        path = Path(file_path)
        if not path.is_file():
            raise FileNotFoundError(f"File not found: {file_path}")

        ext = path.suffix.lower()
        if ext == ".pdf":
            return self._chunk_pdf(path)
        elif ext in [".md", ".markdown"]:
            return self._chunk_markdown(path)
        else:
            return self._chunk_plain_text(path)

    def _chunk_pdf(self, path: Path) -> list[DocumentChunk]:
        """Extract text from PDF pages and chunk with page provenance."""
        chunks: list[DocumentChunk] = []
        chunk_idx = 0

        try:
            reader = pypdf.PdfReader(str(path))
            for page_num, page in enumerate(reader.pages, start=1):
                page_text = page.extract_text() or ""
                page_text = page_text.strip()
                if not page_text:
                    continue

                page_chunks = self._split_text_sliding(page_text)
                for txt in page_chunks:
                    chunks.append(
                        DocumentChunk(
                            chunk_index=chunk_idx,
                            text=txt,
                            page_number=page_num,
                            section_title=f"Page {page_num}",
                        )
                    )
                    chunk_idx += 1
        except Exception as e:
            logger.error(f"Error parsing PDF {path}: {e}")

        return chunks

    def _chunk_markdown(self, path: Path) -> list[DocumentChunk]:
        """Chunk Markdown files preserving headings as section context."""
        try:
            with open(path, "r", encoding="utf-8", errors="replace") as f:
                content = f.read()
        except Exception as e:
            logger.error(f"Error reading Markdown file {path}: {e}")
            return []

        chunks: list[DocumentChunk] = []
        lines = content.splitlines()

        current_heading = "Overview"
        current_buffer: list[str] = []
        chunk_idx = 0

        def flush_buffer(heading: str):
            nonlocal chunk_idx
            text = "\n".join(current_buffer).strip()
            if text:
                sub_chunks = self._split_text_sliding(text)
                for sc in sub_chunks:
                    chunks.append(
                        DocumentChunk(
                            chunk_index=chunk_idx,
                            text=sc,
                            page_number=1,
                            section_title=heading,
                        )
                    )
                    chunk_idx += 1
            current_buffer.clear()

        for line in lines:
            header_match = re.match(r"^(#{1,4})\s+(.+)$", line)
            if header_match:
                flush_buffer(current_heading)
                current_heading = header_match.group(2).strip()
                current_buffer.append(line)
            else:
                current_buffer.append(line)

        flush_buffer(current_heading)
        return chunks

    def _chunk_plain_text(self, path: Path) -> list[DocumentChunk]:
        """Fallback chunker for text, code, or structured logs."""
        try:
            with open(path, "r", encoding="utf-8", errors="replace") as f:
                content = f.read()
        except Exception as e:
            logger.error(f"Error reading text file {path}: {e}")
            return []

        chunks: list[DocumentChunk] = []
        sub_chunks = self._split_text_sliding(content)
        for idx, sc in enumerate(sub_chunks):
            chunks.append(
                DocumentChunk(
                    chunk_index=idx,
                    text=sc,
                    page_number=1,
                    section_title="General",
                )
            )
        return chunks

    def _split_text_sliding(self, text: str) -> list[str]:
        """Sliding window text split with character overlap."""
        text = text.strip()
        if len(text) <= self.chunk_size:
            return [text] if text else []

        results: list[str] = []
        start = 0
        while start < len(text):
            end = start + self.chunk_size
            if end >= len(text):
                results.append(text[start:].strip())
                break

            # Try to break at paragraph or newline or space
            break_pos = text.rfind("\n\n", start, end)
            if break_pos == -1 or break_pos < start + (self.chunk_size // 2):
                break_pos = text.rfind(". ", start, end)
            if break_pos == -1 or break_pos < start + (self.chunk_size // 2):
                break_pos = text.rfind(" ", start, end)
            if break_pos == -1 or break_pos <= start:
                break_pos = end

            results.append(text[start:break_pos].strip())
            start = max(start + 1, break_pos - self.overlap)

        return [r for r in results if r]
