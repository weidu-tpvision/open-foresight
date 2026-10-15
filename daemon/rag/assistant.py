"""Multi-Source Live Assistant answering user queries grounded in local documents and meeting transcripts."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional, Any

from daemon.asr.timestamps import format_timestamp_ms
from daemon.db.repository import DatabaseRepository
from daemon.llm.client import GemmaClient
from daemon.rag.vector_store import VectorStore, SearchResult

logger = logging.getLogger(__name__)

ASSISTANT_SYSTEM_PROMPT = """You are the Live Assistant for Google AI Edge Foresight on NVIDIA Spark.
Your job is to answer the user's questions grounded STRICTLY on the retrieved local documents and meeting transcripts provided below.

INSTRUCTIONS:
1. Provide a direct, concise, and professional answer.
2. Every key factual claim or technical decision MUST include a citation badge:
   - For local documents: [doc: filename.md#Section] or [doc: filename.pdf#Page N]
   - For meeting transcripts: [ref: MM:SS (Speaker)]
3. If the retrieved context does not contain the answer, explicitly state:
   "Based on the indexed local documents and meeting transcripts, I could not find information regarding [topic]."
4. Never invent details, numbers, or architecture topologies not present in the sources."""


@dataclass
class Citation:
    source_type: str  # 'document' or 'transcript'
    source_label: str
    text_snippet: str
    score: float


@dataclass
class AssistantResponse:
    query: str
    answer_markdown: str
    citations: list[Citation]


class LiveAssistant:
    """Multi-source RAG assistant combining local documents and meeting transcripts."""

    def __init__(
        self,
        vector_store: VectorStore,
        repository: DatabaseRepository,
        llm_client: Optional[GemmaClient] = None,
    ):
        self.vector_store = vector_store
        self.repository = repository
        self.llm = llm_client or GemmaClient()

    async def answer_query(
        self,
        query: str,
        session_id: Optional[str] = None,
        top_k_docs: int = 4,
        top_k_transcripts: int = 3,
    ) -> AssistantResponse:
        """Retrieve relevant document chunks and transcript segments, then synthesize answer."""
        citations: list[Citation] = []
        context_blocks: list[str] = []

        # 1. Retrieve from local knowledge documents via 768-D vector search
        doc_results: list[SearchResult] = await self.vector_store.similarity_search(
            query=query, top_k=top_k_docs
        )

        for res in doc_results:
            source_tag = f"{res.file_name}#{res.section_title or f'Page {res.page_number}'}"
            context_blocks.append(
                f"[DOCUMENT CHUNK: {source_tag} (Similarity: {res.score:.2f})]\n{res.text}"
            )
            citations.append(
                Citation(
                    source_type="document",
                    source_label=source_tag,
                    text_snippet=res.text[:200] + "..." if len(res.text) > 200 else res.text,
                    score=res.score,
                )
            )

        # 2. Retrieve from meeting transcripts (current session or recent sessions)
        if session_id:
            # Simple keyword or segment matching in current session
            all_segs = await self.repository.get_transcript_segments(session_id)
            if all_segs:
                # Rank transcript segments by lexical / word overlap with query
                query_words = set(query.lower().split())
                scored_segs = []
                for s in all_segs:
                    seg_words = set(s["text"].lower().split())
                    overlap = len(query_words.intersection(seg_words))
                    if overlap > 0:
                        scored_segs.append((overlap, s))

                scored_segs.sort(key=lambda x: x[0], reverse=True)
                for _, seg in scored_segs[:top_k_transcripts]:
                    ts = format_timestamp_ms(seg["start_ms"])
                    tag = f"{ts} ({seg['speaker_label']})"
                    context_blocks.append(
                        f"[TRANSCRIPT UTTERANCE: ref: {tag}]\n\"{seg['text']}\""
                    )
                    citations.append(
                        Citation(
                            source_type="transcript",
                            source_label=tag,
                            text_snippet=seg["text"],
                            score=1.0,
                        )
                    )

        # 3. If no context found at all
        if not context_blocks:
            prompt = f"""[USER QUESTION]
{query}

[NOTICE]
No relevant local documents or meeting transcripts were found. Reply informing the user that no records contain this information."""
            answer = await self.llm.generate(prompt=prompt, system_prompt=ASSISTANT_SYSTEM_PROMPT)
            return AssistantResponse(query=query, answer_markdown=answer, citations=[])

        # 4. Construct grounded synthesis prompt
        context_text = "\n\n".join(context_blocks)
        user_prompt = f"""[RETRIEVED KNOWLEDGE CONTEXT]
{context_text}

[USER QUESTION]
{query}

[ANSWER INSTRUCTIONS]
Answer the user's question clearly, citing the exact document or transcript sources."""

        answer = await self.llm.generate(
            prompt=user_prompt, system_prompt=ASSISTANT_SYSTEM_PROMPT
        )

        return AssistantResponse(
            query=query,
            answer_markdown=answer,
            citations=citations,
        )
