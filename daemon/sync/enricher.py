"""The Shorthand Synchronization Engine.
Correlates user shorthand notes to conversational transcript windows [T - 45s, T + 15s]
and synthesizes grounded executive meeting minutes using Gemma 4.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional, Any

from daemon.asr.timestamps import format_timestamp_ms
from daemon.db.repository import DatabaseRepository
from daemon.llm.client import GemmaClient

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are an executive note-taking copilot for Google AI Edge Foresight.
Your task is to expand the user's brief, rapid shorthand notes into polished, professional meeting minutes.

CRITICAL RULES:
1. Base your expansion STRICTLY on the facts, technical specifics, names, and decisions in the provided [CONVERSATION WINDOW].
2. Never hallucinate or assume details not spoken in the audio transcript.
3. For each bullet point, produce a clear summary with:
   - Bold title/topic
   - Specific details and context agreed upon in the conversation
   - Owner / Assignee (if mentioned)
   - Deadline / Target date (if mentioned)
   - Audio Citation with timestamp: [ref: MM:SS]
4. Keep the output in clean, crisp Markdown bullet format."""


@dataclass
class EnrichedBullet:
    shorthand: str
    timestamp_ms: int
    enriched_markdown: str
    correlated_segments_count: int


class ShorthandEnricher:
    """Orchestrates temporal window extraction and grounded note synthesis using Gemma 4."""

    def __init__(
        self,
        repository: DatabaseRepository,
        llm_client: Optional[GemmaClient] = None,
    ):
        self.repository = repository
        self.llm = llm_client or GemmaClient()

    async def enrich_single_note(
        self,
        session_id: str,
        note_text: str,
        note_timestamp_ms: int,
        model: Optional[str] = None,
    ) -> EnrichedBullet:
        """Enrich a single shorthand bullet using its 60-second temporal audio window."""
        window_segments = await self.repository.get_temporal_window(
            session_id=session_id,
            center_ms=note_timestamp_ms,
            window_before_ms=45000,
            window_after_ms=15000,
        )

        formatted_time = format_timestamp_ms(note_timestamp_ms)

        if not window_segments:
            # Fallback if no audio was spoken during this note window
            prompt = f"""[USER SHORTHAND (Recorded at {formatted_time})]
{note_text}

[NOTICE]
No dialogue was spoken during the surrounding window. Clean up and format this note professionally as a standalone action item with citation [ref: {formatted_time}]."""

            try:
                expansion = await self.llm.generate(
                    prompt=prompt, system_prompt=SYSTEM_PROMPT, model=model
                )
            except Exception:
                expansion = f"- **{note_text.lstrip('-* ')}** *(Ref: {formatted_time})*"

            return EnrichedBullet(
                shorthand=note_text,
                timestamp_ms=note_timestamp_ms,
                enriched_markdown=expansion,
                correlated_segments_count=0,
            )

        # Build transcript window dialogue text
        first_ms = window_segments[0]["start_ms"]
        last_ms = window_segments[-1]["end_ms"]
        time_range = f"{format_timestamp_ms(first_ms)} - {format_timestamp_ms(last_ms)}"

        transcript_lines = []
        for s in window_segments:
            ts = format_timestamp_ms(s["start_ms"])
            speaker = s.get("speaker_label", "Speaker")
            text = s.get("text", "").strip()
            transcript_lines.append(f"[{ts}] {speaker}: \"{text}\"")

        dialogue_block = "\n".join(transcript_lines)

        user_prompt = f"""[CORRELATED CONVERSATION WINDOW ({time_range})]
{dialogue_block}

[USER SHORTHAND (Typed at {formatted_time})]
{note_text}

[INSTRUCTIONS]
Synthesize the shorthand note into an executive meeting bullet point strictly grounded in the dialogue above. Include Owner, Deadline, and Citation."""

        try:
            expansion = await self.llm.generate(
                prompt=user_prompt, system_prompt=SYSTEM_PROMPT, model=model
            )
        except Exception as e:
            logger.error(f"Gemma 4 expansion failed: {e}")
            expansion = f"- **{note_text.lstrip('-* ')}** *(Correlated with {len(window_segments)} transcript segments | Ref: {formatted_time})*"

        return EnrichedBullet(
            shorthand=note_text,
            timestamp_ms=note_timestamp_ms,
            enriched_markdown=expansion,
            correlated_segments_count=len(window_segments),
        )

    async def enrich_meeting_notes(
        self,
        session_id: str,
        raw_notes: str,
        note_timestamps: Optional[dict[Any, int]] = None,
        model: Optional[str] = None,
    ) -> str:
        """
        Takes multi-line shorthand notes and maps each line to its timestamp offset.
        Returns unified executive meeting minutes while preserving user line indexing.
        """
        raw_lines = raw_notes.splitlines()
        non_empty_indices = [i for i, line in enumerate(raw_lines) if line.strip()]
        if not non_empty_indices:
            return "No notes provided to enrich."

        session = await self.repository.get_session(session_id)
        session_duration_ms = 60000
        if session and session.get("started_at"):
            import time

            now_epoch = int(time.time() * 1000)
            session_duration_ms = max(
                10000,
                (session.get("ended_at") or now_epoch) - session["started_at"],
            )

        enriched_sections: list[str] = []

        for line_idx in non_empty_indices:
            line_text = raw_lines[line_idx].strip()
            # Resolve timestamp using the exact original line index from the editor
            if note_timestamps and str(line_idx) in note_timestamps:
                ts_ms = note_timestamps[str(line_idx)]
            elif note_timestamps and line_idx in note_timestamps:
                ts_ms = note_timestamps[line_idx]
            else:
                # Approximate evenly across session
                ts_ms = int((line_idx / max(1, len(raw_lines))) * session_duration_ms)

            bullet = await self.enrich_single_note(
                session_id=session_id,
                note_text=line_text,
                note_timestamp_ms=ts_ms,
                model=model,
            )
            enriched_sections.append(bullet.enriched_markdown)

        return "\n\n".join(enriched_sections)
