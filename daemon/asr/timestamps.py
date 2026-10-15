"""Audio timestamp utilities for word-level and segment-level provenance."""

from __future__ import annotations


def format_timestamp_ms(ms: int) -> str:
    """Format milliseconds into MM:SS or HH:MM:SS string."""
    total_seconds = max(0, ms // 1000)
    hours = total_seconds // 3600
    minutes = (total_seconds % 3600) // 60
    seconds = total_seconds % 60

    if hours > 0:
        return f"{hours:02d}:{minutes:02d}:{seconds:02d}"
    return f"{minutes:02d}:{seconds:02d}"


def parse_timestamp_str(timestamp_str: str) -> int:
    """Parse MM:SS or HH:MM:SS into milliseconds."""
    parts = timestamp_str.strip().split(":")
    if len(parts) == 2:
        minutes, seconds = int(parts[0]), int(parts[1])
        return (minutes * 60 + seconds) * 1000
    elif len(parts) == 3:
        hours, minutes, seconds = int(parts[0]), int(parts[1]), int(parts[2])
        return (hours * 3600 + minutes * 60 + seconds) * 1000
    raise ValueError(f"Invalid timestamp format: '{timestamp_str}'")
