"""Input helpers for externally generated dense descriptions."""

from __future__ import annotations

import re
from pathlib import Path

from .models import DescriptionSegment

_TIMED_BLOCK_RE = re.compile(
    r"^\s*(?P<start>\d{1,2}:\d{2}(?::\d{2})?)\s*-\s*"
    r"(?P<end>\d{1,2}:\d{2}(?::\d{2})?)\s+(?P<description>.*)$",
)
_MARKDOWN_TIME_RANGE_RE = re.compile(r"^\*\*Time Range:\*\*", re.IGNORECASE)


def _is_ignored_markdown_line(line: str) -> bool:
    """Return whether one Markdown metadata line should be ignored."""

    if line.startswith("#"):
        return True
    if _MARKDOWN_TIME_RANGE_RE.match(line):
        return True
    return False


def _timecode_to_seconds(value: str) -> float:
    """Convert MM:SS or HH:MM:SS into seconds."""

    parts = [int(part) for part in value.split(":")]
    if len(parts) == 2:
        minutes, seconds = parts
        return float(minutes * 60 + seconds)
    if len(parts) == 3:
        hours, minutes, seconds = parts
        return float(hours * 3600 + minutes * 60 + seconds)
    raise ValueError(f"Unsupported timecode: {value}")


def _iter_blocks(text: str) -> list[str]:
    """Split description text into blank-line separated blocks."""

    blocks: list[str] = []
    current: list[str] = []
    for line in text.splitlines():
        stripped = line.strip()
        if stripped and _is_ignored_markdown_line(stripped):
            if current:
                blocks.append("\n".join(current))
                current = []
            continue
        if not stripped:
            if current:
                blocks.append("\n".join(current))
                current = []
            continue
        current.append(stripped)
    if current:
        blocks.append("\n".join(current))
    return blocks


def parse_description_text(text: str) -> list[DescriptionSegment]:
    """Parse timestamped dense descriptions into segment objects."""

    segments: list[DescriptionSegment] = []
    for block in _iter_blocks(text):
        first_line, *continuation = block.splitlines()
        match = _TIMED_BLOCK_RE.match(first_line)
        if not match:
            raise ValueError(f"Description block is missing a leading time range: {first_line}")
        description_parts = [match.group("description"), *continuation]
        description = " ".join(part.strip() for part in description_parts if part.strip())
        segments.append(
            DescriptionSegment(
                start_s=_timecode_to_seconds(match.group("start")),
                end_s=_timecode_to_seconds(match.group("end")),
                description=description,
            ),
        )
    return segments


def load_description_segments(path: str | Path) -> list[DescriptionSegment]:
    """Load timestamped dense descriptions from a text file."""

    return parse_description_text(Path(path).read_text(encoding="utf-8"))
