from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class _ItemBoundary:
    start: int
    content_start: int
    item_part: str | None
    item_id: str
    line_index: int
    confidence: int
    in_toc_range: bool = False
    toc_like_line: bool = False


@dataclass(frozen=True)
class EmbeddedHeadingHit:
    """Diagnostic record for an embedded heading detection inside item text.

    Attributes:
        kind: Heading token family (for example ``item`` or ``part``).
        classification: Hit classifier label (for example ``cross_ref`` or ``toc_row``).
        item_id: Parsed item id when available.
        part: Parsed part label when available.
        line_idx: Zero-based line index within normalized extractor body.
        char_pos: Character offset within normalized extractor body.
        full_text_len: Full extracted item text length used for contextual bucketing.
        snippet: Local text snippet around the hit.
    """
    kind: str
    classification: str
    item_id: str | None
    part: str | None
    line_idx: int
    char_pos: int
    full_text_len: int
    snippet: str
