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
    kind: str
    classification: str
    item_id: str | None
    part: str | None
    line_idx: int
    char_pos: int
    full_text_len: int
    snippet: str
