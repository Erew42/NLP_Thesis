from __future__ import annotations

from dataclasses import dataclass
import re
from typing import cast
from typing import Literal

from thesis_pkg.core.sec.extraction import _strip_edgar_metadata


Full10KCleaningContract = Literal["current", "lm2011_paper"]

FULL_10K_CLEANING_CONTRACTS: tuple[Full10KCleaningContract, ...] = (
    "current",
    "lm2011_paper",
)

_TAIL_START_FRACTION = 0.60
_WEAK_CLUSTER_WINDOW_CHARS = 4_000

_EXHIBIT_INDEX_RE = re.compile(r"(?i)\b(?:exhibit index|index to exhibits)\b")
_EX_TAG_RE = re.compile(r"(?i)</?ex-[a-z0-9][a-z0-9.-]*>")
_WEAK_EXHIBIT_LINE_RE = re.compile(
    r"(?i)^\s*exhibit\s+[A-Za-z0-9][A-Za-z0-9().-]*\b",
)


@dataclass(frozen=True)
class ExhibitTailDetection:
    start: int | None
    reason: str
    anchor_text: str | None


@dataclass(frozen=True)
class _LineCandidate:
    start: int
    text: str
    reason: str


def _validate_cleaning_contract(contract: str) -> Full10KCleaningContract:
    if contract not in FULL_10K_CLEANING_CONTRACTS:
        raise ValueError(
            f"Unsupported full-10-K cleaning contract: {contract!r}. "
            f"Expected one of {list(FULL_10K_CLEANING_CONTRACTS)}."
        )
    return cast(Full10KCleaningContract, contract)


def _iter_lines_with_offsets(text: str) -> list[tuple[int, str]]:
    lines: list[tuple[int, str]] = []
    offset = 0
    for raw_line in text.splitlines(keepends=True):
        lines.append((offset, raw_line.rstrip("\r\n")))
        offset += len(raw_line)
    if not lines and text:
        lines.append((0, text))
    return lines


def _detect_exhibit_tail(stripped_text: str) -> ExhibitTailDetection:
    if not stripped_text:
        return ExhibitTailDetection(start=None, reason="no_tail_anchor", anchor_text=None)

    min_start = int(len(stripped_text) * _TAIL_START_FRACTION)
    strong_candidates: list[_LineCandidate] = []
    weak_candidates: list[_LineCandidate] = []

    for start, line in _iter_lines_with_offsets(stripped_text):
        if not line:
            continue
        if _EXHIBIT_INDEX_RE.search(line):
            strong_candidates.append(
                _LineCandidate(
                    start=start,
                    text=line,
                    reason="strong_anchor_exhibit_index",
                )
            )
            continue
        if _EX_TAG_RE.search(line):
            strong_candidates.append(
                _LineCandidate(
                    start=start,
                    text=line,
                    reason="strong_anchor_ex_tag",
                )
            )
        if _WEAK_EXHIBIT_LINE_RE.search(line):
            weak_candidates.append(
                _LineCandidate(
                    start=start,
                    text=line,
                    reason="weak_anchor_cluster",
                )
            )

    accepted_candidates: list[_LineCandidate] = [
        candidate for candidate in strong_candidates if candidate.start >= min_start
    ]

    for candidate in weak_candidates:
        if candidate.start < min_start:
            continue
        lookahead_end = candidate.start + _WEAK_CLUSTER_WINDOW_CHARS
        weak_hits = sum(
            1
            for other in weak_candidates
            if candidate.start <= other.start <= lookahead_end
        )
        strong_hit = any(
            candidate.start <= other.start <= lookahead_end
            for other in strong_candidates
        )
        if weak_hits >= 2 or (weak_hits >= 1 and strong_hit):
            accepted_candidates.append(candidate)

    if not accepted_candidates:
        return ExhibitTailDetection(start=None, reason="no_tail_anchor", anchor_text=None)

    chosen = min(accepted_candidates, key=lambda candidate: candidate.start)
    return ExhibitTailDetection(
        start=chosen.start,
        reason=chosen.reason,
        anchor_text=chosen.text,
    )


def detect_exhibit_tail_start(text: str) -> int | None:
    return _detect_exhibit_tail(text).start


def _apply_lm2011_paper_cleaning(text: str | None) -> tuple[str | None, ExhibitTailDetection]:
    if text is None:
        return None, ExhibitTailDetection(start=None, reason="no_tail_anchor", anchor_text=None)
    stripped_text = _strip_edgar_metadata(text)
    detection = _detect_exhibit_tail(stripped_text)
    if detection.start is None:
        return stripped_text, detection
    return stripped_text[: detection.start].rstrip(), detection


def clean_full_10k_for_lm2011(
    text: str | None,
    *,
    contract: Full10KCleaningContract,
) -> str | None:
    normalized_contract = _validate_cleaning_contract(contract)
    if normalized_contract == "current":
        return text
    cleaned_text, _ = _apply_lm2011_paper_cleaning(text)
    return cleaned_text
