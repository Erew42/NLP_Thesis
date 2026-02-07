from __future__ import annotations

import warnings

import pytest

from thesis_pkg.core.sec import extraction


def _line_starts(lines: list[str]) -> list[int]:
    starts: list[int] = []
    pos = 0
    for line in lines:
        starts.append(pos)
        pos += len(line) + 1
    return starts


def _require_fast_extension() -> None:
    if extraction._extraction_fast is None:
        pytest.skip("Fast extraction extension is unavailable in this environment.")


def test_scan_part_markers_v2_fallback_matches_python_impl(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    lines = [
        "PART I",
        "ITEM 1. FINANCIAL STATEMENTS",
        "some text",
        "PART II",
        "ITEM 3. QUANTITATIVE AND QUALITATIVE DISCLOSURES ABOUT MARKET RISK",
    ]
    starts = _line_starts(lines)
    kwargs = {
        "allowed_parts": {"I", "II"},
        "scan_sparse_layout": True,
        "toc_mask": set(),
        "is_10q": True,
    }

    monkeypatch.setattr(extraction, "_extraction_fast", None)
    expected = extraction._scan_part_markers_v2_py(lines, starts, **kwargs)
    actual = extraction._scan_part_markers_v2(lines, starts, **kwargs)
    assert actual == expected


def test_scan_part_markers_v2_uses_fast_result_shape(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _FastStub:
        @staticmethod
        def scan_part_markers_v2_fast(*_args: object, **_kwargs: object) -> list[tuple[int, str, int, bool]]:
            return [
                (12, "I", 2, True),
                (128, "II", 20, True),
            ]

    monkeypatch.setattr(extraction, "_extraction_fast", _FastStub())
    markers = extraction._scan_part_markers_v2(
        ["PART I", "PART II"],
        [0, 7],
        allowed_parts={"I", "II"},
        scan_sparse_layout=True,
        toc_mask=set(),
        is_10q=True,
    )
    assert markers == [
        extraction._PartMarker(start=12, part="I", line_index=2, high_confidence=True),
        extraction._PartMarker(start=128, part="II", line_index=20, high_confidence=True),
    ]


def test_scan_item_boundaries_fallback_matches_python_impl(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    lines = [
        "PART I",
        "ITEM 1. BUSINESS",
        "Body text.",
        "ITEM 1A. RISK FACTORS",
    ]
    starts = _line_starts(lines)
    kwargs = {
        "is_10k": True,
        "max_item_number": 20,
        "allowed_parts": {"I", "II", "III", "IV"},
        "scan_sparse_layout": False,
        "toc_mask": set(),
        "toc_window_flags": [False] * len(lines),
        "toc_cache": {},
        "extraction_regime": "legacy",
    }
    body = "\n".join(lines)

    monkeypatch.setattr(extraction, "_extraction_fast", None)
    expected = extraction._scan_item_boundaries_py(lines, starts, body, **kwargs)
    actual = extraction._scan_item_boundaries(lines, starts, body, **kwargs)
    assert actual == expected


def test_scan_item_boundaries_uses_fast_result_shape(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _FastStub:
        @staticmethod
        def scan_item_boundaries_fast(
            *_args: object, **_kwargs: object
        ) -> list[tuple[int, int, str | None, str, int, int, bool, bool]]:
            return [
                (10, 20, "I", "1", 3, 3, False, False),
                (30, 40, None, "1A", 6, 2, True, True),
            ]

    monkeypatch.setattr(extraction, "_extraction_fast", _FastStub())
    boundaries = extraction._scan_item_boundaries(
        ["ITEM 1. TEST"],
        [0],
        "ITEM 1. TEST",
        is_10k=True,
        max_item_number=20,
        allowed_parts={"I", "II", "III", "IV"},
        scan_sparse_layout=False,
        toc_mask=set(),
        toc_window_flags=[False],
        toc_cache={},
        extraction_regime="legacy",
    )
    assert boundaries == [
        extraction._ItemBoundary(
            start=10,
            content_start=20,
            item_part="I",
            item_id="1",
            line_index=3,
            confidence=3,
            in_toc_range=False,
            toc_like_line=False,
        ),
        extraction._ItemBoundary(
            start=30,
            content_start=40,
            item_part=None,
            item_id="1A",
            line_index=6,
            confidence=2,
            in_toc_range=True,
            toc_like_line=True,
        ),
    ]


def test_scan_item_boundaries_tracks_fallback_when_fastpath_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(extraction, "_extraction_fast", None)
    extraction.reset_extraction_fastpath_metrics()

    extraction._scan_item_boundaries(
        ["ITEM 1. TEST"],
        [0],
        "ITEM 1. TEST",
        is_10k=True,
        max_item_number=20,
        allowed_parts={"I", "II", "III", "IV"},
        scan_sparse_layout=False,
        toc_mask=set(),
        toc_window_flags=[False],
        toc_cache={},
        extraction_regime="legacy",
    )
    metrics = extraction.get_extraction_fastpath_metrics()
    assert metrics["scan_item_boundaries_fallbacks"] == 1
    assert metrics["scan_item_boundaries_fast_success"] == 0
    assert metrics["scan_item_boundaries_fast_failures"] == 0


def test_scan_item_boundaries_warns_once_and_falls_back_on_fast_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _FailingFastStub:
        @staticmethod
        def scan_item_boundaries_fast(*_args: object, **_kwargs: object) -> list[tuple[int, int, str | None, str, int, int, bool, bool]]:
            raise RuntimeError("synthetic fast-path failure")

    lines = ["PART I", "ITEM 1. BUSINESS"]
    starts = _line_starts(lines)
    body = "\n".join(lines)
    kwargs = {
        "is_10k": True,
        "max_item_number": 20,
        "allowed_parts": {"I", "II", "III", "IV"},
        "scan_sparse_layout": False,
        "toc_mask": set(),
        "toc_window_flags": [False, False],
        "toc_cache": {},
        "extraction_regime": "legacy",
    }

    monkeypatch.setattr(extraction, "_extraction_fast", _FailingFastStub())
    extraction.reset_extraction_fastpath_metrics()
    expected = extraction._scan_item_boundaries_py(lines, starts, body, **kwargs)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        actual_1 = extraction._scan_item_boundaries(lines, starts, body, **kwargs)
        actual_2 = extraction._scan_item_boundaries(lines, starts, body, **kwargs)

    runtime_warnings = [
        warning
        for warning in caught
        if issubclass(warning.category, RuntimeWarning)
    ]
    assert len(runtime_warnings) == 1
    assert "falling back to Python implementation" in str(runtime_warnings[0].message)
    assert actual_1 == expected
    assert actual_2 == expected

    metrics = extraction.get_extraction_fastpath_metrics()
    assert metrics["scan_item_boundaries_fast_failures"] == 2
    assert metrics["scan_item_boundaries_fallbacks"] == 2
    assert metrics["scan_item_boundaries_fast_success"] == 0


def test_scan_part_markers_v2_fast_parity_mixed_case_part_token() -> None:
    _require_fast_extension()
    extraction.reset_extraction_fastpath_metrics()

    lines = ["pArT I"]
    starts = _line_starts(lines)
    kwargs = {
        "allowed_parts": {"I", "II"},
        "scan_sparse_layout": False,
        "toc_mask": set(),
        "is_10q": True,
    }
    expected = extraction._scan_part_markers_v2_py(lines, starts, **kwargs)
    actual = extraction._scan_part_markers_v2(lines, starts, **kwargs)
    assert actual == expected

    metrics = extraction.get_extraction_fastpath_metrics()
    assert metrics["scan_part_markers_fast_failures"] == 0
    assert metrics["scan_part_markers_fast_success"] == 1


def test_scan_item_boundaries_fast_parity_mixed_case_item_token() -> None:
    _require_fast_extension()
    extraction.reset_extraction_fastpath_metrics()

    lines = ["iTeM 1. Business"]
    starts = _line_starts(lines)
    body = "\n".join(lines)
    kwargs = {
        "is_10k": True,
        "max_item_number": 20,
        "allowed_parts": {"I", "II", "III", "IV"},
        "scan_sparse_layout": False,
        "toc_mask": set(),
        "toc_window_flags": [False],
        "toc_cache": {},
        "extraction_regime": "legacy",
    }
    expected = extraction._scan_item_boundaries_py(lines, starts, body, **kwargs)
    actual = extraction._scan_item_boundaries(lines, starts, body, **kwargs)
    assert actual == expected

    metrics = extraction.get_extraction_fastpath_metrics()
    assert metrics["scan_item_boundaries_fast_failures"] == 0
    assert metrics["scan_item_boundaries_fast_success"] == 1


def test_scan_part_markers_v2_fast_parity_non_sparse_rescue_behavior() -> None:
    _require_fast_extension()
    extraction.reset_extraction_fastpath_metrics()

    lines = ["PART I, ITEM 1. FINANCIAL STATEMENTS"]
    starts = _line_starts(lines)
    kwargs = {
        "allowed_parts": {"I", "II"},
        "scan_sparse_layout": False,
        "toc_mask": set(),
        "is_10q": True,
    }
    expected = extraction._scan_part_markers_v2_py(lines, starts, **kwargs)
    actual = extraction._scan_part_markers_v2(lines, starts, **kwargs)
    assert actual == expected

    metrics = extraction.get_extraction_fastpath_metrics()
    assert metrics["scan_part_markers_fast_failures"] == 0
    assert metrics["scan_part_markers_fast_success"] == 1
