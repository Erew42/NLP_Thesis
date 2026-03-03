import pytest
from hypothesis import given, settings, strategies as st
from thesis_pkg.core.sec.extraction import (
    _scan_item_boundaries_py,
    _scan_part_markers_v2_py,
    _ItemBoundary,
    _PartMarker,
)
from thesis_pkg.core.sec.extraction_fast import (
    scan_item_boundaries_fast,
    scan_part_markers_v2_fast,
)

@st.composite
def sec_document_strategy(draw):
    line_strategies = st.sampled_from([
        "PART I", "PART II", "Item 1. Business.", "Item 1A. Risk Factors.", 
        "Item 7. Management's Discussion and Analysis", "ITEM 1.",
        "    PART II   ", "Table of Contents", "10-Q",
        "This is some prose about the company.",
        " "*20, "\t \t", "123", "", "Page 42", "(continued)",
        "Item 1", "Item2", "Item 15",
        "Part I, Item 1",
        "ITEM 1. BUSINESS", "ITEM 1A. RISK FACTORS",
        "See Part II, Item 7",
        "PART I ITEM 1"
    ])
    
    random_text = st.text(
        alphabet="abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 \n\t.-()[]", 
        min_size=0, 
        max_size=100
    )
    
    lines = draw(st.lists(st.one_of(line_strategies, random_text), min_size=1, max_size=50))
    
    body = "\n".join(lines)
    lines_actual = body.split('\n')
    line_starts = []
    current = 0
    for line in lines_actual:
        line_starts.append(current)
        current += len(line) + 1 # +1 for \n
    
    is_10k = draw(st.booleans())
    max_item_number = 15 if is_10k else 6
    allowed_parts = {"I", "II", "III", "IV"}
    scan_sparse_layout = draw(st.booleans())
    
    toc_indices = draw(st.lists(st.integers(min_value=0, max_value=max(0, len(lines_actual)-1)), max_size=5))
    toc_mask = set(toc_indices)
    
    toc_window_flags = [False] * len(lines_actual)
    for i in range(len(lines_actual)):
        if draw(st.booleans()):
            toc_window_flags[i] = True
            
    toc_cache = {}
    extraction_regime_v2 = draw(st.booleans())
    
    return (lines_actual, line_starts, body, is_10k, max_item_number, allowed_parts, scan_sparse_layout, toc_mask, toc_window_flags, toc_cache, extraction_regime_v2)

@settings(max_examples=500, deadline=None)
@given(doc_args=sec_document_strategy())
def test_scan_item_boundaries_parity(doc_args):
    (lines, line_starts, body, is_10k, max_item_number, allowed_parts, scan_sparse_layout, toc_mask, toc_window_flags, toc_cache, extraction_regime_v2) = doc_args
    
    py_res = _scan_item_boundaries_py(
        lines=lines,
        line_starts=line_starts,
        body=body,
        is_10k=is_10k,
        max_item_number=max_item_number,
        allowed_parts=allowed_parts,
        scan_sparse_layout=scan_sparse_layout,
        toc_mask=toc_mask,
        toc_window_flags=toc_window_flags,
        toc_cache=toc_cache,
        extraction_regime="v2" if extraction_regime_v2 else "legacy"
    )
    
    fast_raw = scan_item_boundaries_fast(
        lines,
        line_starts,
        body,
        is_10k,
        max_item_number,
        allowed_parts,
        scan_sparse_layout,
        toc_mask,
        toc_window_flags,
        toc_cache,
        extraction_regime_v2
    )
    
    fast_res = [
        _ItemBoundary(
            start=int(start),
            content_start=int(content_start),
            item_part=item_part,
            item_id=item_id,
            line_index=int(line_index),
            confidence=int(confidence),
            in_toc_range=bool(in_toc_range),
            toc_like_line=bool(toc_like_line),
        )
        for (
            start,
            content_start,
            item_part,
            item_id,
            line_index,
            confidence,
            in_toc_range,
            toc_like_line,
        ) in fast_raw
    ]
    
    assert py_res == fast_res, f"Mismatch!\nPython: {py_res}\nCython: {fast_res}"

@settings(max_examples=500, deadline=None)
@given(doc_args=sec_document_strategy())
def test_scan_part_markers_parity(doc_args):
    (lines, line_starts, body, is_10k, max_item_number, allowed_parts, scan_sparse_layout, toc_mask, toc_window_flags, toc_cache, extraction_regime_v2) = doc_args
    
    py_res = _scan_part_markers_v2_py(
        lines=lines,
        line_starts=line_starts,
        allowed_parts=allowed_parts,
        scan_sparse_layout=scan_sparse_layout,
        toc_mask=toc_mask,
        is_10q=not is_10k,
    )
    
    fast_raw = scan_part_markers_v2_fast(
        lines,
        line_starts,
        allowed_parts,
        scan_sparse_layout,
        toc_mask,
        not is_10k,
    )
    
    fast_res = [
        _PartMarker(
            start=int(start),
            part=str(part),
            line_index=int(line_index),
            high_confidence=bool(high_confidence),
        )
        for start, part, line_index, high_confidence in fast_raw
    ]
    
    assert py_res == fast_res, f"Mismatch!\nPython: {py_res}\nCython: {fast_res}"
