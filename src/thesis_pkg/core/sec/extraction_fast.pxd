cpdef list scan_part_markers_v2_fast(
    list lines,
    object line_starts,
    set allowed_parts,
    bint scan_sparse_layout,
    set toc_mask,
    bint is_10q=*,
)

cpdef list scan_item_boundaries_fast(
    list lines,
    object line_starts,
    str body,
    bint is_10k,
    int max_item_number,
    set allowed_parts,
    bint scan_sparse_layout,
    set toc_mask,
    list toc_window_flags,
    dict toc_cache,
    bint extraction_regime_v2,
)
