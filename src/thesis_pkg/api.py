from __future__ import annotations

from thesis_pkg.core.ccm.sec_ccm_contracts import (
    MatchReasonCode,
    PhaseBAlignmentMode,
    PhaseBDailyJoinMode,
    SecCcmJoinSpec,
    SecCcmJoinSpecV1,
    SecCcmJoinSpecV2,
    make_sec_ccm_join_spec_preset,
    normalize_sec_ccm_join_spec,
)
from thesis_pkg.core.ccm.sec_ccm_premerge import (
    align_doc_dates_phase_b,
    apply_phase_b_reason_codes,
    build_match_status_doc,
    build_unmatched_diagnostics_doc,
    normalize_sec_filings_phase_a,
    resolve_links_phase_a,
    join_daily_phase_b,
)
from thesis_pkg.core.ccm.transforms import (
    EXCHCD_NAME_MAP,
    SHRCD_FIRST_DIGIT_MAP,
    SHRCD_NAME_MAP,
    SHRCD_SECOND_DIGIT_MAP,
    add_exchcd_name,
    add_final_returns,
    apply_concept_filter_flags_doc,
    filter_us_common_major_exchange,
    add_shrcd_name,
    attach_ccm_links,
    attach_company_description,
    attach_filings,
    build_price_panel,
    exchcd_name_expr,
    map_shrcd_to_name,
    shrcd_name_expr,
    map_exchcd_to_name,
)
from thesis_pkg.core.sec.filing_text import ParsedFilingSchema, RawTextSchema, parse_filename_minimal
from thesis_pkg.io.parquet import load_tables
from thesis_pkg.pipelines.ccm_pipeline import merge_histories
from thesis_pkg.pipelines.sec_ccm_pipeline import run_sec_ccm_premerge_pipeline
from thesis_pkg.pipelines.sec_pipeline import process_zip_year, process_zip_year_raw_text


__all__ = [
    "load_tables",
    "build_price_panel",
    "EXCHCD_NAME_MAP",
    "SHRCD_FIRST_DIGIT_MAP",
    "SHRCD_NAME_MAP",
    "SHRCD_SECOND_DIGIT_MAP",
    "map_exchcd_to_name",
    "map_shrcd_to_name",
    "exchcd_name_expr",
    "shrcd_name_expr",
    "add_exchcd_name",
    "filter_us_common_major_exchange",
    "add_shrcd_name",
    "add_final_returns",
    "attach_filings",
    "attach_ccm_links",
    "attach_company_description",
    "merge_histories",
    "MatchReasonCode",
    "PhaseBAlignmentMode",
    "PhaseBDailyJoinMode",
    "SecCcmJoinSpec",
    "SecCcmJoinSpecV1",
    "SecCcmJoinSpecV2",
    "normalize_sec_ccm_join_spec",
    "make_sec_ccm_join_spec_preset",
    "normalize_sec_filings_phase_a",
    "resolve_links_phase_a",
    "align_doc_dates_phase_b",
    "join_daily_phase_b",
    "apply_phase_b_reason_codes",
    "build_match_status_doc",
    "build_unmatched_diagnostics_doc",
    "apply_concept_filter_flags_doc",
    "run_sec_ccm_premerge_pipeline",
    "parse_filename_minimal",
    "process_zip_year_raw_text",
    "process_zip_year",
    "RawTextSchema",
    "ParsedFilingSchema",
]
