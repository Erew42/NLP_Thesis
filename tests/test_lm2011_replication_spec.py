from __future__ import annotations

from pathlib import Path

import polars as pl
import pytest
import yaml

from thesis_pkg.core.ccm.sec_ccm_contracts import make_sec_ccm_join_spec_preset
from thesis_pkg.core.sec.filing_text import FilingItemSchema, ParsedFilingSchema, RawTextSchema


SPEC_PATH = Path("replication_plan/LM2011/lm2011_replication_spec.yaml")
EXPECTED_RULE_BASIS = [
    "paper_explicit",
    "implementation_convention",
    "known_divergence",
]
EXPECTED_RULE_STATUS = ["active", "blocked", "legacy_diagnostic_only"]
SEC_RAW_YEARLY_DIR = Path("full_data_run/year_merged")


def _load_spec() -> dict:
    return yaml.safe_load(SPEC_PATH.read_text(encoding="utf-8"))


def _get_node(spec: dict, *path: str) -> dict:
    node = spec
    for key in path:
        node = node[key]
    return node


def _paper_target(spec: dict, target_id: str) -> dict:
    for target in spec["paper_targets"]:
        if target["id"] == target_id:
            return target
    raise AssertionError(f"Paper target not found: {target_id}")


def _sample_schema_names(sample_file: str | None) -> list[str]:
    assert sample_file, "Expected a sample-backed contract"
    path = Path(sample_file)
    if not path.exists():
        pytest.skip(f"Sample file not present: {path}")
    return pl.scan_parquet(path).collect_schema().names()


def _unique_values_from_parquet(paths: list[Path], column: str) -> list[str]:
    if not paths:
        pytest.skip(f"No parquet files available for column scan: {column}")
    missing_paths = [path for path in paths if not path.exists()]
    if missing_paths:
        pytest.skip(f"Sample file not present: {missing_paths[0]}")

    values = (
        pl.scan_parquet([str(path) for path in paths])
        .select(pl.col(column).drop_nulls().unique().sort())
        .collect()
        .get_column(column)
        .to_list()
    )
    return [value for value in values if isinstance(value, str)]


def _benchmark_rows_by_id(rows: list[dict]) -> dict[str, dict]:
    return {row["id"]: row for row in rows}


def test_lm2011_spec_loads_with_phase0_phase1_contract_sections() -> None:
    spec = _load_spec()

    assert spec["spec_version"] == "0.5"
    assert spec["scope"] == "limited_replication"
    assert "column_contract_conventions" in spec["public_interfaces"]

    governance = spec["contract_governance"]
    assert governance["rule_basis_allowed"] == EXPECTED_RULE_BASIS
    assert governance["rule_status_allowed"] == EXPECTED_RULE_STATUS
    assert (
        governance["benchmark_pass_formula"]
        == "abs_diff <= max(tolerance_abs, paper_count * tolerance_pct)"
    )

    assert spec["status"]["blocking_prerequisites"] == []

    normalized_fields = spec["public_interfaces"]["normalized_fields"]
    for field_name in (
        "gvkey_int",
        "normalized_form",
        "filing_trade_date",
        "pre_filing_trade_date",
        "accounting_period_end",
        "quarter_report_date",
    ):
        assert field_name in normalized_fields

    derived_outputs = spec["public_interfaces"]["derived_outputs"]
    for output_name in (
        "lm2011_text_features_full_10k",
        "lm2011_text_features_mda",
        "lm2011_table_i_sample_creation",
        "lm2011_table_i_sample_creation_1994_2024",
        "lm2011_event_panel",
        "lm2011_sue_panel",
        "lm2011_trading_strategy_monthly_returns",
        "lm2011_trading_strategy_ff4_summary",
    ):
        assert output_name in derived_outputs

    datasets = spec["datasets"]
    assert "external_required" in datasets
    assert "harvard_negative_word_list" in datasets["external_required"]
    assert "sec_ccm_premerge" in datasets
    assert (
        datasets["crsp_ccm_daily"]["derived_artifacts"]["final_daily_panel_parquet"]["status"]
        == "active"
    )

    validation = spec["validation_and_acceptance"]
    assert "benchmark_targets" in validation


@pytest.mark.parametrize(
    ("spec_path", "expected_schema_name", "expected_columns"),
    [
        (
            ("datasets", "sec_text", "raw_yearly_parquet"),
            "thesis_pkg.core.sec.filing_text.RawTextSchema",
            list(RawTextSchema.schema),
        ),
        (
            ("datasets", "sec_text", "parsed_yearly_parquet"),
            "thesis_pkg.core.sec.filing_text.ParsedFilingSchema",
            list(ParsedFilingSchema.schema),
        ),
        (
            ("datasets", "sec_text", "extracted_10k_item_parquet"),
            "thesis_pkg.core.sec.filing_text.FilingItemSchema",
            list(FilingItemSchema.schema),
        ),
    ],
)
def test_lm2011_spec_sec_schema_contracts_match_library_classes(
    spec_path: tuple[str, ...],
    expected_schema_name: str,
    expected_columns: list[str],
) -> None:
    spec = _load_spec()
    node = _get_node(spec, *spec_path)

    assert node["library_schema_name"] == expected_schema_name
    assert node["library_schema_columns"] == expected_columns


@pytest.mark.parametrize(
    ("spec_path", "contract_key"),
    [
        (("datasets", "sec_text", "raw_yearly_parquet"), "library_schema_columns"),
        (("datasets", "ccm_filingdates"), "library_input_columns"),
        (
            ("datasets", "crsp_ccm_daily", "derived_artifacts", "canonical_link_table_parquet"),
            "library_output_columns",
        ),
        (
            ("datasets", "crsp_ccm_daily", "derived_artifacts", "final_daily_panel_parquet"),
            "library_output_columns",
        ),
        (("datasets", "sec_ccm_premerge", "match_status_parquet"), "library_output_columns"),
        (("datasets", "sec_ccm_premerge", "matched_clean_parquet"), "library_output_columns"),
        (("datasets", "sec_ccm_premerge", "analysis_doc_ids_parquet"), "library_output_columns"),
    ],
)
def test_lm2011_spec_exact_library_column_contracts_match_sample_artifacts(
    spec_path: tuple[str, ...],
    contract_key: str,
) -> None:
    spec = _load_spec()
    node = _get_node(spec, *spec_path)

    assert node[contract_key] == _sample_schema_names(node["sample_file"])


@pytest.mark.parametrize(
    ("spec_path", "contract_key"),
    [
        (("datasets", "sec_text", "raw_yearly_parquet"), "library_schema_columns"),
        (("datasets", "sec_text", "parsed_yearly_parquet"), "library_schema_columns"),
        (("datasets", "sec_text", "extracted_10k_item_parquet"), "library_schema_columns"),
        (("datasets", "ccm_filingdates"), "library_input_columns"),
        (
            ("datasets", "crsp_ccm_daily", "derived_artifacts", "canonical_link_table_parquet"),
            "library_output_columns",
        ),
        (
            ("datasets", "crsp_ccm_daily", "derived_artifacts", "final_daily_panel_parquet"),
            "library_output_columns",
        ),
        (("datasets", "sec_ccm_premerge", "match_status_parquet"), "library_output_columns"),
        (("datasets", "sec_ccm_premerge", "matched_clean_parquet"), "library_output_columns"),
        (("datasets", "sec_ccm_premerge", "analysis_doc_ids_parquet"), "library_output_columns"),
    ],
)
def test_lm2011_required_columns_are_subsets_of_exact_library_contracts(
    spec_path: tuple[str, ...],
    contract_key: str,
) -> None:
    spec = _load_spec()
    node = _get_node(spec, *spec_path)

    assert set(node["lm2011_required_columns"]).issubset(set(node[contract_key]))


@pytest.mark.parametrize(
    ("spec_path", "required_columns"),
    [
        (
            ("datasets", "crsp_ccm_daily", "tables", "sfz_dp_dly"),
            {"KYPERMNO", "CALDT", "PRC", "RET", "RETX", "TCAP", "VOL"},
        ),
        (
            ("datasets", "crsp_ccm_daily", "tables", "sfz_ds_dly"),
            {"KYPERMNO", "CALDT", "BIDLO", "ASKHI"},
        ),
        (
            ("datasets", "crsp_ccm_daily", "tables", "sfz_shr"),
            {"KYPERMNO", "SHRSDT", "SHRSENDDT", "SHROUT"},
        ),
        (
            ("datasets", "crsp_ccm_daily", "tables", "sfz_mth"),
            {"KYPERMNO", "MCALDT", "MRET", "MTCAP"},
        ),
        (
            ("datasets", "compustat_fundamentals", "annual_balance_sheet"),
            {
                "KYGVKEY",
                "KEYSET",
                "FYYYY",
                "fyra",
                "SEQ",
                "CEQ",
                "AT",
                "LT",
                "TXDITC",
                "PSTKL",
                "PSTKRV",
                "PSTK",
            },
        ),
        (
            ("datasets", "compustat_fundamentals", "annual_income_statement"),
            {"KYGVKEY", "KEYSET", "FYYYY", "fyra", "IB", "XINT", "TXDI", "DVP"},
        ),
        (
            ("datasets", "compustat_fundamentals", "annual_period_descriptor"),
            {"KYGVKEY", "KEYSET", "FYEAR", "FYR", "APDEDATE", "FDATE", "PDATE"},
        ),
        (
            ("datasets", "compustat_fundamentals", "quarterly_period_descriptor"),
            {"KYGVKEY", "KEYSET", "FYEARQ", "FQTR", "APDEDATEQ", "FDATEQ", "PDATEQ", "RDQ"},
        ),
        (
            ("datasets", "compustat_fundamentals", "fiscal_market_annual"),
            {"KYGVKEY", "DATADATE", "MKVALT", "PRCC"},
        ),
    ],
)
def test_lm2011_spec_sample_backed_required_columns_exist(
    spec_path: tuple[str, ...],
    required_columns: set[str],
) -> None:
    spec = _load_spec()
    node = _get_node(spec, *spec_path)

    missing = required_columns - set(_sample_schema_names(node["sample_file"]))
    assert not missing, f"{node['sample_file']} missing required columns: {sorted(missing)}"


def test_lm2011_spec_join_preset_matches_library_definition() -> None:
    spec = _load_spec()
    join_spec = spec["sec_ccm_join_spec"]
    preset = make_sec_ccm_join_spec_preset("lm2011_filing_date")

    assert join_spec["preset"] == "lm2011_filing_date"
    assert join_spec["meaning"]["phase_b_alignment_mode"] == preset.phase_b_alignment_mode.value
    assert join_spec["meaning"]["phase_b_daily_join_mode"] == preset.phase_b_daily_join_mode.value
    assert join_spec["daily_feature_columns"] == list(preset.daily_feature_columns)
    assert join_spec["required_daily_non_null_features"] == list(
        preset.required_daily_non_null_features
    )


def test_lm2011_spec_daily_artifact_and_doc_grain_authority_notes_reflect_corrected_state() -> None:
    spec = _load_spec()
    market_core = _get_node(
        spec,
        "datasets",
        "crsp_ccm_daily",
        "derived_artifacts",
        "ccm_daily_market_core_parquet",
    )
    phase_b_surface = _get_node(
        spec,
        "datasets",
        "crsp_ccm_daily",
        "derived_artifacts",
        "ccm_daily_phase_b_surface_parquet",
    )
    bridge_surface = _get_node(
        spec,
        "datasets",
        "crsp_ccm_daily",
        "derived_artifacts",
        "ccm_daily_bridge_surface_parquet",
    )
    final_daily = _get_node(
        spec,
        "datasets",
        "crsp_ccm_daily",
        "derived_artifacts",
        "final_daily_panel_parquet",
    )
    ccm_premerge = _get_node(spec, "datasets", "sec_ccm_premerge")

    assert market_core["status"] == "active"
    assert phase_b_surface["status"] == "active"
    assert bridge_surface["status"] == "active"
    assert (
        "project_ccm_daily_phase_b_surface" in phase_b_surface["produced_by"]
    )
    assert (
        "project_ccm_daily_bridge_surface" in bridge_surface["produced_by"]
    )
    assert "blocking_caveat" not in final_daily
    assert (
        "The corrected sampled daily artifact is authoritative for its filing-array columns."
        in final_daily["artifact_authority_note"]
    )
    assert "legacy compatibility artifact" in final_daily["compatibility_note"]
    assert (
        "Use the archived sec_ccm_premerge doc-grain artifacts as the preferred LM2011 linkage interface for later filing-level work."
        in final_daily["architecture_preference_note"]
    )
    assert "Preferred doc-grain SEC-CCM linkage artifacts for later LM2011 phases." == ccm_premerge["role"]


def test_lm2011_spec_filingdates_is_a_daily_stage_contract_input() -> None:
    spec = _load_spec()
    filingdates = _get_node(spec, "datasets", "ccm_filingdates")

    assert filingdates["stage"] == "daily_market_data_stage"
    assert (
        "daily-stage contract input" in filingdates["role"]
    )


def test_lm2011_spec_resolves_monthly_trading_return_source_to_sfz_mth() -> None:
    spec = _load_spec()
    assert _get_node(spec, "implementation_notes", "unresolved_choices") == []
    resolved = _get_node(spec, "implementation_notes", "resolved_choices")
    monthly_choice = next(entry for entry in resolved if entry["id"] == "lm2011_monthly_stock_return_source")
    assert monthly_choice["status"] == "active"
    assert monthly_choice["selected_input"] == "sfz_mth"
    assert "lm2011_trading_strategy_monthly_returns uses sfz_mth as the monthly stock-return source." == monthly_choice["statement"]


def test_lm2011_spec_records_strategy_defaults_as_explicit_assumptions() -> None:
    spec = _load_spec()
    strategy_defaults = _get_node(spec, "assumptions_and_defaults", "strategy_defaults")
    assert any("lm2011_trading_strategy_monthly_returns uses sfz_mth with MRET" in entry for entry in strategy_defaults)
    assert any("lm2011_trading_strategy_ff4_summary is estimated from lm2011_trading_strategy_monthly_returns" in entry for entry in strategy_defaults)
    assert any("equal-weight is the default assumption" in entry for entry in strategy_defaults)
    assert any("Harvard H4N-Inf comparison signals require an explicit external word-list input." == entry for entry in strategy_defaults)


def test_lm2011_spec_encodes_exact_tfidf_formula_and_signal_surface() -> None:
    spec = _load_spec()
    weighting = _get_node(spec, "derived_variables", "text_weighting_contract")
    screen_count = _get_node(spec, "derived_variables", "screen_count_contract")
    full_cols = _get_node(
        spec, "public_interfaces", "derived_outputs", "lm2011_text_features_full_10k", "required_columns"
    )
    mda_cols = _get_node(
        spec, "public_interfaces", "derived_outputs", "lm2011_text_features_mda", "required_columns"
    )

    assert weighting["basis"] == "paper_explicit"
    assert weighting["formula"] == "w_i,j = ((1 + log(tf_i,j)) / (1 + log(a_j))) * log(N / df_i) when tf_i,j >= 1, else 0."
    assert "a_j = total LM-tokenized length of the relevant analyzed text unit for document j" in weighting["inputs"]
    assert "Do not use smoothed idf." in weighting["implementation_guardrails"]
    assert "Do not add a +1 idf offset." in weighting["implementation_guardrails"]
    assert "total_token_count_full_10k" in full_cols
    assert "h4n_inf_tfidf" in full_cols
    assert "lm_modal_weak_tfidf" in full_cols
    assert "total_token_count_mda" in mda_cols
    assert "h4n_inf_tfidf" in mda_cols
    assert "lm_negative_tfidf" in mda_cols
    assert screen_count["exported_columns"] == ["total_token_count_full_10k", "total_token_count_mda"]
    assert (
        "`token_count_full_10k` and `token_count_mda` remain recognized-word totals for diagnostics and coverage."
        in screen_count["notes"]
    )


def test_lm2011_spec_separates_raw_share_turnover_from_log_regression_transform() -> None:
    spec = _load_spec()
    share_turnover = _get_node(spec, "derived_variables", "lm2011_controls_and_outcomes", "share_turnover")
    log_transform = _get_node(spec, "derived_variables", "regression_layer_transforms", "log_share_turnover")

    assert share_turnover["definition"] == "Share turnover = sum(volume on trading days -252 through -6) / shares_outstanding_on_filing_date."
    assert log_transform["definition"] == "log_share_turnover = ln(share_turnover)"
    assert set(log_transform["usage"]) == {
        "LM2011 Tables IV",
        "LM2011 Tables V",
        "LM2011 Table VI",
        "LM2011 Table VIII",
        "LM2011 Internet Appendix Table IA.I",
    }
    assert "share_turnover remains the raw appendix variable in lm2011_event_panel and lm2011_sue_panel." in log_transform["notes"]


def test_lm2011_spec_limits_pre_filing_trade_date_public_requirement_to_event_panel() -> None:
    spec = _load_spec()
    pre_filing_trade_date = _get_node(spec, "public_interfaces", "normalized_fields", "pre_filing_trade_date")

    assert pre_filing_trade_date["required_on"] == ["lm2011_event_panel"]


def test_lm2011_spec_pins_ibes_exact_then_safe_fallback_matching_contract() -> None:
    spec = _load_spec()
    ibes = _get_node(spec, "datasets", "external_required", "ibes_unadjusted_earnings")

    assert ibes["matching_contract"] == [
        "First match exact on (gvkey_int, quarter_report_date, quarter_fiscal_period_end) against (gvkey_int, announcement_date, fiscal_period_end).",
        "If no exact match exists, allow fallback only to unique (gvkey_int, announcement_date) rows matched on (gvkey_int, quarter_report_date).",
        "Reject ambiguous fallback announcement_date matches rather than selecting one arbitrarily.",
    ]


def test_lm2011_spec_pins_trading_strategy_direction_and_split_outputs() -> None:
    spec = _load_spec()
    derived_outputs = spec["public_interfaces"]["derived_outputs"]
    strategy_contract = _get_node(spec, "implementation_contracts", "trading_strategy_contract")

    assert derived_outputs["lm2011_trading_strategy_monthly_returns"]["required_columns"] == [
        "portfolio_month",
        "sort_signal_name",
        "long_short_return",
    ]
    assert derived_outputs["lm2011_trading_strategy_ff4_summary"]["required_columns"] == [
        "sort_signal_name",
        "alpha_ff3_mom",
        "alpha_ff3_mom_standard_error",
        "alpha_ff3_mom_t_stat",
        "beta_market",
        "beta_market_standard_error",
        "beta_market_t_stat",
        "beta_smb",
        "beta_smb_standard_error",
        "beta_smb_t_stat",
        "beta_hml",
        "beta_hml_standard_error",
        "beta_hml_t_stat",
        "beta_mom",
        "beta_mom_standard_error",
        "beta_mom_t_stat",
        "r2",
    ]
    assert strategy_contract["formation_month"] == "June"
    assert "Use the prior year's accepted 10-K signal to assign annual June sorts." in strategy_contract["portfolio_assignment_rule"]
    assert "Long portfolio = lowest-negative quintile (Q1)." in strategy_contract["portfolio_assignment_rule"]
    assert "Short portfolio = highest-negative quintile (Q5)." in strategy_contract["portfolio_assignment_rule"]
    assert "long_short_return = return(Q1) - return(Q5)." in strategy_contract["portfolio_assignment_rule"]


def test_lm2011_spec_activates_common_equity_market_cap_exchange_and_mda_filters() -> None:
    spec = _load_spec()
    market_filters = {
        row["id"]: row
        for row in spec["filters"]["paper_faithful_lm2011_filters"]["market_and_listing"]
    }
    mda_filters = {
        row["id"]: row
        for row in spec["filters"]["paper_faithful_lm2011_filters"]["mda_subsample"]
    }

    assert market_filters["require_ordinary_common_equity_filter"]["status"] == "active"
    assert market_filters["require_ordinary_common_equity_filter"]["operational_proxy"]["statement"] == "Operationalize ordinary common equity as SHRCD in {10, 11}."
    assert market_filters["require_market_cap_available"]["status"] == "active"
    assert market_filters["require_nyse_amex_nasdaq_listing"]["operational_proxy"]["statement"] == "Operationalize major-exchange listing as EXCHCD in {1, 2, 3}."
    assert mda_filters["require_identifiable_mda_item_7"]["status"] == "active"
    assert mda_filters["require_mda_token_count_ge_250"]["status"] == "active"


def test_lm2011_spec_adds_retained_table_contracts() -> None:
    spec = _load_spec()
    tables = spec["table_and_regression_contracts"]

    assert set(tables) >= {
        "table_iv_full_10k",
        "table_v_mda",
        "table_vi_full_10k_dictionary_surface",
        "table_viii_sue",
        "internet_appendix_table_ia_i",
        "internet_appendix_table_ia_ii",
    }
    assert tables["table_iv_full_10k"]["estimator"] == "quarterly_fama_macbeth"
    assert tables["table_iv_full_10k"]["newey_west_lags"] == 1
    assert tables["table_iv_full_10k"]["quarter_weighting"]["operationalization"]["statement"] == "Operationalize quarter weighting by quarter observation count."
    assert tables["table_viii_sue"]["source_panel"] == "lm2011_sue_panel"
    assert tables["internet_appendix_table_ia_ii"]["source_artifacts"] == [
        "lm2011_trading_strategy_monthly_returns",
        "lm2011_trading_strategy_ff4_summary",
    ]


def test_lm2011_primary_vs_secondary_outputs_are_partitioned_correctly() -> None:
    spec = _load_spec()
    lm2011 = _paper_target(spec, "LM2011")

    assert "MD&A Item 7 word-feature panels" not in lm2011["primary_outputs"]
    assert "MD&A Item 7 word-feature panels" in lm2011["secondary_outputs"]
    assert "MD&A filing-period return regressions (Table V only)" in lm2011["secondary_outputs"]


def test_lm2011_spec_form_normalization_contract_covers_expected_aliases() -> None:
    spec = _load_spec()
    mapping = spec["derived_variables"]["form_normalization"]["canonical_mapping"]

    assert mapping["10K"] == "10-K"
    assert mapping["10Q"] == "10-Q"
    assert mapping["10K/A"] == "10-K/A"
    assert mapping["10Q/A"] == "10-Q/A"
    assert mapping["10-K405"] == "10-K"
    assert mapping["10-K-A"] == "10-K/A"
    assert mapping["10-K405-A"] == "10-K/A"
    assert mapping["10-Q-A"] == "10-Q/A"


def test_lm2011_form_mapping_covers_observed_annual_raw_forms() -> None:
    spec = _load_spec()
    form_rule = spec["derived_variables"]["form_normalization"]["lm2011_main_sample_rule"]

    sec_paths = sorted(SEC_RAW_YEARLY_DIR.glob("*.parquet"))
    sec_forms = _unique_values_from_parquet(sec_paths, "document_type_filename")
    observed_annual_sec_forms = sorted(
        value for value in sec_forms if value.startswith("10-K") or value.startswith("10K")
    )

    sec_allowed = set(form_rule["sec_text_included_raw_forms"]) | set(
        form_rule["sec_text_excluded_raw_forms"]
    )
    missing_sec = sorted(set(observed_annual_sec_forms) - sec_allowed)
    assert not missing_sec, f"Observed annual SEC forms missing from LM2011 rule: {missing_sec}"

    ccm_filingdates_path = Path(_get_node(spec, "datasets", "ccm_filingdates")["sample_file"])
    ccm_forms = _unique_values_from_parquet([ccm_filingdates_path], "SRCTYPE")
    observed_annual_ccm_forms = sorted(value for value in ccm_forms if value.startswith("10K"))

    ccm_allowed = set(form_rule["ccm_filingdates_included_raw_forms"]) | set(
        form_rule["ccm_filingdates_excluded_raw_forms"]
    )
    missing_ccm = sorted(set(observed_annual_ccm_forms) - ccm_allowed)
    assert not missing_ccm, f"Observed annual CCM forms missing from LM2011 rule: {missing_ccm}"


def test_lm2011_main_sample_rule_is_raw_form_based() -> None:
    spec = _load_spec()
    form_rule = spec["derived_variables"]["form_normalization"]["lm2011_main_sample_rule"]

    assert form_rule["sec_text_included_raw_forms"] == ["10-K", "10-K405"]
    assert form_rule["ccm_filingdates_included_raw_forms"] == ["10K"]
    assert "10-KT" in form_rule["sec_text_excluded_raw_forms"]
    assert "10-K-A" in form_rule["sec_text_excluded_raw_forms"]
    assert "10K/A" in form_rule["ccm_filingdates_excluded_raw_forms"]
    assert (
        "The paper-faithful main sample is defined from raw SEC filing form values, not from normalized_form alone."
        in form_rule["predicate_notes"]
    )


def test_lm2011_phase0_phase1_spec_does_not_invent_raw_copy_fields() -> None:
    spec = _load_spec()

    for spec_path in (
        ("datasets", "sec_text", "parsed_yearly_parquet"),
        ("datasets", "sec_text", "extracted_10k_item_parquet"),
        ("datasets", "ccm_filingdates"),
    ):
        normalized_columns = _get_node(spec, *spec_path)["normalized_columns_to_add"]
        assert "document_type_filename_raw" not in normalized_columns
        assert "SRCTYPE_raw" not in normalized_columns


def test_lm2011_table_i_benchmarks_present_with_tolerances() -> None:
    spec = _load_spec()
    benchmarks = spec["validation_and_acceptance"]["benchmark_targets"]["lm2011_table_i"]

    full_10k_expected = {
        "edgar_complete_nonduplicate_sample": 121217,
        "first_filing_per_year": 120290,
        "minimum_180_day_spacing": 120074,
        "crsp_permno_match": 75252,
        "ordinary_common_equity": 70061,
        "market_cap_available": 64227,
        "price_day_minus_one_ge_3": 55946,
        "event_window_returns_and_volume": 55630,
        "major_exchange_listing": 55612,
        "sixty_day_pre_post_coverage": 55038,
        "book_to_market_available_and_book_value_positive": 50268,
        "token_count_ge_2000": 50115,
    }
    mda_expected = {
        "identifiable_mda": 49179,
        "mda_token_count_ge_250": 37287,
    }

    full_10k_rows = _benchmark_rows_by_id(benchmarks["full_10k_sample"])
    for benchmark_id, paper_count in full_10k_expected.items():
        row = full_10k_rows[benchmark_id]
        assert row["paper_count"] == paper_count
        assert row["tolerance_abs"] == 100
        assert row["tolerance_pct"] == 0.005
        assert row["basis"] == "paper_explicit"
        assert row["status"] == "active"

    mda_rows = _benchmark_rows_by_id(benchmarks["mda_subsample"])
    for benchmark_id, paper_count in mda_expected.items():
        row = mda_rows[benchmark_id]
        assert row["paper_count"] == paper_count
        assert row["tolerance_abs"] == 250
        assert row["tolerance_pct"] == 0.01
        assert row["basis"] == "paper_explicit"
        assert row["status"] == "active"


@pytest.mark.parametrize(
    ("output_name", "contract_name"),
    [
        ("lm2011_table_i_sample_creation", "table_i_sample_creation_contract"),
        ("lm2011_table_i_sample_creation_1994_2024", "table_i_sample_creation_1994_2024_contract"),
    ],
)
def test_lm2011_table_i_sample_creation_contract_aligns_with_benchmark_ids(
    output_name: str,
    contract_name: str,
) -> None:
    spec = _load_spec()
    derived_output = spec["public_interfaces"]["derived_outputs"][output_name]
    assert derived_output["grain"] == ["section_id", "row_id"]
    assert derived_output["required_columns"] == [
        "section_id",
        "section_label",
        "section_order",
        "row_order",
        "row_id",
        "display_label",
        "sample_size_kind",
        "sample_size_value",
        "observations_removed",
        "availability_status",
        "availability_reason",
    ]

    contract = spec["implementation_contracts"][contract_name]
    benchmark_rows = spec["validation_and_acceptance"]["benchmark_targets"]["lm2011_table_i"]
    expected_attrition_row_ids = [
        *(row["id"] for row in benchmark_rows["full_10k_sample"]),
        *(row["id"] for row in benchmark_rows["mda_subsample"]),
    ]
    assert contract["output"] == output_name
    assert contract["benchmark_row_ids_in_order"] == expected_attrition_row_ids
    assert contract["firm_year_summary_row_ids"] == [
        "firm_year_sample",
        "unique_firms",
        "average_years_per_firm",
    ]


def test_lm2011_acceptance_scenarios_cover_phase0_phase1_contracts() -> None:
    spec = _load_spec()
    scenarios = {
        scenario["id"]: scenario["description"]
        for scenario in spec["validation_and_acceptance"]["acceptance_scenarios"]
    }

    assert "lm2011_raw_form_rule_contract" in scenarios
    assert "lm2011_table_i_benchmark_contract" in scenarios
    assert "first-filing-per-year" in scenarios["lm2011_sample_filters"]
    assert "raw-form inclusion and exclusion" in scenarios["lm2011_sample_filters"]
    assert "180-day filing spacing" in scenarios["lm2011_sample_filters"]
