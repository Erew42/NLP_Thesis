from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import polars as pl
import pytest
from pytest import MonkeyPatch

from thesis_pkg.notebooks_and_scripts import sec_ccm_unified_runner as runner
from thesis_pkg.pipelines.refinitiv.lseg_ledger import LsegResumeCompatibilityError


def test_doc_ownership_stage_runs_after_sec_ccm_premerge_block() -> None:
    runner_path = Path("src/thesis_pkg/notebooks_and_scripts/sec_ccm_unified_runner.py")
    source = runner_path.read_text(encoding="utf-8")

    premerge_marker = "sec_ccm_paths = run_sec_ccm_premerge_pipeline("
    doc_stage_marker = "if RUN_REFINITIV_DOC_OWNERSHIP_LM2011_EXACT_HANDOFF:"

    assert premerge_marker in source
    assert doc_stage_marker in source
    assert source.index(doc_stage_marker) > source.index(premerge_marker)


def test_doc_analyst_stage_runs_after_sec_ccm_premerge_block() -> None:
    runner_path = Path("src/thesis_pkg/notebooks_and_scripts/sec_ccm_unified_runner.py")
    source = runner_path.read_text(encoding="utf-8")

    premerge_marker = "sec_ccm_paths = run_sec_ccm_premerge_pipeline("
    doc_stage_marker = "if RUN_REFINITIV_DOC_ANALYST_LM2011_ANCHORS:"

    assert premerge_marker in source
    assert doc_stage_marker in source
    assert source.index(doc_stage_marker) > source.index(premerge_marker)


def test_analyst_step1_stage_functions_are_imported_and_referenced() -> None:
    runner_path = Path("src/thesis_pkg/notebooks_and_scripts/sec_ccm_unified_runner.py")
    source = runner_path.read_text(encoding="utf-8")

    required_symbols = [
        "run_refinitiv_step1_instrument_authority_pipeline",
        "run_refinitiv_step1_analyst_request_groups_pipeline",
        "run_refinitiv_step1_analyst_actuals_api_pipeline",
        "run_refinitiv_step1_analyst_estimates_monthly_api_pipeline",
        "run_refinitiv_step1_analyst_normalize_pipeline",
    ]

    for symbol in required_symbols:
        assert symbol in source


def test_analyst_step1_stage_order_precedes_doc_analyst_stages() -> None:
    runner_path = Path("src/thesis_pkg/notebooks_and_scripts/sec_ccm_unified_runner.py")
    source = runner_path.read_text(encoding="utf-8")

    normalize_marker = "if RUN_REFINITIV_ANALYST_NORMALIZE:"
    doc_anchor_marker = "if RUN_REFINITIV_DOC_ANALYST_LM2011_ANCHORS:"
    doc_select_marker = "if RUN_REFINITIV_DOC_ANALYST_LM2011_SELECT:"

    assert normalize_marker in source
    assert doc_anchor_marker in source
    assert doc_select_marker in source
    assert source.index(normalize_marker) < source.index(doc_anchor_marker)
    assert source.index(doc_anchor_marker) < source.index(doc_select_marker)


def test_notebook_defaults_match_script_defaults_for_refinitiv_stage_plan() -> None:
    runner_path = Path("src/thesis_pkg/notebooks_and_scripts/sec_ccm_unified_runner.py")
    source = runner_path.read_text(encoding="utf-8")
    notebook_source = Path(
        "src/thesis_pkg/notebooks_and_scripts/sec_ccm_unified_runner.ipynb"
    ).read_text(encoding="utf-8")

    expected_explicit_flags = [
        "RUN_REFINITIV_STEP1",
        "RUN_REFINITIV_STEP1_RESOLUTION",
        "RUN_REFINITIV_OWNERSHIP_UNIVERSE_HANDOFF",
        "RUN_REFINITIV_OWNERSHIP_UNIVERSE_RESULTS",
        "RUN_REFINITIV_OWNERSHIP_AUTHORITY",
        "RUN_REFINITIV_DOC_OWNERSHIP_LM2011_EXACT_HANDOFF",
        "RUN_REFINITIV_DOC_OWNERSHIP_LM2011_FALLBACK_HANDOFF",
        "RUN_REFINITIV_DOC_OWNERSHIP_LM2011_FINALIZE",
        "RUN_REFINITIV_DOC_ANALYST_LM2011_ANCHORS",
        "RUN_REFINITIV_DOC_ANALYST_LM2011_SELECT",
        "RUN_REFINITIV_INSTRUMENT_AUTHORITY",
        "RUN_REFINITIV_ANALYST_REQUEST_GROUPS",
        "RUN_REFINITIV_ANALYST_ACTUALS",
        "RUN_REFINITIV_ANALYST_ESTIMATES_MONTHLY",
        "RUN_REFINITIV_ANALYST_NORMALIZE",
        "RUN_SEC_CCM_PREMERGE",
    ]

    for flag_name in expected_explicit_flags:
        assert f'{flag_name} = _env_bool(' in source
        assert (
            f'"{flag_name} = True\\n",' in notebook_source
            or f'"{flag_name} = False\\n",' in notebook_source
        )
        env_flag_name = flag_name.removeprefix("RUN_")
        assert f'SEC_CCM_RUN_{env_flag_name}' in notebook_source
        assert f'": {flag_name},\\n"' in notebook_source


def test_runner_references_expected_analyst_output_artifacts() -> None:
    runner_path = Path("src/thesis_pkg/notebooks_and_scripts/sec_ccm_unified_runner.py")
    source = runner_path.read_text(encoding="utf-8")

    expected_artifacts = [
        "refinitiv_analyst_request_group_membership_common_stock.parquet",
        "refinitiv_analyst_request_universe_common_stock.parquet",
        "refinitiv_analyst_actuals_raw.parquet",
        "refinitiv_analyst_estimates_monthly_raw.parquet",
        "refinitiv_analyst_normalized_panel.parquet",
    ]

    for artifact in expected_artifacts:
        assert artifact in source


def test_sec_ccm_unified_runner_exposes_and_threads_ownership_ticker_fallback_flag() -> None:
    runner_path = Path("src/thesis_pkg/notebooks_and_scripts/sec_ccm_unified_runner.py")
    source = runner_path.read_text(encoding="utf-8")

    assert 'SEC_CCM_REFINITIV_OWNERSHIP_INCLUDE_TICKER_FALLBACK' in source
    assert 'REFINITIV_OWNERSHIP_INCLUDE_TICKER_FALLBACK = _env_bool(' in source
    assert 'include_ticker_fallback=REFINITIV_OWNERSHIP_INCLUDE_TICKER_FALLBACK' in source


def test_first_existing_path_prefers_existing_candidate(tmp_path: Path) -> None:
    missing = tmp_path / "missing"
    existing = tmp_path / "existing"
    existing.mkdir()

    assert runner._first_existing_path(missing, existing) == existing


def test_resolve_ccm_parquet_artifact_supports_nested_documents_export_layout(tmp_path: Path) -> None:
    base_dir = tmp_path / "ccm"
    nested_dir = base_dir / "documents-export-2025-3-19"
    nested_dir.mkdir(parents=True)
    target = nested_dir / "filingdates.parquet"
    target.write_text("placeholder", encoding="utf-8")

    assert runner._resolve_ccm_parquet_artifact(base_dir, "filingdates.parquet") == target


def test_resolve_repo_root_honors_env_override(
    monkeypatch: MonkeyPatch,
    tmp_path: Path,
) -> None:
    repo_root = tmp_path / "repo"
    pipeline_path = repo_root / "src" / "thesis_pkg" / "pipeline.py"
    pipeline_path.parent.mkdir(parents=True)
    pipeline_path.write_text("from __future__ import annotations\n", encoding="utf-8")

    monkeypatch.setenv(runner.REPO_ROOT_ENV_VAR, str(repo_root))

    assert runner._resolve_repo_root() == repo_root


def test_resolve_ff48_siccodes_path_honors_env_override(
    monkeypatch: MonkeyPatch,
    tmp_path: Path,
) -> None:
    ff48_path = tmp_path / "FF_Siccodes_48_Industries.txt"
    ff48_path.write_text("placeholder", encoding="utf-8")

    monkeypatch.setenv("SEC_CCM_FF48_SICCODES_PATH", str(ff48_path))

    assert runner._resolve_ff48_siccodes_path(tmp_path) == ff48_path


def test_env_list_and_bool_helpers(monkeypatch: MonkeyPatch) -> None:
    monkeypatch.setenv("TEST_BOOL", "true")
    monkeypatch.setenv("TEST_OPTIONAL_BOOL", "auto")
    monkeypatch.setenv("TEST_STR_LIST", '["a", "b"]')
    monkeypatch.setenv("TEST_INT_LIST", "[1993, 1994]")

    assert runner._env_bool("TEST_BOOL", False) is True
    assert runner._env_optional_bool("TEST_OPTIONAL_BOOL", False) is None
    assert runner._env_str_list("TEST_STR_LIST", ["x"]) == ["a", "b"]
    assert runner._env_int_list("TEST_INT_LIST", [1]) == [1993, 1994]


def test_env_optional_date_helper(monkeypatch: MonkeyPatch) -> None:
    monkeypatch.setenv("TEST_OPTIONAL_DATE", "2024-12-31")
    monkeypatch.setenv("TEST_OPTIONAL_DATE_NONE", "none")

    assert runner._env_optional_date("TEST_OPTIONAL_DATE", None) == runner.dt.date(2024, 12, 31)
    assert runner._env_optional_date("TEST_OPTIONAL_DATE_NONE", runner.dt.date(2024, 1, 1)) is None


def test_runner_exposes_lseg_request_bound_env_defaults() -> None:
    runner_path = Path("src/thesis_pkg/notebooks_and_scripts/sec_ccm_unified_runner.py")
    source = runner_path.read_text(encoding="utf-8")

    assert 'LSEG_REQUEST_MIN_DATE = _env_optional_date(\n        "SEC_CCM_LSEG_REQUEST_MIN_DATE",\n        dt.date(1994, 1, 1),' in source
    assert 'LSEG_REQUEST_MAX_DATE = _env_optional_date(\n        "SEC_CCM_LSEG_REQUEST_MAX_DATE",\n        dt.date(2024, 12, 31),' in source


def test_runner_passes_lseg_request_bounds_to_request_building_stages() -> None:
    runner_path = Path("src/thesis_pkg/notebooks_and_scripts/sec_ccm_unified_runner.py")
    source = runner_path.read_text(encoding="utf-8")

    assert "request_min_date=LSEG_REQUEST_MIN_DATE" in source
    assert "request_max_date=LSEG_REQUEST_MAX_DATE" in source


def test_notebook_config_exports_lseg_request_bound_env_keys() -> None:
    notebook_path = Path("src/thesis_pkg/notebooks_and_scripts/sec_ccm_unified_runner.ipynb")
    source = notebook_path.read_text(encoding="utf-8")

    assert 'LSEG_REQUEST_MIN_DATE = \\"1994-01-01\\"  # ISO date | None' in source
    assert 'LSEG_REQUEST_MAX_DATE = \\"2024-12-31\\"  # ISO date | None' in source
    assert '"    \\"SEC_CCM_LSEG_REQUEST_MIN_DATE\\": LSEG_REQUEST_MIN_DATE,\\n"' in source
    assert '"    \\"SEC_CCM_LSEG_REQUEST_MAX_DATE\\": LSEG_REQUEST_MAX_DATE,\\n"' in source


def test_runner_exposes_finbert_sentence_postprocess_policy_env() -> None:
    runner_path = Path("src/thesis_pkg/notebooks_and_scripts/sec_ccm_unified_runner.py")
    source = runner_path.read_text(encoding="utf-8")

    assert 'SEC_CCM_FINBERT_SENTENCE_POSTPROCESS_POLICY' in source
    assert 'FINBERT_SENTENCE_POSTPROCESS_POLICY = _env_str(' in source
    assert 'postprocess_policy=FINBERT_SENTENCE_POSTPROCESS_POLICY' in source


def test_notebook_config_exports_finbert_sentence_postprocess_policy_env_key() -> None:
    notebook_path = Path("src/thesis_pkg/notebooks_and_scripts/sec_ccm_unified_runner.ipynb")
    source = notebook_path.read_text(encoding="utf-8")

    assert 'FINBERT_SENTENCE_POSTPROCESS_POLICY = \\"reference_stitch_protect_v3\\"' in source
    assert '"    \\"SEC_CCM_FINBERT_SENTENCE_POSTPROCESS_POLICY\\": FINBERT_SENTENCE_POSTPROCESS_POLICY,\\n"' in source


def test_runner_exposes_lm2011_extension_env_flags_and_orders_stage_after_finbert() -> None:
    runner_path = Path("src/thesis_pkg/notebooks_and_scripts/sec_ccm_unified_runner.py")
    source = runner_path.read_text(encoding="utf-8")

    assert 'SEC_CCM_RUN_LM2011_EXTENSION' in source
    assert 'LM2011_EXTENSION_REQUIRE_CLEANED_SCOPE_MATCH = _env_bool(' in source
    assert 'LM2011_EXTENSION_FINBERT_ANALYSIS_RUN_DIR = _env_optional_path(' in source
    assert 'LM2011_EXTENSION_FINBERT_PREPROCESS_RUN_DIR = _env_optional_path(' in source
    assert "run_lm2011_extension_dictionary_family_comparison_pipeline(extension_cfg)" in source

    finbert_marker = "if RUN_FINBERT_PREPROCESS or RUN_FINBERT_ANALYSIS:"
    extension_marker = "if RUN_LM2011_EXTENSION:"
    assert source.index(extension_marker) > source.index(finbert_marker)


def test_notebook_config_exports_lm2011_extension_env_keys() -> None:
    notebook_path = Path("src/thesis_pkg/notebooks_and_scripts/sec_ccm_unified_runner.ipynb")
    source = notebook_path.read_text(encoding="utf-8")

    assert 'RUN_LM2011_EXTENSION = ' in source
    assert 'LM2011_EXTENSION_OUTPUT_DIR = RUN_ROOT / \\"lm2011_extension\\"' in source
    assert 'LM2011_EXTENSION_REQUIRE_CLEANED_SCOPE_MATCH = True' in source
    assert 'LM2011_EXTENSION_FINBERT_ANALYSIS_RUN_DIR = None' in source
    assert 'LM2011_EXTENSION_FINBERT_PREPROCESS_RUN_DIR = None' in source
    assert '"    \\"SEC_CCM_RUN_LM2011_EXTENSION\\": RUN_LM2011_EXTENSION,\\n"' in source
    assert '"    \\"SEC_CCM_LM2011_EXTENSION_OUTPUT_DIR\\": LM2011_EXTENSION_OUTPUT_DIR,\\n"' in source
    assert (
        '"    \\"SEC_CCM_LM2011_EXTENSION_REQUIRE_CLEANED_SCOPE_MATCH\\": '
        'LM2011_EXTENSION_REQUIRE_CLEANED_SCOPE_MATCH,\\n"'
    ) in source
    assert (
        '"    \\"SEC_CCM_LM2011_EXTENSION_FINBERT_ANALYSIS_RUN_DIR\\": '
        'LM2011_EXTENSION_FINBERT_ANALYSIS_RUN_DIR,\\n"'
    ) in source
    assert (
        '"    \\"SEC_CCM_LM2011_EXTENSION_FINBERT_PREPROCESS_RUN_DIR\\": '
        'LM2011_EXTENSION_FINBERT_PREPROCESS_RUN_DIR,\\n"'
    ) in source


def test_build_lm2011_extension_run_config_prefers_same_run_finbert_artifacts() -> None:
    lm2011_paths = runner.LM2011RunnerPaths(
        sample_root=Path("sample"),
        upstream_run_root=Path("upstream"),
        additional_data_dir=Path("additional"),
        output_dir=Path("output"),
        local_work_root=Path("work"),
        year_merged_dir=Path("year_merged"),
        sample_backbone_path=Path("backbone.parquet"),
        daily_panel_path=Path("daily.parquet"),
        text_features_full_10k_path=None,
        text_features_full_10k_path_is_explicit=False,
        text_features_mda_path=None,
        text_features_mda_path_is_explicit=False,
        ccm_base_dir=Path("ccm"),
        matched_clean_path=Path("matched.parquet"),
        items_analysis_dir=Path("items_analysis"),
        doc_ownership_path=Path("doc_ownership.parquet"),
        doc_analyst_selected_path=Path("doc_analyst.parquet"),
        filingdates_path=Path("filingdates.parquet"),
        quarterly_balance_sheet_path=Path("bsq.parquet"),
        quarterly_income_statement_path=Path("isq.parquet"),
        quarterly_period_descriptor_path=Path("pdq.parquet"),
        annual_balance_sheet_path=Path("bsa.parquet"),
        annual_income_statement_path=Path("isa.parquet"),
        annual_period_descriptor_path=Path("pda.parquet"),
        annual_fiscal_market_path=Path("fma.parquet"),
        company_history_path=Path("companyhistory.parquet"),
        company_description_path=Path("companydescription.parquet"),
        ff_daily_csv_path=Path("ff_daily.csv"),
        ff_monthly_csv_path=Path("ff_monthly.csv"),
        momentum_monthly_csv_path=Path("mom.csv"),
        ff48_siccodes_path=Path("ff48.txt"),
        monthly_stock_path=None,
        ff_monthly_with_mom_path=None,
        full_10k_cleaning_contract="lm2011_paper",
        full_10k_text_feature_batch_size=4,
        mda_text_feature_batch_size=20,
        recompute_event_screen_surface=False,
        recompute_event_panel=False,
        recompute_regression_tables=False,
        event_window_doc_batch_size=50,
        print_ram_stats=False,
        ram_log_interval_batches=10,
    )
    analysis_artifacts = SimpleNamespace(
        run_dir=Path("same_run_finbert_analysis"),
        run_manifest_path=Path("same_run_finbert_analysis") / "run_manifest.json",
        item_features_long_path=Path("same_run_finbert_analysis") / "item_features_long.parquet",
    )
    preprocessing_artifacts = SimpleNamespace(
        run_dir=Path("same_run_finbert_preprocess"),
        run_manifest_path=Path("same_run_finbert_preprocess") / "run_manifest.json",
        cleaned_item_scopes_dir=Path("same_run_finbert_preprocess") / "cleaned_item_scopes" / "by_year",
    )

    cfg = runner._build_lm2011_extension_run_config(
        lm2011_paths=lm2011_paths,
        lm2011_output_dir=Path("lm2011_post_refinitiv"),
        output_dir=Path("lm2011_extension"),
        require_cleaned_scope_match=True,
        recompute_text_features_full_10k=True,
        recompute_text_features_mda=True,
        recompute_event_screen_surface=False,
        recompute_event_panel=False,
        finbert_analysis_run_dir=Path("explicit_analysis"),
        finbert_preprocessing_run_dir=Path("explicit_preprocess"),
        finbert_analysis_artifacts=analysis_artifacts,
        finbert_preprocessing_artifacts=preprocessing_artifacts,
    )

    assert cfg.finbert_analysis_run_dir == Path("same_run_finbert_analysis")
    assert cfg.finbert_analysis_manifest_path == Path("same_run_finbert_analysis") / "run_manifest.json"
    assert cfg.finbert_item_features_long_path == (
        Path("same_run_finbert_analysis") / "item_features_long.parquet"
    )
    assert cfg.finbert_preprocessing_run_dir == Path("same_run_finbert_preprocess")
    assert cfg.finbert_preprocessing_manifest_path == (
        Path("same_run_finbert_preprocess") / "run_manifest.json"
    )
    assert cfg.finbert_cleaned_item_scopes_dir == (
        Path("same_run_finbert_preprocess") / "cleaned_item_scopes" / "by_year"
    )
    assert cfg.year_merged_dir == Path("year_merged")
    assert cfg.matched_clean_path == Path("matched.parquet")
    assert cfg.filingdates_path == Path("filingdates.parquet")
    assert cfg.daily_panel_path == Path("daily.parquet")
    assert cfg.doc_ownership_path == Path("doc_ownership.parquet")
    assert cfg.annual_balance_sheet_path == Path("bsa.parquet")
    assert cfg.annual_income_statement_path == Path("isa.parquet")
    assert cfg.annual_period_descriptor_path == Path("pda.parquet")
    assert cfg.annual_fiscal_market_path == Path("fma.parquet")
    assert cfg.ff_daily_csv_path == Path("ff_daily.csv")
    assert cfg.local_work_root == Path("work") / "lm2011_extension"
    assert cfg.full_10k_cleaning_contract == "lm2011_paper"
    assert cfg.full_10k_text_feature_batch_size == 4
    assert cfg.event_window_doc_batch_size == 50
    assert cfg.recompute_text_features_full_10k is True
    assert cfg.recompute_text_features_mda is True
    assert cfg.recompute_event_screen_surface is False
    assert cfg.recompute_event_panel is False
    assert cfg.event_panel_path == Path("lm2011_post_refinitiv") / "lm2011_event_panel.parquet"


def test_runner_and_notebook_share_lm2011_memory_hardened_defaults() -> None:
    runner_source = Path("src/thesis_pkg/notebooks_and_scripts/sec_ccm_unified_runner.py").read_text(
        encoding="utf-8"
    )
    notebook_source = Path("src/thesis_pkg/notebooks_and_scripts/sec_ccm_unified_runner.ipynb").read_text(
        encoding="utf-8"
    )

    assert "DEFAULT_LM2011_FULL_10K_CLEANING_CONTRACT" in runner_source
    assert "DEFAULT_LM2011_FULL_10K_TEXT_FEATURE_BATCH_SIZE" in runner_source
    assert "DEFAULT_LM2011_MDA_TEXT_FEATURE_BATCH_SIZE" in runner_source
    assert "DEFAULT_LM2011_EVENT_WINDOW_DOC_BATCH_SIZE" in runner_source
    assert 'LM2011_FULL_10K_CLEANING_CONTRACT = _env_str(\n        "SEC_CCM_LM2011_FULL_10K_CLEANING_CONTRACT",\n        DEFAULT_LM2011_FULL_10K_CLEANING_CONTRACT,' in runner_source
    assert 'LM2011_FULL_10K_TEXT_FEATURE_BATCH_SIZE = _env_int(\n        "SEC_CCM_LM2011_FULL_10K_TEXT_FEATURE_BATCH_SIZE",' in runner_source
    assert 'LM2011_MDA_TEXT_FEATURE_BATCH_SIZE = _env_int(\n        "SEC_CCM_LM2011_MDA_TEXT_FEATURE_BATCH_SIZE",' in runner_source
    assert 'LM2011_EVENT_WINDOW_DOC_BATCH_SIZE = _env_int(\n        "SEC_CCM_LM2011_EVENT_WINDOW_DOC_BATCH_SIZE",\n        DEFAULT_LM2011_EVENT_WINDOW_DOC_BATCH_SIZE,' in runner_source
    assert 'LM2011_FULL_10K_CLEANING_CONTRACT = \\"lm2011_paper\\"' in notebook_source
    assert 'LM2011_FULL_10K_TEXT_FEATURE_BATCH_SIZE = 4' in notebook_source
    assert 'LM2011_MDA_TEXT_FEATURE_BATCH_SIZE = 20' in notebook_source
    assert 'LM2011_EVENT_WINDOW_DOC_BATCH_SIZE = 50' in notebook_source


def test_runner_and_notebook_export_lm2011_recompute_stage_env_keys() -> None:
    runner_source = Path("src/thesis_pkg/notebooks_and_scripts/sec_ccm_unified_runner.py").read_text(
        encoding="utf-8"
    )
    notebook_source = Path("src/thesis_pkg/notebooks_and_scripts/sec_ccm_unified_runner.ipynb").read_text(
        encoding="utf-8"
    )

    assert 'SEC_CCM_LM2011_RECOMPUTE_EVENT_SCREEN_SURFACE' in runner_source
    assert 'SEC_CCM_LM2011_RECOMPUTE_EVENT_PANEL' in runner_source
    assert 'SEC_CCM_LM2011_RECOMPUTE_REGRESSION_TABLES' in runner_source
    assert 'LM2011_RECOMPUTE_EVENT_SCREEN_SURFACE = False' in notebook_source
    assert 'LM2011_RECOMPUTE_EVENT_PANEL = False' in notebook_source
    assert 'LM2011_RECOMPUTE_REGRESSION_TABLES = False' in notebook_source
    assert (
        '"    \\"SEC_CCM_LM2011_RECOMPUTE_EVENT_SCREEN_SURFACE\\": '
        'LM2011_RECOMPUTE_EVENT_SCREEN_SURFACE,\\n"'
    ) in notebook_source
    assert (
        '"    \\"SEC_CCM_LM2011_RECOMPUTE_EVENT_PANEL\\": '
        'LM2011_RECOMPUTE_EVENT_PANEL,\\n"'
    ) in notebook_source
    assert (
        '"    \\"SEC_CCM_LM2011_RECOMPUTE_REGRESSION_TABLES\\": '
        'LM2011_RECOMPUTE_REGRESSION_TABLES,\\n"'
    ) in notebook_source


def test_runner_and_notebook_export_ram_logging_env_keys() -> None:
    runner_source = Path("src/thesis_pkg/notebooks_and_scripts/sec_ccm_unified_runner.py").read_text(
        encoding="utf-8"
    )
    notebook_source = Path("src/thesis_pkg/notebooks_and_scripts/sec_ccm_unified_runner.ipynb").read_text(
        encoding="utf-8"
    )

    assert 'SEC_CCM_PRINT_RAM_STATS' in runner_source
    assert 'SEC_CCM_RAM_LOG_INTERVAL_BATCHES' in runner_source
    assert 'PRINT_RAM_STATS = True' in notebook_source
    assert 'RAM_LOG_INTERVAL_BATCHES = 10' in notebook_source
    assert '"    \\"SEC_CCM_PRINT_RAM_STATS\\": PRINT_RAM_STATS,\\n"' in notebook_source
    assert '"    \\"SEC_CCM_RAM_LOG_INTERVAL_BATCHES\\": RAM_LOG_INTERVAL_BATCHES,\\n"' in notebook_source


def test_print_rows_table_uses_tabular_ascii_output(capsys) -> None:
    rows = [{"stage": "lookup", "artifact": "out", "path": "C:/tmp/out.parquet"}]

    runner._print_rows_table(rows, sort_by=["stage", "artifact"], empty_message="empty")

    captured = capsys.readouterr().out
    expected = pl.DataFrame(rows).sort(["stage", "artifact"]).write_csv(None, separator="\t")

    assert captured == expected + "\n"
    assert "┌" not in captured


def test_ram_snapshot_reports_linux_style_values(monkeypatch: MonkeyPatch) -> None:
    def _fake_read_proc(path: Path) -> dict[str, int]:
        if path.name == "meminfo":
            return {"MemTotal": 8 * 1024 * 1024, "MemAvailable": 3 * 1024 * 1024}
        return {"VmRSS": 512 * 1024, "VmHWM": 768 * 1024}

    monkeypatch.setattr(runner, "_read_proc_kb_map", _fake_read_proc)
    monkeypatch.setattr(
        runner,
        "_read_cgroup_memory_bytes",
        lambda: {
            "cgroup_limit_bytes": 6 * 1024 * 1024 * 1024,
            "cgroup_used_bytes": 2 * 1024 * 1024 * 1024,
        },
    )

    snapshot = runner._ram_snapshot("unit_test")

    assert snapshot == {
        "label": "unit_test",
        "process_rss_gb": 0.5,
        "process_hwm_gb": 0.75,
        "system_total_gb": 8.0,
        "system_available_gb": 3.0,
        "system_used_gb": 5.0,
        "cgroup_limit_gb": 6.0,
        "cgroup_used_gb": 2.0,
        "cgroup_available_gb": 4.0,
    }


def test_print_ram_snapshot_is_silent_when_disabled(capsys) -> None:
    runner._print_ram_snapshot("unit_test", enabled=False)

    assert capsys.readouterr().out == ""


def test_sec_ccm_unified_runner_notebook_bootstrap_is_valid() -> None:
    notebook_path = Path("src/thesis_pkg/notebooks_and_scripts/sec_ccm_unified_runner.ipynb")
    notebook = json.loads(notebook_path.read_text(encoding="utf-8"))

    assert len(notebook["cells"]) == 4
    bootstrap_cell = "".join(notebook["cells"][1]["source"])
    config_cell = "".join(notebook["cells"][2]["source"])
    run_cell = "".join(notebook["cells"][3]["source"])

    assert 'subprocess.check_call(["git", "clone"' in bootstrap_cell
    assert "CONFIG_ENV = {" in config_cell
    assert 'SEC_CCM_WORK_ROOT' in config_cell
    assert 'SEC_CCM_RUN_SEC_PARSE' in config_cell
    assert 'SEC_CCM_OUTPUT_DIR' in config_cell
    assert 'SEC_CCM_FINBERT_SENTENCE_POSTPROCESS_POLICY' in config_cell
    assert 'SEC_CCM_PRINT_RAM_STATS' in config_cell
    assert 'SEC_CCM_RAM_LOG_INTERVAL_BATCHES' in config_cell
    assert 'SEC_CCM_LM2011_FULL_10K_TEXT_FEATURE_BATCH_SIZE' in config_cell
    assert 'SEC_CCM_LM2011_MDA_TEXT_FEATURE_BATCH_SIZE' in config_cell
    assert 'ram_snapshot' in config_cell
    assert 'print_ram_snapshot("notebook_before_main")' in run_cell
    assert 'main = reload(module).main' in run_cell
    assert 'print_ram_snapshot("notebook_after_main")' in run_cell


def _configure_minimal_main_env(monkeypatch: MonkeyPatch, tmp_path: Path) -> dict[str, Path]:
    root = tmp_path / "runner"
    sample_root = root / "sample"
    run_root = root / "run"
    ccm_base_dir = root / "ccm_base"
    ccm_derived_dir = root / "ccm_derived"
    sec_batch_root = root / "sec_batches"
    sec_year_merged_dir = root / "year_merged"
    refinitiv_step1_dir = root / "refinitiv_step1"
    analyst_dir = refinitiv_step1_dir / "analyst_common_stock"
    local_tmp = root / "local_tmp"
    local_work = root / "local_work"
    local_item_work = root / "local_item_work"
    local_merge_work = root / "local_merge_work"

    for path in (
        sample_root,
        run_root,
        ccm_base_dir,
        ccm_derived_dir,
        sec_batch_root,
        sec_year_merged_dir,
        analyst_dir,
        local_tmp,
        local_work,
        local_item_work,
        local_merge_work,
    ):
        path.mkdir(parents=True, exist_ok=True)

    year_file = sec_year_merged_dir / "1995.parquet"
    pl.DataFrame(
        {
            "doc_id": ["0000000001:1995000001"],
            "cik_10": ["0000000001"],
            "file_date_filename": [runner.dt.date(1995, 1, 31)],
        }
    ).write_parquet(year_file)
    ff48_path = root / "FF_Siccodes_48_Industries.txt"
    ff48_path.write_text("placeholder", encoding="utf-8")

    env_values = {
        "SEC_CCM_DATA_PROFILE": "LOCAL_SAMPLE",
        "SEC_CCM_SAMPLE_ROOT": str(sample_root),
        "SEC_CCM_RUN_ROOT": str(run_root),
        "SEC_CCM_CCM_BASE_DIR": str(ccm_base_dir),
        "SEC_CCM_CCM_DERIVED_DIR": str(ccm_derived_dir),
        "SEC_CCM_SEC_BATCH_ROOT": str(sec_batch_root),
        "SEC_CCM_SEC_YEAR_MERGED_DIR": str(sec_year_merged_dir),
        "SEC_CCM_REFINITIV_STEP1_OUT_DIR": str(refinitiv_step1_dir),
        "SEC_CCM_REFINITIV_ANALYST_COMMON_STOCK_DIR": str(analyst_dir),
        "SEC_CCM_LOCAL_TMP": str(local_tmp),
        "SEC_CCM_LOCAL_WORK": str(local_work),
        "SEC_CCM_LOCAL_ITEM_WORK": str(local_item_work),
        "SEC_CCM_LOCAL_MERGE_WORK": str(local_merge_work),
        "SEC_CCM_FF48_SICCODES_PATH": str(ff48_path),
        "SEC_CCM_RUN_CCM_MODE": "REBUILD",
        "SEC_CCM_RUN_SEC_PARSE": "false",
        "SEC_CCM_RUN_SEC_YEARLY_MERGE": "false",
        "SEC_CCM_RUN_SEC_CCM_PREMERGE": "false",
        "SEC_CCM_RUN_REFINITIV_STEP1": "false",
        "SEC_CCM_RUN_REFINITIV_STEP1_RESOLUTION": "false",
        "SEC_CCM_RUN_REFINITIV_OWNERSHIP_UNIVERSE_HANDOFF": "false",
        "SEC_CCM_RUN_REFINITIV_OWNERSHIP_UNIVERSE_RESULTS": "false",
        "SEC_CCM_RUN_REFINITIV_OWNERSHIP_AUTHORITY": "false",
        "SEC_CCM_RUN_REFINITIV_INSTRUMENT_AUTHORITY": "false",
        "SEC_CCM_RUN_REFINITIV_DOC_OWNERSHIP_LM2011_EXACT_HANDOFF": "false",
        "SEC_CCM_RUN_REFINITIV_DOC_OWNERSHIP_LM2011_FALLBACK_HANDOFF": "false",
        "SEC_CCM_RUN_REFINITIV_DOC_OWNERSHIP_LM2011_FINALIZE": "false",
        "SEC_CCM_RUN_REFINITIV_DOC_ANALYST_LM2011_ANCHORS": "false",
        "SEC_CCM_RUN_REFINITIV_DOC_ANALYST_LM2011_SELECT": "false",
        "SEC_CCM_RUN_GATED_ITEM_EXTRACTION": "false",
        "SEC_CCM_RUN_UNMATCHED_DIAGNOSTIC_TRACK": "false",
        "SEC_CCM_RUN_NO_ITEM_DIAGNOSTICS": "false",
        "SEC_CCM_RUN_BOUNDARY_DIAGNOSTICS": "false",
        "SEC_CCM_RUN_VALIDATION_CHECKS": "false",
        "SEC_CCM_YEARS": "[1995]",
    }
    for key, value in env_values.items():
        monkeypatch.setenv(key, value)

    ccm_daily_path = root / "ccm_daily.parquet"
    ccm_daily_market_core_path = root / "ccm_market_core.parquet"
    ccm_daily_phase_b_surface_path = root / "ccm_phase_b_surface.parquet"
    ccm_daily_bridge_surface_path = root / "ccm_bridge_surface.parquet"
    canonical_link_path = root / "canonical_link.parquet"
    pl.DataFrame(
        {
            "KYPERMNO": [1],
            "CALDT": [runner.dt.date(1995, 1, 3)],
            "FINAL_RET": [0.01],
            "RET": [0.01],
            "FINAL_PRC": [10.0],
            "PRC": [10.0],
            "VOL": [1000.0],
            "SHROUT": [100.0],
            "SHRCD": [10],
            "EXCHCD": [1],
            "data_status": [0],
        }
    ).write_parquet(ccm_daily_path)
    pl.DataFrame({"CALDT": [runner.dt.date(1995, 1, 3)]}).write_parquet(ccm_daily_market_core_path)
    pl.DataFrame(
        {"KYPERMNO": [1], "CALDT": [runner.dt.date(1995, 1, 3)]}
    ).write_parquet(ccm_daily_phase_b_surface_path)
    pl.DataFrame({"dummy": [1]}).write_parquet(ccm_daily_bridge_surface_path)
    pl.DataFrame({"dummy": [1]}).write_parquet(canonical_link_path)

    def _fake_build_or_reuse_ccm_daily_stage(**_: object) -> dict[str, Path]:
        return {
            "ccm_daily_path": ccm_daily_path,
            "ccm_daily_market_core_path": ccm_daily_market_core_path,
            "ccm_daily_phase_b_surface_path": ccm_daily_phase_b_surface_path,
            "ccm_daily_bridge_surface_path": ccm_daily_bridge_surface_path,
            "canonical_link_path": canonical_link_path,
        }

    monkeypatch.setattr(runner, "build_or_reuse_ccm_daily_stage", _fake_build_or_reuse_ccm_daily_stage)
    monkeypatch.setattr(
        runner,
        "summarize_year_parquets",
        lambda _dir: [{"path": str(year_file), "status": "OK"}],
    )

    def _fake_build_light_metadata_dataset(
        *,
        parquet_dir: list[Path] | Path,
        out_path: Path,
        **_: object,
    ) -> Path:
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        pl.DataFrame({"doc_id": ["0000000001:1995000001"]}).write_parquet(out_path)
        return out_path

    monkeypatch.setattr(runner, "build_light_metadata_dataset", _fake_build_light_metadata_dataset)

    return {
        "root": root,
        "run_root": run_root,
        "ccm_base_dir": ccm_base_dir,
        "ccm_daily_path": ccm_daily_path,
        "refinitiv_step1_dir": refinitiv_step1_dir,
        "analyst_dir": analyst_dir,
        "sec_year_merged_dir": sec_year_merged_dir,
    }


def _write_minimal_analyst_request_universe(path: Path) -> None:
    pl.DataFrame(
        [
            {
                "request_group_id": "group-1",
                "gvkey_int": 1000,
                "effective_collection_ric": "AAA.N",
                "member_bridge_row_count": 1,
                "bridge_start_date_min": runner.dt.date(2020, 1, 1),
                "bridge_end_date_max": runner.dt.date(2020, 3, 31),
                "actuals_request_start_date": runner.dt.date(2020, 1, 1),
                "actuals_request_end_date": runner.dt.date(2020, 1, 31),
                "estimates_request_start_date": runner.dt.date(2020, 1, 1),
                "estimates_request_end_date": runner.dt.date(2020, 1, 31),
                "retrieval_eligible": True,
                "retrieval_exclusion_reason": None,
            }
        ]
    ).write_parquet(path)


def _write_minimal_doc_ownership_authority_artifacts(step1_dir: Path) -> dict[str, Path]:
    authority_dir = step1_dir / "ownership_authority_common_stock"
    authority_dir.mkdir(parents=True, exist_ok=True)
    decisions_path = authority_dir / "refinitiv_permno_ownership_authority_decisions.parquet"
    exceptions_path = authority_dir / "refinitiv_permno_ownership_authority_exceptions.parquet"
    pl.DataFrame(
        {
            "KYPERMNO": ["100"],
            "authoritative_ric": ["AAA.N"],
            "authoritative_source_family": ["CONVENTIONAL"],
            "authority_decision_status": ["STATIC_CONVENTIONAL"],
            "requires_review": [False],
        }
    ).write_parquet(decisions_path)
    pl.DataFrame(
        schema={
            "KYPERMNO": pl.Utf8,
            "authoritative_ric": pl.Utf8,
            "authoritative_source_family": pl.Utf8,
            "authority_window_start_date": pl.Date,
            "authority_window_end_date": pl.Date,
            "authority_exception_status": pl.Utf8,
        }
    ).write_parquet(exceptions_path)
    return {
        "decisions_path": decisions_path,
        "exceptions_path": exceptions_path,
    }


def test_main_reports_analyst_actuals_resume_compatibility_when_stage_enabled(
    monkeypatch: MonkeyPatch,
    tmp_path: Path,
) -> None:
    paths = _configure_minimal_main_env(monkeypatch, tmp_path)
    monkeypatch.setenv("SEC_CCM_RUN_REFINITIV_ANALYST_REQUEST_GROUPS", "false")
    monkeypatch.setenv("SEC_CCM_RUN_REFINITIV_ANALYST_ACTUALS", "true")
    monkeypatch.setenv("SEC_CCM_RUN_REFINITIV_ANALYST_ESTIMATES_MONTHLY", "false")
    monkeypatch.setenv("SEC_CCM_RUN_REFINITIV_ANALYST_NORMALIZE", "false")
    monkeypatch.setattr(runner, "LSEG_API_READY", True)

    request_universe_path = paths["analyst_dir"] / "refinitiv_analyst_request_universe_common_stock.parquet"
    _write_minimal_analyst_request_universe(request_universe_path)

    def _raise_actuals(**_: object) -> dict[str, Path]:
        raise LsegResumeCompatibilityError(
            stage="analyst_actuals",
            meta_key="stage:analyst_actuals:batch_plan_fingerprint",
            existing_value="plan_old",
            current_value="plan_new",
            ledger_path=paths["analyst_dir"] / "refinitiv_analyst_actuals_api_ledger.sqlite3",
            explanation="resume state mismatch",
            guidance=[
                str(paths["analyst_dir"] / "staging" / "analyst_actuals"),
                str(paths["analyst_dir"] / "refinitiv_analyst_actuals_raw.parquet"),
            ],
        )

    monkeypatch.setattr(runner, "run_refinitiv_step1_analyst_actuals_api_pipeline", _raise_actuals)

    with pytest.raises(SystemExit) as exc_info:
        runner.main()

    message = str(exc_info.value)
    assert "Refinitiv analyst actuals stage was enabled for this run" in message
    assert str(request_universe_path) in message
    assert "stage:analyst_actuals:batch_plan_fingerprint" in message
    assert "Stored value: 'plan_old'" in message
    assert "Current value: 'plan_new'" in message


def test_main_does_not_touch_actuals_when_actuals_flag_disabled(
    monkeypatch: MonkeyPatch,
    tmp_path: Path,
) -> None:
    paths = _configure_minimal_main_env(monkeypatch, tmp_path)
    monkeypatch.setenv("SEC_CCM_RUN_REFINITIV_ANALYST_REQUEST_GROUPS", "false")
    monkeypatch.setenv("SEC_CCM_RUN_REFINITIV_ANALYST_ACTUALS", "false")
    monkeypatch.setenv("SEC_CCM_RUN_REFINITIV_ANALYST_ESTIMATES_MONTHLY", "false")
    monkeypatch.setenv("SEC_CCM_RUN_REFINITIV_ANALYST_NORMALIZE", "false")
    monkeypatch.setattr(runner, "LSEG_API_READY", True)

    stale_ledger = paths["analyst_dir"] / "refinitiv_analyst_actuals_api_ledger.sqlite3"
    stale_ledger.write_text("stale", encoding="utf-8")

    def _unexpected_actuals(**_: object) -> dict[str, Path]:
        raise AssertionError("actuals stage should not run when disabled")

    monkeypatch.setattr(runner, "run_refinitiv_step1_analyst_actuals_api_pipeline", _unexpected_actuals)

    runner.main()


def test_main_normalize_reuses_existing_raw_artifacts_when_api_flags_disabled(
    monkeypatch: MonkeyPatch,
    tmp_path: Path,
    capsys,
) -> None:
    paths = _configure_minimal_main_env(monkeypatch, tmp_path)
    monkeypatch.setenv("SEC_CCM_RUN_REFINITIV_ANALYST_REQUEST_GROUPS", "false")
    monkeypatch.setenv("SEC_CCM_RUN_REFINITIV_ANALYST_ACTUALS", "false")
    monkeypatch.setenv("SEC_CCM_RUN_REFINITIV_ANALYST_ESTIMATES_MONTHLY", "false")
    monkeypatch.setenv("SEC_CCM_RUN_REFINITIV_ANALYST_NORMALIZE", "true")
    monkeypatch.setattr(runner, "LSEG_API_READY", True)

    actuals_raw_path = paths["analyst_dir"] / "refinitiv_analyst_actuals_raw.parquet"
    estimates_raw_path = paths["analyst_dir"] / "refinitiv_analyst_estimates_monthly_raw.parquet"
    pl.DataFrame({"item_id": ["a"]}).write_parquet(actuals_raw_path)
    pl.DataFrame({"item_id": ["b"]}).write_parquet(estimates_raw_path)

    captured_paths: dict[str, Path] = {}

    def _normalize_stub(
        *,
        actuals_raw_artifact_path: Path | str,
        estimates_raw_artifact_path: Path | str,
        output_dir: Path | str,
    ) -> dict[str, Path]:
        captured_paths["actuals"] = Path(actuals_raw_artifact_path)
        captured_paths["estimates"] = Path(estimates_raw_artifact_path)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        normalized_path = output_dir / "refinitiv_analyst_normalized_panel.parquet"
        rejections_path = output_dir / "refinitiv_analyst_normalization_rejections.parquet"
        pl.DataFrame({"dummy": [1]}).write_parquet(normalized_path)
        pl.DataFrame({"dummy": [1]}).write_parquet(rejections_path)
        return {
            "refinitiv_analyst_normalized_panel_parquet": normalized_path,
            "refinitiv_analyst_normalization_rejections_parquet": rejections_path,
        }

    monkeypatch.setattr(runner, "run_refinitiv_step1_analyst_normalize_pipeline", _normalize_stub)

    runner.main()

    output = capsys.readouterr().out
    assert "reusing existing analyst raw artifacts for normalization" in output
    assert captured_paths["actuals"] == actuals_raw_path
    assert captured_paths["estimates"] == estimates_raw_path


def test_main_rebuilds_lm2011_backbone_when_sec_ccm_premerge_runs(
    monkeypatch: MonkeyPatch,
    tmp_path: Path,
) -> None:
    paths = _configure_minimal_main_env(monkeypatch, tmp_path)
    monkeypatch.setenv("SEC_CCM_RUN_SEC_CCM_PREMERGE", "true")
    monkeypatch.setattr(runner, "LSEG_API_READY", False)

    sec_ccm_output_dir = paths["run_root"] / "sec_ccm_premerge"
    sec_ccm_output_dir.mkdir(parents=True, exist_ok=True)
    matched_clean_path = sec_ccm_output_dir / "sec_ccm_matched_clean.parquet"
    match_status_path = sec_ccm_output_dir / "sec_ccm_match_status.parquet"
    filingdates_path = paths["ccm_base_dir"] / "filingdates.parquet"
    pl.DataFrame({"doc_id": ["0000000001:1995000001"], "KYPERMNO": [100], "gvkey": [1000]}).write_parquet(
        matched_clean_path
    )
    pl.DataFrame(
        {
            "doc_id": ["0000000001:1995000001"],
            "match_reason_code": ["OK"],
            "match_flag": [True],
            "has_acceptance_datetime": [True],
        }
    ).write_parquet(match_status_path)
    pl.DataFrame({"LPERMNO": [100], "FILEDATE": [runner.dt.date(1995, 1, 31)], "SRCTYPE": ["10K"]}).write_parquet(
        filingdates_path
    )

    monkeypatch.setattr(
        runner,
        "run_sec_ccm_premerge_pipeline",
        lambda **_: {
            "sec_ccm_matched_clean": matched_clean_path,
            "sec_ccm_match_status": match_status_path,
        },
    )
    captured: dict[str, object] = {}

    def _write_backbone_stub(
        *,
        sec_year_paths: list[Path],
        matched_clean_path: Path,
        filingdates_path: Path,
        output_path: Path,
    ) -> Path:
        captured["sec_year_paths"] = sec_year_paths
        captured["matched_clean_path"] = matched_clean_path
        captured["filingdates_path"] = filingdates_path
        captured["output_path"] = output_path
        output_path.parent.mkdir(parents=True, exist_ok=True)
        pl.DataFrame(
            {
                "doc_id": ["0000000001:1995000001"],
                "cik_10": ["0000000001"],
                "filing_date": [runner.dt.date(1995, 1, 31)],
                "normalized_form": ["10-K"],
                "KYPERMNO": ["100"],
            }
        ).write_parquet(output_path)
        return output_path

    monkeypatch.setattr(runner, "_write_lm2011_backbone_artifact", _write_backbone_stub)

    runner.main()

    assert captured["matched_clean_path"] == matched_clean_path
    assert captured["filingdates_path"] == filingdates_path
    assert captured["output_path"] == sec_ccm_output_dir / "lm2011_sample_backbone.parquet"
    assert (sec_ccm_output_dir / "lm2011_sample_backbone.parquet").exists()


def test_main_doc_ownership_exact_uses_canonical_lm2011_backbone(
    monkeypatch: MonkeyPatch,
    tmp_path: Path,
) -> None:
    paths = _configure_minimal_main_env(monkeypatch, tmp_path)
    monkeypatch.setenv("SEC_CCM_RUN_REFINITIV_DOC_OWNERSHIP_LM2011_EXACT_HANDOFF", "true")
    monkeypatch.setattr(runner, "LSEG_API_READY", False)

    backbone_path = paths["run_root"] / "sec_ccm_premerge" / "lm2011_sample_backbone.parquet"
    backbone_path.parent.mkdir(parents=True, exist_ok=True)
    pl.DataFrame(
        {
            "doc_id": ["doc_1"],
            "cik_10": ["0000000001"],
            "filing_date": [runner.dt.date(1998, 3, 31)],
            "normalized_form": ["10-K"],
            "KYPERMNO": ["100"],
        }
    ).write_parquet(backbone_path)
    _write_minimal_doc_ownership_authority_artifacts(paths["refinitiv_step1_dir"])

    monkeypatch.setattr(
        runner,
        "_write_lm2011_backbone_artifact",
        lambda **_: (_ for _ in ()).throw(AssertionError("existing backbone artifact should be reused")),
    )

    captured: dict[str, Path] = {}

    def _exact_handoff_stub(
        *,
        doc_filing_artifact_path: Path | str,
        authority_decisions_artifact_path: Path | str,
        authority_exceptions_artifact_path: Path | str,
        output_dir: Path | str,
        request_min_date=None,
        request_max_date=None,
    ) -> dict[str, Path]:
        captured["doc_filing_artifact_path"] = Path(doc_filing_artifact_path)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        requests_path = output_dir / "refinitiv_lm2011_doc_ownership_exact_requests.parquet"
        raw_path = output_dir / "refinitiv_lm2011_doc_ownership_exact_raw.parquet"
        pl.DataFrame({"doc_id": ["doc_1"]}).write_parquet(requests_path)
        pl.DataFrame({"doc_id": ["doc_1"]}).write_parquet(raw_path)
        return {
            "refinitiv_lm2011_doc_ownership_exact_requests_parquet": requests_path,
            "refinitiv_lm2011_doc_ownership_exact_raw_parquet": raw_path,
        }

    monkeypatch.setattr(runner, "run_refinitiv_lm2011_doc_ownership_exact_handoff_pipeline", _exact_handoff_stub)

    runner.main()

    assert captured["doc_filing_artifact_path"] == backbone_path
    preflight_requests_path = (
        paths["run_root"]
        / "refinitiv_doc_ownership_lm2011"
        / "refinitiv_lm2011_doc_ownership_preflight_requests.parquet"
    )
    assert preflight_requests_path.exists()
    assert pl.read_parquet(preflight_requests_path).get_column("doc_id").to_list() == ["doc_1"]


def test_main_doc_ownership_exact_fails_fast_on_preflight_mismatch(
    monkeypatch: MonkeyPatch,
    tmp_path: Path,
) -> None:
    paths = _configure_minimal_main_env(monkeypatch, tmp_path)
    monkeypatch.setenv("SEC_CCM_RUN_REFINITIV_DOC_OWNERSHIP_LM2011_EXACT_HANDOFF", "true")
    monkeypatch.setattr(runner, "LSEG_API_READY", False)

    backbone_path = paths["run_root"] / "sec_ccm_premerge" / "lm2011_sample_backbone.parquet"
    backbone_path.parent.mkdir(parents=True, exist_ok=True)
    pl.DataFrame(
        {
            "doc_id": ["doc_backbone"],
            "cik_10": ["0000000001"],
            "filing_date": [runner.dt.date(1998, 3, 31)],
            "normalized_form": ["10-K"],
            "KYPERMNO": ["100"],
        }
    ).write_parquet(backbone_path)
    _write_minimal_doc_ownership_authority_artifacts(paths["refinitiv_step1_dir"])

    monkeypatch.setattr(
        runner,
        "build_refinitiv_lm2011_doc_ownership_requests",
        lambda *_, **__: pl.DataFrame(
            {
                "doc_id": ["doc_request"],
                "filing_date": [runner.dt.date(1998, 3, 31)],
                "KYPERMNO": ["100"],
                "authoritative_ric": ["AAA.N"],
                "authority_decision_status": ["STATIC_CONVENTIONAL"],
                "target_quarter_end": [runner.dt.date(1997, 12, 31)],
                "target_effective_date": [runner.dt.date(1998, 1, 1)],
                "fallback_window_start": [runner.dt.date(1998, 1, 1)],
                "fallback_window_end": [runner.dt.date(1998, 2, 15)],
                "retrieval_eligible": [True],
                "retrieval_exclusion_reason": [None],
            }
        ),
    )
    monkeypatch.setattr(
        runner,
        "run_refinitiv_lm2011_doc_ownership_exact_handoff_pipeline",
        lambda **_: (_ for _ in ()).throw(AssertionError("exact pipeline should not run after preflight mismatch")),
    )

    with pytest.raises(AssertionError, match="preflight detected a request/backbone universe mismatch"):
        runner.main()


def test_main_doc_analyst_anchors_use_canonical_lm2011_backbone(
    monkeypatch: MonkeyPatch,
    tmp_path: Path,
) -> None:
    paths = _configure_minimal_main_env(monkeypatch, tmp_path)
    monkeypatch.setenv("SEC_CCM_RUN_REFINITIV_DOC_ANALYST_LM2011_ANCHORS", "true")
    monkeypatch.setattr(runner, "LSEG_API_READY", False)

    backbone_path = paths["run_root"] / "sec_ccm_premerge" / "lm2011_sample_backbone.parquet"
    backbone_path.parent.mkdir(parents=True, exist_ok=True)
    pl.DataFrame(
        {
            "doc_id": ["doc_1", "doc_2"],
            "cik_10": ["0000000001", "0000000002"],
            "filing_date": [runner.dt.date(1998, 3, 31), runner.dt.date(1999, 3, 31)],
            "normalized_form": ["10-K", "10-K"],
            "KYPERMNO": [100, 200],
            "gvkey_int": [1000, 2000],
        }
    ).write_parquet(backbone_path)

    quarterly_balance_sheet_path = paths["ccm_base_dir"] / "balancesheetquarterly.parquet"
    quarterly_income_statement_path = paths["ccm_base_dir"] / "incomestatementquarterly.parquet"
    quarterly_period_descriptor_path = paths["ccm_base_dir"] / "perioddescriptorquarterly.parquet"
    pl.DataFrame({"gvkey_int": [1000]}).write_parquet(quarterly_balance_sheet_path)
    pl.DataFrame({"gvkey_int": [1000]}).write_parquet(quarterly_income_statement_path)
    pl.DataFrame({"gvkey_int": [1000]}).write_parquet(quarterly_period_descriptor_path)

    monkeypatch.setattr(runner, "build_quarterly_accounting_panel", lambda *_, **__: pl.DataFrame({"gvkey_int": [1]}).lazy())
    captured: dict[str, list[str]] = {}

    def _anchor_stub(
        *,
        sample_backbone_lf: pl.LazyFrame,
        quarterly_accounting_panel_lf: pl.LazyFrame,
        output_dir: Path | str,
    ) -> dict[str, Path]:
        captured["doc_ids"] = sample_backbone_lf.collect().get_column("doc_id").to_list()
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / "refinitiv_doc_analyst_request_anchors.parquet"
        pl.DataFrame({"doc_id": captured["doc_ids"]}).write_parquet(output_path)
        return {"refinitiv_doc_analyst_request_anchors_parquet": output_path}

    monkeypatch.setattr(runner, "run_refinitiv_lm2011_doc_analyst_anchor_pipeline", _anchor_stub)

    runner.main()

    assert captured["doc_ids"] == ["doc_1", "doc_2"]


def test_resolve_stage_toggle_prefers_explicit_flag_over_umbrella(monkeypatch: MonkeyPatch) -> None:
    monkeypatch.setenv("TEST_STAGE_FLAG", "false")

    assert (
        runner._resolve_stage_toggle(
            "TEST_STAGE_FLAG",
            umbrella_enabled=True,
            default_when_umbrella=True,
        )
        is False
    )


def test_resolve_finbert_batch_config_prefers_raw_overrides() -> None:
    batch_cfg = runner._resolve_finbert_batch_config(
        profile_name="baseline",
        short_batch_size=7,
        medium_batch_size=None,
        long_batch_size=3,
    )

    assert batch_cfg.short_batch_size == 7
    assert batch_cfg.medium_batch_size == runner.FINBERT_BATCH_PRESETS["baseline"].medium_batch_size
    assert batch_cfg.long_batch_size == 3


def test_resolve_finbert_bucket_edges_and_lengths_auto_match() -> None:
    bucket_edges = runner._resolve_finbert_bucket_edges(
        short_edge=64,
        medium_edge=128,
    )
    bucket_lengths = runner._resolve_finbert_bucket_lengths(
        bucket_edges=bucket_edges,
        short_max_length=None,
        medium_max_length=None,
        long_max_length=None,
    )

    assert bucket_edges.short_edge == 64
    assert bucket_edges.medium_edge == 128
    assert bucket_lengths.short_max_length == 64
    assert bucket_lengths.medium_max_length == 128
    assert bucket_lengths.long_max_length == 512


def test_resolve_finbert_bucket_lengths_prefers_explicit_overrides() -> None:
    bucket_lengths = runner._resolve_finbert_bucket_lengths(
        bucket_edges=runner._resolve_finbert_bucket_edges(
            short_edge=64,
            medium_edge=128,
        ),
        short_max_length=80,
        medium_max_length=None,
        long_max_length=400,
    )

    assert bucket_lengths.short_max_length == 80
    assert bucket_lengths.medium_max_length == 128
    assert bucket_lengths.long_max_length == 400


def test_main_runs_downstream_pipelines_and_indexes_manifests(
    monkeypatch: MonkeyPatch,
    tmp_path: Path,
    capsys,
) -> None:
    paths = _configure_minimal_main_env(monkeypatch, tmp_path)
    items_analysis_dir = paths["run_root"] / "items_analysis"
    items_analysis_dir.mkdir(parents=True, exist_ok=True)
    pl.DataFrame({"doc_id": ["0000000001:1995000001"]}).write_parquet(items_analysis_dir / "1995.parquet")
    for year in (2006, 2007, 2008):
        pl.DataFrame({"doc_id": [f"0000000001:{year}000001"]}).write_parquet(items_analysis_dir / f"{year}.parquet")
    doc_ownership_dir = paths["run_root"] / "refinitiv_doc_ownership_lm2011"
    doc_ownership_dir.mkdir(parents=True, exist_ok=True)
    pl.DataFrame({"doc_id": ["0000000001:1995000001"]}).write_parquet(
        doc_ownership_dir / "refinitiv_lm2011_doc_ownership.parquet"
    )
    doc_analyst_dir = paths["run_root"] / "refinitiv_doc_analyst_lm2011"
    doc_analyst_dir.mkdir(parents=True, exist_ok=True)
    pl.DataFrame({"doc_id": ["0000000001:1995000001"]}).write_parquet(
        doc_analyst_dir / "refinitiv_doc_analyst_selected.parquet"
    )
    monkeypatch.setenv("SEC_CCM_RUN_LM2011_POST_REFINITIV", "true")
    monkeypatch.setenv("SEC_CCM_RUN_FINBERT", "true")
    monkeypatch.setenv("SEC_CCM_FINBERT_SHORT_BATCH_SIZE", "5")
    monkeypatch.setenv("SEC_CCM_FINBERT_SHORT_EDGE", "64")
    monkeypatch.setenv("SEC_CCM_FINBERT_MEDIUM_EDGE", "128")
    monkeypatch.setenv("SEC_CCM_FINBERT_RUN_NAME", "shared_finbert")
    monkeypatch.setenv("SEC_CCM_FINBERT_SENTENCE_POSTPROCESS_POLICY", "none")

    captured: dict[str, object] = {}

    def _lm2011_stub(run_cfg):
        captured["lm2011_run_cfg"] = run_cfg
        run_cfg.paths.output_dir.mkdir(parents=True, exist_ok=True)
        (run_cfg.paths.output_dir / "lm2011_sample_run_manifest.json").write_text(
            "{}",
            encoding="utf-8",
        )
        return 0

    def _finbert_stub(*args, **kwargs):
        captured["finbert_args"] = args
        captured["finbert_kwargs"] = kwargs
        analysis_cfg = args[0]
        pre_run_dir = analysis_cfg.out_root / "_staged_intermediates" / "shared_finbert_sentence_preprocessing"
        pre_run_dir.mkdir(parents=True, exist_ok=True)
        (pre_run_dir / "run_manifest.json").write_text("{}", encoding="utf-8")
        analysis_run_dir = analysis_cfg.out_root / "shared_finbert"
        analysis_run_dir.mkdir(parents=True, exist_ok=True)
        (analysis_run_dir / "run_manifest.json").write_text("{}", encoding="utf-8")
        from types import SimpleNamespace

        return SimpleNamespace(
            preprocessing_artifacts=SimpleNamespace(
                run_dir=pre_run_dir,
                run_manifest_path=pre_run_dir / "run_manifest.json",
            ),
            analysis_artifacts=SimpleNamespace(
                run_dir=analysis_run_dir,
                run_manifest_path=analysis_run_dir / "run_manifest.json",
            ),
        )

    monkeypatch.setattr(runner, "run_lm2011_post_refinitiv_pipeline", _lm2011_stub)
    monkeypatch.setattr(runner, "run_finbert_pipeline", _finbert_stub)

    runner.main()

    output = capsys.readouterr().out
    lm2011_run_cfg = captured["lm2011_run_cfg"]
    finbert_analysis_cfg = captured["finbert_args"][0]

    assert lm2011_run_cfg.fail_closed_for_enabled_stages is True
    assert "sample_backbone" in lm2011_run_cfg.enabled_stages
    assert "ff_factors_monthly_with_mom_normalized" not in lm2011_run_cfg.enabled_stages
    assert lm2011_run_cfg.paths.local_work_root == (paths["root"] / "local_work" / "lm2011_post_refinitiv")
    assert finbert_analysis_cfg.batch_config.short_batch_size == 5
    assert finbert_analysis_cfg.sentence_dataset.bucket_edges.short_edge == 64
    assert finbert_analysis_cfg.sentence_dataset.bucket_edges.medium_edge == 128
    assert finbert_analysis_cfg.bucket_lengths.short_max_length == 64
    assert finbert_analysis_cfg.bucket_lengths.medium_max_length == 128
    assert finbert_analysis_cfg.sentence_dataset.postprocess_policy == "none"
    assert finbert_analysis_cfg.year_filter == runner.LOCAL_SAMPLE_FINBERT_YEARS
    assert captured["finbert_kwargs"] == {"preprocessing_cfg": None, "run_preprocess": True, "run_analysis": True}
    assert "lm2011_post_refinitiv_manifest_json" in output
    assert "finbert_analysis_manifest_json" in output


def test_main_lm2011_contract_env_override_wins(monkeypatch: MonkeyPatch, tmp_path: Path) -> None:
    paths = _configure_minimal_main_env(monkeypatch, tmp_path)
    items_analysis_dir = paths["run_root"] / "items_analysis"
    items_analysis_dir.mkdir(parents=True, exist_ok=True)
    pl.DataFrame({"doc_id": ["0000000001:1995000001"]}).write_parquet(items_analysis_dir / "1995.parquet")
    doc_ownership_dir = paths["run_root"] / "refinitiv_doc_ownership_lm2011"
    doc_ownership_dir.mkdir(parents=True, exist_ok=True)
    pl.DataFrame({"doc_id": ["0000000001:1995000001"]}).write_parquet(
        doc_ownership_dir / "refinitiv_lm2011_doc_ownership.parquet"
    )
    doc_analyst_dir = paths["run_root"] / "refinitiv_doc_analyst_lm2011"
    doc_analyst_dir.mkdir(parents=True, exist_ok=True)
    pl.DataFrame({"doc_id": ["0000000001:1995000001"]}).write_parquet(
        doc_analyst_dir / "refinitiv_doc_analyst_selected.parquet"
    )
    monkeypatch.setenv("SEC_CCM_RUN_LM2011_POST_REFINITIV", "true")
    monkeypatch.setenv("SEC_CCM_RUN_FINBERT", "false")
    monkeypatch.setenv("SEC_CCM_LM2011_FULL_10K_CLEANING_CONTRACT", "current")
    monkeypatch.setenv("SEC_CCM_LM2011_TEXT_FEATURE_BATCH_SIZE", "7")
    monkeypatch.setenv("SEC_CCM_LM2011_EVENT_WINDOW_DOC_BATCH_SIZE", "9")

    captured: dict[str, object] = {}

    def _lm2011_stub(run_cfg):
        captured["run_cfg"] = run_cfg
        return 0

    monkeypatch.setattr(runner, "run_lm2011_post_refinitiv_pipeline", _lm2011_stub)

    runner.main()

    lm2011_run_cfg = captured["run_cfg"]
    assert lm2011_run_cfg.paths.full_10k_cleaning_contract == "current"
    assert lm2011_run_cfg.paths.full_10k_text_feature_batch_size == 7
    assert lm2011_run_cfg.paths.mda_text_feature_batch_size == 7
    assert lm2011_run_cfg.paths.event_window_doc_batch_size == 9
    assert lm2011_run_cfg.paths.local_work_root == (paths["root"] / "local_work" / "lm2011_post_refinitiv")


def test_main_runs_pre_extension_gc_barrier_before_extension_entrypoint(
    monkeypatch: MonkeyPatch,
    tmp_path: Path,
) -> None:
    paths = _configure_minimal_main_env(monkeypatch, tmp_path)
    items_analysis_dir = paths["run_root"] / "items_analysis"
    items_analysis_dir.mkdir(parents=True, exist_ok=True)
    pl.DataFrame({"doc_id": ["0000000001:1995000001"]}).write_parquet(items_analysis_dir / "1995.parquet")
    doc_ownership_dir = paths["run_root"] / "refinitiv_doc_ownership_lm2011"
    doc_ownership_dir.mkdir(parents=True, exist_ok=True)
    pl.DataFrame({"doc_id": ["0000000001:1995000001"]}).write_parquet(
        doc_ownership_dir / "refinitiv_lm2011_doc_ownership.parquet"
    )
    doc_analyst_dir = paths["run_root"] / "refinitiv_doc_analyst_lm2011"
    doc_analyst_dir.mkdir(parents=True, exist_ok=True)
    pl.DataFrame({"doc_id": ["0000000001:1995000001"]}).write_parquet(
        doc_analyst_dir / "refinitiv_doc_analyst_selected.parquet"
    )

    monkeypatch.setenv("SEC_CCM_RUN_LM2011_POST_REFINITIV", "true")
    monkeypatch.setenv("SEC_CCM_RUN_FINBERT", "false")
    monkeypatch.setenv("SEC_CCM_RUN_LM2011_EXTENSION", "true")

    events: list[str] = []

    def _lm2011_stub(run_cfg):
        events.append("lm2011")
        run_cfg.paths.output_dir.mkdir(parents=True, exist_ok=True)
        (run_cfg.paths.output_dir / "lm2011_sample_run_manifest.json").write_text("{}", encoding="utf-8")
        return 0

    def _capture_snapshot(label: str, *, enabled: bool) -> None:
        if label in {
            "sec_ccm_unified_runner_after_lm2011",
            "sec_ccm_unified_runner_before_extension_gc",
            "sec_ccm_unified_runner_after_pre_extension_gc",
        }:
            events.append(f"ram:{label}")

    monkeypatch.setattr(runner, "run_lm2011_post_refinitiv_pipeline", _lm2011_stub)
    monkeypatch.setattr(
        runner,
        "_build_lm2011_extension_run_config",
        lambda **_: events.append("build_extension_cfg") or SimpleNamespace(),
    )
    monkeypatch.setattr(
        runner,
        "run_lm2011_extension_dictionary_family_comparison_pipeline",
        lambda _cfg: events.append("extension"),
    )
    monkeypatch.setattr(runner, "_print_ram_snapshot", _capture_snapshot)
    monkeypatch.setattr(runner.gc, "collect", lambda: events.append("gc"))

    runner.main()

    after_lm2011_idx = events.index("ram:sec_ccm_unified_runner_after_lm2011")
    before_extension_gc_idx = events.index("ram:sec_ccm_unified_runner_before_extension_gc")
    extension_gc_idx = next(
        idx for idx, event in enumerate(events) if idx > before_extension_gc_idx and event == "gc"
    )
    after_extension_gc_idx = events.index("ram:sec_ccm_unified_runner_after_pre_extension_gc")
    build_extension_cfg_idx = events.index("build_extension_cfg")
    extension_idx = events.index("extension")

    assert (
        after_lm2011_idx
        < before_extension_gc_idx
        < extension_gc_idx
        < after_extension_gc_idx
        < build_extension_cfg_idx
        < extension_idx
    )


def test_main_lm2011_defaults_daily_panel_to_ccm_daily_path(
    monkeypatch: MonkeyPatch,
    tmp_path: Path,
) -> None:
    paths = _configure_minimal_main_env(monkeypatch, tmp_path)
    items_analysis_dir = paths["run_root"] / "items_analysis"
    items_analysis_dir.mkdir(parents=True, exist_ok=True)
    pl.DataFrame({"doc_id": ["0000000001:1995000001"]}).write_parquet(items_analysis_dir / "1995.parquet")
    doc_ownership_dir = paths["run_root"] / "refinitiv_doc_ownership_lm2011"
    doc_ownership_dir.mkdir(parents=True, exist_ok=True)
    pl.DataFrame({"doc_id": ["0000000001:1995000001"]}).write_parquet(
        doc_ownership_dir / "refinitiv_lm2011_doc_ownership.parquet"
    )
    doc_analyst_dir = paths["run_root"] / "refinitiv_doc_analyst_lm2011"
    doc_analyst_dir.mkdir(parents=True, exist_ok=True)
    pl.DataFrame({"doc_id": ["0000000001:1995000001"]}).write_parquet(
        doc_analyst_dir / "refinitiv_doc_analyst_selected.parquet"
    )
    sec_ccm_output_dir = paths["run_root"] / "sec_ccm_premerge"
    sec_ccm_output_dir.mkdir(parents=True, exist_ok=True)
    pl.DataFrame(
        {
            "doc_id": ["0000000001:1995000001"],
            "filing_date": [runner.dt.date(1995, 1, 31)],
            "aligned_caldt": [runner.dt.date(1995, 1, 31)],
            "kypermno": [1],
        }
    ).write_parquet(sec_ccm_output_dir / "final_flagged_data.parquet")

    monkeypatch.setenv("SEC_CCM_RUN_LM2011_POST_REFINITIV", "true")
    monkeypatch.setenv("SEC_CCM_RUN_FINBERT", "false")

    captured: dict[str, object] = {}

    def _lm2011_stub(run_cfg):
        captured["run_cfg"] = run_cfg
        return 0

    monkeypatch.setattr(runner, "run_lm2011_post_refinitiv_pipeline", _lm2011_stub)

    runner.main()

    lm2011_run_cfg = captured["run_cfg"]
    assert lm2011_run_cfg.paths.daily_panel_path == paths["ccm_daily_path"]


def test_main_lm2011_daily_panel_override_fails_fast_when_not_daily_panel(
    monkeypatch: MonkeyPatch,
    tmp_path: Path,
) -> None:
    paths = _configure_minimal_main_env(monkeypatch, tmp_path)
    items_analysis_dir = paths["run_root"] / "items_analysis"
    items_analysis_dir.mkdir(parents=True, exist_ok=True)
    pl.DataFrame({"doc_id": ["0000000001:1995000001"]}).write_parquet(items_analysis_dir / "1995.parquet")
    doc_ownership_dir = paths["run_root"] / "refinitiv_doc_ownership_lm2011"
    doc_ownership_dir.mkdir(parents=True, exist_ok=True)
    pl.DataFrame({"doc_id": ["0000000001:1995000001"]}).write_parquet(
        doc_ownership_dir / "refinitiv_lm2011_doc_ownership.parquet"
    )
    doc_analyst_dir = paths["run_root"] / "refinitiv_doc_analyst_lm2011"
    doc_analyst_dir.mkdir(parents=True, exist_ok=True)
    pl.DataFrame({"doc_id": ["0000000001:1995000001"]}).write_parquet(
        doc_analyst_dir / "refinitiv_doc_analyst_selected.parquet"
    )
    bad_daily_panel = tmp_path / "bad_daily_panel.parquet"
    pl.DataFrame(
        {
            "doc_id": ["0000000001:1995000001"],
            "filing_date": [runner.dt.date(1995, 1, 31)],
            "aligned_caldt": [runner.dt.date(1995, 1, 31)],
            "kypermno": [1],
        }
    ).write_parquet(bad_daily_panel)

    monkeypatch.setenv("SEC_CCM_RUN_LM2011_POST_REFINITIV", "true")
    monkeypatch.setenv("SEC_CCM_RUN_FINBERT", "false")
    monkeypatch.setenv("SEC_CCM_LM2011_DAILY_PANEL_PATH", str(bad_daily_panel))
    monkeypatch.setattr(
        runner,
        "run_lm2011_post_refinitiv_pipeline",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("LM2011 pipeline should not run")),
    )

    with pytest.raises(ValueError, match="LM2011 daily_panel_path must point to a CCM daily market panel parquet"):
        runner.main()


def test_main_lm2011_auto_detects_existing_text_feature_artifacts_from_output_dir(
    monkeypatch: MonkeyPatch,
    tmp_path: Path,
) -> None:
    paths = _configure_minimal_main_env(monkeypatch, tmp_path)
    items_analysis_dir = paths["run_root"] / "items_analysis"
    items_analysis_dir.mkdir(parents=True, exist_ok=True)
    pl.DataFrame({"doc_id": ["0000000001:1995000001"]}).write_parquet(items_analysis_dir / "1995.parquet")
    lm2011_output_dir = paths["run_root"] / "lm2011_post_refinitiv"
    lm2011_output_dir.mkdir(parents=True, exist_ok=True)
    full_10k_path = lm2011_output_dir / "lm2011_text_features_full_10k.parquet"
    mda_path = lm2011_output_dir / "lm2011_text_features_mda.parquet"
    pl.DataFrame({"doc_id": ["0000000001:1995000001"]}).write_parquet(full_10k_path)
    pl.DataFrame({"doc_id": ["0000000001:1995000001"]}).write_parquet(mda_path)

    monkeypatch.setenv("SEC_CCM_RUN_LM2011_POST_REFINITIV", "false")
    monkeypatch.setenv("SEC_CCM_RUN_LM2011_TEXT_FEATURES_FULL_10K", "true")
    monkeypatch.setenv("SEC_CCM_RUN_LM2011_TEXT_FEATURES_MDA", "true")
    monkeypatch.setenv("SEC_CCM_RUN_FINBERT", "false")

    captured: dict[str, object] = {}

    def _lm2011_stub(run_cfg):
        captured["run_cfg"] = run_cfg
        return 0

    monkeypatch.setattr(runner, "run_lm2011_post_refinitiv_pipeline", _lm2011_stub)

    runner.main()

    lm2011_run_cfg = captured["run_cfg"]
    assert lm2011_run_cfg.paths.text_features_full_10k_path == full_10k_path
    assert lm2011_run_cfg.paths.text_features_full_10k_path_is_explicit is False
    assert lm2011_run_cfg.paths.text_features_mda_path == mda_path
    assert lm2011_run_cfg.paths.text_features_mda_path_is_explicit is False


def test_main_lm2011_text_feature_override_env_paths_take_precedence(
    monkeypatch: MonkeyPatch,
    tmp_path: Path,
) -> None:
    paths = _configure_minimal_main_env(monkeypatch, tmp_path)
    items_analysis_dir = paths["run_root"] / "items_analysis"
    items_analysis_dir.mkdir(parents=True, exist_ok=True)
    pl.DataFrame({"doc_id": ["0000000001:1995000001"]}).write_parquet(items_analysis_dir / "1995.parquet")
    explicit_full_10k = tmp_path / "explicit" / "full_10k.parquet"
    explicit_mda = tmp_path / "explicit" / "mda.parquet"
    explicit_full_10k.parent.mkdir(parents=True, exist_ok=True)
    pl.DataFrame({"doc_id": ["0000000001:1995000001"]}).write_parquet(explicit_full_10k)
    pl.DataFrame({"doc_id": ["0000000001:1995000001"]}).write_parquet(explicit_mda)

    monkeypatch.setenv("SEC_CCM_RUN_LM2011_POST_REFINITIV", "false")
    monkeypatch.setenv("SEC_CCM_RUN_LM2011_TEXT_FEATURES_FULL_10K", "true")
    monkeypatch.setenv("SEC_CCM_RUN_LM2011_TEXT_FEATURES_MDA", "true")
    monkeypatch.setenv("SEC_CCM_RUN_FINBERT", "false")
    monkeypatch.setenv("SEC_CCM_LM2011_TEXT_FEATURES_FULL_10K_PATH", str(explicit_full_10k))
    monkeypatch.setenv("SEC_CCM_LM2011_TEXT_FEATURES_MDA_PATH", str(explicit_mda))

    captured: dict[str, object] = {}

    def _lm2011_stub(run_cfg):
        captured["run_cfg"] = run_cfg
        return 0

    monkeypatch.setattr(runner, "run_lm2011_post_refinitiv_pipeline", _lm2011_stub)

    runner.main()

    lm2011_run_cfg = captured["run_cfg"]
    assert lm2011_run_cfg.paths.text_features_full_10k_path == explicit_full_10k
    assert lm2011_run_cfg.paths.text_features_full_10k_path_is_explicit is True
    assert lm2011_run_cfg.paths.text_features_mda_path == explicit_mda
    assert lm2011_run_cfg.paths.text_features_mda_path_is_explicit is True


def test_main_lm2011_recompute_text_feature_env_disables_auto_reuse(
    monkeypatch: MonkeyPatch,
    tmp_path: Path,
) -> None:
    paths = _configure_minimal_main_env(monkeypatch, tmp_path)
    items_analysis_dir = paths["run_root"] / "items_analysis"
    items_analysis_dir.mkdir(parents=True, exist_ok=True)
    pl.DataFrame({"doc_id": ["0000000001:1995000001"]}).write_parquet(items_analysis_dir / "1995.parquet")
    lm2011_output_dir = paths["run_root"] / "lm2011_post_refinitiv"
    lm2011_output_dir.mkdir(parents=True, exist_ok=True)
    pl.DataFrame({"doc_id": ["0000000001:1995000001"]}).write_parquet(
        lm2011_output_dir / "lm2011_text_features_full_10k.parquet"
    )
    pl.DataFrame({"doc_id": ["0000000001:1995000001"]}).write_parquet(
        lm2011_output_dir / "lm2011_text_features_mda.parquet"
    )

    monkeypatch.setenv("SEC_CCM_RUN_LM2011_POST_REFINITIV", "false")
    monkeypatch.setenv("SEC_CCM_RUN_LM2011_TEXT_FEATURES_FULL_10K", "true")
    monkeypatch.setenv("SEC_CCM_RUN_LM2011_TEXT_FEATURES_MDA", "true")
    monkeypatch.setenv("SEC_CCM_RUN_FINBERT", "false")
    monkeypatch.setenv("SEC_CCM_LM2011_RECOMPUTE_TEXT_FEATURES_FULL_10K", "true")
    monkeypatch.setenv("SEC_CCM_LM2011_RECOMPUTE_TEXT_FEATURES_MDA", "true")

    captured: dict[str, object] = {}

    def _lm2011_stub(run_cfg):
        captured["run_cfg"] = run_cfg
        return 0

    monkeypatch.setattr(runner, "run_lm2011_post_refinitiv_pipeline", _lm2011_stub)

    runner.main()

    lm2011_run_cfg = captured["run_cfg"]
    assert lm2011_run_cfg.paths.text_features_full_10k_path is None
    assert lm2011_run_cfg.paths.text_features_full_10k_path_is_explicit is False
    assert lm2011_run_cfg.paths.text_features_mda_path is None
    assert lm2011_run_cfg.paths.text_features_mda_path_is_explicit is False


def test_main_lm2011_recompute_stage_env_flags_propagate(
    monkeypatch: MonkeyPatch,
    tmp_path: Path,
) -> None:
    paths = _configure_minimal_main_env(monkeypatch, tmp_path)
    items_analysis_dir = paths["run_root"] / "items_analysis"
    items_analysis_dir.mkdir(parents=True, exist_ok=True)
    pl.DataFrame({"doc_id": ["0000000001:1995000001"]}).write_parquet(items_analysis_dir / "1995.parquet")

    monkeypatch.setenv("SEC_CCM_RUN_LM2011_POST_REFINITIV", "false")
    monkeypatch.setenv("SEC_CCM_RUN_LM2011_EVENT_SCREEN_SURFACE", "true")
    monkeypatch.setenv("SEC_CCM_RUN_FINBERT", "false")
    monkeypatch.setenv("SEC_CCM_LM2011_RECOMPUTE_EVENT_SCREEN_SURFACE", "true")
    monkeypatch.setenv("SEC_CCM_LM2011_RECOMPUTE_EVENT_PANEL", "true")
    monkeypatch.setenv("SEC_CCM_LM2011_RECOMPUTE_REGRESSION_TABLES", "true")

    captured: dict[str, object] = {}

    def _lm2011_stub(run_cfg):
        captured["run_cfg"] = run_cfg
        return 0

    monkeypatch.setattr(runner, "run_lm2011_post_refinitiv_pipeline", _lm2011_stub)

    runner.main()

    lm2011_run_cfg = captured["run_cfg"]
    assert lm2011_run_cfg.paths.recompute_event_screen_surface is True
    assert lm2011_run_cfg.paths.recompute_event_panel is True
    assert lm2011_run_cfg.paths.recompute_regression_tables is True


def test_main_lm2011_recompute_text_feature_env_conflicts_with_override(
    monkeypatch: MonkeyPatch,
    tmp_path: Path,
) -> None:
    paths = _configure_minimal_main_env(monkeypatch, tmp_path)
    items_analysis_dir = paths["run_root"] / "items_analysis"
    items_analysis_dir.mkdir(parents=True, exist_ok=True)
    pl.DataFrame({"doc_id": ["0000000001:1995000001"]}).write_parquet(items_analysis_dir / "1995.parquet")
    explicit_full_10k = tmp_path / "explicit" / "full_10k.parquet"
    explicit_full_10k.parent.mkdir(parents=True, exist_ok=True)
    pl.DataFrame({"doc_id": ["0000000001:1995000001"]}).write_parquet(explicit_full_10k)

    monkeypatch.setenv("SEC_CCM_RUN_LM2011_POST_REFINITIV", "false")
    monkeypatch.setenv("SEC_CCM_RUN_LM2011_TEXT_FEATURES_FULL_10K", "true")
    monkeypatch.setenv("SEC_CCM_RUN_FINBERT", "false")
    monkeypatch.setenv("SEC_CCM_LM2011_TEXT_FEATURES_FULL_10K_PATH", str(explicit_full_10k))
    monkeypatch.setenv("SEC_CCM_LM2011_RECOMPUTE_TEXT_FEATURES_FULL_10K", "true")

    with pytest.raises(
        ValueError,
        match="SEC_CCM_LM2011_RECOMPUTE_TEXT_FEATURES_FULL_10K cannot be combined with SEC_CCM_LM2011_TEXT_FEATURES_FULL_10K_PATH",
    ):
        runner.main()
