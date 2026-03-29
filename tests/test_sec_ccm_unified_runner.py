from __future__ import annotations

import json
from pathlib import Path

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


def test_new_analyst_step1_booleans_exist_and_default_false() -> None:
    runner_path = Path("src/thesis_pkg/notebooks_and_scripts/sec_ccm_unified_runner.py")
    source = runner_path.read_text(encoding="utf-8")

    expected_defaults = [
        'RUN_REFINITIV_INSTRUMENT_AUTHORITY = _env_bool(\n        "SEC_CCM_RUN_REFINITIV_INSTRUMENT_AUTHORITY",\n        True,',
        'RUN_REFINITIV_ANALYST_REQUEST_GROUPS = _env_bool(\n        "SEC_CCM_RUN_REFINITIV_ANALYST_REQUEST_GROUPS",\n        False,',
        'RUN_REFINITIV_ANALYST_ACTUALS = _env_bool(\n        "SEC_CCM_RUN_REFINITIV_ANALYST_ACTUALS",\n        False,',
        'RUN_REFINITIV_ANALYST_ESTIMATES_MONTHLY = _env_bool(\n        "SEC_CCM_RUN_REFINITIV_ANALYST_ESTIMATES_MONTHLY",\n        False,',
        'RUN_REFINITIV_ANALYST_NORMALIZE = _env_bool(\n        "SEC_CCM_RUN_REFINITIV_ANALYST_NORMALIZE",\n        False,',
    ]

    for snippet in expected_defaults:
        assert snippet in source


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


def test_print_rows_table_uses_tabular_ascii_output(capsys) -> None:
    rows = [{"stage": "lookup", "artifact": "out", "path": "C:/tmp/out.parquet"}]

    runner._print_rows_table(rows, sort_by=["stage", "artifact"], empty_message="empty")

    captured = capsys.readouterr().out
    expected = pl.DataFrame(rows).sort(["stage", "artifact"]).write_csv(None, separator="\t")

    assert captured == expected + "\n"
    assert "┌" not in captured


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
    assert "from thesis_pkg.notebooks_and_scripts.sec_ccm_unified_runner import main" in run_cell
    assert "main()" in run_cell


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
    pl.DataFrame({"dummy": [1]}).write_parquet(ccm_daily_path)
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
