from __future__ import annotations

import json
from pathlib import Path

import polars as pl
import pytest

from thesis_pkg.notebooks_and_scripts import refinitiv_local_api_runner as runner
from thesis_pkg.pipelines.refinitiv.lseg_stage_audit import StageAuditResult


def _write_parquet(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pl.DataFrame({"row_id": [1]}).write_parquet(path)


def _build_run_root(tmp_path: Path) -> Path:
    run_root = tmp_path / "run_root"
    (run_root / "sec_ccm_premerge").mkdir(parents=True)
    (run_root / "refinitiv_step1" / "analyst_common_stock").mkdir(parents=True)
    (run_root / "refinitiv_step1" / "ownership_universe_common_stock").mkdir(parents=True)
    (run_root / "refinitiv_step1" / "ownership_authority_common_stock").mkdir(parents=True)
    (run_root / "refinitiv_doc_ownership_lm2011").mkdir(parents=True)
    return run_root


def test_parse_args_defaults() -> None:
    args = runner.parse_args(["--run-root", "C:/tmp/run"])

    assert args.stage_start == "lookup_api"
    assert args.stage_stop == "doc_finalize"
    assert args.selected_stages == runner.STAGE_ORDER
    assert args.resume is True
    assert args.audit_only is False
    assert args.recover_mode is None
    assert args.api_stage_mode == "full"
    assert args.batch_profile == "current"
    assert args.preflight_probe is False
    assert args.stage_manifest_required is True
    assert args.provider_session_name == "desktop.workspace"
    assert args.provider_config_name is None
    assert args.provider_timeout_seconds is None
    assert args.lookup_batch_size == runner.LOCAL_LOOKUP_BATCH_SIZE_DEFAULT
    assert args.ownership_batch_size == runner.LOCAL_OWNERSHIP_BATCH_SIZE_DEFAULT
    assert args.ownership_max_batch_items == runner.LOCAL_OWNERSHIP_MAX_BATCH_ITEMS_DEFAULT
    assert args.ownership_max_extra_rows_abs == runner.LOCAL_OWNERSHIP_MAX_EXTRA_ROWS_ABS_DEFAULT
    assert args.ownership_max_extra_rows_ratio == runner.LOCAL_OWNERSHIP_MAX_EXTRA_ROWS_RATIO_DEFAULT
    assert args.ownership_max_union_span_days == runner.LOCAL_OWNERSHIP_MAX_UNION_SPAN_DAYS_DEFAULT
    assert args.ownership_row_density_rows_per_day == runner.LOCAL_OWNERSHIP_ROW_DENSITY_ROWS_PER_DAY_DEFAULT
    assert args.include_ticker_fallback is runner.LOCAL_INCLUDE_TICKER_FALLBACK_DEFAULT
    assert args.analyst_actuals_batch_size == runner.LOCAL_ANALYST_ACTUALS_BATCH_SIZE_DEFAULT
    assert args.analyst_estimates_batch_size == runner.LOCAL_ANALYST_ESTIMATES_BATCH_SIZE_DEFAULT
    assert args.analyst_actuals_max_batch_items == runner.LOCAL_ANALYST_ACTUALS_MAX_BATCH_ITEMS_DEFAULT
    assert args.analyst_actuals_max_extra_rows_abs == runner.LOCAL_ANALYST_ACTUALS_MAX_EXTRA_ROWS_ABS_DEFAULT
    assert args.analyst_actuals_max_extra_rows_ratio == runner.LOCAL_ANALYST_ACTUALS_MAX_EXTRA_ROWS_RATIO_DEFAULT
    assert args.analyst_actuals_max_union_span_days == runner.LOCAL_ANALYST_ACTUALS_MAX_UNION_SPAN_DAYS_DEFAULT
    assert args.analyst_actuals_row_density_rows_per_day == runner.LOCAL_ANALYST_ACTUALS_ROW_DENSITY_ROWS_PER_DAY_DEFAULT
    assert args.analyst_estimates_max_batch_items == runner.LOCAL_ANALYST_ESTIMATES_MAX_BATCH_ITEMS_DEFAULT
    assert (
        args.analyst_estimates_max_extra_rows_abs
        == runner.LOCAL_ANALYST_ESTIMATES_MAX_EXTRA_ROWS_ABS_DEFAULT
    )
    assert (
        args.analyst_estimates_max_extra_rows_ratio
        == runner.LOCAL_ANALYST_ESTIMATES_MAX_EXTRA_ROWS_RATIO_DEFAULT
    )
    assert (
        args.analyst_estimates_max_union_span_days
        == runner.LOCAL_ANALYST_ESTIMATES_MAX_UNION_SPAN_DAYS_DEFAULT
    )
    assert (
        args.analyst_estimates_row_density_rows_per_day
        == runner.LOCAL_ANALYST_ESTIMATES_ROW_DENSITY_ROWS_PER_DAY_DEFAULT
    )
    assert args.doc_exact_batch_size == 15
    assert args.doc_fallback_batch_size == 5
    assert args.min_seconds_between_requests == 2.0
    assert args.max_attempts == 4


def test_parse_args_rejects_stage_list_with_stage_range() -> None:
    with pytest.raises(SystemExit):
        runner.parse_args(
            [
                "--run-root",
                "C:/tmp/run",
                "--stage-list",
                "lookup_api,resolution",
                "--stage-start",
                "lookup_api",
            ]
        )


def test_parse_args_stage_list_can_skip_analyst_normalize() -> None:
    args = runner.parse_args(
        [
            "--run-root",
            "C:/tmp/run",
            "--stage-list",
            (
                "lookup_api,resolution,instrument_authority,ownership_handoff,ownership_api,"
                "authority,analyst_request_groups,analyst_actuals_api,analyst_estimates_api,"
                "doc_exact_api,doc_fallback_api"
            ),
        ]
    )

    assert args.selected_stages == (
        "lookup_api",
        "resolution",
        "instrument_authority",
        "ownership_handoff",
        "ownership_api",
        "authority",
        "analyst_request_groups",
        "analyst_actuals_api",
        "analyst_estimates_api",
        "doc_exact_api",
        "doc_fallback_api",
    )
    assert "analyst_normalize" not in args.selected_stages
    assert args.stage_start == "lookup_api"
    assert args.stage_stop == "doc_fallback_api"


def test_parse_args_local_safe_batch_profile_applies_defaults() -> None:
    args = runner.parse_args(["--run-root", "C:/tmp/run", "--batch-profile", "local_safe"])

    assert args.lookup_batch_size == 25
    assert args.ownership_batch_size == 10
    assert args.ownership_max_batch_items == 10
    assert args.ownership_max_extra_rows_ratio == 0.25
    assert args.ownership_max_union_span_days == 3650
    assert args.analyst_actuals_batch_size == 25
    assert args.analyst_actuals_row_density_rows_per_day == pytest.approx(1.0 / 91.0)
    assert args.analyst_estimates_batch_size == 10
    assert args.analyst_estimates_row_density_rows_per_day == pytest.approx(1.0 / 30.5)
    assert args.doc_exact_batch_size == 15
    assert args.doc_fallback_batch_size == 5


def test_main_passes_analyst_interval_batching_args_and_records_manifest(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run_root = _build_run_root(tmp_path)
    _write_parquet(run_root / "refinitiv_step1" / "analyst_common_stock" / "refinitiv_analyst_request_universe_common_stock.parquet")

    monkeypatch.setattr(runner, "is_lseg_available", lambda: True)
    captured: dict[str, dict[str, Path | int | float | None]] = {}

    def actuals_stub(**kwargs: Path | int | float | None) -> dict[str, Path]:
        captured["actuals"] = kwargs
        output_path = kwargs["output_dir"] / "refinitiv_analyst_actuals_raw.parquet"
        _write_parquet(output_path)
        return {"refinitiv_analyst_actuals_raw_parquet": output_path}

    def estimates_stub(**kwargs: Path | int | float | None) -> dict[str, Path]:
        captured["estimates"] = kwargs
        output_path = kwargs["output_dir"] / "refinitiv_analyst_estimates_monthly_raw.parquet"
        _write_parquet(output_path)
        return {"refinitiv_analyst_estimates_monthly_raw_parquet": output_path}

    monkeypatch.setattr(runner, "run_refinitiv_step1_analyst_actuals_api_pipeline", actuals_stub)
    monkeypatch.setattr(runner, "run_refinitiv_step1_analyst_estimates_monthly_api_pipeline", estimates_stub)

    exit_code = runner.main(
        [
            "--run-root",
            str(run_root),
            "--stage-start",
            "analyst_actuals_api",
            "--stage-stop",
            "analyst_estimates_api",
            "--analyst-actuals-max-batch-items",
            "17",
            "--analyst-actuals-max-extra-rows-abs",
            "12.5",
            "--analyst-actuals-max-extra-rows-ratio",
            "0.2",
            "--analyst-actuals-max-union-span-days",
            "180",
            "--analyst-actuals-row-density-rows-per-day",
            "0.05",
            "--analyst-estimates-max-batch-items",
            "19",
            "--analyst-estimates-max-extra-rows-abs",
            "24.5",
            "--analyst-estimates-max-extra-rows-ratio",
            "0.15",
            "--analyst-estimates-max-union-span-days",
            "210",
            "--analyst-estimates-row-density-rows-per-day",
            "0.07",
        ]
    )

    assert exit_code == 0
    assert captured["actuals"]["max_batch_items"] == 17
    assert captured["actuals"]["max_extra_rows_abs"] == 12.5
    assert captured["actuals"]["max_extra_rows_ratio"] == 0.2
    assert captured["actuals"]["max_union_span_days"] == 180
    assert captured["actuals"]["row_density_rows_per_day"] == 0.05
    assert captured["estimates"]["max_batch_items"] == 19
    assert captured["estimates"]["max_extra_rows_abs"] == 24.5
    assert captured["estimates"]["max_extra_rows_ratio"] == 0.15
    assert captured["estimates"]["max_union_span_days"] == 210
    assert captured["estimates"]["row_density_rows_per_day"] == 0.07

    manifest = json.loads((run_root / "refinitiv_local_api_runner_manifest.json").read_text(encoding="utf-8"))
    assert manifest["batching"]["analyst_actuals_max_batch_items"] == 17
    assert manifest["batching"]["analyst_actuals_max_extra_rows_abs"] == 12.5
    assert manifest["batching"]["analyst_estimates_max_batch_items"] == 19
    assert manifest["batching"]["analyst_estimates_row_density_rows_per_day"] == 0.07


def test_main_passes_ownership_interval_batching_args_and_records_manifest(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run_root = _build_run_root(tmp_path)
    _write_parquet(
        run_root
        / "refinitiv_step1"
        / "ownership_universe_common_stock"
        / "refinitiv_ownership_universe_handoff_common_stock.parquet"
    )

    monkeypatch.setattr(runner, "is_lseg_available", lambda: True)
    captured: dict[str, dict[str, Path | int | float | None]] = {}

    def ownership_stub(**kwargs: Path | int | float | None) -> dict[str, Path]:
        captured["ownership"] = kwargs
        results_path = kwargs["output_dir"] / "refinitiv_ownership_universe_results.parquet"
        summary_path = kwargs["output_dir"] / "refinitiv_ownership_universe_row_summary.parquet"
        _write_parquet(results_path)
        _write_parquet(summary_path)
        return {
            "refinitiv_ownership_universe_results_parquet": results_path,
            "refinitiv_ownership_universe_row_summary_parquet": summary_path,
        }

    monkeypatch.setattr(runner, "run_refinitiv_step1_ownership_universe_api_pipeline", ownership_stub)

    exit_code = runner.main(
        [
            "--run-root",
            str(run_root),
            "--stage-start",
            "ownership_api",
            "--stage-stop",
            "ownership_api",
            "--ownership-max-batch-items",
            "13",
            "--ownership-max-extra-rows-abs",
            "11.5",
            "--ownership-max-extra-rows-ratio",
            "0.2",
            "--ownership-max-union-span-days",
            "180",
            "--ownership-row-density-rows-per-day",
            "0.03",
        ]
    )

    assert exit_code == 0
    assert captured["ownership"]["max_batch_items"] == 13
    assert captured["ownership"]["max_extra_rows_abs"] == 11.5
    assert captured["ownership"]["max_extra_rows_ratio"] == 0.2
    assert captured["ownership"]["max_union_span_days"] == 180
    assert captured["ownership"]["row_density_rows_per_day"] == 0.03

    manifest = json.loads((run_root / "refinitiv_local_api_runner_manifest.json").read_text(encoding="utf-8"))
    assert manifest["batching"]["ownership_max_batch_items"] == 13
    assert manifest["batching"]["ownership_max_extra_rows_abs"] == 11.5
    assert manifest["batching"]["ownership_row_density_rows_per_day"] == 0.03
    assert manifest["batching"]["include_ticker_fallback"] is runner.LOCAL_INCLUDE_TICKER_FALLBACK_DEFAULT


def test_parse_args_rejects_invalid_stage_range() -> None:
    with pytest.raises(SystemExit):
        runner.parse_args(
            [
                "--run-root",
                "C:/tmp/run",
                "--stage-start",
                "authority",
                "--stage-stop",
                "lookup_api",
            ]
        )


def test_main_lookup_stage_writes_manifest_and_uses_canonical_snapshot_path(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run_root = _build_run_root(tmp_path)
    snapshot_path = run_root / "refinitiv_step1" / "refinitiv_ric_lookup_handoff_common_stock_extended_snapshot.parquet"
    _write_parquet(snapshot_path)

    monkeypatch.setattr(runner, "is_lseg_available", lambda: True)
    captured: dict[str, Path] = {}

    def lookup_stub(**kwargs: Path) -> dict[str, Path]:
        captured["snapshot_parquet_path"] = kwargs["snapshot_parquet_path"]
        captured["output_dir"] = kwargs["output_dir"]
        output_path = kwargs["output_dir"] / "refinitiv_ric_lookup_handoff_common_stock_extended.parquet"
        _write_parquet(output_path)
        return {
            "refinitiv_ric_lookup_handoff_common_stock_extended_parquet": output_path,
        }

    monkeypatch.setattr(runner, "run_refinitiv_step1_lookup_api_pipeline", lookup_stub)

    exit_code = runner.main(
        [
            "--run-root",
            str(run_root),
            "--stage-start",
            "lookup_api",
            "--stage-stop",
            "lookup_api",
        ]
    )

    assert exit_code == 0
    assert captured["snapshot_parquet_path"] == snapshot_path
    assert captured["output_dir"] == run_root / "refinitiv_step1"

    manifest = json.loads((run_root / "refinitiv_local_api_runner_manifest.json").read_text(encoding="utf-8"))
    assert manifest["selected_stages"] == ["lookup_api"]
    assert "lookup_api.refinitiv_ric_lookup_handoff_common_stock_extended_parquet" in manifest["row_counts"]


def test_main_full_chain_uses_canonical_stage_paths(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run_root = _build_run_root(tmp_path)
    _write_parquet(run_root / "refinitiv_step1" / "refinitiv_bridge_universe.parquet")
    _write_parquet(run_root / "refinitiv_step1" / "refinitiv_ric_lookup_handoff_common_stock_extended_snapshot.parquet")
    _write_parquet(run_root / "sec_ccm_premerge" / "sec_ccm_matched_clean_filtered.parquet")

    monkeypatch.setattr(runner, "is_lseg_available", lambda: True)
    calls: list[tuple[str, dict[str, Path | int | float | None]]] = []

    def lookup_stub(**kwargs: Path) -> dict[str, Path]:
        calls.append(("lookup_api", kwargs))
        output_path = kwargs["output_dir"] / "refinitiv_ric_lookup_handoff_common_stock_extended.parquet"
        _write_parquet(output_path)
        return {"refinitiv_ric_lookup_handoff_common_stock_extended_parquet": output_path}

    def resolution_stub(**kwargs: Path) -> dict[str, Path]:
        calls.append(("resolution", kwargs))
        output_path = kwargs["output_dir"] / "refinitiv_ric_resolution_common_stock.parquet"
        _write_parquet(output_path)
        return {"refinitiv_ric_resolution_common_stock_parquet": output_path}

    def instrument_authority_stub(**kwargs: Path) -> dict[str, Path]:
        calls.append(("instrument_authority", kwargs))
        output_path = kwargs["output_dir"] / "refinitiv_instrument_authority_common_stock.parquet"
        _write_parquet(output_path)
        return {"refinitiv_instrument_authority_common_stock_parquet": output_path}

    def ownership_handoff_stub(**kwargs: Path) -> dict[str, Path]:
        calls.append(("ownership_handoff", kwargs))
        output_path = kwargs["output_dir"] / "refinitiv_ownership_universe_handoff_common_stock.parquet"
        _write_parquet(output_path)
        return {"refinitiv_ownership_universe_handoff_common_stock_parquet": output_path}

    def ownership_api_stub(**kwargs: Path | int | float) -> dict[str, Path]:
        calls.append(("ownership_api", kwargs))
        results_path = kwargs["output_dir"] / "refinitiv_ownership_universe_results.parquet"
        summary_path = kwargs["output_dir"] / "refinitiv_ownership_universe_row_summary.parquet"
        _write_parquet(results_path)
        _write_parquet(summary_path)
        return {
            "refinitiv_ownership_universe_results_parquet": results_path,
            "refinitiv_ownership_universe_row_summary_parquet": summary_path,
        }

    def authority_stub(**kwargs: Path) -> dict[str, Path]:
        calls.append(("authority", kwargs))
        decisions_path = kwargs["output_dir"] / "refinitiv_permno_ownership_authority_decisions.parquet"
        exceptions_path = kwargs["output_dir"] / "refinitiv_permno_ownership_authority_exceptions.parquet"
        _write_parquet(decisions_path)
        _write_parquet(exceptions_path)
        return {
            "refinitiv_permno_ownership_authority_decisions_parquet": decisions_path,
            "refinitiv_permno_ownership_authority_exceptions_parquet": exceptions_path,
        }

    def analyst_request_groups_stub(**kwargs: Path) -> dict[str, Path]:
        calls.append(("analyst_request_groups", kwargs))
        membership_path = kwargs["output_dir"] / "refinitiv_analyst_request_group_membership_common_stock.parquet"
        request_path = kwargs["output_dir"] / "refinitiv_analyst_request_universe_common_stock.parquet"
        _write_parquet(membership_path)
        _write_parquet(request_path)
        return {
            "refinitiv_analyst_request_group_membership_common_stock_parquet": membership_path,
            "refinitiv_analyst_request_universe_common_stock_parquet": request_path,
        }

    def analyst_actuals_stub(**kwargs: Path | int | float) -> dict[str, Path]:
        calls.append(("analyst_actuals_api", kwargs))
        raw_path = kwargs["output_dir"] / "refinitiv_analyst_actuals_raw.parquet"
        _write_parquet(raw_path)
        return {"refinitiv_analyst_actuals_raw_parquet": raw_path}

    def analyst_estimates_stub(**kwargs: Path | int | float) -> dict[str, Path]:
        calls.append(("analyst_estimates_api", kwargs))
        raw_path = kwargs["output_dir"] / "refinitiv_analyst_estimates_monthly_raw.parquet"
        _write_parquet(raw_path)
        return {"refinitiv_analyst_estimates_monthly_raw_parquet": raw_path}

    def analyst_normalize_stub(**kwargs: Path) -> dict[str, Path]:
        calls.append(("analyst_normalize", kwargs))
        panel_path = kwargs["output_dir"] / "refinitiv_analyst_normalized_panel.parquet"
        rejection_path = kwargs["output_dir"] / "refinitiv_analyst_normalization_rejections.parquet"
        _write_parquet(panel_path)
        _write_parquet(rejection_path)
        return {
            "refinitiv_analyst_normalized_panel_parquet": panel_path,
            "refinitiv_analyst_normalization_rejections_parquet": rejection_path,
        }

    def doc_exact_stub(**kwargs: Path | int | float) -> dict[str, Path]:
        calls.append(("doc_exact_api", kwargs))
        requests_path = kwargs["output_dir"] / "refinitiv_lm2011_doc_ownership_exact_requests.parquet"
        raw_path = kwargs["output_dir"] / "refinitiv_lm2011_doc_ownership_exact_raw.parquet"
        _write_parquet(requests_path)
        _write_parquet(raw_path)
        return {
            "refinitiv_lm2011_doc_ownership_exact_requests_parquet": requests_path,
            "refinitiv_lm2011_doc_ownership_exact_raw_parquet": raw_path,
        }

    def doc_fallback_stub(**kwargs: Path | int | float) -> dict[str, Path]:
        calls.append(("doc_fallback_api", kwargs))
        requests_path = kwargs["output_dir"] / "refinitiv_lm2011_doc_ownership_fallback_requests.parquet"
        raw_path = kwargs["output_dir"] / "refinitiv_lm2011_doc_ownership_fallback_raw.parquet"
        _write_parquet(requests_path)
        _write_parquet(raw_path)
        return {
            "refinitiv_lm2011_doc_ownership_fallback_requests_parquet": requests_path,
            "refinitiv_lm2011_doc_ownership_fallback_raw_parquet": raw_path,
        }

    def doc_finalize_stub(**kwargs: Path) -> dict[str, Path]:
        calls.append(("doc_finalize", kwargs))
        raw_path = kwargs["output_dir"] / "refinitiv_lm2011_doc_ownership_raw.parquet"
        final_path = kwargs["output_dir"] / "refinitiv_lm2011_doc_ownership.parquet"
        _write_parquet(raw_path)
        _write_parquet(final_path)
        return {
            "refinitiv_lm2011_doc_ownership_raw_parquet": raw_path,
            "refinitiv_lm2011_doc_ownership_parquet": final_path,
        }

    monkeypatch.setattr(runner, "run_refinitiv_step1_lookup_api_pipeline", lookup_stub)
    monkeypatch.setattr(runner, "run_refinitiv_step1_resolution_pipeline", resolution_stub)
    monkeypatch.setattr(runner, "run_refinitiv_step1_instrument_authority_pipeline", instrument_authority_stub)
    monkeypatch.setattr(runner, "run_refinitiv_step1_ownership_universe_handoff_pipeline", ownership_handoff_stub)
    monkeypatch.setattr(runner, "run_refinitiv_step1_ownership_universe_api_pipeline", ownership_api_stub)
    monkeypatch.setattr(runner, "run_refinitiv_step1_ownership_authority_pipeline", authority_stub)
    monkeypatch.setattr(runner, "run_refinitiv_step1_analyst_request_groups_pipeline", analyst_request_groups_stub)
    monkeypatch.setattr(runner, "run_refinitiv_step1_analyst_actuals_api_pipeline", analyst_actuals_stub)
    monkeypatch.setattr(runner, "run_refinitiv_step1_analyst_estimates_monthly_api_pipeline", analyst_estimates_stub)
    monkeypatch.setattr(runner, "run_refinitiv_step1_analyst_normalize_pipeline", analyst_normalize_stub)
    monkeypatch.setattr(runner, "run_refinitiv_lm2011_doc_ownership_exact_api_pipeline", doc_exact_stub)
    monkeypatch.setattr(runner, "run_refinitiv_lm2011_doc_ownership_fallback_api_pipeline", doc_fallback_stub)
    monkeypatch.setattr(runner, "run_refinitiv_lm2011_doc_ownership_finalize_pipeline", doc_finalize_stub)

    exit_code = runner.main(["--run-root", str(run_root)])

    assert exit_code == 0
    assert [stage for stage, _ in calls] == list(runner.STAGE_ORDER)
    assert calls[1][1]["filled_lookup_workbook_path"] == run_root / "refinitiv_step1" / "refinitiv_ric_lookup_handoff_common_stock_extended.parquet"
    assert calls[2][1]["bridge_artifact_path"] == run_root / "refinitiv_step1" / "refinitiv_bridge_universe.parquet"
    assert calls[3][1]["include_ticker_fallback"] is runner.LOCAL_INCLUDE_TICKER_FALLBACK_DEFAULT
    assert calls[4][1]["handoff_parquet_path"] == run_root / "refinitiv_step1" / "ownership_universe_common_stock" / "refinitiv_ownership_universe_handoff_common_stock.parquet"
    assert calls[5][1]["resolution_artifact_path"] == run_root / "refinitiv_step1" / "refinitiv_ric_resolution_common_stock.parquet"
    assert calls[5][1]["ownership_results_artifact_path"] == run_root / "refinitiv_step1" / "ownership_universe_common_stock" / "refinitiv_ownership_universe_results.parquet"
    assert calls[6][1]["instrument_authority_artifact_path"] == run_root / "refinitiv_step1" / "refinitiv_instrument_authority_common_stock.parquet"
    assert calls[7][1]["request_universe_parquet_path"] == run_root / "refinitiv_step1" / "analyst_common_stock" / "refinitiv_analyst_request_universe_common_stock.parquet"
    assert calls[8][1]["request_universe_parquet_path"] == run_root / "refinitiv_step1" / "analyst_common_stock" / "refinitiv_analyst_request_universe_common_stock.parquet"
    assert calls[10][1]["doc_filing_artifact_path"] == run_root / "sec_ccm_premerge" / "sec_ccm_matched_clean_filtered.parquet"
    assert calls[10][1]["authority_decisions_artifact_path"] == run_root / "refinitiv_step1" / "ownership_authority_common_stock" / "refinitiv_permno_ownership_authority_decisions.parquet"
    assert calls[11][1]["output_dir"] == run_root / "refinitiv_doc_ownership_lm2011"


def test_main_passes_no_ticker_fallback_to_ownership_handoff_stage(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run_root = _build_run_root(tmp_path)
    _write_parquet(run_root / "refinitiv_step1" / "refinitiv_ric_resolution_common_stock.parquet")

    monkeypatch.setattr(runner, "is_lseg_available", lambda: True)
    captured: dict[str, object] = {}

    def ownership_handoff_stub(**kwargs: Path | bool) -> dict[str, Path]:
        captured.update(kwargs)
        output_path = kwargs["output_dir"] / "refinitiv_ownership_universe_handoff_common_stock.parquet"
        _write_parquet(output_path)
        return {"refinitiv_ownership_universe_handoff_common_stock_parquet": output_path}

    monkeypatch.setattr(runner, "run_refinitiv_step1_ownership_universe_handoff_pipeline", ownership_handoff_stub)

    exit_code = runner.main(
        [
            "--run-root",
            str(run_root),
            "--stage-start",
            "ownership_handoff",
            "--stage-stop",
            "ownership_handoff",
            "--no-stage-manifest-required",
            "--no-ticker-fallback",
        ]
    )

    assert exit_code == 0
    assert captured["include_ticker_fallback"] is False


def test_main_stage_range_skips_earlier_stages_and_stops_at_stage_stop(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run_root = _build_run_root(tmp_path)
    _write_parquet(run_root / "refinitiv_step1" / "refinitiv_ric_resolution_common_stock.parquet")
    _write_parquet(run_root / "refinitiv_step1" / "ownership_universe_common_stock" / "refinitiv_ownership_universe_handoff_common_stock.parquet")

    monkeypatch.setattr(runner, "is_lseg_available", lambda: True)
    called_stages: list[str] = []

    def ownership_api_stub(**kwargs: Path | int | float) -> dict[str, Path]:
        called_stages.append("ownership_api")
        results_path = kwargs["output_dir"] / "refinitiv_ownership_universe_results.parquet"
        summary_path = kwargs["output_dir"] / "refinitiv_ownership_universe_row_summary.parquet"
        _write_parquet(results_path)
        _write_parquet(summary_path)
        return {
            "refinitiv_ownership_universe_results_parquet": results_path,
            "refinitiv_ownership_universe_row_summary_parquet": summary_path,
        }

    def authority_stub(**kwargs: Path) -> dict[str, Path]:
        called_stages.append("authority")
        decisions_path = kwargs["output_dir"] / "refinitiv_permno_ownership_authority_decisions.parquet"
        exceptions_path = kwargs["output_dir"] / "refinitiv_permno_ownership_authority_exceptions.parquet"
        _write_parquet(decisions_path)
        _write_parquet(exceptions_path)
        return {
            "refinitiv_permno_ownership_authority_decisions_parquet": decisions_path,
            "refinitiv_permno_ownership_authority_exceptions_parquet": exceptions_path,
        }

    monkeypatch.setattr(runner, "run_refinitiv_step1_ownership_universe_api_pipeline", ownership_api_stub)
    monkeypatch.setattr(runner, "run_refinitiv_step1_ownership_authority_pipeline", authority_stub)

    exit_code = runner.main(
        [
            "--run-root",
            str(run_root),
            "--stage-start",
            "ownership_api",
            "--stage-stop",
            "authority",
        ]
    )

    assert exit_code == 0
    assert called_stages == ["ownership_api", "authority"]


def test_main_stage_list_can_skip_analyst_normalize_and_continue_to_doc_api(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run_root = _build_run_root(tmp_path)
    _write_parquet(run_root / "refinitiv_step1" / "refinitiv_ric_lookup_handoff_common_stock_extended_snapshot.parquet")
    _write_parquet(run_root / "refinitiv_step1" / "refinitiv_bridge_universe.parquet")
    _write_parquet(run_root / "sec_ccm_premerge" / "sec_ccm_matched_clean_filtered.parquet")

    monkeypatch.setattr(runner, "is_lseg_available", lambda: True)
    called_stages: list[str] = []

    def _record(stage_name: str, output_names: list[str]):
        def _stub(**kwargs: Path | int | float | str | None) -> dict[str, Path]:
            called_stages.append(stage_name)
            artifacts: dict[str, Path] = {}
            for output_name in output_names:
                output_path = kwargs["output_dir"] / output_name
                _write_parquet(output_path)
                artifacts[output_name.replace(".parquet", "_parquet")] = output_path
            return artifacts

        return _stub

    monkeypatch.setattr(
        runner,
        "run_refinitiv_step1_lookup_api_pipeline",
        lambda **kwargs: (
            called_stages.append("lookup_api") or {
                "refinitiv_ric_lookup_handoff_common_stock_extended_parquet": (
                    _write_parquet(kwargs["output_dir"] / "refinitiv_ric_lookup_handoff_common_stock_extended.parquet")
                    or kwargs["output_dir"] / "refinitiv_ric_lookup_handoff_common_stock_extended.parquet"
                )
            }
        ),
    )
    monkeypatch.setattr(
        runner,
        "run_refinitiv_step1_resolution_pipeline",
        lambda **kwargs: (
            called_stages.append("resolution") or {
                "refinitiv_ric_resolution_common_stock_parquet": (
                    _write_parquet(kwargs["output_dir"] / "refinitiv_ric_resolution_common_stock.parquet")
                    or kwargs["output_dir"] / "refinitiv_ric_resolution_common_stock.parquet"
                )
            }
        ),
    )
    monkeypatch.setattr(
        runner,
        "run_refinitiv_step1_instrument_authority_pipeline",
        lambda **kwargs: (
            called_stages.append("instrument_authority") or {
                "refinitiv_instrument_authority_common_stock_parquet": (
                    _write_parquet(kwargs["output_dir"] / "refinitiv_instrument_authority_common_stock.parquet")
                    or kwargs["output_dir"] / "refinitiv_instrument_authority_common_stock.parquet"
                )
            }
        ),
    )
    monkeypatch.setattr(
        runner,
        "run_refinitiv_step1_ownership_universe_handoff_pipeline",
        lambda **kwargs: (
            called_stages.append("ownership_handoff") or {
                "refinitiv_ownership_universe_handoff_common_stock_parquet": (
                    _write_parquet(
                        kwargs["output_dir"] / "refinitiv_ownership_universe_handoff_common_stock.parquet"
                    )
                    or kwargs["output_dir"] / "refinitiv_ownership_universe_handoff_common_stock.parquet"
                )
            }
        ),
    )
    monkeypatch.setattr(
        runner,
        "run_refinitiv_step1_ownership_universe_api_pipeline",
        lambda **kwargs: (
            called_stages.append("ownership_api") or {
                "refinitiv_ownership_universe_results_parquet": (
                    _write_parquet(kwargs["output_dir"] / "refinitiv_ownership_universe_results.parquet")
                    or kwargs["output_dir"] / "refinitiv_ownership_universe_results.parquet"
                ),
                "refinitiv_ownership_universe_row_summary_parquet": (
                    _write_parquet(kwargs["output_dir"] / "refinitiv_ownership_universe_row_summary.parquet")
                    or kwargs["output_dir"] / "refinitiv_ownership_universe_row_summary.parquet"
                ),
            }
        ),
    )
    monkeypatch.setattr(
        runner,
        "run_refinitiv_step1_ownership_authority_pipeline",
        lambda **kwargs: (
            called_stages.append("authority") or {
                "refinitiv_permno_ownership_authority_decisions_parquet": (
                    _write_parquet(kwargs["output_dir"] / "refinitiv_permno_ownership_authority_decisions.parquet")
                    or kwargs["output_dir"] / "refinitiv_permno_ownership_authority_decisions.parquet"
                ),
                "refinitiv_permno_ownership_authority_exceptions_parquet": (
                    _write_parquet(kwargs["output_dir"] / "refinitiv_permno_ownership_authority_exceptions.parquet")
                    or kwargs["output_dir"] / "refinitiv_permno_ownership_authority_exceptions.parquet"
                ),
            }
        ),
    )
    monkeypatch.setattr(
        runner,
        "run_refinitiv_step1_analyst_request_groups_pipeline",
        lambda **kwargs: (
            called_stages.append("analyst_request_groups") or {
                "refinitiv_analyst_request_group_membership_common_stock_parquet": (
                    _write_parquet(
                        kwargs["output_dir"] / "refinitiv_analyst_request_group_membership_common_stock.parquet"
                    )
                    or kwargs["output_dir"] / "refinitiv_analyst_request_group_membership_common_stock.parquet"
                ),
                "refinitiv_analyst_request_universe_common_stock_parquet": (
                    _write_parquet(kwargs["output_dir"] / "refinitiv_analyst_request_universe_common_stock.parquet")
                    or kwargs["output_dir"] / "refinitiv_analyst_request_universe_common_stock.parquet"
                ),
            }
        ),
    )
    monkeypatch.setattr(
        runner,
        "run_refinitiv_step1_analyst_actuals_api_pipeline",
        lambda **kwargs: (
            called_stages.append("analyst_actuals_api") or {
                "refinitiv_analyst_actuals_raw_parquet": (
                    _write_parquet(kwargs["output_dir"] / "refinitiv_analyst_actuals_raw.parquet")
                    or kwargs["output_dir"] / "refinitiv_analyst_actuals_raw.parquet"
                )
            }
        ),
    )
    monkeypatch.setattr(
        runner,
        "run_refinitiv_step1_analyst_estimates_monthly_api_pipeline",
        lambda **kwargs: (
            called_stages.append("analyst_estimates_api") or {
                "refinitiv_analyst_estimates_monthly_raw_parquet": (
                    _write_parquet(kwargs["output_dir"] / "refinitiv_analyst_estimates_monthly_raw.parquet")
                    or kwargs["output_dir"] / "refinitiv_analyst_estimates_monthly_raw.parquet"
                )
            }
        ),
    )
    monkeypatch.setattr(
        runner,
        "run_refinitiv_lm2011_doc_ownership_exact_api_pipeline",
        lambda **kwargs: (
            called_stages.append("doc_exact_api") or {
                "refinitiv_lm2011_doc_ownership_exact_requests_parquet": (
                    _write_parquet(
                        kwargs["output_dir"] / "refinitiv_lm2011_doc_ownership_exact_requests.parquet"
                    )
                    or kwargs["output_dir"] / "refinitiv_lm2011_doc_ownership_exact_requests.parquet"
                ),
                "refinitiv_lm2011_doc_ownership_exact_raw_parquet": (
                    _write_parquet(kwargs["output_dir"] / "refinitiv_lm2011_doc_ownership_exact_raw.parquet")
                    or kwargs["output_dir"] / "refinitiv_lm2011_doc_ownership_exact_raw.parquet"
                ),
            }
        ),
    )
    monkeypatch.setattr(
        runner,
        "run_refinitiv_lm2011_doc_ownership_fallback_api_pipeline",
        lambda **kwargs: (
            called_stages.append("doc_fallback_api") or {
                "refinitiv_lm2011_doc_ownership_fallback_requests_parquet": (
                    _write_parquet(
                        kwargs["output_dir"] / "refinitiv_lm2011_doc_ownership_fallback_requests.parquet"
                    )
                    or kwargs["output_dir"] / "refinitiv_lm2011_doc_ownership_fallback_requests.parquet"
                ),
                "refinitiv_lm2011_doc_ownership_fallback_raw_parquet": (
                    _write_parquet(kwargs["output_dir"] / "refinitiv_lm2011_doc_ownership_fallback_raw.parquet")
                    or kwargs["output_dir"] / "refinitiv_lm2011_doc_ownership_fallback_raw.parquet"
                ),
            }
        ),
    )

    exit_code = runner.main(
        [
            "--run-root",
            str(run_root),
            "--stage-list",
            (
                "lookup_api,resolution,instrument_authority,ownership_handoff,ownership_api,"
                "authority,analyst_request_groups,analyst_actuals_api,analyst_estimates_api,"
                "doc_exact_api,doc_fallback_api"
            ),
        ]
    )

    assert exit_code == 0
    assert called_stages == [
        "lookup_api",
        "resolution",
        "instrument_authority",
        "ownership_handoff",
        "ownership_api",
        "authority",
        "analyst_request_groups",
        "analyst_actuals_api",
        "analyst_estimates_api",
        "doc_exact_api",
        "doc_fallback_api",
    ]


def test_main_missing_lookup_snapshot_fails_fast(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run_root = _build_run_root(tmp_path)
    monkeypatch.setattr(runner, "is_lseg_available", lambda: True)

    with pytest.raises(FileNotFoundError, match="lookup_snapshot_parquet"):
        runner.main(
            [
                "--run-root",
                str(run_root),
                "--stage-start",
                "lookup_api",
                "--stage-stop",
                "lookup_api",
            ]
        )


def test_main_missing_doc_filing_artifact_fails_fast(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run_root = _build_run_root(tmp_path)
    _write_parquet(run_root / "refinitiv_step1" / "ownership_authority_common_stock" / "refinitiv_permno_ownership_authority_decisions.parquet")
    _write_parquet(run_root / "refinitiv_step1" / "ownership_authority_common_stock" / "refinitiv_permno_ownership_authority_exceptions.parquet")
    monkeypatch.setattr(runner, "is_lseg_available", lambda: True)

    with pytest.raises(FileNotFoundError, match="doc_filing_artifact_parquet"):
        runner.main(
            [
                "--run-root",
                str(run_root),
                "--stage-start",
                "doc_exact_api",
                "--stage-stop",
                "doc_exact_api",
            ]
        )


def test_main_missing_authority_exception_fails_fast(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run_root = _build_run_root(tmp_path)
    _write_parquet(run_root / "sec_ccm_premerge" / "sec_ccm_matched_clean_filtered.parquet")
    _write_parquet(run_root / "refinitiv_step1" / "ownership_authority_common_stock" / "refinitiv_permno_ownership_authority_decisions.parquet")
    monkeypatch.setattr(runner, "is_lseg_available", lambda: True)

    with pytest.raises(FileNotFoundError, match="authority_exceptions_parquet"):
        runner.main(
            [
                "--run-root",
                str(run_root),
                "--stage-start",
                "doc_exact_api",
                "--stage-stop",
                "doc_exact_api",
            ]
        )


def test_main_audit_only_writes_manifest(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run_root = _build_run_root(tmp_path)
    _write_parquet(run_root / "refinitiv_step1" / "refinitiv_ric_lookup_handoff_common_stock_extended_snapshot.parquet")

    monkeypatch.setattr(
        runner,
        "_audit_stage",
        lambda stage, paths: StageAuditResult(stage_name=stage, passed=True, issues=(), metrics={"ok": True}),
    )

    exit_code = runner.main(
        [
            "--run-root",
            str(run_root),
            "--stage-start",
            "lookup_api",
            "--stage-stop",
            "lookup_api",
            "--audit-only",
        ]
    )

    assert exit_code == 0
    manifest = json.loads((run_root / "refinitiv_local_api_runner_manifest.json").read_text(encoding="utf-8"))
    assert manifest["mode"] == "audit"
    assert manifest["audit_results"]["lookup_api"]["passed"] is True


def test_main_recover_mode_writes_lookup_unresolved_artifact(tmp_path: Path) -> None:
    run_root = _build_run_root(tmp_path)
    recovery_dir = run_root / "refinitiv_recovery"
    recovery_dir.mkdir(parents=True, exist_ok=True)
    pl.DataFrame(
        {
            "bridge_row_id": ["a", "b"],
            "effective_collection_ric": [None, "AAA.N"],
        }
    ).write_parquet(run_root / "refinitiv_step1" / "refinitiv_ric_resolution_common_stock.parquet")

    exit_code = runner.main(
        [
            "--run-root",
            str(run_root),
            "--recover-mode",
            "lookup_unresolved",
        ]
    )

    assert exit_code == 0
    recovery_path = recovery_dir / "lookup_unresolved.parquet"
    assert recovery_path.exists()
    recovered = pl.read_parquet(recovery_path)
    assert recovered.height == 1
    assert recovered.item(0, "bridge_row_id") == "a"

    manifest = json.loads((run_root / "refinitiv_local_api_runner_manifest.json").read_text(encoding="utf-8"))
    assert manifest["mode"] == "recover"
    assert manifest["recovery_result"]["mode"] == "lookup_unresolved"


def test_main_recover_mode_writes_lookup_stage_manifest_from_existing_artifacts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run_root = _build_run_root(tmp_path)
    refinitiv_step1_dir = run_root / "refinitiv_step1"
    _write_parquet(refinitiv_step1_dir / "refinitiv_ric_lookup_handoff_common_stock_extended_snapshot.parquet")
    _write_parquet(refinitiv_step1_dir / "refinitiv_ric_lookup_handoff_common_stock_extended.parquet")
    staging_dir = refinitiv_step1_dir / "staging" / runner.LOOKUP_STAGE
    staging_dir.mkdir(parents=True, exist_ok=True)
    _write_parquet(staging_dir / "batch_example.parquet")
    (refinitiv_step1_dir / "refinitiv_lookup_api_ledger.sqlite3").write_text("", encoding="utf-8")
    (refinitiv_step1_dir / "refinitiv_lookup_api_requests.jsonl").write_text("", encoding="utf-8")

    monkeypatch.setattr(
        runner,
        "_audit_stage",
        lambda stage, paths: StageAuditResult(
            stage_name=runner.LOOKUP_STAGE,
            passed=True,
            issues=(),
            metrics={
                "run_session_ids": ["run_test_lookup"],
                "rebuild_row_counts": {"lookup_extended_parquet": 1},
            },
        ),
    )
    monkeypatch.setattr(
        runner,
        "_run_stage",
        lambda *args, **kwargs: pytest.fail("recover_mode should not call _run_stage"),
    )

    exit_code = runner.main(
        [
            "--run-root",
            str(run_root),
            "--recover-mode",
            "lookup_stage_manifest_only",
        ]
    )

    assert exit_code == 0
    stage_manifest_path = refinitiv_step1_dir / "refinitiv_lookup_stage_manifest.json"
    assert stage_manifest_path.exists()
    stage_manifest = json.loads(stage_manifest_path.read_text(encoding="utf-8"))
    assert stage_manifest["manifest_role"] == "stage_completion"
    assert stage_manifest["stage_name"] == runner.LOOKUP_STAGE
    assert stage_manifest["summary"]["recovered_from_existing_artifacts"] is True
    assert stage_manifest["summary"]["source_stage"] == "lookup_api"
    assert stage_manifest["summary"]["run_session_ids"] == ["run_test_lookup"]

    manifest = json.loads((run_root / "refinitiv_local_api_runner_manifest.json").read_text(encoding="utf-8"))
    assert manifest["mode"] == "recover"
    assert manifest["recovery_result"]["mode"] == "lookup_stage_manifest_only"
    assert manifest["recovery_result"]["stage"] == "lookup_api"


def test_main_can_resume_from_ownership_api_without_rerunning_previous_stages(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run_root = _build_run_root(tmp_path)
    _write_parquet(run_root / "refinitiv_step1" / "ownership_universe_common_stock" / "refinitiv_ownership_universe_handoff_common_stock.parquet")
    monkeypatch.setattr(runner, "is_lseg_available", lambda: True)

    called: list[str] = []

    def ownership_api_stub(**kwargs: Path | int | float) -> dict[str, Path]:
        called.append("ownership_api")
        results_path = kwargs["output_dir"] / "refinitiv_ownership_universe_results.parquet"
        summary_path = kwargs["output_dir"] / "refinitiv_ownership_universe_row_summary.parquet"
        _write_parquet(results_path)
        _write_parquet(summary_path)
        return {
            "refinitiv_ownership_universe_results_parquet": results_path,
            "refinitiv_ownership_universe_row_summary_parquet": summary_path,
        }

    monkeypatch.setattr(runner, "run_refinitiv_step1_ownership_universe_api_pipeline", ownership_api_stub)

    exit_code = runner.main(
        [
            "--run-root",
            str(run_root),
            "--stage-start",
            "ownership_api",
            "--stage-stop",
            "ownership_api",
        ]
    )

    assert exit_code == 0
    assert called == ["ownership_api"]
