from __future__ import annotations

import json
from pathlib import Path

import polars as pl
import pytest

from thesis_pkg.notebooks_and_scripts import refinitiv_local_api_runner as runner


def _write_parquet(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pl.DataFrame({"row_id": [1]}).write_parquet(path)


def _build_run_root(tmp_path: Path) -> Path:
    run_root = tmp_path / "run_root"
    (run_root / "sec_ccm_premerge").mkdir(parents=True)
    (run_root / "refinitiv_step1" / "ownership_universe_common_stock").mkdir(parents=True)
    (run_root / "refinitiv_step1" / "ownership_authority_common_stock").mkdir(parents=True)
    (run_root / "refinitiv_doc_ownership_lm2011").mkdir(parents=True)
    return run_root


def test_parse_args_defaults() -> None:
    args = runner.parse_args(["--run-root", "C:/tmp/run"])

    assert args.stage_start == "lookup_api"
    assert args.stage_stop == "doc_finalize"
    assert args.resume is True
    assert args.lookup_batch_size == 25
    assert args.ownership_batch_size == 10
    assert args.doc_exact_batch_size == 15
    assert args.doc_fallback_batch_size == 5
    assert args.min_seconds_between_requests == 2.0
    assert args.max_attempts == 4


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
    monkeypatch.setattr(runner, "run_refinitiv_step1_ownership_universe_handoff_pipeline", ownership_handoff_stub)
    monkeypatch.setattr(runner, "run_refinitiv_step1_ownership_universe_api_pipeline", ownership_api_stub)
    monkeypatch.setattr(runner, "run_refinitiv_step1_ownership_authority_pipeline", authority_stub)
    monkeypatch.setattr(runner, "run_refinitiv_lm2011_doc_ownership_exact_api_pipeline", doc_exact_stub)
    monkeypatch.setattr(runner, "run_refinitiv_lm2011_doc_ownership_fallback_api_pipeline", doc_fallback_stub)
    monkeypatch.setattr(runner, "run_refinitiv_lm2011_doc_ownership_finalize_pipeline", doc_finalize_stub)

    exit_code = runner.main(["--run-root", str(run_root)])

    assert exit_code == 0
    assert [stage for stage, _ in calls] == list(runner.STAGE_ORDER)
    assert calls[1][1]["filled_lookup_workbook_path"] == run_root / "refinitiv_step1" / "refinitiv_ric_lookup_handoff_common_stock_extended.parquet"
    assert calls[3][1]["handoff_parquet_path"] == run_root / "refinitiv_step1" / "ownership_universe_common_stock" / "refinitiv_ownership_universe_handoff_common_stock.parquet"
    assert calls[4][1]["resolution_artifact_path"] == run_root / "refinitiv_step1" / "refinitiv_ric_resolution_common_stock.parquet"
    assert calls[4][1]["ownership_results_artifact_path"] == run_root / "refinitiv_step1" / "ownership_universe_common_stock" / "refinitiv_ownership_universe_results.parquet"
    assert calls[5][1]["doc_filing_artifact_path"] == run_root / "sec_ccm_premerge" / "sec_ccm_matched_clean_filtered.parquet"
    assert calls[5][1]["authority_decisions_artifact_path"] == run_root / "refinitiv_step1" / "ownership_authority_common_stock" / "refinitiv_permno_ownership_authority_decisions.parquet"
    assert calls[6][1]["output_dir"] == run_root / "refinitiv_doc_ownership_lm2011"


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
