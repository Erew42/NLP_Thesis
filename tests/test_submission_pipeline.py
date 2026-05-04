from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import polars as pl
import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]


def _load_submission_tool():
    module_path = REPO_ROOT / "tools" / "run_submission_pipeline.py"
    spec = importlib.util.spec_from_file_location("test_run_submission_pipeline_tool", module_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _write_parquet(path: Path, df: pl.DataFrame | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if df is None:
        df = pl.DataFrame({"placeholder": [1]})
    df.write_parquet(path)


def _write_text(path: Path, text: str = "x\n") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _write_minimal_package_manifest(root: Path) -> None:
    payload = {
        "schema_version": 1,
        "profile": "SUBMISSION",
        "path_semantics": "relative_to_submission_root",
        "code_version": {
            "git_commit": "unit-test",
            "dirty": True,
        },
        "artifact_groups": {
            "sec_year_merged": {
                "paths": ["data/sec/year_merged/1994.parquet"],
                "schema_summary": {
                    "required_columns": ["doc_id", "cik_10", "full_text"],
                    "required_column_groups": {"filing_date": ["filing_date", "file_date_filename"]},
                },
            },
            "sec_items_analysis": {
                "paths": ["data/sec/items_analysis/2009.parquet"],
                "schema_summary": {
                    "required_columns": [
                        "doc_id",
                        "item_id",
                        "full_text",
                        "item_status",
                        "exists_by_regime",
                        "boundary_authority_status",
                    ],
                },
            },
            "sec_ccm_matched_clean": {
                "paths": ["data/sec/sec_ccm_matched_clean.parquet"],
                "schema_summary": {"required_columns": ["doc_id"]},
            },
            "ccm_crsp_compustat": {
                "paths": [
                    "data/ccm_crsp_compustat/final_flagged_data_compdesc_added.parquet",
                    "data/ccm_crsp_compustat/sfz_mth.parquet",
                ],
                "schema_summary": {"key_columns": ["KYPERMNO", "CALDT"]},
            },
            "refinitiv_finalized": {
                "paths": [
                    "data/refinitiv_finalized/refinitiv_lm2011_doc_ownership.parquet",
                    "data/refinitiv_finalized/refinitiv_doc_analyst_selected.parquet",
                ],
                "schema_summary": {"required_columns": ["doc_id"]},
            },
            "lm2011_additional_data": {
                "paths": [
                    "data/LM2011_additional_data/LM2011_MasterDictionary.txt",
                    "data/LM2011_additional_data/Loughran-McDonald_MasterDictionary_1993-2024.csv",
                ],
                "schema_summary": {"dictionary_family_inputs": ["replication", "extended"]},
            },
        },
    }
    _write_text(root / "submission_package_manifest.json", json.dumps(payload, indent=2, sort_keys=True))


def _write_minimal_submission_inputs(root: Path, *, write_package_manifest: bool = True) -> None:
    _write_parquet(
        root / "data" / "sec" / "year_merged" / "1994.parquet",
        pl.DataFrame(
            {
                "doc_id": ["d1"],
                "cik_10": ["0000000001"],
                "filing_date": ["1994-03-01"],
                "full_text": ["Item 7. Management discussion text."],
            }
        ),
    )
    _write_parquet(
        root / "data" / "sec" / "items_analysis" / "2009.parquet",
        pl.DataFrame(
            {
                "doc_id": ["d1"],
                "cik_10": ["0000000001"],
                "accession_nodash": ["000000000123000001"],
                "filing_date": ["2009-03-01"],
                "document_type_filename": ["10-K"],
                "item_id": ["7"],
                "full_text": ["Management discussion text."],
                "item_status": ["active"],
                "exists_by_regime": [True],
                "boundary_authority_status": ["accepted"],
            }
        ),
    )
    _write_parquet(root / "data" / "sec" / "sec_ccm_matched_clean.parquet", pl.DataFrame({"doc_id": ["d1"]}))
    _write_parquet(
        root / "data" / "refinitiv_finalized" / "refinitiv_lm2011_doc_ownership.parquet",
        pl.DataFrame({"doc_id": ["d1"]}),
    )
    _write_parquet(
        root / "data" / "refinitiv_finalized" / "refinitiv_doc_analyst_selected.parquet",
        pl.DataFrame({"doc_id": ["d1"]}),
    )
    ccm_dir = root / "data" / "ccm_crsp_compustat"
    for filename in (
        "filingdates.parquet",
        "balancesheetquarterly.parquet",
        "incomestatementquarterly.parquet",
        "perioddescriptorquarterly.parquet",
        "balancesheetindustrialannual.parquet",
        "incomestatementindustrialannual.parquet",
        "perioddescriptorannual.parquet",
        "fiscalmarketdataannual.parquet",
        "companyhistory.parquet",
        "companydescription.parquet",
        "sfz_mth.parquet",
    ):
        _write_parquet(ccm_dir / filename)
    _write_parquet(
        ccm_dir / "final_flagged_data_compdesc_added.parquet",
        pl.DataFrame(
            {
                "KYPERMNO": [1],
                "CALDT": ["2020-01-01"],
                "FINAL_RET": [0.01],
                "FINAL_PRC": [10.0],
                "VOL": [100],
                "SHROUT": [1000],
                "SHRCD": [10],
                "EXCHCD": [1],
            }
        ),
    )
    additional = root / "data" / "LM2011_additional_data"
    for filename in (
        "F-F_Research_Data_Factors_daily.csv",
        "F-F_Research_Data_Factors.csv",
        "F-F_Momentum_Factor.csv",
        "FF_Siccodes_48_Industries.txt",
        "Fin-Neg.txt",
        "Fin-Pos.txt",
        "Fin-Unc.txt",
        "Fin-Lit.txt",
        "MW-Strong.txt",
        "MW-Weak.txt",
        "Harvard_IV_NEG_Inf.txt",
    ):
        _write_text(additional / filename)
    _write_text(additional / "LM2011_MasterDictionary.txt", "Word\nLOSS\n")
    _write_text(
        additional / "Loughran-McDonald_MasterDictionary_1993-2024.csv",
        "Word,Negative,Positive,Uncertainty,Litigious,Strong_Modal,Weak_Modal\nLOSS,2009,0,0,0,0,0\n",
    )
    if write_package_manifest:
        _write_minimal_package_manifest(root)


def _write_minimal_seed_inputs(
    root: Path,
    *,
    write_package_manifest: bool = False,
    include_canonical_link: bool = True,
    include_daily_rebuild_inputs: bool = True,
    include_year_text: bool = True,
) -> None:
    year_payload = {
        "doc_id": ["d1"],
        "cik_10": ["0000000001"],
        "filing_date": ["1994-03-01"],
    }
    if include_year_text:
        year_payload["full_text"] = ["Item 7. Management discussion text."]
    _write_parquet(root / "data" / "sec" / "year_merged" / "1994.parquet", pl.DataFrame(year_payload))
    ccm_dir = root / "data" / "ccm_crsp_compustat"
    if include_daily_rebuild_inputs:
        for filename in (
            "filingdates.parquet",
            "linkhistory.parquet",
            "linkfiscalperiodall.parquet",
            "companydescription.parquet",
            "companyhistory.parquet",
            "securityheader.parquet",
            "securityheaderhistory.parquet",
            "sfz_ds_dly.parquet",
            "sfz_dp_dly.parquet",
            "sfz_del.parquet",
            "sfz_nam.parquet",
            "sfz_hdr.parquet",
            "sfz_shr.parquet",
        ):
            _write_parquet(ccm_dir / filename)
    if include_canonical_link:
        _write_parquet(ccm_dir / "canonical_link_table.parquet")
    if write_package_manifest:
        _write_minimal_package_manifest(root)


def test_submission_profile_default_layout_validates(tmp_path: Path) -> None:
    tool = _load_submission_tool()
    root = tmp_path / "submission"
    _write_minimal_submission_inputs(root)

    profile = tool.resolve_submission_profile(submission_root=root)
    tool.validate_submission_profile(profile)

    assert profile.year_merged_dir == (root / "data" / "sec" / "year_merged").resolve()
    assert profile.doc_ownership_path == (
        root / "data" / "refinitiv_finalized" / "refinitiv_lm2011_doc_ownership.parquet"
    ).resolve()


def test_submission_profile_requires_package_manifest(tmp_path: Path) -> None:
    tool = _load_submission_tool()
    root = tmp_path / "submission"
    _write_minimal_submission_inputs(root, write_package_manifest=False)

    profile = tool.resolve_submission_profile(submission_root=root)
    with pytest.raises(tool.SubmissionPipelineError, match="submission package manifest"):
        tool.validate_submission_profile(profile)


def test_submission_profile_validates_every_items_analysis_shard(tmp_path: Path) -> None:
    tool = _load_submission_tool()
    root = tmp_path / "submission"
    _write_minimal_submission_inputs(root)
    _write_parquet(root / "data" / "sec" / "items_analysis" / "2010.parquet", pl.DataFrame({"doc_id": ["d2"]}))

    profile = tool.resolve_submission_profile(submission_root=root)
    with pytest.raises(tool.SubmissionPipelineError, match="items_analysis shard schema is invalid"):
        tool.validate_submission_profile(profile)


def test_submission_package_manifest_rejects_escaped_paths(tmp_path: Path) -> None:
    tool = _load_submission_tool()
    root = tmp_path / "submission"
    _write_minimal_submission_inputs(root)
    manifest_path = root / "submission_package_manifest.json"
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    payload["artifact_groups"]["sec_year_merged"]["paths"] = ["../outside.parquet"]
    manifest_path.write_text(json.dumps(payload), encoding="utf-8")

    profile = tool.resolve_submission_profile(submission_root=root)
    with pytest.raises(tool.SubmissionPipelineError, match="escapes submission_root"):
        tool.validate_submission_profile(profile)


def test_submission_config_rejects_escaped_artifact_override(tmp_path: Path) -> None:
    tool = _load_submission_tool()
    root = tmp_path / "submission"
    root.mkdir()
    config_path = root / "submission_pipeline_config.json"
    config_path.write_text(
        json.dumps(
            {
                "artifact_overrides": [
                    {
                        "artifact_key": "lm2011_table_ia_ii_results",
                        "path": "../outside.parquet",
                        "reason": "bad escape",
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(tool.SubmissionPipelineError, match="escapes submission_root"):
        tool.resolve_submission_profile(submission_root=root)


def test_submission_config_allows_omitted_artifact_overrides(tmp_path: Path) -> None:
    tool = _load_submission_tool()
    root = tmp_path / "submission"
    root.mkdir()
    (root / "submission_pipeline_config.json").write_text(
        json.dumps({"run_id": "unit_run", "years": [2007, 2008], "nw_lags": [1, 2]}),
        encoding="utf-8",
    )

    profile = tool.resolve_submission_profile(submission_root=root)

    assert profile.config.run_id == "unit_run"
    assert profile.config.years == (2007, 2008)
    assert profile.config.nw_lags == (1, 2)
    assert profile.config.artifact_overrides == ()


def test_dry_run_all_emits_deterministic_stage_order(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    tool = _load_submission_tool()
    root = tmp_path / "submission"
    root.mkdir()

    exit_code = tool.main(["--submission-root", str(root), "--stage", "all", "--dry-run"])

    assert exit_code == 0
    payload = json.loads(capsys.readouterr().out)
    assert [row["stage"] for row in payload["plan"]] == list(tool.ORDERED_STAGES)
    rendered = json.dumps(payload)
    assert "LOCAL_REPO" not in rendered
    assert "COLAB_DRIVE" not in rendered


def test_dry_run_all_with_raw_zips_prepends_raw_sec_stages(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    tool = _load_submission_tool()
    root = tmp_path / "submission"
    root.mkdir()
    (root / "submission_pipeline_config.json").write_text(
        json.dumps({"source_mode": "from_raw_zips", "run_raw_sec_stages": True}),
        encoding="utf-8",
    )

    exit_code = tool.main(["--submission-root", str(root), "--stage", "all", "--dry-run"])

    assert exit_code == 0
    payload = json.loads(capsys.readouterr().out)
    assert [row["stage"] for row in payload["plan"]] == list(tool.RAW_RECOMPUTE_STAGES)


def test_retained_dry_run_uses_legacy_stage_order(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    tool = _load_submission_tool()
    root = tmp_path / "submission"
    root.mkdir()

    exit_code = tool.main(["--submission-root", str(root), "--stage", "retained", "--dry-run"])

    assert exit_code == 0
    payload = json.loads(capsys.readouterr().out)
    assert [row["stage"] for row in payload["plan"]] == list(tool.RETAINED_STAGES)
    lm2011_command = next(row["command"] for row in payload["plan"] if row["stage"] == tool.STAGE_LM2011)
    assert str((root / "data" / "sec" / "items_analysis").resolve()) in lm2011_command
    assert str((root / "data" / "sec" / "sec_ccm_matched_clean.parquet").resolve()) in lm2011_command
    assert str((root / "analysis_outputs" / "items_analysis").resolve()) not in lm2011_command


def test_default_dry_run_uses_submission_readiness_stages(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    tool = _load_submission_tool()
    root = tmp_path / "submission"
    root.mkdir()

    exit_code = tool.main(["--submission-root", str(root), "--dry-run"])

    assert exit_code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["stage"] == tool.STAGE_READINESS
    assert [row["stage"] for row in payload["plan"]] == list(tool.READINESS_STAGES)


def test_all_spawns_child_stage_commands_in_order(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    tool = _load_submission_tool()
    root = tmp_path / "submission"
    root.mkdir()
    calls: list[list[str]] = []

    def _fake_run(command, cwd=None, check=False):
        calls.append([str(part) for part in command])
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(tool.subprocess, "run", _fake_run)

    exit_code = tool.main(["--submission-root", str(root), "--stage", "all", "--run-id", "unit_run"])

    assert exit_code == 0
    assert [command[command.index("--stage") + 1] for command in calls] == list(tool.ORDERED_STAGES)
    assert all("--run-id" in command and "unit_run" in command for command in calls)


def test_retained_spawns_child_commands_with_retained_source_mode(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tool = _load_submission_tool()
    root = tmp_path / "submission"
    root.mkdir()
    calls: list[list[str]] = []

    def _fake_run(command, cwd=None, check=False):
        calls.append([str(part) for part in command])
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(tool.subprocess, "run", _fake_run)

    exit_code = tool.main(["--submission-root", str(root), "--stage", "retained", "--run-id", "unit_run"])

    assert exit_code == 0
    assert [command[command.index("--stage") + 1] for command in calls] == list(tool.RETAINED_STAGES)
    assert all(command[command.index("--source-mode") + 1] == tool.SOURCE_MODE_RETAINED for command in calls)


def test_default_readiness_spawns_only_readiness_child_stages(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tool = _load_submission_tool()
    root = tmp_path / "submission"
    root.mkdir()
    calls: list[list[str]] = []

    def _fake_run(command, cwd=None, check=False):
        calls.append([str(part) for part in command])
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(tool.subprocess, "run", _fake_run)

    exit_code = tool.main(["--submission-root", str(root), "--run-id", "unit_run"])

    assert exit_code == 0
    assert [command[command.index("--stage") + 1] for command in calls] == list(tool.READINESS_STAGES)
    assert all("--run-id" in command and "unit_run" in command for command in calls)


def test_package_manifest_stage_writes_current_submission_manifest(tmp_path: Path) -> None:
    tool = _load_submission_tool()
    root = tmp_path / "submission"
    _write_minimal_submission_inputs(root, write_package_manifest=False)
    profile = tool.resolve_submission_profile(submission_root=root)

    exit_code = tool.run_stage(profile, "package-manifest", force=False, dry_run=False)

    assert exit_code == 0
    payload = json.loads(profile.package_manifest_path.read_text(encoding="utf-8"))
    assert payload["profile"] == "SUBMISSION"
    assert payload["path_semantics"] == "relative_to_submission_root"
    assert set(tool.PACKAGE_MANIFEST_REQUIRED_GROUPS) <= set(payload["artifact_groups"])
    assert payload["artifact_groups"]["sec_items_analysis"]["paths"] == [
        "data/sec/items_analysis/2009.parquet"
    ]


def test_package_manifest_allows_missing_generated_groups_before_recompute(tmp_path: Path) -> None:
    tool = _load_submission_tool()
    root = tmp_path / "submission"
    _write_minimal_seed_inputs(root)
    profile = tool.resolve_submission_profile(submission_root=root)

    exit_code = tool.run_stage(profile, "package-manifest", force=False, dry_run=False)

    assert exit_code == 0
    payload = json.loads(profile.package_manifest_path.read_text(encoding="utf-8"))
    assert "sec_year_merged" in payload["artifact_groups"]
    assert "ccm_crsp_compustat" in payload["artifact_groups"]
    assert "sec_items_analysis" not in payload["artifact_groups"]
    assert "sec_ccm_matched_clean" not in payload["artifact_groups"]


def test_seed_validation_does_not_require_derived_or_refinitiv_inputs(tmp_path: Path) -> None:
    tool = _load_submission_tool()
    root = tmp_path / "submission"
    _write_minimal_seed_inputs(root)
    profile = tool.resolve_submission_profile(submission_root=root)
    tool.run_stage(profile, "package-manifest", force=False, dry_run=False)

    tool.validate_submission_seed_inputs(profile)


def test_seed_validation_rejects_missing_daily_rebuild_input(tmp_path: Path) -> None:
    tool = _load_submission_tool()
    root = tmp_path / "submission"
    _write_minimal_seed_inputs(root)
    profile = tool.resolve_submission_profile(submission_root=root)
    (profile.ccm_dir / "sfz_ds_dly.parquet").unlink()
    tool.run_stage(profile, "package-manifest", force=False, dry_run=False)

    with pytest.raises(tool.SubmissionPipelineError, match="sfz_ds_dly"):
        tool.validate_submission_seed_inputs(profile)


def test_seed_validation_rejects_textless_year_merged(tmp_path: Path) -> None:
    tool = _load_submission_tool()
    root = tmp_path / "submission"
    _write_minimal_seed_inputs(root, include_year_text=False)
    profile = tool.resolve_submission_profile(submission_root=root)
    tool.run_stage(profile, "package-manifest", force=False, dry_run=False)

    with pytest.raises(tool.SubmissionPipelineError, match="full_text"):
        tool.validate_submission_seed_inputs(profile)


def test_derived_validation_requires_generated_outputs(tmp_path: Path) -> None:
    tool = _load_submission_tool()
    root = tmp_path / "submission"
    _write_minimal_seed_inputs(root)
    profile = tool.resolve_submission_profile(submission_root=root)
    tool.run_stage(profile, "package-manifest", force=False, dry_run=False)

    with pytest.raises(tool.SubmissionPipelineError, match="items_analysis|matched-clean"):
        tool.validate_submission_derived_inputs(profile)


def test_lm2011_command_uses_only_explicit_submission_paths(tmp_path: Path) -> None:
    tool = _load_submission_tool()
    root = tmp_path / "submission"
    root.mkdir()
    profile = tool.resolve_submission_profile(submission_root=root)

    command = tool.build_lm2011_command(profile, force=True)
    command_text = " ".join(command)

    assert "--doc-ownership-path" in command
    assert str(profile.doc_ownership_path) in command
    assert "--doc-analyst-selected-path" in command
    assert str(profile.doc_analyst_selected_path) in command
    assert "--daily-panel-path" in command
    assert str(profile.active_daily_panel_path) in command
    assert str(profile.generated_items_analysis_dir) in command
    assert str(profile.generated_matched_clean_path) in command
    assert command[command.index("--nw-lags") + 1 : command.index("--nw-lags") + 5] == ["1", "2", "3", "4"]
    assert "full_data_run" not in command_text


def test_lm2011_event_window_sensitivity_command_writes_locked_root(tmp_path: Path) -> None:
    tool = _load_submission_tool()
    root = tmp_path / "submission"
    root.mkdir()
    profile = tool.resolve_submission_profile(submission_root=root)

    command = tool.build_lm2011_event_window_sensitivity_command(profile, force=False)

    assert "--event-window-sensitivity-only" in command
    assert command[
        command.index("--event-window-sensitivity-days") + 1 : command.index("--event-window-sensitivity-days") + 4
    ] == ["6", "12", "18"]
    assert command[command.index("--event-window-sensitivity-output-dir") + 1] == str(
        profile.event_window_sensitivity_dir
    )


def test_event_window_force_removes_existing_sensitivity_output(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tool = _load_submission_tool()
    root = tmp_path / "submission"
    root.mkdir()
    profile = tool.resolve_submission_profile(submission_root=root)
    stale_path = profile.event_window_sensitivity_dir / "stale.parquet"
    _write_parquet(stale_path)
    calls: list[list[str]] = []

    def _fake_run(command, cwd=None, check=False):
        calls.append([str(part) for part in command])
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(tool.subprocess, "run", _fake_run)

    exit_code = tool.run_stage(profile, "event-window-sensitivity", force=True, dry_run=False)

    assert exit_code == 0
    assert calls
    assert not stale_path.exists()


def test_lm2011_command_retained_mode_uses_packaged_derived_inputs(tmp_path: Path) -> None:
    tool = _load_submission_tool()
    root = tmp_path / "submission"
    root.mkdir()
    (root / "submission_pipeline_config.json").write_text(
        json.dumps({"source_mode": "retained"}),
        encoding="utf-8",
    )
    profile = tool.resolve_submission_profile(submission_root=root)

    command = tool.build_lm2011_command(profile, force=False)

    assert str(profile.items_analysis_dir) in command
    assert str(profile.matched_clean_path) in command


def test_lm2011_dry_run_source_mode_override_uses_packaged_derived_inputs(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    tool = _load_submission_tool()
    root = tmp_path / "submission"
    root.mkdir()

    exit_code = tool.main(
        [
            "--submission-root",
            str(root),
            "--stage",
            "lm2011",
            "--source-mode",
            "retained",
            "--dry-run",
        ]
    )

    assert exit_code == 0
    payload = json.loads(capsys.readouterr().out)
    command = payload["plan"][0]["command"]
    assert str((root / "data" / "sec" / "items_analysis").resolve()) in command
    assert str((root / "data" / "sec" / "sec_ccm_matched_clean.parquet").resolve()) in command


def test_generated_canonical_link_is_replaced_from_packaged_seed(tmp_path: Path) -> None:
    tool = _load_submission_tool()
    root = tmp_path / "submission"
    _write_minimal_seed_inputs(root)
    profile = tool.resolve_submission_profile(submission_root=root)
    _write_parquet(
        profile.packaged_canonical_link_path,
        pl.DataFrame({"source_marker": ["packaged"]}),
    )
    _write_parquet(
        profile.generated_canonical_link_path,
        pl.DataFrame({"source_marker": ["stale"]}),
    )

    target = tool._ensure_submission_canonical_link_table(profile, force=False)

    assert target == profile.generated_canonical_link_path
    assert pl.read_parquet(target)["source_marker"].to_list() == ["packaged"]


def test_sec_ccm_premerge_stage_uses_generated_output_dir(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tool = _load_submission_tool()
    root = tmp_path / "submission"
    _write_minimal_seed_inputs(root)
    profile = tool.resolve_submission_profile(submission_root=root)

    import thesis_pkg.pipelines.ccm_pipeline as ccm_pipeline
    import thesis_pkg.pipelines.sec_ccm_pipeline as sec_ccm_pipeline

    captured: dict[str, object] = {}

    def _fake_build_or_reuse_ccm_daily_stage(**kwargs):
        captured["ccm_kwargs"] = kwargs
        market_core_path = profile.ccm_derived_dir / "ccm_daily_market_core.parquet"
        phase_b_path = profile.ccm_derived_dir / "ccm_daily_phase_b_surface.parquet"
        bridge_path = profile.ccm_derived_dir / "ccm_daily_bridge_surface.parquet"
        _write_parquet(profile.generated_canonical_link_path)
        _write_parquet(profile.generated_daily_panel_path)
        _write_parquet(market_core_path, pl.DataFrame({"CALDT": ["1994-03-01"]}))
        _write_parquet(
            phase_b_path,
            pl.DataFrame({"KYPERMNO": [1], "CALDT": ["1994-03-01"], "RET": [0.01]}),
        )
        _write_parquet(bridge_path)
        return {
            "ccm_daily_path": profile.generated_daily_panel_path,
            "canonical_link_path": profile.generated_canonical_link_path,
            "ccm_daily_market_core_path": market_core_path,
            "ccm_daily_phase_b_surface_path": phase_b_path,
            "ccm_daily_bridge_surface_path": bridge_path,
        }

    def _fake_run_sec_ccm_premerge_pipeline(**kwargs):
        captured["premerge_kwargs"] = kwargs
        _write_parquet(profile.generated_matched_clean_path, pl.DataFrame({"doc_id": ["d1"]}))
        _write_parquet(profile.sec_ccm_analysis_doc_ids_path, pl.DataFrame({"doc_id": ["d1"]}))
        return {
            "sec_ccm_matched_clean": profile.generated_matched_clean_path,
            "sec_ccm_analysis_doc_ids": profile.sec_ccm_analysis_doc_ids_path,
        }

    monkeypatch.setattr(ccm_pipeline, "build_or_reuse_ccm_daily_stage", _fake_build_or_reuse_ccm_daily_stage)
    monkeypatch.setattr(sec_ccm_pipeline, "run_sec_ccm_premerge_pipeline", _fake_run_sec_ccm_premerge_pipeline)

    paths = tool._run_sec_ccm_premerge_stage(profile, force=False)

    assert profile.generated_canonical_link_path.exists()
    assert captured["ccm_kwargs"]["run_mode"] == "REBUILD"
    assert captured["ccm_kwargs"]["ccm_derived_dir"] == profile.ccm_derived_dir
    assert captured["ccm_kwargs"]["ccm_reuse_daily_path"] == profile.generated_daily_panel_path
    assert captured["premerge_kwargs"]["output_dir"] == profile.sec_ccm_premerge_dir
    assert paths["sec_ccm_matched_clean"] == profile.generated_matched_clean_path


def test_sec_items_analysis_stage_uses_generated_output_dir(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tool = _load_submission_tool()
    root = tmp_path / "submission"
    _write_minimal_seed_inputs(root)
    profile = tool.resolve_submission_profile(submission_root=root)
    _write_parquet(profile.sec_ccm_analysis_doc_ids_path, pl.DataFrame({"doc_id": ["d1"]}))

    import thesis_pkg.pipelines.sec_pipeline as sec_pipeline

    captured: dict[str, object] = {}

    def _fake_process_year_dir_extract_items_gated(**kwargs):
        captured.update(kwargs)
        out_path = kwargs["out_dir"] / "1994.parquet"
        _write_parquet(
            out_path,
            pl.DataFrame(
                {
                    "doc_id": ["d1"],
                    "cik_10": ["0000000001"],
                    "accession_nodash": ["000000000123000001"],
                    "filing_date": ["1994-03-01"],
                    "document_type_filename": ["10-K"],
                    "item_id": ["7"],
                    "full_text": ["Management discussion text."],
                    "item_status": ["active"],
                    "exists_by_regime": [True],
                    "boundary_authority_status": ["accepted"],
                }
            ),
        )
        return [out_path]

    monkeypatch.setattr(sec_pipeline, "process_year_dir_extract_items_gated", _fake_process_year_dir_extract_items_gated)

    paths = tool._run_sec_items_analysis_stage(profile, force=False)

    assert captured["year_dir"] == profile.active_year_merged_dir
    assert captured["out_dir"] == profile.generated_items_analysis_dir
    assert captured["doc_id_allowlist"] == profile.sec_ccm_analysis_doc_ids_path
    assert captured["extraction_regime"] == "legacy"
    assert paths == [profile.generated_items_analysis_dir / "1994.parquet"]


def test_finbert_commands_pin_revision_and_analysis_requires_checkpoint(tmp_path: Path) -> None:
    tool = _load_submission_tool()
    root = tmp_path / "submission"
    root.mkdir()
    profile = tool.resolve_submission_profile(submission_root=root)

    preprocess_command = tool.build_finbert_preprocess_command(profile, force=False)
    analysis_command = tool.build_finbert_analysis_command(profile, force=True)

    assert "--model-revision" in preprocess_command
    assert preprocess_command[preprocess_command.index("--model-revision") + 1] == tool.DEFAULT_FINBERT_REVISION
    assert "--tokenizer-revision" in analysis_command
    assert analysis_command[analysis_command.index("--tokenizer-revision") + 1] == tool.DEFAULT_FINBERT_REVISION
    assert "--analysis-only" in analysis_command
    assert "--write-sentence-scores" in analysis_command
    assert "--overwrite" in analysis_command
    with pytest.raises(tool.SubmissionPipelineError, match="requires the named preprocessing checkpoint"):
        tool.run_stage(profile, "finbert-analysis", force=False, dry_run=False)


def test_finbert_commands_follow_retained_manifest_null_revision(tmp_path: Path) -> None:
    tool = _load_submission_tool()
    root = tmp_path / "submission"
    root.mkdir()
    profile = tool.resolve_submission_profile(submission_root=root)
    manifest = {
        "semantic_reuse_guard": {
            "payload": {
                "authority": {
                    "model_name": "yiyanghkust/finbert-tone",
                    "model_revision": None,
                    "tokenizer_revision": None,
                }
            }
        }
    }
    _write_text(profile.finbert_preprocessing_run_dir / "run_manifest.json", json.dumps(manifest))

    preprocess_command = tool.build_finbert_preprocess_command(profile, force=False)
    analysis_command = tool.build_finbert_analysis_command(profile, force=False)

    assert preprocess_command[preprocess_command.index("--model-revision") + 1] == "none"
    assert preprocess_command[preprocess_command.index("--tokenizer-revision") + 1] == "none"
    assert analysis_command[analysis_command.index("--model-revision") + 1] == "none"
    assert analysis_command[analysis_command.index("--tokenizer-revision") + 1] == "none"
    assert profile.config.finbert.model_revision == tool.DEFAULT_FINBERT_REVISION
    assert profile.config.finbert.tokenizer_revision == tool.DEFAULT_FINBERT_REVISION


def test_visible_prefix_extension_stage_builds_separate_sensitivity_config(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tool = _load_submission_tool()
    root = tmp_path / "submission"
    root.mkdir()
    profile = tool.resolve_submission_profile(submission_root=root)

    from thesis_pkg.notebooks_and_scripts import lm2011_sample_post_refinitiv_runner as lm_runner

    captured: dict[str, object] = {}
    monkeypatch.setattr(
        lm_runner,
        "run_lm2011_extension_dictionary_family_comparison_pipeline",
        lambda cfg: captured.update({"cfg": cfg}),
    )

    tool._run_visible_prefix_extension_stage(profile, force=False)

    cfg = captured["cfg"]
    assert cfg.output_dir == profile.visible_prefix_extension_output_dir
    assert cfg.local_work_root == profile.local_work_root / "lm2011_extension_finbert_visible_prefix"
    assert cfg.dictionary_source_mode == lm_runner.EXTENSION_DICTIONARY_SOURCE_FINBERT_VISIBLE_PREFIX
    assert cfg.text_scopes == ("item_1a_risk_factors", "item_7_mda", "items_1a_7_combined")
    assert cfg.finbert_sentence_scores_dir == profile.finbert_analysis_run_dir / "sentence_scores" / "by_year"
    assert cfg.finbert_visible_prefix_model_revision == profile.config.finbert.model_revision
    assert "full_10k" not in cfg.text_scopes


def test_nw_sensitivity_promotion_writes_manifest(tmp_path: Path) -> None:
    tool = _load_submission_tool()
    root = tmp_path / "submission"
    root.mkdir()
    profile = tool.resolve_submission_profile(submission_root=root)
    _write_parquet(profile.lm2011_output_dir / "core_tables_nw_lag_sensitivity.parquet")
    _write_parquet(profile.extension_output_dir / "extension_results_nw_lag_sensitivity.parquet")
    _write_parquet(profile.extension_output_dir / "extension_fit_comparisons_nw_lag_sensitivity.parquet")
    _write_text(profile.extension_output_dir / "nw_lag_sensitivity_summary.csv", "lag\n1\n")

    tool._promote_nw_sensitivity_pack(profile, require_core=True, require_extension=True)

    manifest = json.loads((profile.nw_sensitivity_dir / "lm2011_nw_lag_sensitivity_run_manifest.json").read_text())
    assert manifest["complete"] is True
    assert manifest["nw_lags"] == [1, 2, 3, 4]
    assert "core_tables_nw_lag_sensitivity.parquet" in manifest["artifacts"]


def test_thesis_asset_stage_creates_lock_and_uses_submission_lock(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    tool = _load_submission_tool()
    root = tmp_path / "submission"
    root.mkdir()
    profile = tool.resolve_submission_profile(submission_root=root)

    _write_parquet(profile.lm2011_output_dir / "lm2011_table_i_sample_creation.parquet")
    _write_parquet(profile.nw_sensitivity_dir / "core_tables_nw_lag_sensitivity.parquet")
    _write_parquet(profile.nw_sensitivity_dir / "extension_results_nw_lag_sensitivity.parquet")
    _write_parquet(profile.nw_sensitivity_dir / "extension_fit_comparisons_nw_lag_sensitivity.parquet")
    (profile.nw_sensitivity_dir / "lm2011_nw_lag_sensitivity_run_manifest.json").write_text("{}", encoding="utf-8")
    _write_parquet(profile.event_window_sensitivity_dir / "lm2011_event_window_sensitivity_results.parquet")
    _write_parquet(profile.extension_output_dir / "lm2011_extension_fit_summary.parquet")
    _write_parquet(profile.extension_output_dir / "lm2011_extension_fit_comparisons.parquet")
    (profile.extension_output_dir / "lm2011_extension_run_manifest.json").write_text("{}", encoding="utf-8")
    _write_parquet(profile.visible_prefix_extension_output_dir / "lm2011_extension_fit_summary.parquet")
    _write_parquet(profile.visible_prefix_extension_output_dir / "lm2011_extension_fit_comparisons.parquet")
    _write_parquet(
        profile.visible_prefix_extension_output_dir
        / "lm2011_extension_finbert_visible_prefix_audit.parquet"
    )
    (profile.visible_prefix_extension_output_dir / "lm2011_extension_run_manifest.json").write_text(
        "{}",
        encoding="utf-8",
    )
    (profile.finbert_analysis_run_dir / "run_manifest.json").parent.mkdir(parents=True, exist_ok=True)
    (profile.finbert_analysis_run_dir / "run_manifest.json").write_text(
        json.dumps(
            {
                "authority": {
                    "model_name": tool.DEFAULT_FINBERT_MODEL_NAME,
                    "model_revision": tool.DEFAULT_FINBERT_REVISION,
                    "tokenizer_revision": tool.DEFAULT_FINBERT_REVISION,
                }
            }
        ),
        encoding="utf-8",
    )
    _write_parquet(
        profile.finbert_analysis_run_dir / "item_features_long.parquet",
        pl.DataFrame({"model_version": [tool.DEFAULT_FINBERT_REVISION]}),
    )
    for filename in (
        "finbert_robustness_existing_scale_coefficients.parquet",
        "finbert_robustness_tail_coefficients.parquet",
        "finbert_robustness_quantile_coefficients.parquet",
    ):
        _write_parquet(profile.finbert_robustness_output_dir / filename)
    (profile.finbert_robustness_output_dir / "finbert_robustness_run_manifest.json").write_text("{}", encoding="utf-8")
    _write_parquet(profile.finbert_secondary_outcomes_output_dir / "finbert_secondary_outcome_coefficients.parquet")
    (profile.finbert_secondary_outcomes_output_dir / "finbert_secondary_outcome_run_manifest.json").write_text(
        "{}",
        encoding="utf-8",
    )

    calls: list[list[str]] = []

    def _fake_run(command, cwd=None, check=False):
        calls.append([str(part) for part in command])
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(tool.subprocess, "run", _fake_run)

    exit_code = tool.run_stage(profile, "thesis-assets", force=False, dry_run=False)

    assert exit_code == 0
    assert profile.submission_lock_path.exists()
    lock = json.loads(profile.submission_lock_path.read_text(encoding="utf-8"))
    assert sorted(lock["run_roots"]) == [
        "finbert_robustness",
        "finbert_run",
        "finbert_secondary_outcomes",
        "lm2011_event_window_sensitivity",
        "lm2011_extension",
        "lm2011_extension_finbert_visible_prefix",
        "lm2011_nw_lag_sensitivity",
        "lm2011_post_refinitiv",
    ]
    assert lock["provenance_disclosures"][0]["id"] == tool.FINBERT_INFERRED_REVISION_DISCLOSURE_ID
    assert any("--submission-lock" in command for command in calls)
