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
                "schema_summary": {"required_columns": ["doc_id"]},
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
    _write_parquet(root / "data" / "sec" / "year_merged" / "1994.parquet", pl.DataFrame({"doc_id": ["d1"]}))
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
    assert str(profile.daily_panel_path) in command
    assert command[command.index("--nw-lags") + 1 : command.index("--nw-lags") + 5] == ["1", "2", "3", "4"]
    assert "full_data_run" not in command_text


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
        "lm2011_extension",
        "lm2011_extension_finbert_visible_prefix",
        "lm2011_nw_lag_sensitivity",
        "lm2011_post_refinitiv",
    ]
    assert any("--submission-lock" in command for command in calls)
