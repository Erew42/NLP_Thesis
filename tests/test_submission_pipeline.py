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


def _write_minimal_submission_inputs(root: Path) -> None:
    _write_parquet(root / "data" / "sec" / "year_merged" / "1994.parquet", pl.DataFrame({"doc_id": ["d1"]}))
    _write_parquet(root / "data" / "sec" / "items_analysis" / "2009.parquet", pl.DataFrame({"doc_id": ["d1"]}))
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
        "lm2011_nw_lag_sensitivity",
        "lm2011_post_refinitiv",
    ]
    assert any("--submission-lock" in command for command in calls)
