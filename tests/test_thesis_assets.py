from __future__ import annotations

import json
import importlib.util
import sys
from datetime import date
from pathlib import Path

import polars as pl
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from thesis_assets.api import build_single_asset
from thesis_assets.builders.sample_contracts import common_row_comparison
from thesis_assets.builders.sample_contracts import common_success_comparison
from thesis_assets.builders.sample_contracts import ownership_common_support
from thesis_assets.builders.sample_contracts import raw_available
from thesis_assets.builders.sample_contracts import regression_eligible
from thesis_assets.cli.__main__ import main as cli_main
from thesis_assets.config.constants import DEFAULT_COMMON_SUCCESS_POLICY
from thesis_assets.errors import RegistryError
from thesis_assets.errors import SampleContractError
from thesis_assets.registry import loader
from thesis_assets.specs import BuildResult
from thesis_assets.specs import BuildSessionResult
from thesis_assets.usage import resolve_usage_run_paths


def test_registry_loader_imports_expected_assets() -> None:
    assets = loader.load_registry()
    assert [asset.asset_id for asset in assets] == [
        "ch4_sample_attrition_lm2011_1994_2008",
        "ch5_fit_horserace_item7_c0",
        "ch5_concordance_item7_common_sample",
    ]


def test_registry_loader_rejects_duplicate_asset_ids(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(loader, "REGISTRY_MODULES", ("chapter4_descriptives", "chapter4_descriptives"))
    with pytest.raises(RegistryError, match="Duplicate asset_id"):
        loader.load_registry()


def test_sample_contract_helpers_apply_expected_selection() -> None:
    base_lf = pl.DataFrame(
        {
            "id": [1, 2, 3],
            "value": [10.0, None, 30.0],
            "ownership_flag": [True, False, True],
        }
    ).lazy()
    raw_df = raw_available(base_lf, filters=(pl.col("id") >= 2,)).collect()
    assert raw_df.get_column("id").to_list() == [2, 3]

    eligible_df = regression_eligible(base_lf, required_columns=("value",)).collect()
    assert eligible_df.get_column("id").to_list() == [1, 3]

    ownership_df = ownership_common_support(
        base_lf,
        ownership_flag_column="ownership_flag",
    ).collect()
    assert ownership_df.get_column("id").to_list() == [1, 3]


def test_common_row_comparison_and_common_success_validation() -> None:
    left_lf = pl.DataFrame(
        {
            "doc_id": ["a", "b", "c"],
            "filing_date": [date(2024, 1, 1), date(2024, 1, 2), date(2024, 1, 3)],
            "text_scope": ["item_7_mda", "item_7_mda", "item_7_mda"],
            "cleaning_policy_id": ["clean", "clean", "clean"],
            "lm_negative_tfidf": [1.0, None, 3.0],
        }
    ).lazy()
    right_lf = pl.DataFrame(
        {
            "doc_id": ["a", "b", "d"],
            "filing_date": [date(2024, 1, 1), date(2024, 1, 2), date(2024, 1, 4)],
            "text_scope": ["item_7_mda", "item_7_mda", "item_7_mda"],
            "cleaning_policy_id": ["clean", "clean", "clean"],
            "finbert_neg_prob_lenw_mean": [0.1, 0.2, 0.4],
        }
    ).lazy()

    joined = common_row_comparison(
        left_lf,
        right_lf,
        join_keys=("doc_id", "filing_date", "text_scope", "cleaning_policy_id"),
        left_signal_columns=("lm_negative_tfidf",),
        right_signal_columns=("finbert_neg_prob_lenw_mean",),
    ).collect()
    assert joined.get_column("doc_id").to_list() == ["a"]

    fit_lf = pl.DataFrame(
        {
            "common_success_policy": [DEFAULT_COMMON_SUCCESS_POLICY, DEFAULT_COMMON_SUCCESS_POLICY],
            "specification_name": ["dictionary_only", "finbert_only"],
        }
    ).lazy()
    assert common_success_comparison(
        fit_lf,
        expected_policy=DEFAULT_COMMON_SUCCESS_POLICY,
    ).collect().height == 2

    bad_fit_lf = pl.DataFrame(
        {
            "common_success_policy": ["wrong_policy"],
            "specification_name": ["dictionary_only"],
        }
    ).lazy()
    with pytest.raises(SampleContractError, match="Expected common-success policy"):
        common_success_comparison(
            bad_fit_lf,
            expected_policy=DEFAULT_COMMON_SUCCESS_POLICY,
        ).collect()


def test_manifest_writing_for_single_asset(tmp_path: Path) -> None:
    repo_root = tmp_path / "repo"
    run_root = repo_root / "inputs" / "lm2011_post_refinitiv"
    run_root.mkdir(parents=True)
    _write_sample_attrition_parquet(run_root / "lm2011_table_i_sample_creation.parquet")

    result = build_single_asset(
        asset_id="ch4_sample_attrition_lm2011_1994_2008",
        run_id="unit_manifest",
        repo_root=repo_root,
        lm2011_post_refinitiv_dir=run_root,
    )

    asset_result = result.asset_results["ch4_sample_attrition_lm2011_1994_2008"]
    assert asset_result.status == "completed"
    assert Path(asset_result.output_paths["csv"]).exists()
    assert Path(asset_result.output_paths["tex"]).exists()
    assert Path(asset_result.output_paths["table_preview"]).exists()

    manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))
    assert manifest["asset_statuses"]["ch4_sample_attrition_lm2011_1994_2008"] == "completed"
    assert manifest["assets"]["ch4_sample_attrition_lm2011_1994_2008"]["sample_contract_id"] == "raw_available"


def test_missing_artifact_failure_is_recorded_in_manifest(tmp_path: Path) -> None:
    repo_root = tmp_path / "repo"
    run_root = repo_root / "inputs" / "lm2011_post_refinitiv"
    run_root.mkdir(parents=True)

    result = build_single_asset(
        asset_id="ch4_sample_attrition_lm2011_1994_2008",
        run_id="unit_missing",
        repo_root=repo_root,
        lm2011_post_refinitiv_dir=run_root,
    )

    asset_result = result.asset_results["ch4_sample_attrition_lm2011_1994_2008"]
    assert asset_result.status == "failed"
    assert "could not be resolved" in (asset_result.failure_reason or "")

    manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))
    assert manifest["asset_statuses"]["ch4_sample_attrition_lm2011_1994_2008"] == "failed"
    assert "could not be resolved" in manifest["assets"]["ch4_sample_attrition_lm2011_1994_2008"]["failure_reason"]


def test_cli_and_api_use_same_build_path(tmp_path: Path) -> None:
    repo_root = tmp_path / "repo"
    run_root = repo_root / "inputs" / "lm2011_post_refinitiv"
    run_root.mkdir(parents=True)
    _write_sample_attrition_parquet(run_root / "lm2011_table_i_sample_creation.parquet")

    api_result = build_single_asset(
        asset_id="ch4_sample_attrition_lm2011_1994_2008",
        run_id="api_run",
        repo_root=repo_root,
        lm2011_post_refinitiv_dir=run_root,
    )
    cli_exit = cli_main(
        [
            "build-asset",
            "--asset-id",
            "ch4_sample_attrition_lm2011_1994_2008",
            "--run-id",
            "cli_run",
            "--repo-root",
            str(repo_root),
            "--lm2011-post-refinitiv-dir",
            str(run_root),
        ]
    )
    assert cli_exit == 0

    api_manifest = json.loads(api_result.manifest_path.read_text(encoding="utf-8"))
    cli_manifest = json.loads((repo_root / "output" / "thesis_assets" / "cli_run" / "manifest.json").read_text(encoding="utf-8"))
    assert api_manifest["asset_statuses"] == cli_manifest["asset_statuses"]
    assert set(api_manifest["assets"]) == set(cli_manifest["assets"])


def test_usage_run_paths_prefer_unified_runner_layout(tmp_path: Path) -> None:
    repo_root = tmp_path / "repo"
    unified_root = (
        repo_root
        / "full_data_run"
        / "sample_5pct_seed42"
        / "results"
        / "sec_ccm_unified_runner"
        / "local_sample"
    )
    (unified_root / "lm2011_post_refinitiv").mkdir(parents=True)
    (unified_root / "lm2011_extension").mkdir(parents=True)
    finbert_run = unified_root / "finbert_item_analysis" / "run_a"
    finbert_run.mkdir(parents=True)
    (finbert_run / "run_manifest.json").write_text("{}", encoding="utf-8")
    (finbert_run / "item_features_long.parquet").touch()

    fallback_post = repo_root / "full_data_run" / "lm2011_post_refinitiv"
    fallback_ext = repo_root / "full_data_run" / "lm2011_extension"
    fallback_post.mkdir(parents=True)
    fallback_ext.mkdir(parents=True)

    resolved = resolve_usage_run_paths(
        repo_root=repo_root,
        data_profile="LOCAL_REPO",
    )

    assert resolved["lm2011_post_refinitiv_dir"] == (unified_root / "lm2011_post_refinitiv").resolve()
    assert resolved["lm2011_extension_dir"] == (unified_root / "lm2011_extension").resolve()
    assert resolved["finbert_run_dir"] == finbert_run.resolve()


def test_usage_run_paths_support_versioned_snapshot_and_drive_layouts(tmp_path: Path) -> None:
    repo_root = tmp_path / "repo"
    full_data_root = repo_root / "full_data_run"
    versioned_post = full_data_root / "lm2011_post_refinitiv-20260421T173344Z-3-001" / "lm2011_post_refinitiv"
    versioned_ext = full_data_root / "lm2011_extension-20260421T114544Z-3-001" / "lm2011_extension"
    versioned_post.mkdir(parents=True)
    versioned_ext.mkdir(parents=True)

    local_resolved = resolve_usage_run_paths(
        repo_root=repo_root,
        data_profile="LOCAL_REPO",
    )
    assert local_resolved["lm2011_post_refinitiv_dir"] == versioned_post.resolve()
    assert local_resolved["lm2011_extension_dir"] == versioned_ext.resolve()

    drive_data_root = tmp_path / "content" / "drive" / "MyDrive" / "Data_LM"
    unified_drive_root = drive_data_root / "results" / "sec_ccm_unified_runner"
    (unified_drive_root / "lm2011_post_refinitiv").mkdir(parents=True)
    (unified_drive_root / "lm2011_extension").mkdir(parents=True)
    drive_finbert_run = unified_drive_root / "finbert_item_analysis" / "finbert_item_analysis_2026-04-20T105101+0000"
    drive_finbert_run.mkdir(parents=True)
    (drive_finbert_run / "run_manifest.json").write_text("{}", encoding="utf-8")
    (drive_finbert_run / "item_features_long.parquet").touch()

    colab_resolved = resolve_usage_run_paths(
        repo_root=repo_root,
        data_profile="COLAB_DRIVE",
        drive_data_root=drive_data_root,
    )
    assert colab_resolved["lm2011_post_refinitiv_dir"] == (unified_drive_root / "lm2011_post_refinitiv").resolve()
    assert colab_resolved["lm2011_extension_dir"] == (unified_drive_root / "lm2011_extension").resolve()
    assert colab_resolved["finbert_run_dir"] == drive_finbert_run.resolve()


def test_tools_entrypoint_build_asset_emits_json_and_allows_failures(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    module_path = REPO_ROOT / "tools" / "build_thesis_assets.py"
    spec = importlib.util.spec_from_file_location("test_build_thesis_assets_tool", module_path)
    assert spec is not None
    assert spec.loader is not None
    tool_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(tool_module)

    repo_root = tmp_path / "repo"
    output_root = tmp_path / "drive" / "Data_LM" / "results" / "thesis_assets" / "tool_run"
    manifest_path = output_root / "manifest.json"
    resolved_paths = {
        "lm2011_post_refinitiv_dir": tmp_path / "inputs" / "lm2011_post_refinitiv",
        "lm2011_extension_dir": tmp_path / "inputs" / "lm2011_extension",
        "finbert_run_dir": tmp_path / "inputs" / "finbert_item_analysis",
    }

    monkeypatch.setattr(tool_module, "_resolve_run_paths", lambda **_: resolved_paths)

    captured_kwargs: dict[str, object] = {}

    def _fake_build_single_asset(**kwargs: object) -> BuildSessionResult:
        captured_kwargs.update(kwargs)
        return BuildSessionResult(
            run_id="tool_run",
            output_root=output_root,
            manifest_path=manifest_path,
            asset_results={
                "ch5_fit_horserace_item7_c0": BuildResult(
                    asset_id="ch5_fit_horserace_item7_c0",
                    chapter="chapter5",
                    asset_kind="table",
                    sample_contract_id="common_success_comparison",
                    status="failed",
                    resolved_inputs={"fit_summary": "C:/tmp/lm2011_extension_fit_summary.parquet"},
                    output_paths={},
                    row_counts={},
                    failure_reason="required artifact missing",
                )
            },
        )

    monkeypatch.setattr(tool_module, "build_single_asset", _fake_build_single_asset)

    exit_code = tool_module.main(
        [
            "build-asset",
            "--asset-id",
            "ch5_fit_horserace_item7_c0",
            "--run-id",
            "tool_run",
            "--repo-root",
            str(repo_root),
            "--output-root",
            str(output_root),
            "--allow-failures",
        ]
    )

    assert exit_code == 0
    assert captured_kwargs["repo_root"] == repo_root.resolve()
    assert captured_kwargs["output_root"] == output_root.resolve()
    assert captured_kwargs["run_id"] == "tool_run"

    payload = json.loads(capsys.readouterr().out)
    assert payload["command"] == "build-asset"
    assert payload["run_id"] == "tool_run"
    assert payload["output_root"] == str(output_root)
    assert payload["manifest_path"] == str(manifest_path)
    assert payload["asset_statuses"] == {"ch5_fit_horserace_item7_c0": "failed"}
    assert payload["resolved_paths"] == {key: str(value) for key, value in resolved_paths.items()}


def _write_sample_attrition_parquet(path: Path) -> None:
    df = pl.DataFrame(
        {
            "section_id": ["full_10k_document", "full_10k_document"],
            "section_label": ["Full 10-K Document", "Full 10-K Document"],
            "section_order": [1, 1],
            "row_order": [1, 2],
            "row_id": ["edgar_complete_nonduplicate_sample", "first_filing_per_year"],
            "display_label": [
                "EDGAR 10-K/10-K405 1994-2008 complete sample (excluding duplicates)",
                "Include only first filing in a given year",
            ],
            "sample_size_kind": ["count", "count"],
            "sample_size_value": [121995.0, 120350.0],
            "observations_removed": [None, 1645],
            "availability_status": ["available", "available"],
            "availability_reason": [None, None],
        }
    )
    df.write_parquet(path)
