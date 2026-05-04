from __future__ import annotations

import datetime as dt
import importlib.util
import json
import random
from pathlib import Path

import polars as pl


ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = ROOT / "tools" / "run_finbert_secondary_outcome_regressions.py"
SPEC = importlib.util.spec_from_file_location("run_finbert_secondary_outcome_regressions", SCRIPT_PATH)
assert SPEC is not None
runner = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(runner)


def _extension_panel() -> pl.DataFrame:
    rows: list[dict[str, object]] = []
    rng = random.Random(20260411)
    filing_dates = [dt.date(2009, 2, 16), dt.date(2009, 5, 15)]
    for quarter_idx, filing_date in enumerate(filing_dates):
        for industry_idx, industry_id in enumerate((1, 12)):
            for obs_idx in range(8):
                doc_index = len(rows) + 1
                log_size = 4.0 + rng.random() + 0.05 * quarter_idx
                log_bm = -0.2 + 0.5 * rng.random() + 0.03 * industry_idx
                log_turnover = -3.0 + 0.7 * rng.random() + 0.01 * obs_idx
                pre_ffalpha = -0.02 + 0.04 * rng.random() + 0.002 * quarter_idx
                ownership = 20.0 + 30.0 * rng.random() + 2.0 * industry_idx
                lm_signal = 0.02 + 0.08 * rng.random() + 0.002 * industry_idx
                finbert_signal = 0.25 + 0.35 * rng.random() + 0.01 * quarter_idx
                nasdaq = int((obs_idx + industry_idx) % 2)
                filing_return = (
                    0.01
                    + 0.7 * lm_signal
                    - 0.04 * finbert_signal
                    + 0.003 * log_size
                    - 0.002 * log_bm
                    + 0.001 * log_turnover
                    + 0.02 * pre_ffalpha
                    + 0.0003 * ownership
                    + 0.002 * nasdaq
                    + 0.001 * quarter_idx
                    + 0.002 * industry_idx
                )
                rows.append(
                    {
                        "doc_id": f"doc_{doc_index}",
                        "sample_window": "2009_2024",
                        "text_scope": "item_7_mda",
                        "filing_date": filing_date,
                        "ff48_industry_id": industry_id,
                        "log_size": log_size,
                        "log_book_to_market": log_bm,
                        "log_share_turnover": log_turnover,
                        "pre_ffalpha": pre_ffalpha,
                        "nasdaq_dummy": nasdaq,
                        "institutional_ownership_proxy_refinitiv": ownership,
                        "common_support_flag_ownership": True,
                        "filing_period_excess_return": filing_return,
                        "abnormal_volume": 0.1 + 0.001 * obs_idx + 0.01 * finbert_signal,
                        "postevent_return_volatility": 0.04 + 0.001 * obs_idx + 0.01 * lm_signal,
                        "lm_negative_tfidf": lm_signal,
                        "finbert_neg_prob_lenw_mean": finbert_signal,
                    }
                )
    return pl.DataFrame(rows)


def test_parse_args_defaults_to_secondary_outcomes() -> None:
    args = runner.parse_args(["--extension-panel-path", "panel.parquet"])

    assert tuple(args.outcomes) == runner.DEFAULT_SECONDARY_OUTCOMES
    assert tuple(args.text_scopes) == runner.DEFAULT_TEXT_SCOPES
    assert tuple(args.control_set_ids) == runner.DEFAULT_CONTROL_SET_IDS
    assert tuple(args.specification_names) == runner.DEFAULT_SPECIFICATION_NAMES
    assert args.nw_lags == 1
    assert args.include_primary_return is False


def test_secondary_outcome_tool_writes_expected_artifacts(tmp_path: Path) -> None:
    panel_path = tmp_path / "extension_panel.parquet"
    output_dir = tmp_path / "secondary_outputs"
    _extension_panel().write_parquet(panel_path)

    args = runner.parse_args(
        [
            "--extension-panel-path",
            str(panel_path),
            "--output-dir",
            str(output_dir),
            "--text-scopes",
            "item_7_mda",
        ]
    )
    result = runner.run_secondary_outcome_regressions(args)

    expected_files = [
        runner.COEFFICIENTS_FILENAME,
        runner.FIT_SUMMARY_FILENAME,
        runner.FIT_COMPARISONS_FILENAME,
        runner.FIT_SKIPPED_QUARTERS_FILENAME,
    ]
    for filename in expected_files:
        parquet_path = output_dir / filename
        assert parquet_path.exists()
        assert parquet_path.with_suffix(".csv").exists()

    manifest_path = output_dir / runner.MANIFEST_FILENAME
    assert manifest_path.exists()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    coefficients = pl.read_parquet(output_dir / runner.COEFFICIENTS_FILENAME)
    fit_summary = pl.read_parquet(output_dir / runner.FIT_SUMMARY_FILENAME)
    fit_comparisons = pl.read_parquet(output_dir / runner.FIT_COMPARISONS_FILENAME)

    assert result["row_counts"]["coefficients"] == coefficients.height
    assert manifest["resolved_inputs"]["extension_panel_path"] == str(panel_path.resolve())
    assert manifest["config"]["outcomes"] == list(runner.DEFAULT_SECONDARY_OUTCOMES)
    assert manifest["config"]["text_scopes"] == ["item_7_mda"]
    assert manifest["config"]["control_set_ids"] == ["C0"]
    assert set(coefficients.get_column("outcome_name").unique().to_list()) == set(
        runner.DEFAULT_SECONDARY_OUTCOMES
    )
    assert set(fit_summary.get_column("estimator_status").unique().to_list()) == {"estimated"}
    assert set(fit_comparisons.get_column("estimator_status").unique().to_list()) == {"estimated"}
    assert manifest["row_counts"]["fit_summary"] == fit_summary.height
    assert manifest["row_counts"]["fit_comparisons"] == fit_comparisons.height
    assert manifest["status_counts"]["coefficients"]
