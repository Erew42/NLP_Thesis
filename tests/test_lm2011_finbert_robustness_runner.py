from __future__ import annotations

import datetime as dt
import json
from pathlib import Path
import random

import polars as pl
import pytest
from polars.testing import assert_frame_equal

from thesis_pkg.benchmarking.finbert_tail_features import TAIL_FEATURE_COLUMNS
from thesis_pkg.benchmarking.manifest_contracts import MANIFEST_PATH_SEMANTICS_RELATIVE
from thesis_pkg.notebooks_and_scripts import lm2011_finbert_robustness_runner as runner


def _write_parquet(path: Path, df: pl.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.write_parquet(path)


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _build_extension_panel() -> pl.DataFrame:
    rows: list[dict[str, object]] = []
    rng = random.Random(20260423)
    filing_dates = [
        dt.date(2009, 2, 16),
        dt.date(2009, 5, 15),
        dt.date(2010, 2, 16),
        dt.date(2010, 5, 15),
    ]
    scope_specs = (
        ("item_7_mda", 0.0),
        ("item_1a_risk_factors", 0.03),
    )
    for quarter_idx, filing_date in enumerate(filing_dates):
        for scope_idx, (text_scope, scope_shift) in enumerate(scope_specs):
            for industry_idx, industry_id in enumerate((1, 12)):
                for obs_idx in range(6):
                    doc_id = (
                        f"{filing_date.year}_{filing_date.month:02d}_{text_scope}_"
                        f"{industry_id}_{obs_idx}"
                    )
                    log_size = 4.0 + rng.random() + 0.05 * quarter_idx + scope_shift
                    log_bm = -0.2 + 0.5 * rng.random() + 0.03 * industry_idx + (0.25 * scope_shift)
                    log_turnover = -3.0 + 0.7 * rng.random() + 0.01 * obs_idx
                    pre_ffalpha = -0.02 + 0.04 * rng.random() + 0.002 * quarter_idx
                    ownership = 20.0 + 30.0 * rng.random() + 2.0 * industry_idx
                    lm_signal = 0.02 + 0.08 * rng.random() + 0.002 * industry_idx + scope_shift
                    finbert_neg = 0.25 + 0.35 * rng.random() + 0.01 * quarter_idx + scope_shift
                    finbert_pos = 0.08 + 0.18 * rng.random() + 0.004 * industry_idx + (0.2 * scope_shift)
                    finbert_net = finbert_neg - finbert_pos
                    finbert_neg_dominant_share = 0.15 + 0.55 * rng.random() + (0.3 * scope_shift)
                    nasdaq_dummy = int((obs_idx + industry_idx + scope_idx) % 2)
                    outcome = (
                        0.01
                        + 0.7 * lm_signal
                        - 0.05 * finbert_neg
                        + 0.003 * log_size
                        - 0.002 * log_bm
                        + 0.0015 * log_turnover
                        + 0.02 * pre_ffalpha
                        + 0.0003 * ownership
                        + 0.002 * nasdaq_dummy
                        + 0.001 * quarter_idx
                        + 0.0015 * industry_idx
                        + 0.0007 * scope_idx
                    )
                    rows.append(
                        {
                            "doc_id": doc_id,
                            "sample_window": "2009_2024",
                            "text_scope": text_scope,
                            "filing_date": filing_date,
                            "ff48_industry_id": industry_id,
                            "log_size": log_size,
                            "log_book_to_market": log_bm,
                            "log_share_turnover": log_turnover,
                            "pre_ffalpha": pre_ffalpha,
                            "nasdaq_dummy": nasdaq_dummy,
                            "institutional_ownership_proxy_refinitiv": ownership,
                            "common_support_flag_ownership": True,
                            "filing_period_excess_return": outcome,
                            "abnormal_volume": 0.12 + 0.005 * obs_idx + 0.003 * scope_idx,
                            "postevent_return_volatility": 0.04 + 0.002 * obs_idx + 0.001 * scope_idx,
                            "lm_negative_tfidf": lm_signal,
                            "finbert_neg_prob_lenw_mean": finbert_neg,
                            "finbert_pos_prob_lenw_mean": finbert_pos,
                            "finbert_neu_prob_lenw_mean": 1.0 - finbert_neg - finbert_pos,
                            "finbert_net_negative_lenw_mean": finbert_net,
                            "finbert_neg_dominant_share": finbert_neg_dominant_share,
                            "dictionary_family_source": "replication",
                        }
                    )
    return pl.DataFrame(rows)


def _build_sentence_scores_by_year(panel_df: pl.DataFrame) -> dict[int, pl.DataFrame]:
    rows_by_year: dict[int, list[dict[str, object]]] = {}
    sentence_weight_template = [12, 10, 8, 6, 4, 2]
    sentence_offsets = [0.24, 0.16, 0.08, -0.02, -0.12, -0.22]
    for row in panel_df.iter_rows(named=True):
        year = row["filing_date"].year
        year_rows = rows_by_year.setdefault(year, [])
        center = float(row["finbert_neg_prob_lenw_mean"])
        for sentence_idx, (token_weight, offset) in enumerate(
            zip(sentence_weight_template, sentence_offsets, strict=True)
        ):
            negative_prob = min(max(center + offset, 0.01), 0.99)
            neutral_prob = min(max(0.8 - (0.4 * negative_prob), 0.01), 0.98)
            positive_prob = max(0.01, 1.0 - negative_prob - neutral_prob)
            year_rows.append(
                {
                    "doc_id": row["doc_id"],
                    "filing_date": row["filing_date"],
                    "text_scope": row["text_scope"],
                    "cleaning_policy_id": "item_text_clean_v2",
                    "model_name": "yiyanghkust/finbert-tone",
                    "model_version": "rev-a",
                    "segment_policy_id": "sentence_dataset_v1_finbert_token_512",
                    "sentence_index": sentence_idx,
                    "benchmark_sentence_id": f"{row['doc_id']}:{row['text_scope']}:{sentence_idx}",
                    "negative_prob": negative_prob,
                    "neutral_prob": neutral_prob,
                    "positive_prob": positive_prob,
                    "predicted_label": "negative" if negative_prob >= positive_prob else "neutral",
                    "finbert_token_count_512": token_weight + sentence_idx,
                }
            )
    return {
        year: pl.DataFrame(year_rows).sort(
            "doc_id",
            "text_scope",
            "negative_prob",
            "sentence_index",
            descending=[False, False, True, False],
        )
        for year, year_rows in rows_by_year.items()
    }


def test_parse_args_rejects_removed_finbert_item_features_override() -> None:
    with pytest.raises(SystemExit):
        runner.parse_args(
            [
                "--extension-run-dir",
                "extension",
                "--finbert-analysis-run-dir",
                "finbert",
                "--output-dir",
                "output",
                "--finbert-item-features-long-path",
                "unused.parquet",
            ]
        )


def test_lm2011_finbert_robustness_runner_emits_variant_tagged_outputs(tmp_path: Path) -> None:
    extension_run_dir = tmp_path / "extension_run"
    finbert_run_dir = tmp_path / "finbert_run"
    output_dir = tmp_path / "robustness_output"

    extension_panel = _build_extension_panel()
    sentence_scores_by_year = _build_sentence_scores_by_year(extension_panel)

    extension_panel_path = extension_run_dir / "custom_panels" / "analysis_panel.parquet"
    sentence_scores_dir = finbert_run_dir / "tail_inputs" / "by_year"

    _write_parquet(extension_panel_path, extension_panel)
    for year, year_df in sentence_scores_by_year.items():
        _write_parquet(sentence_scores_dir / f"{year}.parquet", year_df)

    _write_json(
        extension_run_dir / runner.EXTENSION_MANIFEST_FILENAME,
        {
            "path_semantics": MANIFEST_PATH_SEMANTICS_RELATIVE,
            "stages": {
                "extension_analysis_panel": {
                    "artifact_path": "custom_panels/analysis_panel.parquet",
                }
            },
        },
    )
    _write_json(
        finbert_run_dir / runner.FINBERT_ANALYSIS_MANIFEST_FILENAME,
        {
            "path_semantics": MANIFEST_PATH_SEMANTICS_RELATIVE,
            "artifacts": {
                "item_features_long_path": "analysis_outputs/item_features_long.parquet",
                "sentence_scores_dir": "tail_inputs/by_year",
            },
        },
    )

    artifacts = runner.run_lm2011_finbert_robustness(
        runner.FinbertRobustnessRunConfig(
            extension_run_dir=extension_run_dir,
            finbert_analysis_run_dir=finbert_run_dir,
            output_dir=output_dir,
            run_name="unit_test_finbert_robustness",
        )
    )

    existing_scale_coefficients = pl.read_parquet(artifacts.existing_scale_coefficients_path)
    tail_coefficients = pl.read_parquet(artifacts.tail_coefficients_path)
    tail_doc_surface = pl.read_parquet(artifacts.tail_doc_surface_path)
    candidate_summary = pl.read_parquet(artifacts.candidate_summary_path)

    assert artifacts.manifest_path.exists()
    assert artifacts.existing_scale_fit_summary_path.exists()
    assert artifacts.existing_scale_fit_comparisons_path.exists()
    assert artifacts.tail_fit_summary_path.exists()
    assert artifacts.tail_fit_comparisons_path.exists()
    assert artifacts.tail_doc_surface_by_year_dir.exists()
    assert (artifacts.tail_doc_surface_by_year_dir / "2009.parquet").exists()
    assert (artifacts.tail_doc_surface_by_year_dir / "2010.parquet").exists()

    assert set(existing_scale_coefficients.get_column("variant_id").unique().to_list()) == {
        variant.variant_id for variant in runner.EXISTING_SCALE_VARIANTS
    }
    assert set(existing_scale_coefficients.get_column("weighting_rule").drop_nulls().unique().to_list()) == {
        "quarter_observation_count",
        "equal_quarter",
    }
    assert set(tail_coefficients.get_column("variant_id").unique().to_list()) == set(TAIL_FEATURE_COLUMNS)
    assert set(tail_coefficients.get_column("weighting_rule").drop_nulls().unique().to_list()) == {
        "quarter_observation_count",
        "equal_quarter",
    }
    assert tail_doc_surface.height == extension_panel.height
    assert set(tail_doc_surface.get_column("text_scope").unique().to_list()) == set(runner.PRIMARY_TEXT_SCOPES)
    assert candidate_summary.height > 0
    assert set(candidate_summary.get_column("variant_family").unique().to_list()) == {
        runner.EXISTING_SCALE_FAMILY,
        runner.TAIL_SIGNAL_FAMILY,
    }
    stacked_from_years = pl.concat(
        [
            pl.read_parquet(artifacts.tail_doc_surface_by_year_dir / "2009.parquet"),
            pl.read_parquet(artifacts.tail_doc_surface_by_year_dir / "2010.parquet"),
        ],
        how="vertical_relaxed",
    ).sort("doc_id", "text_scope")
    assert_frame_equal(
        tail_doc_surface.sort("doc_id", "text_scope"),
        stacked_from_years,
    )

    manifest = json.loads(artifacts.manifest_path.read_text(encoding="utf-8"))
    assert manifest["path_semantics"] == MANIFEST_PATH_SEMANTICS_RELATIVE
    assert manifest["effective_dictionary_family_source"] == "replication"
    assert manifest["resolved_inputs"]["extension_analysis_panel_path"] == str(extension_panel_path.resolve())
    assert "finbert_item_features_long_path" not in manifest["resolved_inputs"]
    assert manifest["resolved_inputs"]["finbert_sentence_scores_dir"] == str(sentence_scores_dir.resolve())
    assert len(manifest["variant_registry"]["existing_scale"]) == 5
    assert len(manifest["variant_registry"]["tail_signal"]) == len(TAIL_FEATURE_COLUMNS)
    assert manifest["row_counts"]["tail_doc_surface"] == extension_panel.height
    assert not Path(manifest["artifacts"]["candidate_summary_path"]).is_absolute()


def test_build_tail_doc_surfaces_rejects_cross_year_duplicate_doc_scope(tmp_path: Path) -> None:
    sentence_scores_dir = tmp_path / "sentence_scores" / "by_year"
    output_dir = tmp_path / "tail_output"
    duplicate_rows = [
        {
            "doc_id": "duplicate_doc",
            "filing_date": dt.date(2010, 3, 1),
            "text_scope": "item_7_mda",
            "cleaning_policy_id": "item_text_clean_v2",
            "model_name": "yiyanghkust/finbert-tone",
            "model_version": "rev-a",
            "segment_policy_id": "sentence_dataset_v1_finbert_token_512",
            "sentence_index": 0,
            "benchmark_sentence_id": "dup:0",
            "negative_prob": 0.91,
            "finbert_token_count_512": 12,
        }
    ]
    _write_parquet(sentence_scores_dir / "2009.parquet", pl.DataFrame(duplicate_rows))
    _write_parquet(
        sentence_scores_dir / "2010.parquet",
        pl.DataFrame(duplicate_rows).with_columns(pl.lit(dt.date(2011, 3, 1)).alias("filing_date")),
    )

    with pytest.raises(ValueError, match="tail_doc_surface"):
        runner._build_tail_doc_surfaces(
            sentence_scores_dir=sentence_scores_dir,
            output_dir=output_dir,
        )
