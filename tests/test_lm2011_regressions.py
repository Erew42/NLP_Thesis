from __future__ import annotations

import datetime as dt
import math

import polars as pl
import pytest

from thesis_pkg.pipeline import (
    build_lm2011_normalized_difference_panel,
    build_lm2011_return_regression_panel,
    build_lm2011_sue_regression_panel,
    build_lm2011_table_ia_i_results,
    build_lm2011_table_ia_ii_results,
    build_lm2011_table_iv_results,
    build_lm2011_table_v_results,
    build_lm2011_table_vi_results,
    build_lm2011_table_viii_results,
    run_lm2011_quarterly_fama_macbeth,
)
from thesis_pkg.pipelines.lm2011_pipeline import (
    build_lm2011_trading_strategy_ff4_summary,
    build_lm2011_trading_strategy_monthly_returns,
)


def _write_ff48_mapping(tmp_path) -> str:
    ff48_path = tmp_path / "ff48.txt"
    ff48_path.write_text(
        "\n".join(
            [
                " 1 Agric  Agriculture",
                "          0100-0199 Agricultural production - crops",
                "12 MedEq  Medical Equipment",
                "          3840-3849 Surgical, medical, and dental instruments and supplies",
            ]
        ),
        encoding="utf-8",
    )
    return str(ff48_path)


def _lm_dictionary_lists() -> dict[str, list[str]]:
    return {
        "negative": ["loss"],
        "positive": ["gain"],
        "uncertainty": ["uncertain"],
        "litigious": ["lawsuit"],
        "modal_strong": ["must"],
        "modal_weak": ["may"],
    }


def _harvard_negative_word_list() -> list[str]:
    return ["bad"]


def _master_dictionary_words() -> list[str]:
    return [
        "loss",
        "gain",
        "uncertain",
        "lawsuit",
        "must",
        "may",
        "bad",
        "neutral",
        "recognized",
    ]


def _strategy_months() -> list[dt.date]:
    return [
        dt.date(1997, 7, 31),
        dt.date(1997, 8, 31),
        dt.date(1997, 9, 30),
        dt.date(1997, 10, 31),
        dt.date(1997, 11, 30),
        dt.date(1997, 12, 31),
        dt.date(1998, 1, 31),
        dt.date(1998, 2, 28),
        dt.date(1998, 3, 31),
        dt.date(1998, 4, 30),
        dt.date(1998, 5, 31),
        dt.date(1998, 6, 30),
    ]


def _build_strategy_inputs() -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    event_panel = pl.DataFrame(
        {
            "doc_id": [f"doc_{idx}" for idx in range(10)],
            "KYPERMNO": [100 + idx for idx in range(10)],
            "filing_date": [dt.date(1996, 3, 1) + dt.timedelta(days=idx) for idx in range(10)],
        }
    )
    sec_parsed = pl.DataFrame(
        {
            "doc_id": [f"doc_{idx}" for idx in range(10)],
            "cik_10": [f"{idx:010d}" for idx in range(10)],
            "filing_date": [dt.date(1996, 3, 1) + dt.timedelta(days=idx) for idx in range(10)],
            "document_type_filename": ["10-K"] * 10,
            "full_text": [
                " ".join((["loss"] * (idx + 1)) + (["bad"] * max(0, 9 - idx)) + (["neutral"] * 20))
                for idx in range(10)
            ],
        }
    )
    monthly_rows: list[dict[str, object]] = []
    stock_months = [dt.date(1997, 6, 30), *_strategy_months()]
    for permno_idx, permno in enumerate(range(100, 110)):
        for month_idx, month_end in enumerate(stock_months):
            monthly_rows.append(
                {
                    "KYPERMNO": permno,
                    "MCALDT": month_end,
                    "MRET": 0.01 + 0.001 * permno_idx + 0.0002 * month_idx,
                    "MTCAP": 100.0 + 10.0 * permno_idx + month_idx,
                }
            )
    monthly_stock = pl.DataFrame(monthly_rows)
    monthly_factors = pl.DataFrame(
        {
            "month_end": _strategy_months(),
            "mkt_rf": [0.01, 0.015, -0.01, 0.02, 0.005, -0.005, 0.012, 0.011, -0.006, 0.008, 0.007, 0.009],
            "smb": [0.002, -0.001, 0.001, 0.0, 0.003, -0.002, 0.002, -0.001, 0.001, 0.002, 0.0, -0.001],
            "hml": [0.001, 0.002, -0.002, 0.001, 0.0, -0.001, 0.001, 0.002, -0.001, 0.0, 0.001, -0.002],
            "rf": [0.0003] * 12,
            "mom": [0.004, 0.005, -0.003, 0.006, 0.002, -0.001, 0.003, 0.004, -0.002, 0.001, 0.002, 0.003],
        }
    )
    return event_panel, sec_parsed, monthly_stock, monthly_factors


def _regression_test_inputs(tmp_path) -> dict[str, object]:
    event_rows: list[dict[str, object]] = []
    sue_rows: list[dict[str, object]] = []
    full_rows: list[dict[str, object]] = []
    mda_rows: list[dict[str, object]] = []
    company_history_rows: list[dict[str, object]] = []
    company_description_rows: list[dict[str, object]] = []
    quarter_dates = [
        dt.date(2020, 2, 14),
        dt.date(2020, 5, 15),
        dt.date(2021, 2, 12),
        dt.date(2021, 5, 14),
    ]

    doc_index = 0
    for quarter_idx, filing_date in enumerate(quarter_dates):
        for industry_idx, sic_code in enumerate((111, 3845)):
            for obs_idx in range(6):
                doc_index += 1
                doc_id = f"doc_{doc_index}"
                gvkey_int = 1000 + doc_index
                signal_base = (
                    0.02
                    + 0.0032 * obs_idx
                    + 0.00035 * (obs_idx**2)
                    + 0.0025 * quarter_idx
                    + 0.006 * industry_idx
                )
                h4n_prop = (
                    0.015
                    + 0.0026 * obs_idx
                    + 0.00045 * ((obs_idx + 1) ** 2)
                    + 0.0017 * quarter_idx
                    + 0.004 * industry_idx
                )
                lm_negative_prop = signal_base
                lm_negative_tfidf = 1.5 * lm_negative_prop + 0.01
                h4n_inf_tfidf = 1.6 * h4n_prop + 0.005
                size_event = 90.0 + 5.0 * doc_index + 1.4 * (obs_idx**2) + 3.0 * industry_idx
                bm_event = 0.5 + 0.018 * obs_idx + 0.0035 * (obs_idx**2) + 0.01 * quarter_idx + 0.02 * industry_idx
                share_turnover = (
                    0.08
                    + 0.0065 * obs_idx
                    + 0.0011 * (obs_idx**2)
                    + 0.004 * quarter_idx
                    + 0.002 * industry_idx
                )
                pre_ffalpha = (
                    -0.02
                    + 0.0022 * obs_idx
                    + 0.0004 * (obs_idx**2)
                    + 0.001 * quarter_idx
                    + 0.00035 * ((obs_idx + quarter_idx + (2 * industry_idx)) % 3)
                )
                institutional_ownership = (
                    20.0
                    + 1.4 * obs_idx
                    + 0.5 * (obs_idx**2)
                    + 2.5 * quarter_idx
                    + 0.8 * industry_idx
                    + 0.6 * ((obs_idx + quarter_idx + industry_idx) % 3)
                )
                nasdaq_dummy = int((obs_idx + industry_idx) % 2)
                analyst_dispersion = (
                    0.01
                    + 0.0008 * obs_idx
                    + 0.00025 * (obs_idx**2)
                    + 0.0005 * quarter_idx
                    + 0.0002 * industry_idx
                    + 0.00017 * ((obs_idx + quarter_idx + industry_idx) % 2)
                    + 0.000019 * (((obs_idx + 1) * (industry_idx + 1) * (quarter_idx + 1)) % 5)
                )
                analyst_revisions = (
                    -0.02
                    + 0.0015 * obs_idx
                    + 0.0003 * (obs_idx**2)
                    + 0.001 * quarter_idx
                    + 0.00035 * industry_idx
                    + 0.00021 * ((obs_idx + (2 * quarter_idx) + industry_idx) % 3)
                    + 0.000023 * (((obs_idx + 2) * (quarter_idx + 1) + industry_idx) % 4)
                )
                quarter_effect = 0.002 * quarter_idx
                industry_effect = 0.003 * industry_idx

                filing_period_excess_return = (
                    0.02
                    + 0.35 * lm_negative_prop
                    - 0.18 * h4n_prop
                    + 0.001 * math.log(size_event)
                    + 0.002 * math.log(bm_event)
                    + 0.004 * math.log(share_turnover)
                    + 0.03 * pre_ffalpha
                    + 0.0005 * institutional_ownership
                    + 0.005 * nasdaq_dummy
                    + quarter_effect
                    + industry_effect
                )
                sue = (
                    0.08
                    + 0.4 * lm_negative_prop
                    + 0.1 * h4n_prop
                    + 0.2 * analyst_dispersion
                    - 0.15 * analyst_revisions
                    + 0.0007 * institutional_ownership
                    + 0.01 * pre_ffalpha
                    + quarter_effect
                    + industry_effect
                )

                full_signal_row = {
                    "doc_id": doc_id,
                    "token_count_full_10k": 2500 + doc_index,
                    "total_token_count_full_10k": 2500 + doc_index,
                    "h4n_inf_prop": h4n_prop,
                    "h4n_inf_tfidf": h4n_inf_tfidf,
                    "lm_negative_prop": lm_negative_prop,
                    "lm_negative_tfidf": lm_negative_tfidf,
                    "lm_positive_prop": 0.01 + 0.002 * obs_idx + 0.001 * industry_idx,
                    "lm_positive_tfidf": 0.04 + 0.003 * obs_idx + 0.002 * industry_idx,
                    "lm_uncertainty_prop": 0.008 + 0.0015 * obs_idx + 0.0007 * quarter_idx,
                    "lm_uncertainty_tfidf": 0.03 + 0.002 * obs_idx + 0.001 * quarter_idx,
                    "lm_litigious_prop": 0.006 + 0.001 * obs_idx + 0.001 * industry_idx,
                    "lm_litigious_tfidf": 0.025 + 0.0015 * obs_idx + 0.0008 * industry_idx,
                    "lm_modal_strong_prop": 0.012 + 0.001 * obs_idx + 0.0004 * quarter_idx,
                    "lm_modal_strong_tfidf": 0.03 + 0.002 * obs_idx + 0.0012 * quarter_idx,
                    "lm_modal_weak_prop": 0.014 + 0.0013 * obs_idx + 0.0006 * industry_idx,
                    "lm_modal_weak_tfidf": 0.035 + 0.0022 * obs_idx + 0.001 * industry_idx,
                }
                mda_signal_row = {
                    "doc_id": doc_id,
                    "token_count_mda": 600 + doc_index,
                    "total_token_count_mda": 600 + doc_index,
                    "h4n_inf_prop": h4n_prop + 0.002,
                    "h4n_inf_tfidf": h4n_inf_tfidf + 0.003,
                    "lm_negative_prop": lm_negative_prop + 0.001,
                    "lm_negative_tfidf": lm_negative_tfidf + 0.004,
                    "lm_positive_prop": full_signal_row["lm_positive_prop"] + 0.001,
                    "lm_positive_tfidf": full_signal_row["lm_positive_tfidf"] + 0.002,
                    "lm_uncertainty_prop": full_signal_row["lm_uncertainty_prop"] + 0.0005,
                    "lm_uncertainty_tfidf": full_signal_row["lm_uncertainty_tfidf"] + 0.001,
                    "lm_litigious_prop": full_signal_row["lm_litigious_prop"] + 0.0004,
                    "lm_litigious_tfidf": full_signal_row["lm_litigious_tfidf"] + 0.001,
                    "lm_modal_strong_prop": full_signal_row["lm_modal_strong_prop"] + 0.0005,
                    "lm_modal_strong_tfidf": full_signal_row["lm_modal_strong_tfidf"] + 0.001,
                    "lm_modal_weak_prop": full_signal_row["lm_modal_weak_prop"] + 0.0004,
                    "lm_modal_weak_tfidf": full_signal_row["lm_modal_weak_tfidf"] + 0.001,
                }

                event_rows.append(
                    {
                        "doc_id": doc_id,
                        "gvkey_int": gvkey_int,
                        "KYPERMNO": 5000 + doc_index,
                        "filing_date": filing_date,
                        "filing_trade_date": filing_date,
                        "pre_filing_trade_date": filing_date - dt.timedelta(days=1),
                        "size_event": size_event,
                        "bm_event": bm_event,
                        "share_turnover": share_turnover,
                        "pre_ffalpha": pre_ffalpha,
                        "institutional_ownership": institutional_ownership,
                        "nasdaq_dummy": nasdaq_dummy,
                        "filing_period_excess_return": filing_period_excess_return,
                        "abnormal_volume": 0.02 + 0.001 * obs_idx,
                        "postevent_return_volatility": 0.03 + 0.001 * obs_idx,
                    }
                )
                sue_rows.append(
                    {
                        "doc_id": doc_id,
                        "gvkey_int": gvkey_int,
                        "KYPERMNO": 5000 + doc_index,
                        "filing_date": filing_date,
                        "quarter_report_date": filing_date + dt.timedelta(days=20),
                        "size_event": size_event,
                        "bm_event": bm_event,
                        "share_turnover": share_turnover,
                        "sue": sue,
                        "analyst_dispersion": analyst_dispersion,
                        "analyst_revisions": analyst_revisions,
                        "pre_ffalpha": pre_ffalpha,
                        "institutional_ownership": institutional_ownership,
                        "nasdaq_dummy": nasdaq_dummy,
                    }
                )
                full_rows.append(full_signal_row)
                mda_rows.append(mda_signal_row)
                company_history_rows.append(
                    {
                        "KYGVKEY": gvkey_int,
                        "HCHGDT": dt.date(2010, 1, 1),
                        "HCHGENDDT": None,
                        "HSIC": sic_code,
                    }
                )
                company_description_rows.append(
                    {
                        "KYGVKEY": gvkey_int,
                        "SIC": str(sic_code),
                    }
                )

    return {
        "event_panel": pl.DataFrame(event_rows),
        "sue_panel": pl.DataFrame(sue_rows),
        "full_text_features": pl.DataFrame(full_rows),
        "mda_text_features": pl.DataFrame(mda_rows),
        "company_history": pl.DataFrame(company_history_rows),
        "company_description": pl.DataFrame(company_description_rows),
        "ff48_path": _write_ff48_mapping(tmp_path),
    }


def test_run_lm2011_quarterly_fama_macbeth_weights_quarters_and_hides_industry_dummies() -> None:
    rows: list[dict[str, object]] = []
    for filing_date, slope, x_values in (
        (dt.date(2021, 2, 15), 1.0, [0.0, 1.0]),
        (dt.date(2021, 5, 15), 3.0, [0.0, 1.0, 2.0]),
    ):
        for industry_id in (1, 12):
            for x_value in x_values:
                rows.append(
                    {
                        "filing_date": filing_date,
                        "ff48_industry_id": industry_id,
                        "signal": x_value,
                        "dependent": 1.0 + slope * x_value + (5.0 if industry_id == 12 else 0.0),
                    }
                )

    results = run_lm2011_quarterly_fama_macbeth(
        pl.DataFrame(rows).lazy(),
        table_id="unit_test_table",
        text_scope="full_10k",
        dependent_variable="dependent",
        signal_column="signal",
        control_columns=(),
    ).sort("coefficient_name")

    by_name = {row["coefficient_name"]: row for row in results.to_dicts()}
    assert math.isclose(by_name["signal"]["estimate"], 2.2, rel_tol=0.0, abs_tol=1e-12)
    assert by_name["signal"]["n_quarters"] == 2
    assert math.isclose(by_name["signal"]["mean_quarter_n"], 5.0, rel_tol=0.0, abs_tol=1e-12)
    assert by_name["signal"]["weighting_rule"] == "quarter_observation_count"
    assert all(not row["coefficient_name"].startswith("_industry_dummy_") for row in results.to_dicts())


def test_run_lm2011_quarterly_fama_macbeth_raises_on_rank_deficient_quarter() -> None:
    rows: list[dict[str, object]] = []
    filing_date = dt.date(2021, 2, 15)
    for industry_id in (1, 12):
        for obs_idx in range(2):
            rows.append(
                {
                    "filing_date": filing_date,
                    "ff48_industry_id": industry_id,
                    "signal": 1.0 if industry_id == 12 else 0.0,
                    "dependent": 1.0 + float(obs_idx) + (3.0 if industry_id == 12 else 0.0),
                }
            )

    with pytest.raises(ValueError, match=r"Quarterly Fama-MacBeth quarter=2021-01-01 .*rank-deficient OLS design"):
        run_lm2011_quarterly_fama_macbeth(
            pl.DataFrame(rows).lazy(),
            table_id="unit_test_table",
            text_scope="full_10k",
            dependent_variable="dependent",
            signal_column="signal",
            control_columns=(),
        )


def test_run_lm2011_quarterly_fama_macbeth_keeps_near_singular_small_sample_quarter() -> None:
    rows: list[dict[str, object]] = []
    filing_date = dt.date(2021, 2, 15)
    for industry_id, industry_effect in ((1, 0.0), (12, 0.25)):
        for signal_value, epsilon in zip((0.0, 1.0, 2.0), (1e-6, 2e-6, 4e-6), strict=True):
            control = signal_value + (epsilon if industry_id == 1 else -epsilon)
            rows.append(
                {
                    "filing_date": filing_date,
                    "ff48_industry_id": industry_id,
                    "signal": signal_value,
                    "control": control,
                    "dependent": 1.0 + 1.75 * signal_value - 0.5 * control + industry_effect,
                }
            )

    results = run_lm2011_quarterly_fama_macbeth(
        pl.DataFrame(rows).lazy(),
        table_id="unit_test_table",
        text_scope="full_10k",
        dependent_variable="dependent",
        signal_column="signal",
        control_columns=("control",),
    ).sort("coefficient_name")

    by_name = {row["coefficient_name"]: row for row in results.to_dicts()}
    assert by_name["intercept"]["estimate"] == pytest.approx(1.0, abs=1e-6)
    assert by_name["signal"]["estimate"] == pytest.approx(1.75, abs=1e-6)
    assert by_name["control"]["estimate"] == pytest.approx(-0.5, abs=1e-6)
    assert by_name["signal"]["n_quarters"] == 1


def test_build_lm2011_normalized_difference_panel_uses_prior_year_industry_prop_stats() -> None:
    panel = pl.DataFrame(
        {
            "doc_id": ["a20", "b20", "c21", "d20", "e21", "f21"],
            "filing_date": [
                dt.date(2020, 2, 15),
                dt.date(2020, 5, 15),
                dt.date(2021, 2, 15),
                dt.date(2020, 3, 15),
                dt.date(2021, 3, 15),
                dt.date(2021, 4, 15),
            ],
            "ff48_industry_id": [1, 1, 1, 12, 12, None],
            "lm_negative_prop": [0.1, 0.2, 0.3, 0.15, 0.25, 0.4],
            "h4n_inf_prop": [0.2, 0.4, 0.5, 0.25, 0.35, 0.6],
            "lm_negative_tfidf": [10.0, 20.0, 30.0, 15.0, 25.0, 40.0],
            "h4n_inf_tfidf": [50.0, 60.0, 70.0, 80.0, 90.0, 100.0],
        }
    )

    normalized = build_lm2011_normalized_difference_panel(panel.lazy()).collect().sort("doc_id")

    assert normalized.get_column("doc_id").to_list() == ["c21"]
    row = normalized.row(0, named=True)
    assert math.isclose(row["normalized_difference_negative"], 2.1213203435596424, rel_tol=0.0, abs_tol=1e-12)
    assert math.isclose(row["normalized_difference_h4n_inf"], 1.414213562373095, rel_tol=0.0, abs_tol=1e-12)


def test_build_lm2011_return_regression_panel_adds_transforms_and_ff48_labels(tmp_path) -> None:
    inputs = _regression_test_inputs(tmp_path)

    panel = build_lm2011_return_regression_panel(
        inputs["event_panel"].lazy(),
        inputs["full_text_features"].lazy(),
        inputs["company_history"].lazy(),
        inputs["company_description"].lazy(),
        ff48_siccodes_path=inputs["ff48_path"],
        text_scope="full_10k",
    ).collect().sort("doc_id")

    assert {"log_size", "log_book_to_market", "log_share_turnover"}.issubset(set(panel.columns))
    assert {"ff48_industry_id", "ff48_industry_short", "lm_negative_prop", "h4n_inf_tfidf", "text_scope"}.issubset(
        set(panel.columns)
    )
    first_row = panel.row(0, named=True)
    assert first_row["text_scope"] == "full_10k"
    assert math.isclose(first_row["log_share_turnover"], math.log(first_row["share_turnover"]), rel_tol=0.0, abs_tol=1e-12)


def test_build_lm2011_sue_regression_panel_is_self_sufficient_for_table_viii(tmp_path) -> None:
    inputs = _regression_test_inputs(tmp_path)

    panel = build_lm2011_sue_regression_panel(
        inputs["sue_panel"].lazy(),
        inputs["full_text_features"].lazy(),
        inputs["company_history"].lazy(),
        inputs["company_description"].lazy(),
        ff48_siccodes_path=inputs["ff48_path"],
    ).collect().sort("doc_id")

    assert {"analyst_dispersion", "analyst_revisions", "log_size", "log_book_to_market", "log_share_turnover"}.issubset(
        set(panel.columns)
    )
    assert {"ff48_industry_id", "lm_negative_prop", "h4n_inf_prop", "text_scope"}.issubset(set(panel.columns))
    assert panel.get_column("text_scope").unique().to_list() == ["full_10k"]


@pytest.mark.parametrize(
    ("builder_name", "input_key"),
    (
        ("return", "event_panel"),
        ("sue", "sue_panel"),
    ),
)
def test_lm2011_regression_builders_nullify_infinite_float_outputs(tmp_path, builder_name: str, input_key: str) -> None:
    inputs = _regression_test_inputs(tmp_path)
    doc_id = "doc_1"
    if builder_name == "return":
        inputs[input_key] = inputs[input_key].with_columns(
            pl.when(pl.col("doc_id") == doc_id).then(pl.lit(float("inf"))).otherwise(pl.col("size_event")).alias(
                "size_event"
            )
        )
        panel = build_lm2011_return_regression_panel(
            inputs["event_panel"].lazy(),
            inputs["full_text_features"].lazy(),
            inputs["company_history"].lazy(),
            inputs["company_description"].lazy(),
            ff48_siccodes_path=inputs["ff48_path"],
            text_scope="full_10k",
        ).collect()
    else:
        inputs[input_key] = inputs[input_key].with_columns(
            pl.when(pl.col("doc_id") == doc_id)
            .then(pl.lit(float("-inf")))
            .otherwise(pl.col("analyst_dispersion"))
            .alias("analyst_dispersion")
        )
        panel = build_lm2011_sue_regression_panel(
            inputs["sue_panel"].lazy(),
            inputs["full_text_features"].lazy(),
            inputs["company_history"].lazy(),
            inputs["company_description"].lazy(),
            ff48_siccodes_path=inputs["ff48_path"],
        ).collect()

    row = panel.filter(pl.col("doc_id") == doc_id).row(0, named=True)
    if builder_name == "return":
        assert row["size_event"] is None
        assert row["log_size"] is None
    else:
        assert row["analyst_dispersion"] is None


def test_table_wrappers_follow_revised_spec_signal_scopes(tmp_path) -> None:
    inputs = _regression_test_inputs(tmp_path)

    table_iv = build_lm2011_table_iv_results(
        inputs["event_panel"].lazy(),
        inputs["full_text_features"].lazy(),
        inputs["company_history"].lazy(),
        inputs["company_description"].lazy(),
        ff48_siccodes_path=inputs["ff48_path"],
    )
    table_v = build_lm2011_table_v_results(
        inputs["event_panel"].lazy(),
        inputs["mda_text_features"].lazy(),
        inputs["company_history"].lazy(),
        inputs["company_description"].lazy(),
        ff48_siccodes_path=inputs["ff48_path"],
    )
    table_vi = build_lm2011_table_vi_results(
        inputs["event_panel"].lazy(),
        inputs["full_text_features"].lazy(),
        inputs["company_history"].lazy(),
        inputs["company_description"].lazy(),
        ff48_siccodes_path=inputs["ff48_path"],
    )
    table_viii = build_lm2011_table_viii_results(
        inputs["sue_panel"].lazy(),
        inputs["full_text_features"].lazy(),
        inputs["company_history"].lazy(),
        inputs["company_description"].lazy(),
        ff48_siccodes_path=inputs["ff48_path"],
    )

    assert set(table_iv.get_column("signal_name").unique().to_list()) == {
        "h4n_inf_prop",
        "lm_negative_prop",
        "h4n_inf_tfidf",
        "lm_negative_tfidf",
    }
    assert table_iv.get_column("text_scope").unique().to_list() == ["full_10k"]
    assert table_iv.get_column("dependent_variable").unique().to_list() == ["filing_period_excess_return"]

    assert set(table_v.get_column("signal_name").unique().to_list()) == {
        "h4n_inf_prop",
        "lm_negative_prop",
        "h4n_inf_tfidf",
        "lm_negative_tfidf",
    }
    assert table_v.get_column("text_scope").unique().to_list() == ["mda_item_7"]

    assert set(table_vi.get_column("signal_name").unique().to_list()) == {
        "lm_negative_prop",
        "lm_negative_tfidf",
        "lm_positive_prop",
        "lm_positive_tfidf",
        "lm_uncertainty_prop",
        "lm_uncertainty_tfidf",
        "lm_litigious_prop",
        "lm_litigious_tfidf",
        "lm_modal_strong_prop",
        "lm_modal_strong_tfidf",
        "lm_modal_weak_prop",
        "lm_modal_weak_tfidf",
    }
    assert table_vi.filter(pl.col("signal_name").str.contains("h4n")).height == 0
    assert table_viii.get_column("dependent_variable").unique().to_list() == ["sue"]
    assert set(table_viii.get_column("signal_name").unique().to_list()) == {
        "h4n_inf_prop",
        "lm_negative_prop",
        "h4n_inf_tfidf",
        "lm_negative_tfidf",
    }


def test_build_lm2011_table_v_results_enforces_mda_token_count_floor(tmp_path) -> None:
    inputs = _regression_test_inputs(tmp_path)

    low_token_table_v = build_lm2011_table_v_results(
        inputs["event_panel"].lazy(),
        inputs["mda_text_features"].with_columns(pl.lit(249).alias("total_token_count_mda")).lazy(),
        inputs["company_history"].lazy(),
        inputs["company_description"].lazy(),
        ff48_siccodes_path=inputs["ff48_path"],
    )
    boundary_table_v = build_lm2011_table_v_results(
        inputs["event_panel"].lazy(),
        inputs["mda_text_features"].with_columns(pl.lit(250).alias("total_token_count_mda")).lazy(),
        inputs["company_history"].lazy(),
        inputs["company_description"].lazy(),
        ff48_siccodes_path=inputs["ff48_path"],
    )

    assert low_token_table_v.height == 0
    assert boundary_table_v.height > 0


def test_build_lm2011_table_v_results_enforces_builder_linked_total_token_floor(tmp_path) -> None:
    from thesis_pkg.pipeline import build_lm2011_text_features_mda

    inputs = _regression_test_inputs(tmp_path)
    event_panel = inputs["event_panel"]

    def _build_mda_features(target_token_count: int) -> pl.DataFrame:
        rows: list[dict[str, object]] = []
        for idx, row in enumerate(event_panel.iter_rows(named=True), start=1):
            negative_count = 1 + (idx % 5)
            harvard_count = 2 + (idx % 3)
            filler_count = target_token_count - negative_count - harvard_count
            assert filler_count > 0
            rows.append(
                {
                    "doc_id": row["doc_id"],
                    "cik_10": f"{idx:010d}",
                    "filing_date": row["filing_date"],
                    "document_type_filename": "10-K",
                    "item_id": "7",
                    "full_text": " ".join(
                        [
                            *(["recognized"] * filler_count),
                            *(["loss"] * negative_count),
                            *(["bad"] * harvard_count),
                        ]
                    ),
                }
            )
        built_counts = build_lm2011_text_features_mda(
            pl.DataFrame(rows).lazy(),
            dictionary_lists=_lm_dictionary_lists(),
            harvard_negative_word_list=_harvard_negative_word_list(),
            master_dictionary_words=["recognized", "loss", "bad"],
        ).collect().select("doc_id", "token_count_mda", "total_token_count_mda")
        return (
            inputs["mda_text_features"]
            .drop("token_count_mda", "total_token_count_mda")
            .join(built_counts, on="doc_id", how="inner")
        )

    low_token_table_v = build_lm2011_table_v_results(
        inputs["event_panel"].lazy(),
        _build_mda_features(249).lazy(),
        inputs["company_history"].lazy(),
        inputs["company_description"].lazy(),
        ff48_siccodes_path=inputs["ff48_path"],
    )
    boundary_table_v = build_lm2011_table_v_results(
        inputs["event_panel"].lazy(),
        _build_mda_features(250).lazy(),
        inputs["company_history"].lazy(),
        inputs["company_description"].lazy(),
        ff48_siccodes_path=inputs["ff48_path"],
    )

    assert low_token_table_v.height == 0
    assert boundary_table_v.height > 0


def test_build_lm2011_table_v_results_requires_total_token_screen_count_column(tmp_path) -> None:
    inputs = _regression_test_inputs(tmp_path)

    with pytest.raises(ValueError, match="missing required columns"):
        build_lm2011_table_v_results(
            inputs["event_panel"].lazy(),
            inputs["mda_text_features"].drop("total_token_count_mda").lazy(),
            inputs["company_history"].lazy(),
            inputs["company_description"].lazy(),
            ff48_siccodes_path=inputs["ff48_path"],
        )


def test_build_lm2011_table_ia_i_results_uses_normalized_difference_signals(tmp_path) -> None:
    inputs = _regression_test_inputs(tmp_path)

    table_ia_i = build_lm2011_table_ia_i_results(
        inputs["event_panel"].lazy(),
        inputs["full_text_features"].lazy(),
        inputs["company_history"].lazy(),
        inputs["company_description"].lazy(),
        ff48_siccodes_path=inputs["ff48_path"],
    )

    assert table_ia_i.height > 0
    assert set(table_ia_i.get_column("signal_name").unique().to_list()) == {
        "normalized_difference_negative",
        "normalized_difference_h4n_inf",
    }
    assert table_ia_i.get_column("table_id").unique().to_list() == ["internet_appendix_table_ia_i"]


def test_build_lm2011_table_ia_ii_results_matches_strategy_artifacts() -> None:
    event_panel, sec_parsed, monthly_stock, monthly_factors = _build_strategy_inputs()
    monthly_returns = build_lm2011_trading_strategy_monthly_returns(
        event_panel.lazy(),
        sec_parsed.lazy(),
        monthly_stock.lazy(),
        lm_dictionary_lists=_lm_dictionary_lists(),
        harvard_negative_word_list=_harvard_negative_word_list(),
        master_dictionary_words=_master_dictionary_words(),
    ).collect().sort("portfolio_month", "sort_signal_name")
    summary = build_lm2011_trading_strategy_ff4_summary(
        monthly_returns.lazy(),
        monthly_factors.lazy(),
    ).collect().sort("sort_signal_name")

    table_ia_ii = build_lm2011_table_ia_ii_results(
        event_panel.lazy(),
        sec_parsed.lazy(),
        monthly_stock.lazy(),
        monthly_factors.lazy(),
        lm_dictionary_lists=_lm_dictionary_lists(),
        harvard_negative_word_list=_harvard_negative_word_list(),
        master_dictionary_words=_master_dictionary_words(),
    ).sort("signal_name", "coefficient_name")

    assert set(table_ia_ii.get_column("coefficient_name").unique().to_list()) == {
        "mean_long_short_return",
        "alpha_ff3_mom",
        "beta_market",
        "beta_smb",
        "beta_hml",
        "beta_mom",
        "r2",
    }
    assert set(table_ia_ii.get_column("signal_name").unique().to_list()) == {
        "fin_neg_prop",
        "fin_neg_tfidf",
        "h4n_inf_prop",
        "h4n_inf_tfidf",
    }

    mean_returns = monthly_returns.group_by("sort_signal_name").agg(
        pl.col("long_short_return").mean().alias("mean_long_short_return")
    )
    combined = mean_returns.join(summary, on="sort_signal_name", how="inner").sort("sort_signal_name")
    for row in combined.iter_rows(named=True):
        signal_rows = table_ia_ii.filter(pl.col("signal_name") == row["sort_signal_name"])
        assert math.isclose(
            signal_rows.filter(pl.col("coefficient_name") == "mean_long_short_return").select("estimate").item(),
            row["mean_long_short_return"],
            rel_tol=0.0,
            abs_tol=1e-12,
        )
        for coefficient_name in ("alpha_ff3_mom", "beta_market", "beta_smb", "beta_hml", "beta_mom", "r2"):
            assert math.isclose(
                signal_rows.filter(pl.col("coefficient_name") == coefficient_name).select("estimate").item(),
                row[coefficient_name],
                rel_tol=0.0,
                abs_tol=1e-12,
            )
