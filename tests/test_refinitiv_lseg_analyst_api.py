from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Any, Callable

import polars as pl

from thesis_pkg.pipelines.refinitiv.lseg_analyst_api import (
    run_refinitiv_step1_analyst_actuals_api_pipeline,
    run_refinitiv_step1_analyst_estimates_monthly_api_pipeline,
)
from thesis_pkg.pipelines.refinitiv.lseg_provider import LsegDataResponse, LsegResponseMetadata


class FakeProvider:
    def __init__(
        self,
        responses: list[pl.DataFrame] | None = None,
        *,
        response_factory: Callable[[list[str], list[str], dict[str, Any]], pl.DataFrame] | None = None,
    ) -> None:
        self._responses = list(responses or [])
        self._response_factory = response_factory
        self.calls: list[dict[str, Any]] = []

    def open(self) -> None:
        return None

    def close(self) -> None:
        return None

    def get_data(
        self,
        *,
        universe: list[str],
        fields: list[str],
        parameters: dict[str, Any] | None = None,
    ) -> LsegDataResponse:
        self.calls.append(
            {
                "universe": list(universe),
                "fields": list(fields),
                "parameters": dict(parameters or {}),
            }
        )
        if self._response_factory is not None:
            frame = self._response_factory(list(universe), list(fields), dict(parameters or {}))
        else:
            frame = self._responses.pop(0) if self._responses else pl.DataFrame()
        return LsegDataResponse(
            frame=frame,
            metadata=LsegResponseMetadata(
                status_code=200,
                headers={"X-Request-Limit-Remaining": "100"},
                latency_ms=25,
                response_bytes=256,
                fingerprint="fake",
            ),
        )


def _write_request_universe(path: Path) -> None:
    pl.DataFrame(
        {
            "request_group_id": ["group-1", "group-2"],
            "gvkey_int": [1000, 1001],
            "effective_collection_ric": ["AAA.N", "BBB.N"],
            "member_bridge_row_count": [2, 1],
            "bridge_start_date_min": [date(2020, 1, 1), date(2021, 1, 1)],
            "bridge_end_date_max": [date(2020, 12, 31), date(2021, 12, 31)],
            "actuals_request_start_date": [date(2019, 12, 1), date(2020, 12, 1)],
            "actuals_request_end_date": [date(2021, 4, 30), date(2022, 4, 30)],
            "estimates_request_start_date": [date(2019, 4, 6), date(2020, 4, 6)],
            "estimates_request_end_date": [date(2021, 1, 31), date(2022, 1, 31)],
            "retrieval_eligible": [True, True],
            "retrieval_exclusion_reason": [None, None],
        }
    ).write_parquet(path)


def test_run_refinitiv_step1_analyst_actuals_api_pipeline_uses_request_groups(tmp_path: Path) -> None:
    request_path = tmp_path / "requests.parquet"
    _write_request_universe(request_path)
    provider = FakeProvider(
        response_factory=lambda universe, _fields, _parameters: pl.DataFrame(
            {
                "Instrument": [universe[0]],
                "TR.EPSActValue": [1.5 if universe[0] == "AAA.N" else 2.0],
                "TR.EPSActValue.date": [date(2023, 10, 15) if universe[0] == "AAA.N" else date(2023, 10, 20)],
                "TR.EPSActValue.periodenddate": [date(2023, 9, 30)],
                "TR.EPSActValue.fperiod": ["FY2023Q3"],
            }
        )
    )

    out = run_refinitiv_step1_analyst_actuals_api_pipeline(
        request_universe_parquet_path=request_path,
        output_dir=tmp_path,
        provider=provider,
        min_seconds_between_requests=0.0,
    )

    raw_df = pl.read_parquet(out["refinitiv_analyst_actuals_raw_parquet"])
    assert raw_df.height == 2
    assert len(provider.calls) == 2
    assert all(call["parameters"]["Frq"] == "FQ" for call in provider.calls)
    assert all(call["parameters"]["Period"] == "FI0" for call in provider.calls)
    assert sorted(call["universe"][0] for call in provider.calls) == ["AAA.N", "BBB.N"]
    assert raw_df.columns == [
        "item_id",
        "response_row_index",
        "request_group_id",
        "gvkey_int",
        "effective_collection_ric",
        "announcement_date",
        "fiscal_period_end",
        "actual_eps",
        "raw_fperiod",
        "row_parse_status",
    ]
    assert sorted(raw_df.get_column("request_group_id").to_list()) == ["group-1", "group-2"]
    assert raw_df.get_column("response_row_index").to_list() == [0, 0]
    assert raw_df.get_column("fiscal_period_end").to_list() == [date(2023, 9, 30), date(2023, 9, 30)]
    assert raw_df.get_column("raw_fperiod").to_list() == ["FY2023Q3", "FY2023Q3"]
    assert raw_df.get_column("row_parse_status").to_list() == ["OK", "OK"]


def test_run_refinitiv_step1_analyst_estimates_monthly_api_pipeline_uses_request_groups(tmp_path: Path) -> None:
    request_path = tmp_path / "requests.parquet"
    _write_request_universe(request_path)
    provider = FakeProvider(
        response_factory=lambda universe, _fields, parameters: pl.DataFrame(
            {
                "Instrument": [universe[0]],
                "TR.EPSMean": [
                    1.2
                    if universe[0] == "AAA.N" and parameters["Period"] == "FQ1"
                    else 1.1
                    if universe[0] == "AAA.N"
                    else 1.8
                    if parameters["Period"] == "FQ1"
                    else 1.6
                ],
                "TR.EPSMean.calcdate": [
                    date(2023, 10, 1) if parameters["Period"] == "FQ1" else date(2023, 6, 1)
                ],
                "TR.EPSMean.periodenddate": [date(2023, 9, 30)],
                "TR.EPSMean.fperiod": ["FY2023Q3"],
                "TR.EPSStdDev": [
                    0.2
                    if universe[0] == "AAA.N" and parameters["Period"] == "FQ1"
                    else 0.18
                    if universe[0] == "AAA.N"
                    else 0.3
                    if parameters["Period"] == "FQ1"
                    else 0.28
                ],
                "TR.EPSNumberofEstimates": [
                    10
                    if universe[0] == "AAA.N" and parameters["Period"] == "FQ1"
                    else 9
                    if universe[0] == "AAA.N"
                    else 12
                    if parameters["Period"] == "FQ1"
                    else 11
                ],
            }
        )
    )

    out = run_refinitiv_step1_analyst_estimates_monthly_api_pipeline(
        request_universe_parquet_path=request_path,
        output_dir=tmp_path,
        provider=provider,
        min_seconds_between_requests=0.0,
    )

    raw_df = pl.read_parquet(out["refinitiv_analyst_estimates_monthly_raw_parquet"])
    assert raw_df.height == 4
    assert len(provider.calls) == 4
    assert all(call["parameters"]["Frq"] == "M" for call in provider.calls)
    assert sorted(call["parameters"]["Period"] for call in provider.calls) == ["FQ1", "FQ1", "FQ2", "FQ2"]
    assert sorted(call["universe"][0] for call in provider.calls) == ["AAA.N", "AAA.N", "BBB.N", "BBB.N"]
    assert raw_df.columns == [
        "item_id",
        "response_row_index",
        "request_group_id",
        "request_period",
        "gvkey_int",
        "effective_collection_ric",
        "calc_date",
        "fiscal_period_end",
        "raw_fperiod",
        "forecast_consensus_mean",
        "forecast_dispersion",
        "estimate_count",
        "row_parse_status",
    ]
    assert sorted(raw_df.get_column("request_period").unique().to_list()) == ["FQ1", "FQ2"]
    assert sorted(raw_df.get_column("estimate_count").to_list()) == [9, 10, 11, 12]
    assert raw_df.get_column("response_row_index").to_list() == [0, 0, 0, 0]
