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


def _write_request_universe(path: Path, rows: list[dict[str, Any]]) -> None:
    pl.DataFrame(rows).write_parquet(path)


def test_run_refinitiv_step1_analyst_actuals_api_pipeline_batches_close_request_groups(tmp_path: Path) -> None:
    request_path = tmp_path / "requests.parquet"
    _write_request_universe(
        request_path,
        [
            {
                "request_group_id": "group-1",
                "gvkey_int": 1000,
                "effective_collection_ric": "AAA.N",
                "member_bridge_row_count": 2,
                "bridge_start_date_min": date(2020, 1, 1),
                "bridge_end_date_max": date(2020, 12, 31),
                "actuals_request_start_date": date(2020, 1, 1),
                "actuals_request_end_date": date(2020, 3, 31),
                "estimates_request_start_date": date(2020, 1, 1),
                "estimates_request_end_date": date(2020, 3, 31),
                "retrieval_eligible": True,
                "retrieval_exclusion_reason": None,
            },
            {
                "request_group_id": "group-2",
                "gvkey_int": 1001,
                "effective_collection_ric": "BBB.N",
                "member_bridge_row_count": 1,
                "bridge_start_date_min": date(2020, 1, 15),
                "bridge_end_date_max": date(2020, 12, 31),
                "actuals_request_start_date": date(2020, 1, 15),
                "actuals_request_end_date": date(2020, 3, 31),
                "estimates_request_start_date": date(2020, 1, 15),
                "estimates_request_end_date": date(2020, 3, 31),
                "retrieval_eligible": True,
                "retrieval_exclusion_reason": None,
            },
        ],
    )
    provider = FakeProvider(
        response_factory=lambda universe, _fields, _parameters: pl.DataFrame(
            {
                "Instrument": list(universe),
                "TR.EPSActValue": [1.5 if instrument == "AAA.N" else 2.0 for instrument in universe],
                "TR.EPSActValue.date": [
                    date(2020, 3, 15) if instrument == "AAA.N" else date(2020, 3, 20)
                    for instrument in universe
                ],
                "TR.EPSActValue.periodenddate": [date(2019, 12, 31) for _ in universe],
                "TR.EPSActValue.fperiod": ["FY2019Q4" for _ in universe],
            }
        )
    )

    out = run_refinitiv_step1_analyst_actuals_api_pipeline(
        request_universe_parquet_path=request_path,
        output_dir=tmp_path,
        provider=provider,
        min_seconds_between_requests=0.0,
    )

    raw_df = pl.read_parquet(out["refinitiv_analyst_actuals_raw_parquet"]).sort("request_group_id")
    assert raw_df.height == 2
    assert len(provider.calls) == 1
    assert provider.calls[0]["parameters"] == {
        "Frq": "FQ",
        "Period": "FI0",
        "SDate": "2020-01-01",
        "EDate": "2020-03-31",
    }
    assert sorted(provider.calls[0]["universe"]) == ["AAA.N", "BBB.N"]
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
    assert raw_df.get_column("request_group_id").to_list() == ["group-1", "group-2"]
    assert raw_df.get_column("response_row_index").to_list() == [0, 0]
    assert raw_df.get_column("row_parse_status").to_list() == ["OK", "OK"]


def test_run_refinitiv_step1_analyst_estimates_monthly_api_pipeline_batches_by_period(tmp_path: Path) -> None:
    request_path = tmp_path / "requests.parquet"
    _write_request_universe(
        request_path,
        [
            {
                "request_group_id": "group-1",
                "gvkey_int": 1000,
                "effective_collection_ric": "AAA.N",
                "member_bridge_row_count": 2,
                "bridge_start_date_min": date(2020, 1, 1),
                "bridge_end_date_max": date(2020, 12, 31),
                "actuals_request_start_date": date(2020, 1, 1),
                "actuals_request_end_date": date(2020, 3, 31),
                "estimates_request_start_date": date(2020, 1, 1),
                "estimates_request_end_date": date(2020, 3, 31),
                "retrieval_eligible": True,
                "retrieval_exclusion_reason": None,
            },
            {
                "request_group_id": "group-2",
                "gvkey_int": 1001,
                "effective_collection_ric": "BBB.N",
                "member_bridge_row_count": 1,
                "bridge_start_date_min": date(2020, 1, 15),
                "bridge_end_date_max": date(2020, 12, 31),
                "actuals_request_start_date": date(2020, 1, 15),
                "actuals_request_end_date": date(2020, 3, 31),
                "estimates_request_start_date": date(2020, 1, 15),
                "estimates_request_end_date": date(2020, 3, 31),
                "retrieval_eligible": True,
                "retrieval_exclusion_reason": None,
            },
        ],
    )
    provider = FakeProvider(
        response_factory=lambda universe, _fields, parameters: pl.DataFrame(
            {
                "Instrument": list(universe),
                "TR.EPSMean": [
                    1.2
                    if instrument == "AAA.N" and parameters["Period"] == "FQ1"
                    else 1.1
                    if instrument == "AAA.N"
                    else 1.8
                    if parameters["Period"] == "FQ1"
                    else 1.6
                    for instrument in universe
                ],
                "TR.EPSMean.calcdate": [
                    date(2020, 3, 1) if parameters["Period"] == "FQ1" else date(2020, 2, 1)
                    for _ in universe
                ],
                "TR.EPSMean.periodenddate": [date(2020, 3, 31) for _ in universe],
                "TR.EPSMean.fperiod": ["FY2020Q1" for _ in universe],
                "TR.EPSStdDev": [
                    0.2
                    if instrument == "AAA.N" and parameters["Period"] == "FQ1"
                    else 0.18
                    if instrument == "AAA.N"
                    else 0.3
                    if parameters["Period"] == "FQ1"
                    else 0.28
                    for instrument in universe
                ],
                "TR.EPSNumberofEstimates": [
                    10
                    if instrument == "AAA.N" and parameters["Period"] == "FQ1"
                    else 9
                    if instrument == "AAA.N"
                    else 12
                    if parameters["Period"] == "FQ1"
                    else 11
                    for instrument in universe
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

    raw_df = pl.read_parquet(out["refinitiv_analyst_estimates_monthly_raw_parquet"]).sort(
        ["request_period", "request_group_id"]
    )
    assert raw_df.height == 4
    assert len(provider.calls) == 2
    assert sorted(call["parameters"]["Period"] for call in provider.calls) == ["FQ1", "FQ2"]
    assert all(call["parameters"]["Frq"] == "M" for call in provider.calls)
    assert all(sorted(call["universe"]) == ["AAA.N", "BBB.N"] for call in provider.calls)
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
    assert raw_df.get_column("request_period").to_list() == ["FQ1", "FQ1", "FQ2", "FQ2"]
    assert raw_df.get_column("response_row_index").to_list() == [0, 0, 0, 0]
    assert sorted(raw_df.get_column("estimate_count").to_list()) == [9, 10, 11, 12]


def test_analyst_actuals_pipeline_filters_widened_batch_rows_per_item_window(tmp_path: Path) -> None:
    request_path = tmp_path / "requests.parquet"
    _write_request_universe(
        request_path,
        [
            {
                "request_group_id": "group-jan",
                "gvkey_int": 1000,
                "effective_collection_ric": "AAA.N",
                "member_bridge_row_count": 1,
                "bridge_start_date_min": date(2020, 1, 1),
                "bridge_end_date_max": date(2020, 1, 31),
                "actuals_request_start_date": date(2020, 1, 1),
                "actuals_request_end_date": date(2020, 1, 31),
                "estimates_request_start_date": date(2020, 1, 1),
                "estimates_request_end_date": date(2020, 1, 31),
                "retrieval_eligible": True,
                "retrieval_exclusion_reason": None,
            },
            {
                "request_group_id": "group-feb",
                "gvkey_int": 1000,
                "effective_collection_ric": "AAA.N",
                "member_bridge_row_count": 1,
                "bridge_start_date_min": date(2020, 2, 1),
                "bridge_end_date_max": date(2020, 2, 29),
                "actuals_request_start_date": date(2020, 2, 1),
                "actuals_request_end_date": date(2020, 2, 29),
                "estimates_request_start_date": date(2020, 2, 1),
                "estimates_request_end_date": date(2020, 2, 29),
                "retrieval_eligible": True,
                "retrieval_exclusion_reason": None,
            },
        ],
    )
    provider = FakeProvider(
        response_factory=lambda universe, _fields, _parameters: pl.DataFrame(
            {
                "Instrument": ["AAA.N", "AAA.N", "AAA.N", "AAA.N"],
                "TR.EPSActValue": [1.0, 1.1, 1.2, 1.3],
                "TR.EPSActValue.date": [
                    date(2020, 1, 15),
                    date(2020, 1, 31),
                    date(2020, 2, 10),
                    date(2020, 3, 1),
                ],
                "TR.EPSActValue.periodenddate": [date(2019, 12, 31)] * 4,
                "TR.EPSActValue.fperiod": ["FY2019Q4"] * 4,
            }
        )
    )

    out = run_refinitiv_step1_analyst_actuals_api_pipeline(
        request_universe_parquet_path=request_path,
        output_dir=tmp_path,
        provider=provider,
        min_seconds_between_requests=0.0,
    )

    raw_df = pl.read_parquet(out["refinitiv_analyst_actuals_raw_parquet"]).sort(
        ["request_group_id", "response_row_index"]
    )
    assert len(provider.calls) == 1
    assert raw_df.height == 3
    assert raw_df.filter(pl.col("request_group_id") == "group-jan").get_column("announcement_date").to_list() == [
        date(2020, 1, 15),
        date(2020, 1, 31),
    ]
    assert raw_df.filter(pl.col("request_group_id") == "group-jan").get_column("response_row_index").to_list() == [
        0,
        1,
    ]
    assert raw_df.filter(pl.col("request_group_id") == "group-feb").get_column("announcement_date").to_list() == [
        date(2020, 2, 10)
    ]
    assert raw_df.filter(pl.col("request_group_id") == "group-feb").get_column("response_row_index").to_list() == [
        0
    ]
