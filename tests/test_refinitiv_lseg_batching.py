from __future__ import annotations

from datetime import date
import random
from typing import Any

from thesis_pkg.pipelines.refinitiv.lseg_batching import (
    IntervalBatchPlannerConfig,
    RequestItem,
    build_batch_definition,
    plan_interval_batches,
    request_signature,
)


def _make_item(
    *,
    item_id: str,
    instrument: str,
    start_date: date,
    end_date: date,
    stage: str = "analyst_actuals",
    period: str = "FI0",
    frq: str = "FQ",
) -> RequestItem:
    start_text = start_date.isoformat()
    end_text = end_date.isoformat()
    batch_key = f"{start_text}|{end_text}" if stage == "analyst_actuals" else f"{start_text}|{end_text}|{period}"
    return RequestItem(
        item_id=item_id,
        stage=stage,
        instrument=instrument,
        batch_key=batch_key,
        fields=("TR.Field",),
        parameters={"Frq": frq, "Period": period, "SDate": start_text, "EDate": end_text},
        payload={},
    )


def _make_ownership_item(
    *,
    item_id: str,
    instrument: str,
    start_date: date,
    end_date: date,
) -> RequestItem:
    start_text = start_date.isoformat()
    end_text = end_date.isoformat()
    return RequestItem(
        item_id=item_id,
        stage="ownership_universe",
        instrument=instrument,
        batch_key=f"{start_text}|{end_text}",
        fields=(
            "TR.CategoryOwnershipPct.Date",
            "TR.CategoryOwnershipPct",
            "TR.InstrStatTypeValue",
        ),
        parameters={"StatType": 7, "SDate": start_text, "EDate": end_text},
        payload={},
    )


def _signature(item: RequestItem) -> str:
    return request_signature(
        stage=item.stage,
        fields=item.fields,
        parameters=item.parameters,
        excluded_parameter_keys=("SDate", "EDate"),
    )


def _interval(item: RequestItem) -> tuple[date, date]:
    return (
        date.fromisoformat(str(item.parameters["SDate"])),
        date.fromisoformat(str(item.parameters["EDate"])),
    )


def _batch_builder(items: list[RequestItem], start_date: date, end_date: date) -> Any:
    first = items[0]
    start_text = start_date.isoformat()
    end_text = end_date.isoformat()
    if first.stage == "ownership_universe":
        return build_batch_definition(
            items,
            batch_key=f"{start_text}|{end_text}",
            parameters={
                "StatType": first.parameters["StatType"],
                "SDate": start_text,
                "EDate": end_text,
            },
        )
    request_period = str(first.parameters["Period"])
    batch_key = (
        f"{start_text}|{end_text}"
        if first.stage == "analyst_actuals"
        else f"{start_text}|{end_text}|{request_period}"
    )
    return build_batch_definition(
        items,
        batch_key=batch_key,
        parameters={
            "Frq": first.parameters["Frq"],
            "Period": request_period,
            "SDate": start_text,
            "EDate": end_text,
        },
    )


def _plan(items: list[RequestItem], *, config: IntervalBatchPlannerConfig) -> tuple[tuple[str, ...], ...]:
    plan = plan_interval_batches(
        items,
        config=config,
        planner_version="test_planner_v1",
        signature_fn=_signature,
        interval_fn=_interval,
        batch_builder=_batch_builder,
    )
    return tuple(batch.item_ids for batch in plan.batches)


def test_request_signature_can_exclude_date_parameters() -> None:
    actuals_a = _make_item(
        item_id="item-a",
        instrument="AAA.N",
        start_date=date(2020, 1, 1),
        end_date=date(2020, 3, 31),
    )
    actuals_b = _make_item(
        item_id="item-b",
        instrument="BBB.N",
        start_date=date(2020, 1, 15),
        end_date=date(2020, 3, 31),
    )
    estimates = _make_item(
        item_id="item-c",
        instrument="AAA.N",
        start_date=date(2020, 1, 1),
        end_date=date(2020, 3, 31),
        stage="analyst_estimates_monthly",
        period="FQ1",
        frq="M",
    )

    assert request_signature(
        stage=actuals_a.stage,
        fields=actuals_a.fields,
        parameters=actuals_a.parameters,
    ) != request_signature(
        stage=actuals_b.stage,
        fields=actuals_b.fields,
        parameters=actuals_b.parameters,
    )
    assert _signature(actuals_a) == _signature(actuals_b)
    assert _signature(actuals_a) != _signature(estimates)


def test_ownership_request_signature_excludes_dates_but_keeps_stat_type() -> None:
    ownership_a = _make_ownership_item(
        item_id="item-own-a",
        instrument="AAA.N",
        start_date=date(2020, 1, 1),
        end_date=date(2020, 3, 31),
    )
    ownership_b = _make_ownership_item(
        item_id="item-own-b",
        instrument="BBB.N",
        start_date=date(2020, 1, 15),
        end_date=date(2020, 3, 31),
    )
    ownership_other_stat = RequestItem(
        item_id="item-own-c",
        stage="ownership_universe",
        instrument="CCC.N",
        batch_key="2020-01-01|2020-03-31",
        fields=ownership_a.fields,
        parameters={"StatType": 8, "SDate": "2020-01-01", "EDate": "2020-03-31"},
        payload={},
    )

    assert _signature(ownership_a) == _signature(ownership_b)
    assert _signature(ownership_a) != _signature(ownership_other_stat)


def test_plan_interval_batches_keeps_mixed_request_periods_separate() -> None:
    items = [
        _make_item(
            item_id="item-fq1",
            instrument="AAA.N",
            start_date=date(2020, 1, 1),
            end_date=date(2020, 3, 31),
            stage="analyst_estimates_monthly",
            period="FQ1",
            frq="M",
        ),
        _make_item(
            item_id="item-fq2",
            instrument="AAA.N",
            start_date=date(2020, 1, 15),
            end_date=date(2020, 3, 31),
            stage="analyst_estimates_monthly",
            period="FQ2",
            frq="M",
        ),
    ]

    planned = _plan(
        items,
        config=IntervalBatchPlannerConfig(
            max_batch_size=10,
            max_batch_items=10,
            max_extra_rows_abs=100.0,
            max_extra_rows_ratio=1.0,
            row_density_rows_per_day=1.0,
        ),
    )

    assert set(planned) == {("item-fq1",), ("item-fq2",)}


def test_plan_interval_batches_prefers_smallest_incremental_extra_rows() -> None:
    items = [
        _make_item(
            item_id="item-1",
            instrument="AAA.N",
            start_date=date(2020, 1, 1),
            end_date=date(2020, 3, 31),
        ),
        _make_item(
            item_id="item-2",
            instrument="BBB.N",
            start_date=date(2020, 1, 15),
            end_date=date(2020, 3, 31),
        ),
        _make_item(
            item_id="item-3",
            instrument="CCC.N",
            start_date=date(2020, 3, 15),
            end_date=date(2020, 4, 30),
        ),
    ]

    planned = _plan(
        items,
        config=IntervalBatchPlannerConfig(
            max_batch_size=2,
            max_batch_items=2,
            max_extra_rows_abs=20.0,
            max_extra_rows_ratio=0.2,
            row_density_rows_per_day=1.0 / 91.0,
        ),
    )

    assert planned == (("item-1", "item-2"), ("item-3",))


def test_plan_interval_batches_uses_unique_instrument_cost_not_raw_item_count() -> None:
    items = [
        _make_item(
            item_id="item-1",
            instrument="AAA.N",
            start_date=date(2020, 1, 1),
            end_date=date(2020, 1, 31),
        ),
        _make_item(
            item_id="item-2",
            instrument="AAA.N",
            start_date=date(2020, 2, 1),
            end_date=date(2020, 2, 29),
        ),
    ]

    plan = plan_interval_batches(
        items,
        config=IntervalBatchPlannerConfig(
            max_batch_size=1,
            max_batch_items=2,
            max_extra_rows_abs=0.0,
            max_extra_rows_ratio=0.0,
            row_density_rows_per_day=1.0,
        ),
        planner_version="test_planner_v1",
        signature_fn=_signature,
        interval_fn=_interval,
        batch_builder=_batch_builder,
    )

    assert tuple(batch.item_ids for batch in plan.batches) == (("item-1", "item-2"),)
    metrics = plan.batch_metrics_by_id[plan.batches[0].batch_id]
    assert metrics["unique_instrument_count"] == 1
    assert metrics["estimated_extra_rows_abs"] == 0.0


def test_plan_interval_batches_enforces_absolute_relative_and_hard_caps() -> None:
    far_apart = [
        _make_item(
            item_id="item-1",
            instrument="AAA.N",
            start_date=date(2020, 1, 1),
            end_date=date(2020, 3, 31),
        ),
        _make_item(
            item_id="item-2",
            instrument="BBB.N",
            start_date=date(2021, 1, 1),
            end_date=date(2021, 3, 31),
        ),
    ]
    same_signature = [
        _make_item(
            item_id="item-3",
            instrument="AAA.N",
            start_date=date(2020, 1, 1),
            end_date=date(2020, 1, 31),
        ),
        _make_item(
            item_id="item-4",
            instrument="BBB.N",
            start_date=date(2020, 1, 1),
            end_date=date(2020, 1, 31),
        ),
    ]
    same_instrument = [
        _make_item(
            item_id="item-5",
            instrument="AAA.N",
            start_date=date(2020, 1, 1),
            end_date=date(2020, 1, 31),
        ),
        _make_item(
            item_id="item-6",
            instrument="AAA.N",
            start_date=date(2020, 2, 1),
            end_date=date(2020, 2, 29),
        ),
        _make_item(
            item_id="item-7",
            instrument="AAA.N",
            start_date=date(2020, 3, 1),
            end_date=date(2020, 3, 31),
        ),
    ]

    assert _plan(
        far_apart,
        config=IntervalBatchPlannerConfig(
            max_batch_size=10,
            max_batch_items=10,
            max_extra_rows_abs=120.0,
            max_extra_rows_ratio=0.25,
            row_density_rows_per_day=1.0 / 91.0,
        ),
    ) == (("item-1",), ("item-2",))
    assert _plan(
        same_signature,
        config=IntervalBatchPlannerConfig(
            max_batch_size=1,
            max_batch_items=10,
            max_extra_rows_abs=100.0,
            max_extra_rows_ratio=10.0,
            row_density_rows_per_day=1.0,
        ),
    ) == (("item-3",), ("item-4",))
    assert _plan(
        same_instrument,
        config=IntervalBatchPlannerConfig(
            max_batch_size=1,
            max_batch_items=2,
            max_extra_rows_abs=100.0,
            max_extra_rows_ratio=10.0,
            row_density_rows_per_day=1.0,
        ),
    ) == (("item-5", "item-6"), ("item-7",))


def test_plan_interval_batches_is_deterministic_across_input_order() -> None:
    items = [
        _make_item(
            item_id="item-1",
            instrument="AAA.N",
            start_date=date(2020, 1, 1),
            end_date=date(2020, 1, 31),
        ),
        _make_item(
            item_id="item-2",
            instrument="AAA.N",
            start_date=date(2020, 2, 1),
            end_date=date(2020, 2, 29),
        ),
        _make_item(
            item_id="item-3",
            instrument="BBB.N",
            start_date=date(2020, 1, 10),
            end_date=date(2020, 1, 31),
        ),
    ]
    config = IntervalBatchPlannerConfig(
        max_batch_size=2,
        max_batch_items=3,
        max_extra_rows_abs=20.0,
        max_extra_rows_ratio=0.5,
        row_density_rows_per_day=1.0 / 91.0,
    )

    base_plan = plan_interval_batches(
        items,
        config=config,
        planner_version="test_planner_v1",
        signature_fn=_signature,
        interval_fn=_interval,
        batch_builder=_batch_builder,
    )
    shuffled_items = list(items)
    random.Random(7).shuffle(shuffled_items)
    shuffled_plan = plan_interval_batches(
        shuffled_items,
        config=config,
        planner_version="test_planner_v1",
        signature_fn=_signature,
        interval_fn=_interval,
        batch_builder=_batch_builder,
    )

    assert tuple(batch.item_ids for batch in base_plan.batches) == tuple(
        batch.item_ids for batch in shuffled_plan.batches
    )
    assert base_plan.fingerprint == shuffled_plan.fingerprint


def test_plan_interval_batches_batches_close_ownership_windows() -> None:
    items = [
        _make_ownership_item(
            item_id="own-1",
            instrument="AAA.N",
            start_date=date(2020, 1, 1),
            end_date=date(2020, 3, 31),
        ),
        _make_ownership_item(
            item_id="own-2",
            instrument="BBB.N",
            start_date=date(2020, 1, 15),
            end_date=date(2020, 3, 31),
        ),
        _make_ownership_item(
            item_id="own-3",
            instrument="CCC.N",
            start_date=date(2021, 1, 1),
            end_date=date(2021, 3, 31),
        ),
    ]

    planned = _plan(
        items,
        config=IntervalBatchPlannerConfig(
            max_batch_size=2,
            max_batch_items=2,
            max_extra_rows_abs=120.0,
            max_extra_rows_ratio=0.25,
            row_density_rows_per_day=1.0 / 91.0,
        ),
    )

    assert planned == (("own-1", "own-2"), ("own-3",))
