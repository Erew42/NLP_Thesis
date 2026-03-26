from __future__ import annotations

from dataclasses import dataclass
import datetime as dt
import hashlib
import json
from typing import Any, Callable, Iterable, TypeVar


T = TypeVar("T")


def stable_json_dumps(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)


def stable_hash_id(*parts: Any, prefix: str) -> str:
    payload = stable_json_dumps(parts).encode("utf-8")
    digest = hashlib.sha1(payload).hexdigest()
    return f"{prefix}_{digest[:16]}"


def request_signature(
    *,
    stage: str,
    fields: Iterable[str],
    parameters: dict[str, Any] | None,
    excluded_parameter_keys: Iterable[str] | None = None,
) -> str:
    excluded = set(excluded_parameter_keys or ())
    filtered_parameters = {
        key: value
        for key, value in (parameters or {}).items()
        if key not in excluded
    }
    return stable_hash_id(
        stage,
        tuple(fields),
        filtered_parameters,
        prefix="sig",
    )


@dataclass(frozen=True)
class RequestItem:
    item_id: str
    stage: str
    instrument: str
    batch_key: str
    fields: tuple[str, ...]
    parameters: dict[str, Any]
    payload: dict[str, Any]


@dataclass(frozen=True)
class BatchDefinition:
    batch_id: str
    stage: str
    batch_key: str
    fields: tuple[str, ...]
    parameters: dict[str, Any]
    item_ids: tuple[str, ...]
    instruments: tuple[str, ...]


@dataclass(frozen=True)
class IntervalBatchPlannerConfig:
    max_batch_size: int
    max_batch_items: int | None = None
    max_extra_rows_abs: float | None = None
    max_extra_rows_ratio: float | None = None
    max_union_span_days: int | None = None
    row_density_rows_per_day: float = 1.0

    def to_serializable_dict(self) -> dict[str, Any]:
        return {
            "max_batch_size": self.max_batch_size,
            "max_batch_items": self.max_batch_items,
            "max_extra_rows_abs": self.max_extra_rows_abs,
            "max_extra_rows_ratio": self.max_extra_rows_ratio,
            "max_union_span_days": self.max_union_span_days,
            "row_density_rows_per_day": self.row_density_rows_per_day,
        }


@dataclass(frozen=True)
class IntervalBatchPlan:
    planner_version: str
    config: IntervalBatchPlannerConfig
    batches: tuple[BatchDefinition, ...]
    batch_metrics_by_id: dict[str, dict[str, Any]]
    fingerprint: str


def chunked(values: list[T], chunk_size: int) -> list[list[T]]:
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    return [values[idx : idx + chunk_size] for idx in range(0, len(values), chunk_size)]


def batch_items(
    items: Iterable[RequestItem],
    *,
    max_batch_size: int,
    unique_instrument_limit: bool = False,
) -> list[BatchDefinition]:
    grouped: dict[tuple[str, str, str], list[RequestItem]] = {}
    for item in items:
        grouped.setdefault(
            (
                item.stage,
                item.batch_key,
                request_signature(stage=item.stage, fields=item.fields, parameters=item.parameters),
            ),
            [],
        ).append(item)

    batches: list[BatchDefinition] = []
    for (_, _, _), grouped_items in grouped.items():
        ordered_items = sorted(grouped_items, key=lambda item: item.item_id)
        if not ordered_items:
            continue
        if unique_instrument_limit:
            current: list[RequestItem] = []
            current_instruments: set[str] = set()
            for item in ordered_items:
                if current and item.instrument not in current_instruments and len(current_instruments) >= max_batch_size:
                    batches.append(_build_batch_definition(current))
                    current = []
                    current_instruments = set()
                current.append(item)
                current_instruments.add(item.instrument)
            if current:
                batches.append(_build_batch_definition(current))
        else:
            for batch_items_chunk in chunked(ordered_items, max_batch_size):
                batches.append(_build_batch_definition(batch_items_chunk))
    return batches


def split_batch(batch: BatchDefinition) -> list[BatchDefinition]:
    if len(batch.item_ids) <= 1:
        return [batch]

    midpoint = max(1, len(batch.item_ids) // 2)
    left_item_ids = batch.item_ids[:midpoint]
    right_item_ids = batch.item_ids[midpoint:]
    left_instruments = batch.instruments[:midpoint]
    right_instruments = batch.instruments[midpoint:]

    children: list[BatchDefinition] = []
    for suffix, item_ids, instruments in (
        ("a", left_item_ids, left_instruments),
        ("b", right_item_ids, right_instruments),
    ):
        child_batch_id = stable_hash_id(batch.batch_id, suffix, prefix="batch")
        children.append(
            BatchDefinition(
                batch_id=child_batch_id,
                stage=batch.stage,
                batch_key=batch.batch_key,
                fields=batch.fields,
                parameters=dict(batch.parameters),
                item_ids=item_ids,
                instruments=instruments,
            )
        )
    return children


def build_batch_definition(
    items: list[RequestItem],
    *,
    batch_key: str,
    parameters: dict[str, Any],
) -> BatchDefinition:
    if not items:
        raise ValueError("items must be non-empty")
    first = items[0]
    item_ids = tuple(item.item_id for item in items)
    instruments = tuple(item.instrument for item in items)
    batch_id = stable_hash_id(
        first.stage,
        batch_key,
        item_ids,
        prefix="batch",
    )
    return BatchDefinition(
        batch_id=batch_id,
        stage=first.stage,
        batch_key=batch_key,
        fields=first.fields,
        parameters=dict(parameters),
        item_ids=item_ids,
        instruments=instruments,
    )


def plan_interval_batches(
    items: Iterable[RequestItem],
    *,
    config: IntervalBatchPlannerConfig,
    planner_version: str,
    signature_fn: Callable[[RequestItem], str],
    interval_fn: Callable[[RequestItem], tuple[dt.date, dt.date]],
    batch_builder: Callable[[list[RequestItem], dt.date, dt.date], BatchDefinition],
) -> IntervalBatchPlan:
    if config.max_batch_size <= 0:
        raise ValueError("max_batch_size must be positive")
    if config.max_batch_items is not None and config.max_batch_items <= 0:
        raise ValueError("max_batch_items must be positive when set")
    if config.row_density_rows_per_day <= 0:
        raise ValueError("row_density_rows_per_day must be positive")
    if config.max_union_span_days is not None and config.max_union_span_days <= 0:
        raise ValueError("max_union_span_days must be positive when set")

    prepared_items: list[_PreparedIntervalItem] = []
    for item in items:
        start_date, end_date = interval_fn(item)
        prepared_items.append(
            _PreparedIntervalItem(
                request_item=item,
                signature=signature_fn(item),
                start_date=start_date,
                end_date=end_date,
            )
        )
    for prepared in prepared_items:
        if prepared.end_date < prepared.start_date:
            raise ValueError(
                f"interval end date precedes start date for item_id={prepared.request_item.item_id}"
            )

    grouped: dict[tuple[str, str], list[_PreparedIntervalItem]] = {}
    for prepared in prepared_items:
        grouped.setdefault((prepared.request_item.stage, prepared.signature), []).append(prepared)

    planned_batches: list[BatchDefinition] = []
    batch_metrics_by_id: dict[str, dict[str, Any]] = {}
    for (stage, signature), group_items in sorted(grouped.items(), key=lambda value: value[0]):
        ordered = sorted(
            group_items,
            key=lambda prepared: (
                prepared.start_date,
                prepared.end_date,
                prepared.request_item.item_id,
            ),
        )
        remaining = list(ordered)
        while remaining:
            batch_items_rows: list[_PreparedIntervalItem] = [remaining.pop(0)]
            batch_state = _BatchState.from_items(batch_items_rows, row_density_rows_per_day=config.row_density_rows_per_day)
            while True:
                best_candidate: _CandidateDecision | None = None
                for candidate in remaining:
                    decision = _evaluate_candidate(candidate, batch_state=batch_state, config=config)
                    if decision is None:
                        continue
                    if best_candidate is None or decision.sort_key() < best_candidate.sort_key():
                        best_candidate = decision
                if best_candidate is None:
                    break
                batch_items_rows.append(best_candidate.candidate)
                remaining.remove(best_candidate.candidate)
                batch_state = best_candidate.resulting_state
            built_batch = batch_builder(
                [prepared.request_item for prepared in batch_items_rows],
                batch_state.start_date,
                batch_state.end_date,
            )
            planned_batches.append(built_batch)
            batch_metrics_by_id[built_batch.batch_id] = {
                "planner_version": planner_version,
                "stage": stage,
                "signature": signature,
                "item_count": len(batch_items_rows),
                "unique_instrument_count": len(batch_state.unique_instruments),
                "union_start_date": batch_state.start_date.isoformat(),
                "union_end_date": batch_state.end_date.isoformat(),
                "union_span_days": batch_state.union_span_days,
                "estimated_row_density_rows_per_day": config.row_density_rows_per_day,
                "estimated_standalone_rows": batch_state.standalone_rows_estimate,
                "estimated_batched_rows": batch_state.batched_rows_estimate,
                "estimated_extra_rows_abs": batch_state.extra_rows_abs,
                "estimated_extra_rows_ratio": batch_state.extra_rows_ratio,
            }

    fingerprint = stable_hash_id(
        "interval_batch_plan",
        planner_version,
        config.to_serializable_dict(),
        [
            {
                "batch_id": batch.batch_id,
                "stage": batch.stage,
                "batch_key": batch.batch_key,
                "parameters": batch.parameters,
                "item_ids": batch.item_ids,
                "instruments": batch.instruments,
            }
            for batch in planned_batches
        ],
        prefix="plan",
    )

    return IntervalBatchPlan(
        planner_version=planner_version,
        config=config,
        batches=tuple(planned_batches),
        batch_metrics_by_id=batch_metrics_by_id,
        fingerprint=fingerprint,
    )


@dataclass(frozen=True)
class _PreparedIntervalItem:
    request_item: RequestItem
    signature: str
    start_date: dt.date
    end_date: dt.date

    @property
    def standalone_rows_estimate(self) -> float:
        return float(self.span_days)

    @property
    def span_days(self) -> int:
        return _span_days(self.start_date, self.end_date)


@dataclass(frozen=True)
class _BatchState:
    start_date: dt.date
    end_date: dt.date
    item_ids: tuple[str, ...]
    unique_instruments: frozenset[str]
    standalone_rows_estimate: float
    batched_rows_estimate: float
    extra_rows_abs: float
    extra_rows_ratio: float
    row_density_rows_per_day: float

    @property
    def union_span_days(self) -> int:
        return _span_days(self.start_date, self.end_date)

    @property
    def item_count(self) -> int:
        return len(self.item_ids)

    @classmethod
    def from_items(
        cls,
        items: list[_PreparedIntervalItem],
        *,
        row_density_rows_per_day: float,
    ) -> _BatchState:
        if not items:
            raise ValueError("items must be non-empty")
        start_date = min(item.start_date for item in items)
        end_date = max(item.end_date for item in items)
        unique_instruments = frozenset(item.request_item.instrument for item in items)
        standalone_rows = sum(
            row_density_rows_per_day * item.span_days
            for item in items
        )
        batched_rows = row_density_rows_per_day * _span_days(start_date, end_date) * len(unique_instruments)
        extra_rows_abs = batched_rows - standalone_rows
        extra_rows_ratio = 0.0 if standalone_rows <= 0 else extra_rows_abs / standalone_rows
        return cls(
            start_date=start_date,
            end_date=end_date,
            item_ids=tuple(item.request_item.item_id for item in items),
            unique_instruments=unique_instruments,
            standalone_rows_estimate=standalone_rows,
            batched_rows_estimate=batched_rows,
            extra_rows_abs=extra_rows_abs,
            extra_rows_ratio=extra_rows_ratio,
            row_density_rows_per_day=row_density_rows_per_day,
        )


@dataclass(frozen=True)
class _CandidateDecision:
    candidate: _PreparedIntervalItem
    delta_rows: float
    resulting_state: _BatchState

    def sort_key(self) -> tuple[float, float, int, str]:
        return (
            self.delta_rows,
            self.resulting_state.extra_rows_abs,
            self.resulting_state.union_span_days,
            self.candidate.request_item.item_id,
        )


def _evaluate_candidate(
    candidate: _PreparedIntervalItem,
    *,
    batch_state: _BatchState,
    config: IntervalBatchPlannerConfig,
) -> _CandidateDecision | None:
    new_start = min(batch_state.start_date, candidate.start_date)
    new_end = max(batch_state.end_date, candidate.end_date)
    unique_instruments = set(batch_state.unique_instruments)
    unique_instruments.add(candidate.request_item.instrument)
    new_standalone_rows = (
        batch_state.standalone_rows_estimate
        + config.row_density_rows_per_day * candidate.span_days
    )
    new_batched_rows = (
        config.row_density_rows_per_day
        * _span_days(new_start, new_end)
        * len(unique_instruments)
    )
    extra_rows_abs = new_batched_rows - new_standalone_rows
    extra_rows_ratio = 0.0 if new_standalone_rows <= 0 else extra_rows_abs / new_standalone_rows
    resulting_state = _BatchState(
        start_date=new_start,
        end_date=new_end,
        item_ids=batch_state.item_ids + (candidate.request_item.item_id,),
        unique_instruments=frozenset(unique_instruments),
        standalone_rows_estimate=new_standalone_rows,
        batched_rows_estimate=new_batched_rows,
        extra_rows_abs=extra_rows_abs,
        extra_rows_ratio=extra_rows_ratio,
        row_density_rows_per_day=config.row_density_rows_per_day,
    )
    if len(resulting_state.unique_instruments) > config.max_batch_size:
        return None
    if config.max_batch_items is not None and resulting_state.item_count > config.max_batch_items:
        return None
    if config.max_union_span_days is not None and resulting_state.union_span_days > config.max_union_span_days:
        return None
    if config.max_extra_rows_abs is not None and resulting_state.extra_rows_abs > config.max_extra_rows_abs:
        return None
    if config.max_extra_rows_ratio is not None and resulting_state.extra_rows_ratio > config.max_extra_rows_ratio:
        return None
    candidate_standalone_rows = config.row_density_rows_per_day * candidate.span_days
    delta_rows = (
        resulting_state.batched_rows_estimate
        - batch_state.batched_rows_estimate
        - candidate_standalone_rows
    )
    return _CandidateDecision(
        candidate=candidate,
        delta_rows=delta_rows,
        resulting_state=resulting_state,
    )


def _build_batch_definition(items: list[RequestItem]) -> BatchDefinition:
    first = items[0]
    return build_batch_definition(items, batch_key=first.batch_key, parameters=first.parameters)


def _span_days(start_date: dt.date, end_date: dt.date) -> int:
    return (end_date - start_date).days + 1


__all__ = [
    "BatchDefinition",
    "IntervalBatchPlan",
    "IntervalBatchPlannerConfig",
    "RequestItem",
    "batch_items",
    "build_batch_definition",
    "plan_interval_batches",
    "request_signature",
    "split_batch",
    "stable_hash_id",
    "stable_json_dumps",
]
