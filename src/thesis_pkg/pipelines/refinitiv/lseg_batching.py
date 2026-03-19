from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from typing import Any, Iterable, TypeVar


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
) -> str:
    return stable_hash_id(
        stage,
        tuple(fields),
        parameters or {},
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


def _build_batch_definition(items: list[RequestItem]) -> BatchDefinition:
    first = items[0]
    item_ids = tuple(item.item_id for item in items)
    instruments = tuple(item.instrument for item in items)
    batch_id = stable_hash_id(
        first.stage,
        first.batch_key,
        item_ids,
        prefix="batch",
    )
    return BatchDefinition(
        batch_id=batch_id,
        stage=first.stage,
        batch_key=first.batch_key,
        fields=first.fields,
        parameters=dict(first.parameters),
        item_ids=item_ids,
        instruments=instruments,
    )
