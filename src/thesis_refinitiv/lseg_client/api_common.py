from __future__ import annotations

import datetime as dt
import json
from dataclasses import dataclass
from pathlib import Path
import random
from typing import Any

import polars as pl

try:
    from thesis_native import _lm2011_rust
except Exception as exc:  # pragma: no cover - optional native extension
    _lm2011_rust = None
    _LSEG_API_COMMON_RUST_IMPORT_ERROR: str | None = f"{type(exc).__name__}: {exc}"
else:
    _LSEG_API_COMMON_RUST_IMPORT_ERROR = None

from thesis_refinitiv.lseg_client.ledger import (
    LEDGER_DEFERRED_DAILY_LIMIT,
    LEDGER_FATAL_ERROR,
    LEDGER_RETRYABLE_ERROR,
    utc_now,
)
from thesis_refinitiv.lseg_client.provider import (
    LsegRequestError,
    classify_lseg_error_message,
)


_LSEG_API_COMMON_RUST_METRICS: dict[str, int] = {
    "overload_like_fast_success": 0,
    "overload_like_fast_failures": 0,
    "overload_like_fallbacks": 0,
    "daily_limit_fast_success": 0,
    "daily_limit_fast_failures": 0,
    "daily_limit_fallbacks": 0,
    "indicates_daily_limit_fast_success": 0,
    "indicates_daily_limit_fast_failures": 0,
    "indicates_daily_limit_fallbacks": 0,
    "empty_result_fast_success": 0,
    "empty_result_fast_failures": 0,
    "empty_result_fallbacks": 0,
    "classify_policy_fast_success": 0,
    "classify_policy_fast_failures": 0,
    "classify_policy_fallbacks": 0,
}


def get_lseg_api_common_rust_accel_metrics() -> dict[str, int | str | bool | None]:
    metrics: dict[str, int | str | bool | None] = dict(_LSEG_API_COMMON_RUST_METRICS)
    metrics["rust_accel_available"] = _lm2011_rust is not None
    metrics["rust_accel_import_error"] = _LSEG_API_COMMON_RUST_IMPORT_ERROR
    return metrics


def reset_lseg_api_common_rust_accel_metrics() -> None:
    for key in _LSEG_API_COMMON_RUST_METRICS:
        _LSEG_API_COMMON_RUST_METRICS[key] = 0


@dataclass(frozen=True)
class BatchErrorPolicy:
    state: str
    split_batch: bool
    stop_stage: bool
    defer_stage: bool

    def as_dict(self) -> dict[str, Any]:
        return {
            "state": self.state,
            "split_batch": self.split_batch,
            "stop_stage": self.stop_stage,
            "defer_stage": self.defer_stage,
        }


def write_parquet_atomic(df: pl.DataFrame, output_path: Path) -> None:
    temp_path = output_path.with_suffix(output_path.suffix + ".tmp")
    df.write_parquet(temp_path, compression="zstd")
    temp_path.replace(output_path)


def candidate_output_path(output_path: Path) -> Path:
    return output_path.with_suffix(output_path.suffix + ".candidate")


def promote_candidate_output(candidate_path: Path, output_path: Path) -> None:
    candidate_path.replace(output_path)


def append_json_log(path: Path, payload: dict[str, Any]) -> None:
    payload = {
        "logged_at_utc": utc_now().strftime("%Y-%m-%dT%H:%M:%SZ"),
        **payload,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(payload, sort_keys=True, default=str) + "\n"
    lock_path = path.with_suffix(path.suffix + ".lock")
    with lock_path.open("a+b") as lock_handle:
        _acquire_file_lock(lock_handle)
        try:
            with path.open("a", encoding="utf-8") as handle:
                handle.write(line)
        finally:
            _release_file_lock(lock_handle)


def _acquire_file_lock(handle: Any) -> None:
    handle.seek(0, 2)
    if handle.tell() == 0:
        handle.write(b"0")
        handle.flush()
    handle.seek(0)
    if handle.closed:
        raise ValueError("cannot lock a closed file handle")
    try:
        import msvcrt

        msvcrt.locking(handle.fileno(), msvcrt.LK_LOCK, 1)
    except ImportError:
        import fcntl

        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)


def _release_file_lock(handle: Any) -> None:
    handle.seek(0)
    try:
        import msvcrt

        msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
    except ImportError:
        import fcntl

        fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def standardize_field_frame(
    frame: pl.DataFrame,
    *,
    expected_fields: tuple[str, ...],
    field_aliases: dict[str, tuple[str, ...]] | None = None,
) -> pl.DataFrame:
    if frame.height == 0:
        return pl.DataFrame(schema={"instrument": pl.Utf8, **{field: pl.Utf8 for field in expected_fields}})

    field_aliases = field_aliases or {}
    columns = list(frame.columns)
    rename_map: dict[str, str] = {}
    instrument_column = next(
        (name for name in ("instrument", "Instrument") if name in columns),
        None,
    )
    if instrument_column is None and columns:
        instrument_column = columns[0]
    if instrument_column is not None and instrument_column != "instrument":
        rename_map[instrument_column] = "instrument"
    renamed = frame.rename(rename_map) if rename_map else frame
    current_columns = set(renamed.columns)

    if all(field in current_columns for field in expected_fields):
        return renamed.select(["instrument", *expected_fields])

    exprs = [pl.col("instrument").cast(pl.Utf8, strict=False).alias("instrument")]
    for idx, field in enumerate(expected_fields):
        candidate_columns = (field, *field_aliases.get(field, ()))
        matched_column = next((name for name in candidate_columns if name in current_columns), None)
        if matched_column is not None:
            exprs.append(pl.col(matched_column).cast(pl.Utf8, strict=False).alias(field))
            continue
        available_value_columns = [name for name in renamed.columns if name != "instrument"]
        if idx < len(available_value_columns):
            exprs.append(pl.col(available_value_columns[idx]).cast(pl.Utf8, strict=False).alias(field))
            continue
        exprs.append(pl.lit(None, dtype=pl.Utf8).alias(field))
    return renamed.select(exprs)


def _classify_error_policy_py(
    *,
    status_code: int | None,
    headers: dict[str, Any],
    error_kind: str | None,
    message: str,
    batch_size: int,
    attempt_no: int,
    max_attempts: int,
    split_after_attempt: int,
    fatal_exception: bool,
) -> dict[str, Any]:
    if error_kind == "unresolved_identifiers":
        if batch_size == 1 and attempt_no < max_attempts:
            return BatchErrorPolicy(
                state=LEDGER_RETRYABLE_ERROR,
                split_batch=False,
                stop_stage=False,
                defer_stage=False,
            ).as_dict()
        return BatchErrorPolicy(
            state=LEDGER_RETRYABLE_ERROR if batch_size > 1 else LEDGER_FATAL_ERROR,
            split_batch=batch_size > 1,
            stop_stage=False,
            defer_stage=False,
        ).as_dict()

    if _indicates_daily_limit_py(status_code, headers, message):
        return BatchErrorPolicy(
            state=LEDGER_DEFERRED_DAILY_LIMIT,
            split_batch=False,
            stop_stage=False,
            defer_stage=True,
        ).as_dict()

    if status_code == 403 or error_kind == "session_open_failed":
        return BatchErrorPolicy(
            state=LEDGER_FATAL_ERROR,
            split_batch=False,
            stop_stage=True,
            defer_stage=False,
        ).as_dict()

    if _is_overload_like_py(status_code, error_kind, message):
        if batch_size > 1 and attempt_no >= split_after_attempt:
            return BatchErrorPolicy(
                state=LEDGER_RETRYABLE_ERROR,
                split_batch=True,
                stop_stage=False,
                defer_stage=False,
            ).as_dict()
        if attempt_no >= max_attempts:
            return BatchErrorPolicy(
                state=LEDGER_FATAL_ERROR,
                split_batch=False,
                stop_stage=True,
                defer_stage=False,
            ).as_dict()
        return BatchErrorPolicy(
            state=LEDGER_RETRYABLE_ERROR,
            split_batch=False,
            stop_stage=False,
            defer_stage=False,
        ).as_dict()

    if fatal_exception:
        return BatchErrorPolicy(
            state=LEDGER_FATAL_ERROR,
            split_batch=False,
            stop_stage=True,
            defer_stage=False,
        ).as_dict()

    if attempt_no >= max_attempts:
        return BatchErrorPolicy(
            state=LEDGER_FATAL_ERROR,
            split_batch=False,
            stop_stage=True,
            defer_stage=False,
        ).as_dict()

    return BatchErrorPolicy(
        state=LEDGER_RETRYABLE_ERROR,
        split_batch=False,
        stop_stage=False,
        defer_stage=False,
    ).as_dict()


def _classify_error_policy(
    *,
    status_code: int | None,
    headers: dict[str, Any],
    error_kind: str | None,
    message: str,
    batch_size: int,
    attempt_no: int,
    max_attempts: int,
    split_after_attempt: int,
    fatal_exception: bool,
) -> dict[str, Any]:
    if _lm2011_rust is not None:
        try:
            out = _lm2011_rust.lseg_classify_batch_error_policy(
                status_code,
                error_kind,
                message,
                headers,
                batch_size,
                attempt_no,
                max_attempts,
                split_after_attempt,
                fatal_exception,
            )
            _LSEG_API_COMMON_RUST_METRICS["classify_policy_fast_success"] += 1
            return dict(out)
        except Exception:
            _LSEG_API_COMMON_RUST_METRICS["classify_policy_fast_failures"] += 1
            _LSEG_API_COMMON_RUST_METRICS["classify_policy_fallbacks"] += 1
    else:
        _LSEG_API_COMMON_RUST_METRICS["classify_policy_fallbacks"] += 1
    return _classify_error_policy_py(
        status_code=status_code,
        headers=headers,
        error_kind=error_kind,
        message=message,
        batch_size=batch_size,
        attempt_no=attempt_no,
        max_attempts=max_attempts,
        split_after_attempt=split_after_attempt,
        fatal_exception=fatal_exception,
    )


def classify_error(
    exc: Exception,
    *,
    batch_size: int,
    attempt_no: int,
    max_attempts: int,
    split_after_attempt: int = 2,
) -> dict[str, Any]:
    error_details = error_details_for_exception(exc)
    return _classify_error_policy(
        status_code=error_details["status_code"],
        headers=error_details["headers"],
        error_kind=error_details["error_kind"],
        message=str(exc).lower(),
        batch_size=batch_size,
        attempt_no=attempt_no,
        max_attempts=max_attempts,
        split_after_attempt=split_after_attempt,
        fatal_exception=isinstance(exc, (ValueError, TypeError)),
    )


def error_details_for_exception(exc: Exception) -> dict[str, Any]:
    status_code, headers, response_bytes = error_metadata(exc)
    if isinstance(exc, LsegRequestError):
        error_kind = exc.error_kind
        unresolved_identifiers = list(exc.unresolved_identifiers)
    else:
        error_kind, parsed_identifiers = classify_lseg_error_message(str(exc))
        unresolved_identifiers = list(parsed_identifiers)
    return {
        "status_code": status_code,
        "headers": headers,
        "response_bytes": response_bytes,
        "error_kind": error_kind,
        "unresolved_identifiers": unresolved_identifiers,
    }


def error_metadata(exc: Exception) -> tuple[int | None, dict[str, Any], int | None]:
    if isinstance(exc, LsegRequestError):
        return exc.status_code, exc.headers, exc.response_bytes
    return None, {}, None


def _should_treat_as_empty_result_py(
    *,
    error_kind: str | None,
    unresolved_identifiers: list[str],
    universe: list[str],
    attempt_no: int,
    max_attempts: int,
    normalizer: Any,
) -> bool:
    if error_kind != "unresolved_identifiers":
        return False
    if attempt_no < max_attempts:
        return False
    unresolved_set = {
        normalized
        for identifier in unresolved_identifiers
        if (normalized := normalizer(identifier)) is not None
    }
    requested_set = {
        normalized
        for identifier in universe
        if (normalized := normalizer(identifier)) is not None
    }
    return bool(requested_set) and requested_set.issubset(unresolved_set)


def should_treat_as_empty_result(
    *,
    error_kind: str | None,
    unresolved_identifiers: list[str],
    universe: list[str],
    attempt_no: int,
    max_attempts: int,
    normalizer: Any,
) -> bool:
    if _lm2011_rust is not None:
        try:
            out = bool(
                _lm2011_rust.lseg_should_treat_as_empty_result(
                    error_kind,
                    unresolved_identifiers,
                    universe,
                    attempt_no,
                    max_attempts,
                    normalizer,
                )
            )
            _LSEG_API_COMMON_RUST_METRICS["empty_result_fast_success"] += 1
            return out
        except Exception:
            _LSEG_API_COMMON_RUST_METRICS["empty_result_fast_failures"] += 1
            _LSEG_API_COMMON_RUST_METRICS["empty_result_fallbacks"] += 1
    else:
        _LSEG_API_COMMON_RUST_METRICS["empty_result_fallbacks"] += 1
    return _should_treat_as_empty_result_py(
        error_kind=error_kind,
        unresolved_identifiers=unresolved_identifiers,
        universe=universe,
        attempt_no=attempt_no,
        max_attempts=max_attempts,
        normalizer=normalizer,
    )


def retry_delay_seconds(attempt_no: int) -> float:
    ladder = [5.0, 15.0, 45.0, 120.0, 300.0]
    base = ladder[min(max(attempt_no - 1, 0), len(ladder) - 1)]
    return min(600.0, random.uniform(0.5 * base, 1.5 * base))


def _daily_limit_likely_exhausted_py(headers: dict[str, Any]) -> bool:
    numeric_values: list[int] = []
    for key, value in headers.items():
        lowered = key.lower()
        if "remaining" not in lowered and "daily" not in lowered and "limit" not in lowered and "usage" not in lowered:
            continue
        try:
            numeric_values.append(int(value))
        except (TypeError, ValueError):
            continue
    return bool(numeric_values) and min(numeric_values) <= 25


def daily_limit_likely_exhausted(headers: dict[str, Any]) -> bool:
    if _lm2011_rust is not None:
        try:
            out = bool(_lm2011_rust.lseg_daily_limit_likely_exhausted(headers))
            _LSEG_API_COMMON_RUST_METRICS["daily_limit_fast_success"] += 1
            return out
        except Exception:
            _LSEG_API_COMMON_RUST_METRICS["daily_limit_fast_failures"] += 1
            _LSEG_API_COMMON_RUST_METRICS["daily_limit_fallbacks"] += 1
    else:
        _LSEG_API_COMMON_RUST_METRICS["daily_limit_fallbacks"] += 1
    return _daily_limit_likely_exhausted_py(headers)


def _indicates_daily_limit_py(status_code: int | None, headers: dict[str, Any], message: str) -> bool:
    if "daily" in message and "limit" in message:
        return True
    if status_code == 429 and _daily_limit_likely_exhausted_py(headers):
        return True
    return False


def indicates_daily_limit(status_code: int | None, headers: dict[str, Any], message: str) -> bool:
    if _lm2011_rust is not None:
        try:
            out = bool(_lm2011_rust.lseg_indicates_daily_limit(status_code, headers, message))
            _LSEG_API_COMMON_RUST_METRICS["indicates_daily_limit_fast_success"] += 1
            return out
        except Exception:
            _LSEG_API_COMMON_RUST_METRICS["indicates_daily_limit_fast_failures"] += 1
            _LSEG_API_COMMON_RUST_METRICS["indicates_daily_limit_fallbacks"] += 1
    else:
        _LSEG_API_COMMON_RUST_METRICS["indicates_daily_limit_fallbacks"] += 1
    return _indicates_daily_limit_py(status_code, headers, message)


def next_daily_resume_utc(now: dt.datetime) -> dt.datetime:
    tomorrow = (now + dt.timedelta(days=1)).astimezone(dt.timezone.utc).date()
    return dt.datetime.combine(tomorrow, dt.time(hour=0, minute=5), tzinfo=dt.timezone.utc)


def _is_overload_like_py(status_code: int | None, error_kind: str | None, message: str) -> bool:
    if status_code in {429, 500, 502, 503, 504}:
        return True
    if error_kind in {"transport_timeout", "workspace_proxy_timeout", "backend_overload"}:
        return True
    return any(token in message for token in ("timeout", "timed out", "overload", "service unavailable", "backend"))


def _is_overload_like(status_code: int | None, error_kind: str | None, message: str) -> bool:
    if _lm2011_rust is not None:
        try:
            out = bool(_lm2011_rust.is_lseg_overload_like(status_code, error_kind, message))
            _LSEG_API_COMMON_RUST_METRICS["overload_like_fast_success"] += 1
            return out
        except Exception:
            _LSEG_API_COMMON_RUST_METRICS["overload_like_fast_failures"] += 1
            _LSEG_API_COMMON_RUST_METRICS["overload_like_fallbacks"] += 1
    else:
        _LSEG_API_COMMON_RUST_METRICS["overload_like_fallbacks"] += 1
    return _is_overload_like_py(status_code, error_kind, message)
