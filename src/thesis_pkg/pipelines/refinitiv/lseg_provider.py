from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import re
import time
from typing import Any

import polars as pl


SENSITIVE_HEADER_NAMES = frozenset(
    {
        "authorization",
        "cookie",
        "set-cookie",
        "x-api-key",
        "x-auth-token",
        "proxy-authorization",
    }
)

SESSION_NOT_OPENED_MARKERS = (
    "session is not opened",
    "can't send any request",
)

WORKSPACE_PROXY_MARKERS = (
    "localhost:9000",
    "/api/udf",
    "refinitivworkspace.exe",
)

UNRESOLVED_IDENTIFIERS_RE = re.compile(
    r"Unable to resolve all requested identifiers in \[(?P<identifiers>.*?)\]\.",
    re.IGNORECASE,
)


class LsegProviderImportError(RuntimeError):
    pass


class LsegSessionError(RuntimeError):
    pass


class LsegRequestError(RuntimeError):
    def __init__(
        self,
        message: str,
        *,
        status_code: int | None = None,
        headers: dict[str, Any] | None = None,
        response_bytes: int | None = None,
        error_kind: str | None = None,
        unresolved_identifiers: tuple[str, ...] | None = None,
    ) -> None:
        super().__init__(message)
        parsed_error_kind, parsed_unresolved_identifiers = classify_lseg_error_message(message)
        self.status_code = status_code
        self.headers = headers or {}
        self.response_bytes = response_bytes
        self.unresolved_identifiers = tuple(unresolved_identifiers or parsed_unresolved_identifiers)
        self.error_kind = error_kind or parsed_error_kind or (
            "unresolved_identifiers" if self.unresolved_identifiers else None
        )


@dataclass(frozen=True)
class LsegResponseMetadata:
    status_code: int | None
    headers: dict[str, Any]
    latency_ms: int
    response_bytes: int | None
    fingerprint: str | None


@dataclass(frozen=True)
class LsegDataResponse:
    frame: pl.DataFrame
    metadata: LsegResponseMetadata


def is_lseg_available() -> bool:
    try:
        import lseg.data  # noqa: F401
    except ImportError:
        return False
    return True


class LsegDataProvider:
    def __init__(
        self,
        *,
        session_name: str = "desktop.workspace",
        config_name: str | None = None,
        request_timeout: float | None = None,
    ) -> None:
        self.session_name = session_name
        self.config_name = config_name
        self.request_timeout = request_timeout
        self._ld: Any | None = None
        self._config: Any | None = None
        self._previous_request_timeout: Any | None = None
        self._session_open = False

    def open(self) -> None:
        if self._session_open:
            return
        try:
            import lseg.data as ld
        except ImportError as exc:
            raise LsegProviderImportError("lseg.data is not installed in this runtime") from exc

        self._ld = ld
        self._config = self._load_config(ld)
        self._apply_request_timeout()
        try:
            open_kwargs: dict[str, Any] = {"name": self.session_name}
            if self.config_name is not None:
                open_kwargs["config_name"] = self.config_name
            ld.open_session(**open_kwargs)
        except Exception as exc:  # pragma: no cover - exercised only with the real library
            self._restore_request_timeout()
            config_suffix = "" if self.config_name is None else f" using config {self.config_name!r}"
            raise LsegSessionError(f"failed to open LSEG session {self.session_name!r}{config_suffix}") from exc
        self._session_open = True

    def close(self) -> None:
        if self._ld is None:
            self._restore_request_timeout()
            return
        try:
            if self._session_open:
                self._ld.close_session()
        except Exception:
            pass
        finally:
            self._session_open = False
            self._restore_request_timeout()

    def probe(
        self,
        *,
        universe: list[str],
        fields: list[str],
        parameters: dict[str, Any] | None = None,
    ) -> LsegDataResponse:
        return self.get_data(universe=universe, fields=fields, parameters=parameters)

    def get_data(
        self,
        *,
        universe: list[str],
        fields: list[str],
        parameters: dict[str, Any] | None = None,
    ) -> LsegDataResponse:
        if not self._session_open or self._ld is None:
            self.open()
        refresh_attempt = 0
        while True:
            try:
                return self._get_data_once(universe=universe, fields=fields, parameters=parameters)
            except LsegRequestError as exc:
                if not _should_refresh_and_retry_request(
                    exc=exc,
                    universe=universe,
                    refresh_attempt=refresh_attempt,
                ):
                    raise
                self._reset_session()
                refresh_attempt += 1

    def _get_data_once(
        self,
        *,
        universe: list[str],
        fields: list[str],
        parameters: dict[str, Any] | None = None,
    ) -> LsegDataResponse:
        assert self._ld is not None

        started = time.perf_counter()
        response: Any
        try:
            response = self._ld.get_data(universe=universe, fields=fields, parameters=parameters or {})
        except Exception as exc:  # pragma: no cover - exercised only with the real library
            headers, status_code, response_bytes = _extract_http_metadata(exc)
            error_kind, unresolved_identifiers = classify_lseg_error_message(str(exc))
            raise LsegRequestError(
                str(exc),
                status_code=status_code,
                headers=headers,
                response_bytes=response_bytes,
                error_kind=error_kind,
                unresolved_identifiers=unresolved_identifiers,
            ) from exc

        latency_ms = int((time.perf_counter() - started) * 1000)
        headers, status_code, response_bytes = _extract_http_metadata(response)
        try:
            frame = _to_polars_frame(response)
        except Exception as exc:
            raise LsegRequestError(
                f"failed to convert LSEG response to Polars: {exc}",
                status_code=status_code,
                headers=headers,
                response_bytes=response_bytes,
                error_kind="response_parse_error",
            ) from exc
        fingerprint = _frame_fingerprint(frame)
        return LsegDataResponse(
            frame=frame,
            metadata=LsegResponseMetadata(
                status_code=status_code,
                headers=headers,
                latency_ms=latency_ms,
                response_bytes=response_bytes,
                fingerprint=fingerprint,
            ),
        )

    def _reset_session(self) -> None:
        self.close()
        self.open()

    def _load_config(self, ld: Any) -> Any | None:
        getter = getattr(ld, "get_config", None)
        if getter is None:
            return None
        try:
            return getter()
        except Exception:
            return None

    def _apply_request_timeout(self) -> None:
        if self.request_timeout is None or self._config is None:
            return
        getter = getattr(self._config, "get_param", None)
        setter = getattr(self._config, "set_param", None)
        if getter is None or setter is None:
            return
        try:
            self._previous_request_timeout = getter("http.request-timeout")
        except Exception:
            self._previous_request_timeout = None
        try:
            setter("http.request-timeout", self.request_timeout)
        except Exception:
            self._previous_request_timeout = None

    def _restore_request_timeout(self) -> None:
        if self._config is None or self._previous_request_timeout is None:
            return
        setter = getattr(self._config, "set_param", None)
        if setter is None:
            return
        try:
            setter("http.request-timeout", self._previous_request_timeout)
        except Exception:
            pass
        finally:
            self._previous_request_timeout = None


def classify_lseg_error_message(message: str) -> tuple[str | None, tuple[str, ...]]:
    unresolved_identifiers = _parse_unresolved_identifiers(message)
    if unresolved_identifiers:
        return "unresolved_identifiers", unresolved_identifiers
    normalized = message.lower()
    if _is_session_not_opened_message(message):
        return "session_open_failed", ()
    if "timed out" in normalized or "timeout" in normalized:
        if any(marker in normalized for marker in WORKSPACE_PROXY_MARKERS):
            return "workspace_proxy_timeout", ()
        return "transport_timeout", ()
    if any(token in normalized for token in ("service unavailable", "overload", "backend", "502", "503", "504")):
        return "backend_overload", ()
    return None, ()


def _is_session_not_opened_message(message: str) -> bool:
    normalized = message.lower()
    return all(marker in normalized for marker in SESSION_NOT_OPENED_MARKERS)


def _should_refresh_and_retry_request(
    *,
    exc: LsegRequestError,
    universe: list[str],
    refresh_attempt: int,
) -> bool:
    if refresh_attempt > 0:
        return False
    if exc.error_kind in {"session_open_failed", "workspace_proxy_timeout"} or _is_session_not_opened_message(str(exc)):
        return True
    return len(universe) == 1 and exc.error_kind == "unresolved_identifiers"


def _extract_http_metadata(source: Any) -> tuple[dict[str, Any], int | None, int | None]:
    raw = getattr(source, "raw", None)
    status_code = None
    response_bytes = None
    headers: dict[str, Any] = {}

    if raw is not None:
        status_code = _coerce_int(getattr(raw, "status_code", None))
        raw_headers = getattr(raw, "headers", None)
        if raw_headers is not None:
            headers = _sanitize_headers(dict(raw_headers))
        response_bytes = _coerce_int(getattr(raw, "content_length", None))
        if response_bytes is None:
            content = getattr(raw, "content", None)
            if content is not None:
                try:
                    response_bytes = len(content)
                except TypeError:
                    response_bytes = None

    if not headers:
        maybe_headers = getattr(source, "headers", None)
        if maybe_headers is not None:
            headers = _sanitize_headers(dict(maybe_headers))
    if status_code is None:
        status_code = _coerce_int(getattr(source, "status_code", None))
    return headers, status_code, response_bytes


def _sanitize_headers(headers: dict[str, Any]) -> dict[str, Any]:
    sanitized: dict[str, Any] = {}
    for key, value in headers.items():
        lowered = str(key).strip().lower()
        if lowered in SENSITIVE_HEADER_NAMES:
            continue
        sanitized[str(key)] = value
    return sanitized


def _to_polars_frame(response: Any) -> pl.DataFrame:
    data = getattr(response, "data", response)
    candidate = getattr(data, "df", data)
    if hasattr(candidate, "reset_index") and hasattr(candidate, "columns"):
        pandas_df = candidate if "Instrument" in list(candidate.columns) else candidate.reset_index(drop=False)
        try:
            return pl.from_pandas(pandas_df)
        except Exception:
            return _coerce_pandas_frame_to_utf8(candidate=pandas_df)
    if isinstance(candidate, pl.DataFrame):
        return candidate
    if isinstance(candidate, list):
        return pl.DataFrame(candidate)
    if hasattr(candidate, "to_dict"):
        maybe_records = candidate.to_dict(orient="records")
        return pl.DataFrame(maybe_records)
    raise TypeError(f"unsupported LSEG response payload type: {type(candidate)!r}")


def _frame_fingerprint(frame: pl.DataFrame) -> str | None:
    if frame.height == 0:
        return hashlib.sha1(b"empty").hexdigest()
    try:
        payload = json.dumps(frame.to_dicts(), default=str, separators=(",", ":"), sort_keys=True).encode("utf-8")
    except Exception:
        return None
    return hashlib.sha1(payload).hexdigest()


def _coerce_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _parse_unresolved_identifiers(message: str) -> tuple[str, ...]:
    match = UNRESOLVED_IDENTIFIERS_RE.search(message)
    if match is None:
        return ()
    raw_identifiers = match.group("identifiers").strip()
    if not raw_identifiers:
        return ()
    identifiers: list[str] = []
    for token in raw_identifiers.split(","):
        normalized = token.strip().strip("'\"")
        if normalized:
            identifiers.append(normalized)
    return tuple(identifiers)


def _coerce_pandas_frame_to_utf8(*, candidate: Any) -> pl.DataFrame:
    columns = [str(column) for column in list(candidate.columns)]
    safe_candidate = candidate.where(candidate.notna(), None)
    records = safe_candidate.to_dict(orient="records")
    normalized_records = [
        {
            str(column): None if value is None else str(value)
            for column, value in record.items()
        }
        for record in records
    ]
    return pl.DataFrame(
        normalized_records,
        schema={column: pl.Utf8 for column in columns},
    )
