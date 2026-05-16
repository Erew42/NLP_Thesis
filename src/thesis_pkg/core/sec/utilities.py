from __future__ import annotations

import re
from datetime import date, datetime

try:
    from thesis_native import _lm2011_rust
except Exception as exc:  # pragma: no cover - optional native extension
    _lm2011_rust = None
    _SEC_UTILITIES_RUST_IMPORT_ERROR: str | None = f"{type(exc).__name__}: {exc}"
else:
    _SEC_UTILITIES_RUST_IMPORT_ERROR = None


_SEC_UTILITIES_RUST_METRICS: dict[str, int] = {
    "digits_only_fast_success": 0,
    "digits_only_fast_failures": 0,
    "digits_only_fallbacks": 0,
    "cik10_fast_success": 0,
    "cik10_fast_failures": 0,
    "cik10_fallbacks": 0,
    "doc_id_fast_success": 0,
    "doc_id_fast_failures": 0,
    "doc_id_fallbacks": 0,
    "normalize_newlines_fast_success": 0,
    "normalize_newlines_fast_failures": 0,
    "normalize_newlines_fallbacks": 0,
    "parse_date_fast_success": 0,
    "parse_date_fast_failures": 0,
    "parse_date_fallbacks": 0,
    "roman_to_int_fast_success": 0,
    "roman_to_int_fast_failures": 0,
    "roman_to_int_fallbacks": 0,
    "default_part_fast_success": 0,
    "default_part_fast_failures": 0,
    "default_part_fallbacks": 0,
    "prefix_is_bullet_fast_success": 0,
    "prefix_is_bullet_fast_failures": 0,
    "prefix_is_bullet_fallbacks": 0,
}


def get_sec_utilities_rust_accel_metrics() -> dict[str, int | str | bool | None]:
    metrics: dict[str, int | str | bool | None] = dict(_SEC_UTILITIES_RUST_METRICS)
    metrics["rust_accel_available"] = _lm2011_rust is not None
    metrics["rust_accel_import_error"] = _SEC_UTILITIES_RUST_IMPORT_ERROR
    return metrics


def reset_sec_utilities_rust_accel_metrics() -> None:
    for key in _SEC_UTILITIES_RUST_METRICS:
        _SEC_UTILITIES_RUST_METRICS[key] = 0


def _digits_only_py(s: str | None) -> str | None:
    if not s:
        return None
    d = re.sub(r"\D", "", s)
    return d or None


def _digits_only(s: str | None) -> str | None:
    if _lm2011_rust is not None:
        try:
            out = _lm2011_rust.sec_digits_only_value(s)
            _SEC_UTILITIES_RUST_METRICS["digits_only_fast_success"] += 1
            return None if out is None else str(out)
        except Exception:
            _SEC_UTILITIES_RUST_METRICS["digits_only_fast_failures"] += 1
    _SEC_UTILITIES_RUST_METRICS["digits_only_fallbacks"] += 1
    return _digits_only_py(s)


def _cik_10_py(cik: int | None) -> str | None:
    if cik is None:
        return None
    return str(int(cik)).zfill(10)


def _cik_10(cik: int | None) -> str | None:
    if _lm2011_rust is not None:
        try:
            out = _lm2011_rust.sec_cik_10_value(cik)
            _SEC_UTILITIES_RUST_METRICS["cik10_fast_success"] += 1
            return None if out is None else str(out)
        except Exception:
            _SEC_UTILITIES_RUST_METRICS["cik10_fast_failures"] += 1
    _SEC_UTILITIES_RUST_METRICS["cik10_fallbacks"] += 1
    return _cik_10_py(cik)


def _make_doc_id_py(cik10: str | None, acc: str | None) -> str | None:
    # Keep this stable and readable; you can later switch to a hash if desired.
    if acc is None:
        return None
    return f"{cik10}:{acc}" if cik10 else f"UNK:{acc}"


def _make_doc_id(cik10: str | None, acc: str | None) -> str | None:
    if _lm2011_rust is not None:
        try:
            out = _lm2011_rust.sec_make_doc_id_value(cik10, acc)
            _SEC_UTILITIES_RUST_METRICS["doc_id_fast_success"] += 1
            return None if out is None else str(out)
        except Exception:
            _SEC_UTILITIES_RUST_METRICS["doc_id_fast_failures"] += 1
    _SEC_UTILITIES_RUST_METRICS["doc_id_fallbacks"] += 1
    return _make_doc_id_py(cik10, acc)


def _normalize_newlines_py(text: str) -> str:
    return text.replace("\r\n", "\n").replace("\r", "\n")


def _normalize_newlines(text: str) -> str:
    if _lm2011_rust is not None:
        try:
            out = _lm2011_rust.normalize_newlines_value(text)
            _SEC_UTILITIES_RUST_METRICS["normalize_newlines_fast_success"] += 1
            return str(out)
        except Exception:
            _SEC_UTILITIES_RUST_METRICS["normalize_newlines_fast_failures"] += 1
    _SEC_UTILITIES_RUST_METRICS["normalize_newlines_fallbacks"] += 1
    return _normalize_newlines_py(text)


def _parse_date_py(value: str | date | datetime | None) -> date | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    if isinstance(value, str):
        s = value.strip()
        if not s:
            return None
        if re.fullmatch(r"\d{8}", s):
            try:
                return datetime.strptime(s, "%Y%m%d").date()
            except Exception:
                return None
        if re.fullmatch(r"\d{4}-\d{2}-\d{2}", s):
            try:
                return datetime.strptime(s, "%Y-%m-%d").date()
            except Exception:
                return None
    return None


def _parse_date(value: str | date | datetime | None) -> date | None:
    if _lm2011_rust is not None:
        try:
            out = _lm2011_rust.sec_parse_date_value(value)
            _SEC_UTILITIES_RUST_METRICS["parse_date_fast_success"] += 1
            return out
        except Exception:
            _SEC_UTILITIES_RUST_METRICS["parse_date_fast_failures"] += 1
    _SEC_UTILITIES_RUST_METRICS["parse_date_fallbacks"] += 1
    return _parse_date_py(value)


def _roman_to_int_py(s: str) -> int | None:
    """
    Convert a roman numeral (I, II, IV, ...) to int.
    Returns None if the string is not a valid roman numeral.
    """
    s = s.strip().upper()
    if not s or not re.fullmatch(r"[IVXLCDM]+", s):
        return None

    values = {"I": 1, "V": 5, "X": 10, "L": 50, "C": 100, "D": 500, "M": 1000}
    total = 0
    prev = 0
    for ch in reversed(s):
        v = values[ch]
        if v < prev:
            total -= v
        else:
            total += v
            prev = v
    # Validate by round-trip to reduce false positives.
    if total <= 0:
        return None
    return total


def _roman_to_int(s: str) -> int | None:
    if _lm2011_rust is not None:
        try:
            out = _lm2011_rust.sec_roman_to_int_value(s)
            _SEC_UTILITIES_RUST_METRICS["roman_to_int_fast_success"] += 1
            return None if out is None else int(out)
        except Exception:
            _SEC_UTILITIES_RUST_METRICS["roman_to_int_fast_failures"] += 1
    _SEC_UTILITIES_RUST_METRICS["roman_to_int_fallbacks"] += 1
    return _roman_to_int_py(s)


def _default_part_for_item_id_py(item_id: str | None) -> str | None:
    if not item_id or not item_id[0].isdigit():
        return None
    m = re.match(r"^(?P<num>\d{1,2})", item_id)
    if not m:
        return None
    n = int(m.group("num"))
    if 1 <= n <= 4:
        return "I"
    if 5 <= n <= 9:
        return "II"
    if 10 <= n <= 14:
        return "III"
    if 15 <= n <= 16:
        return "IV"
    return None


def _default_part_for_item_id(item_id: str | None) -> str | None:
    if _lm2011_rust is not None:
        try:
            out = _lm2011_rust.sec_default_part_for_item_id_value(item_id)
            _SEC_UTILITIES_RUST_METRICS["default_part_fast_success"] += 1
            return None if out is None else str(out)
        except Exception:
            _SEC_UTILITIES_RUST_METRICS["default_part_fast_failures"] += 1
    _SEC_UTILITIES_RUST_METRICS["default_part_fallbacks"] += 1
    return _default_part_for_item_id_py(item_id)


def _prefix_is_bullet_py(prefix: str) -> bool:
    if not prefix:
        return False
    return bool(re.fullmatch(r"[\s\-\*\u2022\u00b7\u2013\u2014]+", prefix))


def _prefix_is_bullet(prefix: str) -> bool:
    if _lm2011_rust is not None:
        try:
            out = bool(_lm2011_rust.sec_prefix_is_bullet_value(str(prefix)))
            _SEC_UTILITIES_RUST_METRICS["prefix_is_bullet_fast_success"] += 1
            return out
        except Exception:
            _SEC_UTILITIES_RUST_METRICS["prefix_is_bullet_fast_failures"] += 1
    _SEC_UTILITIES_RUST_METRICS["prefix_is_bullet_fallbacks"] += 1
    return _prefix_is_bullet_py(prefix)
