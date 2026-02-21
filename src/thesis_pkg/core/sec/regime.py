from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from functools import lru_cache
from importlib import resources
from pathlib import Path


ITEM_TITLES_10K = {
    "1": ["BUSINESS"],
    "1A": ["RISK FACTORS"],
    "1B": ["UNRESOLVED STAFF COMMENTS"],
    "1C": ["CYBERSECURITY"],
    "2": ["PROPERTIES", "PROPERTY"],
    "3": ["LEGAL PROCEEDINGS"],
    "4": [
        "MINE SAFETY DISCLOSURES",
        "SUBMISSION OF MATTERS TO A VOTE OF SECURITY HOLDERS",
        "SUBMISSION OF MATTERS TO A VOTE OF SHAREHOLDERS",
        "RESERVED",
    ],
    "5": [
        "MARKET FOR REGISTRANT'S COMMON EQUITY",
        "MARKET FOR REGISTRANT S COMMON EQUITY",
        "MARKET FOR REGISTRANTS COMMON EQUITY",
        "MARKET FOR REGISTRANT'S COMMON EQUITY, RELATED STOCKHOLDER MATTERS AND ISSUER PURCHASES OF EQUITY SECURITIES",
        "MARKET FOR REGISTRANT S COMMON EQUITY, RELATED STOCKHOLDER MATTERS AND ISSUER PURCHASES OF EQUITY SECURITIES",
    ],
    "6": ["SELECTED FINANCIAL DATA", "RESERVED"],
    "7": [
        "MANAGEMENT'S DISCUSSION AND ANALYSIS",
        "MANAGEMENT S DISCUSSION AND ANALYSIS",
        "MANAGEMENTS DISCUSSION AND ANALYSIS",
        "MANAGEMENT'S DISCUSSION AND ANALYSIS OF FINANCIAL CONDITION AND RESULTS OF OPERATIONS",
        "MANAGEMENT S DISCUSSION AND ANALYSIS OF FINANCIAL CONDITION AND RESULTS OF OPERATIONS",
    ],
    "7A": [
        "QUANTITATIVE AND QUALITATIVE DISCLOSURES ABOUT MARKET RISK",
        "QUANTITATIVE AND QUALITATIVE DISCLOSURES",
    ],
    "8": [
        "FINANCIAL STATEMENTS AND SUPPLEMENTARY DATA",
        "FINANCIAL STATEMENTS",
        "CONSOLIDATED FINANCIAL STATEMENTS",
        "CONSOLIDATED FINANCIAL STATEMENTS AND SUPPLEMENTARY DATA",
        "NOTES TO CONSOLIDATED FINANCIAL STATEMENTS",
        "NOTES TO THE CONSOLIDATED FINANCIAL STATEMENTS",
    ],
    "9": [
        "CHANGES IN AND DISAGREEMENTS WITH ACCOUNTANTS",
        "CHANGES IN AND DISAGREEMENTS WITH ACCOUNTANTS ON ACCOUNTING AND FINANCIAL DISCLOSURE",
        "CHANGES IN AND DISAGREEMENTS WITH ACCOUNTANTS ON ACCOUNTING AND FINANCIAL DISCLOSURES",
    ],
    "9A": ["CONTROLS AND PROCEDURES"],
    "9B": ["OTHER INFORMATION"],
    "9C": ["DISCLOSURE REGARDING FOREIGN JURISDICTIONS THAT PREVENT INSPECTIONS"],
    "10": [
        "DIRECTORS, EXECUTIVE OFFICERS AND CORPORATE GOVERNANCE",
        "DIRECTORS AND EXECUTIVE OFFICERS",
    ],
    "11": ["EXECUTIVE COMPENSATION"],
    "12": [
        "SECURITY OWNERSHIP OF CERTAIN BENEFICIAL OWNERS AND MANAGEMENT",
        "SECURITY OWNERSHIP OF CERTAIN BENEFICIAL OWNERS AND MANAGEMENT AND RELATED STOCKHOLDER MATTERS",
    ],
    "13": [
        "CERTAIN RELATIONSHIPS AND RELATED TRANSACTIONS",
        "CERTAIN RELATIONSHIPS AND RELATED TRANSACTIONS AND DIRECTOR INDEPENDENCE",
    ],
    "14": ["PRINCIPAL ACCOUNTANT FEES AND SERVICES"],
    "15": [
        "EXHIBITS",
        "EXHIBITS AND FINANCIAL STATEMENT SCHEDULES",
        "EXHIBITS AND FINANCIAL STATEMENTS",
        "EXHIBITS FINANCIAL STATEMENT SCHEDULES",
        "INDEX TO EXHIBITS",
    ],
    "SIGNATURES": ["SIGNATURES"],
}

ITEM_TITLES_10K_BY_CANONICAL = {
    "I:4_VOTING_RESULTS_LEGACY": [
        "SUBMISSION OF MATTERS TO A VOTE OF SECURITY HOLDERS",
        "SUBMISSION OF MATTERS TO A VOTE OF SHAREHOLDERS",
    ],
    "I:4_RESERVED": ["RESERVED"],
    "I:4_MINE_SAFETY": ["MINE SAFETY DISCLOSURES"],
    "II:6_SELECTED_FINANCIAL_DATA": ["SELECTED FINANCIAL DATA"],
    "II:6_RESERVED": ["RESERVED"],
    "III:14_CONTROLS_AND_PROCEDURES_LEGACY": ["CONTROLS AND PROCEDURES"],
    "III:14_PRINCIPAL_ACCOUNTANT_FEES": ["PRINCIPAL ACCOUNTANT FEES AND SERVICES"],
    "IV:14_EXHIBITS_SCHEDULES_REPORTS": [
        "EXHIBITS",
        "EXHIBITS AND FINANCIAL STATEMENT SCHEDULES",
        "EXHIBITS AND FINANCIAL STATEMENTS",
        "EXHIBITS FINANCIAL STATEMENT SCHEDULES",
        "INDEX TO EXHIBITS",
    ],
}


@lru_cache(maxsize=1)
def _load_item_regime_spec() -> dict | None:
    try:
        try:
            data_path = resources.files(__package__).joinpath("item_regime_10k.json")
            with data_path.open("r", encoding="utf-8") as fh:
                return json.load(fh)
        except Exception:
            path = Path(__file__).resolve().parent / "item_regime_10k.json"
            with path.open("r", encoding="utf-8") as fh:
                return json.load(fh)
    except Exception:
        return None


_ALLOWED_ITEM_LETTERS_10K: dict[int, set[str]] = {
    1: {"A", "B", "C"},
    7: {"A"},
    9: {"A", "B", "C"},
}

_ITEM_REGIME_SPEC = _load_item_regime_spec()
# Regime spec is optional; missing or unreadable specs fall back to permissive behavior.
_ITEM_REGIME_ITEMS = _ITEM_REGIME_SPEC.get("items", {}) if _ITEM_REGIME_SPEC else {}
_ITEM_REGIME_LEGACY = (
    {entry["slot"]: entry for entry in _ITEM_REGIME_SPEC.get("legacy_slots", [])}
    if _ITEM_REGIME_SPEC
    else {}
)
_ITEM_REGIME_BY_ID: dict[str, list[tuple[str, dict]]] = {}
if _ITEM_REGIME_SPEC:
    combined = dict(_ITEM_REGIME_ITEMS)
    combined.update(_ITEM_REGIME_LEGACY)
    for key, entry in combined.items():
        item_id = entry.get("item_id")
        if not item_id and ":" in key:
            item_id = key.split(":", 1)[1]
        if not item_id:
            continue
        _ITEM_REGIME_BY_ID.setdefault(item_id, []).append((key, entry))


@dataclass(frozen=True)
class RegimeSpec:
    """Container for a parsed regime specification payload.

    Attributes:
        form: Form label from the spec payload.
        spec_version: Integer spec version.
        triggers: Trigger metadata block from the spec.
        items: Item definition payload keyed by item slot.
    """
    form: str
    spec_version: int
    triggers: dict
    items: dict[str, dict]
    recommended_metadata_fields: dict | None = None
    recommended_output_annotations: dict | None = None


@dataclass(frozen=True)
class RegimeIndex:
    """Compiled lookup structure derived from regime JSON.

    Attributes:
        form: Normalized form label used by the index.
        items_by_key: Item definitions keyed by canonical item key.
        requires_part: Whether part labels are required when validating items.
        triggers: Trigger metadata copied from the source spec.
        spec_version: Integer spec version.
        items_by_id: Optional item-id lookup map (used for 10-K).
    """
    form: str
    items_by_key: dict[str, dict] = field(default_factory=dict)
    requires_part: bool = False
    triggers: dict = field(default_factory=dict)
    spec_version: int = 0
    items_by_id: dict[str, dict] | None = None


def _is_amendment_form(form: str) -> bool:
    if "/A" in form:
        return True
    if form.endswith("-A") or form.endswith("/A"):
        return True
    if re.search(r"[-/]\s*A$", form):
        return True
    return False


def normalize_form_type(form_type: str | None) -> str | None:
    """
    Normalize SEC form labels for regime lookup.

    Supported outputs are ``"10-K"`` and ``"10-Q"``. Amendment forms
    (for example ``10-K/A``) and unsupported forms return ``None``.

    Args:
        form_type: Raw form type string.

    Returns:
        str | None: ``"10-K"``, ``"10-Q"``, or ``None``.
    """
    if not form_type:
        return None
    form = str(form_type).strip().upper()
    if not form:
        return None
    if _is_amendment_form(form):
        return None
    if form.startswith("10-K") or form.startswith("10K"):
        return "10-K"
    if form.startswith("10-Q") or form.startswith("10Q"):
        return "10-Q"
    return None


def load_regime_spec(form: str) -> dict | None:
    """
    Load packaged regime JSON for a normalized form.

    Args:
        form: Form type string accepted by :func:`normalize_form_type`.

    Returns:
        dict | None: Parsed regime payload, or ``None`` when form is unsupported
        or packaged spec loading fails.
    """
    normalized = normalize_form_type(form)
    if not normalized:
        return None
    filename = "item_regime_10k.json" if normalized == "10-K" else "item_regime_10q.json"
    try:
        data_path = resources.files(__package__).joinpath(filename)
        with data_path.open("r", encoding="utf-8") as fh:
            return json.load(fh)
    except Exception:
        return None


def _coerce_spec_version(value: object) -> int:
    try:
        return int(value)
    except Exception:
        return 0


def _derive_item_key(raw_key: object, entry: dict) -> str | None:
    if not isinstance(raw_key, str):
        return None
    key = raw_key.strip()
    if not key:
        return None
    if ":" in key:
        return key
    part = entry.get("part")
    if isinstance(part, str) and part.strip():
        return f"{part.strip()}:{key}"
    return key


def _iter_spec_items(spec: dict) -> list[tuple[str, dict]]:
    items: list[tuple[str, dict]] = []
    raw_items = spec.get("items") or {}
    if isinstance(raw_items, dict):
        for key, entry in raw_items.items():
            if isinstance(entry, dict):
                items.append((key, entry))
    raw_legacy = spec.get("legacy_slots") or []
    if isinstance(raw_legacy, list):
        for entry in raw_legacy:
            if not isinstance(entry, dict):
                continue
            slot = entry.get("slot")
            if isinstance(slot, str) and slot.strip():
                items.append((slot, entry))
    return items


def build_regime_index(spec: dict) -> RegimeIndex:
    """
    Compile a raw regime payload into a :class:`RegimeIndex`.

    Args:
        spec: Parsed regime dictionary.

    Returns:
        RegimeIndex: Compiled lookup object used by extraction heuristics.

    Raises:
        TypeError: If ``spec`` is not a dictionary.
    """
    if not isinstance(spec, dict):
        raise TypeError("spec must be a dict")
    raw_form = spec.get("form")
    normalized_form = normalize_form_type(raw_form) if isinstance(raw_form, str) else None
    form = normalized_form or (raw_form.strip() if isinstance(raw_form, str) else "")
    triggers = spec.get("triggers") if isinstance(spec.get("triggers"), dict) else {}
    spec_version = _coerce_spec_version(spec.get("spec_version"))
    items_by_key: dict[str, dict] = {}
    for raw_key, entry in _iter_spec_items(spec):
        item_key = _derive_item_key(raw_key, entry)
        if not item_key:
            continue
        items_by_key[item_key] = entry
    requires_part = normalized_form == "10-Q"
    items_by_id: dict[str, dict] | None = None
    if normalized_form == "10-K":
        items_by_id = {}
        for item_key, entry in items_by_key.items():
            item_id = entry.get("item_id")
            if not item_id and ":" in item_key:
                item_id = item_key.split(":", 1)[1]
            if not item_id or item_id in items_by_id:
                continue
            items_by_id[item_id] = entry
    return RegimeIndex(
        form=form,
        items_by_key=items_by_key,
        requires_part=requires_part,
        triggers=triggers,
        spec_version=spec_version,
        items_by_id=items_by_id,
    )


@lru_cache(maxsize=4)
def _get_regime_index_cached(normalized_form: str) -> RegimeIndex | None:
    spec = load_regime_spec(normalized_form)
    if not spec:
        return None
    return build_regime_index(spec)


def get_regime_index(form_type: str | None) -> RegimeIndex | None:
    """
    Return cached regime index for a form type.

    Args:
        form_type: Raw form type string.

    Returns:
        RegimeIndex | None: Cached compiled index, or ``None`` when no regime is available.
    """
    normalized = normalize_form_type(form_type)
    if not normalized:
        return None
    return _get_regime_index_cached(normalized)
