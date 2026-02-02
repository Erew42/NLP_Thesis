from __future__ import annotations

import html
import random
import re
from collections import defaultdict
from pathlib import Path

from .extraction import _strip_edgar_metadata
from .heuristics import _repair_wrapped_headings
from .utilities import _normalize_newlines


def normalize_extractor_body(text: str) -> str:
    return _repair_wrapped_headings(_strip_edgar_metadata(_normalize_newlines(text)))


STATUS_PASS = "pass"
STATUS_WARNING = "warning"
STATUS_FAIL = "fail"

DEFAULT_SAMPLE_WEIGHTS: dict[str, float] = {
    STATUS_PASS: 0.5,
    STATUS_WARNING: 0.3,
    STATUS_FAIL: 0.2,
}


def _safe_slug(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_-]+", "_", value).strip("_") or "unknown"


def _html_escape(value: object) -> str:
    if value is None:
        return ""
    return html.escape(str(value))


def _parse_bool(value: object) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    return str(value).strip().lower() in {"1", "true", "yes", "y", "t"}


def _parse_int(value: object, *, default: int = 0) -> int:
    try:
        return int(str(value).strip())
    except Exception:
        return default


def _part_rank(value: str) -> int:
    return {"I": 1, "II": 2, "III": 3, "IV": 4}.get(value.strip().upper(), 99)


def _item_id_sort_key(value: str) -> tuple[int, str]:
    cleaned = value.strip().upper()
    match = re.match(r"^(\d+)([A-Z]?)$", cleaned)
    if match:
        return (int(match.group(1)), match.group(2))
    return (999, cleaned)


def _filing_filename(doc_id: str, accession: str) -> str:
    return f"{_safe_slug(doc_id)}_{_safe_slug(accession)}.html"


def _asset_link(path: str, *, depth: int) -> str:
    if not path:
        return ""
    prefix = "../" * max(depth, 0)
    return f"{prefix}{path}"


def normalize_sample_weights(weights: dict[str, float] | None) -> dict[str, float]:
    merged = dict(DEFAULT_SAMPLE_WEIGHTS)
    if weights:
        for key, value in weights.items():
            if value is None:
                continue
            normalized_key = str(key).strip().lower()
            if normalized_key in merged:
                merged[normalized_key] = float(value)
    return {key: max(0.0, value) for key, value in merged.items()}


def classify_filing_status(row: dict[str, object]) -> str:
    any_fail = _parse_bool(row.get("any_fail"))
    exclusion = str(row.get("filing_exclusion_reason") or "").strip()
    if any_fail or exclusion:
        return STATUS_FAIL
    if _parse_bool(row.get("any_warn")):
        return STATUS_WARNING
    return STATUS_PASS


def _status_label(status: str) -> str:
    return {
        STATUS_PASS: "PASS",
        STATUS_WARNING: "WARNING",
        STATUS_FAIL: "FAIL",
    }.get(status, status.upper())


def _status_badge(status: str) -> str:
    badge_class = "ok"
    if status == STATUS_WARNING:
        badge_class = "warn"
    elif status == STATUS_FAIL:
        badge_class = "fail"
    return f"<span class=\"badge {badge_class}\">{_status_label(status)}</span>"


def _quartile_edges(values: list[int]) -> tuple[int, int, int]:
    if not values:
        return (0, 0, 0)
    ordered = sorted(values)
    n = len(ordered)
    q1 = ordered[int(0.25 * (n - 1))]
    q2 = ordered[int(0.50 * (n - 1))]
    q3 = ordered[int(0.75 * (n - 1))]
    return (q1, q2, q3)


def _quartile_bucket(value: int, edges: tuple[int, int, int]) -> str:
    q1, q2, q3 = edges
    if value <= q1:
        return "Q1"
    if value <= q2:
        return "Q2"
    if value <= q3:
        return "Q3"
    return "Q4"


def _stratified_sample(
    strata: dict[tuple[str, bool], list[dict[str, object]]],
    *,
    sample_size: int,
    seed: int,
) -> list[dict[str, object]]:
    rng = random.Random(seed)
    total = sum(len(rows) for rows in strata.values())
    if total == 0 or sample_size <= 0:
        return []
    if sample_size > total:
        sample_size = total

    targets: dict[tuple[str, bool], int] = {}
    fractional: list[tuple[float, tuple[str, bool]]] = []
    for key, rows in strata.items():
        raw = (sample_size * len(rows)) / total
        base = int(raw)
        targets[key] = min(base, len(rows))
        fractional.append((raw - base, key))

    remaining = sample_size - sum(targets.values())
    fractional.sort(reverse=True, key=lambda t: t[0])
    for _, key in fractional:
        if remaining <= 0:
            break
        if targets[key] < len(strata[key]):
            targets[key] += 1
            remaining -= 1

    while remaining > 0:
        available = [key for key in strata if targets[key] < len(strata[key])]
        if not available:
            break
        key = rng.choice(available)
        targets[key] += 1
        remaining -= 1

    sampled: list[dict[str, object]] = []
    for key in sorted(strata.keys()):
        rows = strata[key]
        k = targets.get(key, 0)
        if k <= 0:
            continue
        sampled.extend(rng.sample(rows, k))
    return sampled


def _sample_stratified_rows(
    rows: list[dict[str, object]],
    *,
    sample_size: int,
    seed: int,
) -> list[dict[str, object]]:
    if not rows or sample_size <= 0:
        return []
    edges = _quartile_edges(
        [_parse_int(row.get("n_items_extracted"), default=0) for row in rows]
    )
    strata: dict[tuple[str, bool], list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        bucket = _quartile_bucket(
            _parse_int(row.get("n_items_extracted"), default=0), edges
        )
        missing_core = bool(str(row.get("missing_core_items") or "").strip())
        strata[(bucket, missing_core)].append(row)
    return _stratified_sample(strata, sample_size=sample_size, seed=seed)


def sample_filings_by_status(
    rows: list[dict[str, object]],
    *,
    sample_size: int,
    seed: int,
    weights: dict[str, float] | None = None,
) -> list[dict[str, object]]:
    if not rows or sample_size <= 0:
        return []

    status_rows = {
        STATUS_PASS: [],
        STATUS_WARNING: [],
        STATUS_FAIL: [],
    }
    for row in rows:
        status_rows[classify_filing_status(row)].append(row)

    total_available = sum(len(rows) for rows in status_rows.values())
    if total_available <= 0:
        return []
    if sample_size > total_available:
        sample_size = total_available

    normalized_weights = normalize_sample_weights(weights)
    weight_sum = sum(
        normalized_weights[status]
        for status, rows_for_status in status_rows.items()
        if rows_for_status
    )
    if weight_sum <= 0:
        weight_sum = sum(1 for rows_for_status in status_rows.values() if rows_for_status)

    targets: dict[str, int] = {}
    fractional: list[tuple[float, str]] = []
    for status, rows_for_status in status_rows.items():
        if not rows_for_status:
            targets[status] = 0
            continue
        raw = sample_size * (normalized_weights[status] / weight_sum)
        base = int(raw)
        targets[status] = min(base, len(rows_for_status))
        fractional.append((raw - base, status))

    remaining = sample_size - sum(targets.values())
    fractional.sort(reverse=True, key=lambda entry: entry[0])
    for _, status in fractional:
        if remaining <= 0:
            break
        if targets[status] < len(status_rows[status]):
            targets[status] += 1
            remaining -= 1

    rng = random.Random(seed)
    while remaining > 0:
        available = [
            status
            for status, rows_for_status in status_rows.items()
            if targets[status] < len(rows_for_status)
        ]
        if not available:
            break
        status = rng.choice(available)
        targets[status] += 1
        remaining -= 1

    sampled: list[dict[str, object]] = []
    for idx, status in enumerate((STATUS_PASS, STATUS_WARNING, STATUS_FAIL)):
        target = targets.get(status, 0)
        if target <= 0:
            continue
        sampled.extend(
            _sample_stratified_rows(
                status_rows[status],
                sample_size=target,
                seed=seed + (idx + 1) * 101,
            )
        )
    return sampled


def write_html_audit(
    index_rows: list[dict[str, object]],
    items_by_filing: dict[str, list[dict[str, object]]],
    out_dir: Path,
    scope_label: str,
    metadata: dict[str, object],
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    filings_dir = out_dir / "filings"
    filings_dir.mkdir(parents=True, exist_ok=True)

    style = """
    :root {
      --bg: #f6f7f9;
      --card: #ffffff;
      --text: #1f2a33;
      --muted: #5e6b76;
      --border: #d9e0e6;
      --accent: #2f6fad;
      --warn: #b06a00;
      --fail: #9b2226;
      --ok: #2d6a4f;
      --mono: "Consolas", "SFMono-Regular", Menlo, Monaco, "Liberation Mono", monospace;
    }
    * { box-sizing: border-box; }
    body {
      margin: 24px;
      background: var(--bg);
      color: var(--text);
      font-family: "Segoe UI", Tahoma, Arial, sans-serif;
    }
    h1, h2, h3 { margin: 0 0 12px 0; }
    a { color: var(--accent); text-decoration: none; }
    a:hover { text-decoration: underline; }
    .card {
      background: var(--card);
      border: 1px solid var(--border);
      border-radius: 10px;
      padding: 16px;
      margin-bottom: 16px;
      box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }
    .summary-grid {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
      gap: 12px;
      margin-top: 8px;
    }
    .summary-item {
      padding: 10px;
      border: 1px solid var(--border);
      border-radius: 8px;
      background: #fbfcfd;
      font-size: 13px;
    }
    .summary-item .label { color: var(--muted); display: block; font-size: 11px; }
    table {
      width: 100%;
      border-collapse: collapse;
      background: var(--card);
      border: 1px solid var(--border);
      border-radius: 10px;
      overflow: hidden;
    }
    th, td {
      padding: 10px 12px;
      border-bottom: 1px solid var(--border);
      text-align: left;
      font-size: 13px;
    }
    th { background: #f0f3f6; color: var(--muted); font-size: 12px; }
    tr:hover td { background: #f9fbfc; }
    .badge {
      display: inline-block;
      padding: 2px 8px;
      border-radius: 999px;
      font-size: 11px;
      text-transform: uppercase;
      letter-spacing: 0.4px;
      font-weight: 600;
    }
    .badge.ok { background: #d8f3dc; color: var(--ok); }
    .badge.warn { background: #ffead1; color: var(--warn); }
    .badge.fail { background: #f8d7da; color: var(--fail); }
    .mono { font-family: var(--mono); }
    .kv {
      width: 100%;
      border-collapse: collapse;
      margin: 8px 0 0;
      font-size: 13px;
    }
    .kv th, .kv td {
      border-bottom: 1px solid var(--border);
      padding: 6px 8px;
    }
    .kv th { width: 190px; color: var(--muted); font-weight: 600; }
    details {
      background: var(--card);
      border: 1px solid var(--border);
      border-radius: 10px;
      padding: 10px 12px;
      margin-bottom: 12px;
    }
    summary {
      cursor: pointer;
      font-weight: 600;
      font-size: 14px;
    }
    pre {
      background: #0f172a;
      color: #e2e8f0;
      padding: 10px;
      border-radius: 8px;
      overflow-x: auto;
      font-family: var(--mono);
      font-size: 12px;
      white-space: pre-wrap;
    }
    .section-title { margin-top: 18px; font-size: 15px; color: var(--muted); }
    .toolbar { display: flex; gap: 12px; align-items: center; margin-bottom: 16px; }
    .muted { color: var(--muted); font-size: 12px; }
    """.strip()

    def _badge(value: object, *, fail: bool = False) -> str:
        if _parse_bool(value):
            return f"<span class=\"badge {'fail' if fail else 'warn'}\">yes</span>"
        return "<span class=\"badge ok\">no</span>"

    def _render_kv(rows: list[tuple[str, object]]) -> str:
        lines = ["<table class=\"kv\">"]
        for key, val in rows:
            lines.append(
                f"<tr><th>{_html_escape(key)}</th><td>{_html_escape(val)}</td></tr>"
            )
        lines.append("</table>")
        return "\n".join(lines)

    def _item_sort_key(row: dict[str, object]) -> tuple[int, int, str, int]:
        part = _part_rank(str(row.get("item_part") or ""))
        item_num, item_letter = _item_id_sort_key(str(row.get("item_id") or ""))
        content_start = _parse_int(row.get("content_start"), default=0)
        return (part, item_num, item_letter, content_start)

    index_lines = [
        "<!doctype html>",
        "<html lang=\"en\">",
        "<head>",
        "  <meta charset=\"utf-8\">",
        "  <meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">",
        "  <title>10-K Extraction Manual Review</title>",
        f"  <style>{style}</style>",
        "</head>",
        "<body>",
        "  <div class=\"card\">",
        "    <h1>10-K Extraction Manual Review</h1>",
        "    <div class=\"summary-grid\">",
    ]

    summary_fields = [
        ("Pass definition", metadata.get("pass_definition", "")),
        ("Total filings", metadata.get("total_filings", "")),
        ("Total items", metadata.get("total_items", "")),
        ("Pass filings", metadata.get("total_pass_filings", "")),
        ("Warning filings", metadata.get("total_warning_filings", "")),
        ("Fail filings", metadata.get("total_fail_filings", "")),
        ("Scope", scope_label),
        ("Sample size", metadata.get("sample_size", "")),
        ("Sample weights", metadata.get("sample_weights", "")),
        ("Generated at", metadata.get("generated_at", "")),
        ("Offset basis", metadata.get("offset_basis", "")),
    ]
    for label, value in summary_fields:
        index_lines.append(
            "      <div class=\"summary-item\">"
            f"<span class=\"label\">{_html_escape(label)}</span>"
            f"{_html_escape(value)}</div>"
        )
    index_lines.extend(
        [
            "    </div>",
            "  </div>",
            "  <table>",
            "    <thead>",
            "      <tr>",
            "        <th>doc_id</th>",
            "        <th>accession</th>",
            "        <th>form</th>",
            "        <th>filing_date</th>",
            "        <th>n_items_extracted</th>",
            "        <th>status</th>",
            "        <th>any_warn</th>",
            "        <th>missing_core_items</th>",
            "        <th>filing_exclusion_reason</th>",
            "      </tr>",
            "    </thead>",
            "    <tbody>",
        ]
    )

    for row in index_rows:
        doc_id = str(row.get("doc_id") or "")
        accession = str(row.get("accession") or "")
        filename = _filing_filename(doc_id, accession)
        status = classify_filing_status(row)
        index_lines.append("      <tr>")
        index_lines.append(
            f"        <td><a href=\"filings/{filename}\">{_html_escape(doc_id)}</a></td>"
        )
        index_lines.append(f"        <td>{_html_escape(accession)}</td>")
        index_lines.append(f"        <td>{_html_escape(row.get('form',''))}</td>")
        index_lines.append(f"        <td>{_html_escape(row.get('filing_date',''))}</td>")
        index_lines.append(
            f"        <td>{_html_escape(row.get('n_items_extracted',''))}</td>"
        )
        index_lines.append(f"        <td>{_status_badge(status)}</td>")
        index_lines.append(f"        <td>{_badge(row.get('any_warn'))}</td>")
        index_lines.append(
            f"        <td>{_html_escape(row.get('missing_core_items',''))}</td>"
        )
        index_lines.append(
            f"        <td>{_html_escape(row.get('filing_exclusion_reason',''))}</td>"
        )
        index_lines.append("      </tr>")

    index_lines.extend(["    </tbody>", "  </table>", "</body>", "</html>"])
    (out_dir / "index.html").write_text("\n".join(index_lines), encoding="utf-8")

    for row in index_rows:
        doc_id = str(row.get("doc_id") or "")
        accession = str(row.get("accession") or "")
        filing_items = items_by_filing.get(doc_id, [])
        filing_items = sorted(filing_items, key=_item_sort_key)
        filename = _filing_filename(doc_id, accession)
        status = classify_filing_status(row)
        file_lines = [
            "<!doctype html>",
            "<html lang=\"en\">",
            "<head>",
            "  <meta charset=\"utf-8\">",
            "  <meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">",
            f"  <title>{_html_escape(doc_id)} manual review</title>",
            f"  <style>{style}</style>",
            "</head>",
            "<body>",
            "  <div class=\"toolbar\">",
            "    <a href=\"../index.html\">Back to index</a>",
            f"    <span class=\"muted\">{_html_escape(doc_id)}</span>",
            "  </div>",
            "  <div class=\"card\">",
            f"    <h2>{_html_escape(doc_id)}</h2>",
            f"    <div class=\"muted\">Accession: {_html_escape(accession)}</div>",
            "    <div class=\"summary-grid\">",
        ]
        header_fields = [
            ("Status", _status_label(status)),
            ("Form", row.get("form", "")),
            ("Filing date", row.get("filing_date", "")),
            ("Period end", row.get("period_end", "")),
            ("CIK", row.get("cik", "")),
            ("Items extracted", row.get("items_extracted", "")),
            ("Missing core items", row.get("missing_core_items", "")),
            ("Any warn", "yes" if _parse_bool(row.get("any_warn")) else "no"),
            ("Any fail", "yes" if _parse_bool(row.get("any_fail")) else "no"),
            ("Filing exclusion reason", row.get("filing_exclusion_reason", "")),
            ("Item count", row.get("n_items_extracted", "")),
        ]
        for label, value in header_fields:
            file_lines.append(
                "      <div class=\"summary-item\">"
                f"<span class=\"label\">{_html_escape(label)}</span>"
                f"{_html_escape(value)}</div>"
            )
        file_lines.extend(["    </div>", "  </div>"])

        filing_text_asset = str(row.get("filing_text_asset") or "")
        filing_text_preview = str(row.get("filing_text_preview") or "")
        if filing_text_asset or filing_text_preview:
            file_lines.append(
                "<div class=\"section-title\">Filing text (normalized extractor body)</div>"
            )
            if filing_text_asset:
                asset_link = _asset_link(filing_text_asset, depth=1)
                file_lines.append(
                    f"<div class=\"muted\"><a href=\"{asset_link}\">Download full text</a></div>"
                )
            if filing_text_preview:
                file_lines.append(f"<pre>{_html_escape(filing_text_preview)}</pre>")

        if not filing_items:
            file_lines.append("<div class=\"card\">No items found for this filing.</div>")
        else:
            file_lines.append(
                f"<div class=\"section-title\">Items ({len(filing_items)})</div>"
            )
            for item in filing_items:
                item_part = str(item.get("item_part") or "").strip()
                item_id = str(item.get("item_id") or "").strip()
                item_label = " ".join([part for part in [item_part, item_id] if part])
                item_status = str(item.get("item_status") or "")
                length_chars = str(item.get("length_chars") or "")
                item_title = str(item.get("item") or "")
                summary = f"{item_label} - {item_title}".strip(" -")
                if item_status:
                    summary += f" ({item_status})"
                if length_chars:
                    summary += f" - {length_chars} chars"

                file_lines.append("<details>")
                file_lines.append(f"  <summary>{_html_escape(summary)}</summary>")

                file_lines.append(
                    _render_kv(
                        [
                            ("item_part", item_part),
                            ("item_id", item_id),
                            ("item", item_title),
                            ("item_status", item_status),
                            ("length_chars", length_chars),
                            ("heading_start", item.get("heading_start", "")),
                            ("heading_end", item.get("heading_end", "")),
                            ("content_start", item.get("content_start", "")),
                            ("content_end", item.get("content_end", "")),
                            ("heading_line_clean", item.get("heading_line_clean", "")),
                            ("heading_line_raw", item.get("heading_line_raw", "")),
                        ]
                    )
                )

                embedded_fields = [
                    ("embedded_heading_warn", item.get("embedded_heading_warn", "")),
                    ("embedded_heading_fail", item.get("embedded_heading_fail", "")),
                    ("first_embedded_kind", item.get("first_embedded_kind", "")),
                    (
                        "first_embedded_classification",
                        item.get("first_embedded_classification", ""),
                    ),
                    ("first_embedded_item_id", item.get("first_embedded_item_id", "")),
                    ("first_embedded_part", item.get("first_embedded_part", "")),
                    ("first_embedded_line_idx", item.get("first_embedded_line_idx", "")),
                    ("first_embedded_char_pos", item.get("first_embedded_char_pos", "")),
                    ("first_embedded_snippet", item.get("first_embedded_snippet", "")),
                    ("first_fail_kind", item.get("first_fail_kind", "")),
                    ("first_fail_classification", item.get("first_fail_classification", "")),
                    ("first_fail_item_id", item.get("first_fail_item_id", "")),
                    ("first_fail_part", item.get("first_fail_part", "")),
                    ("first_fail_line_idx", item.get("first_fail_line_idx", "")),
                    ("first_fail_char_pos", item.get("first_fail_char_pos", "")),
                    ("first_fail_snippet", item.get("first_fail_snippet", "")),
                ]
                if any(str(value).strip() for _, value in embedded_fields):
                    file_lines.append(
                        "<div class=\"section-title\">Embedded flags</div>"
                    )
                    file_lines.append(_render_kv(embedded_fields))

                item_text_asset = str(item.get("item_text_asset") or "")
                item_text_preview = str(item.get("item_text_preview") or "")
                if item_text_asset or item_text_preview:
                    file_lines.append(
                        "<div class=\"section-title\">Item text (normalized extractor body)</div>"
                    )
                    if item_text_asset:
                        asset_link = _asset_link(item_text_asset, depth=1)
                        file_lines.append(
                            f"<div class=\"muted\"><a href=\"{asset_link}\">Download item text</a></div>"
                        )
                    if item_text_preview:
                        file_lines.append(f"<pre>{_html_escape(item_text_preview)}</pre>")

                doc_head = str(item.get("doc_head_200") or "")
                doc_tail = str(item.get("doc_tail_200") or "")
                file_lines.append(
                    "<div class=\"section-title\">doc_head_200</div>"
                )
                file_lines.append(f"<pre>{_html_escape(doc_head)}</pre>")
                file_lines.append(
                    "<div class=\"section-title\">doc_tail_200</div>"
                )
                file_lines.append(f"<pre>{_html_escape(doc_tail)}</pre>")
                file_lines.append("</details>")

        file_lines.extend(["</body>", "</html>"])
        (filings_dir / filename).write_text("\n".join(file_lines), encoding="utf-8")


__all__ = [
    "DEFAULT_SAMPLE_WEIGHTS",
    "STATUS_FAIL",
    "STATUS_PASS",
    "STATUS_WARNING",
    "classify_filing_status",
    "normalize_extractor_body",
    "normalize_sample_weights",
    "sample_filings_by_status",
    "write_html_audit",
]
