from __future__ import annotations

import html
import re
from pathlib import Path

from .extraction import _strip_edgar_metadata
from .heuristics import _repair_wrapped_headings
from .utilities import _normalize_newlines


def normalize_extractor_body(text: str) -> str:
    return _repair_wrapped_headings(_strip_edgar_metadata(_normalize_newlines(text)))


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
        ("Scope", scope_label),
        ("Sample size", metadata.get("sample_size", "")),
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


__all__ = ["normalize_extractor_body", "write_html_audit"]
