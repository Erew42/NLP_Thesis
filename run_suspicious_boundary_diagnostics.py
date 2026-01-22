from __future__ import annotations

import argparse
import csv
import re
import sys
from collections import Counter, defaultdict
from datetime import date, datetime
from pathlib import Path

import polars as pl
import pyarrow as pa
import pyarrow.parquet as pq

# Make local `src` importable when running from repo checkout
ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if SRC.exists():
    sys.path.insert(0, str(SRC))

from thesis_pkg.core.sec.filing_text import (  # noqa: E402
    ITEM_LINESTART_PATTERN,
    TOC_DOT_LEADER_PATTERN,
    _default_part_for_item_id,
    _looks_like_toc_heading_line,
    extract_filing_items,
    parse_header,
)


DEFAULT_PARQUET_DIR = Path(
    r"C:\Users\erik9\Documents\SEC_Data\Data\Sample_Filings\parquet_batches"
)

DEFAULT_OUT_PATH = Path("results/suspicious_boundaries_v3_pre.csv")
DEFAULT_REPORT_PATH = Path("results/suspicious_boundaries_report_v3_pre.txt")
DEFAULT_SAMPLES_DIR = Path("results/Suspicious_Filings_Demo")


def _parse_date(value: str | date | datetime | None) -> date | None:
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
        if len(s) == 8 and s.isdigit():
            try:
                return datetime.strptime(s, "%Y%m%d").date()
            except Exception:
                return None
        if len(s) == 10 and s[4] == "-" and s[7] == "-":
            try:
                return datetime.strptime(s, "%Y-%m-%d").date()
            except Exception:
                return None
    return None


def _is_10k(form_type: str | None) -> bool:
    form = (form_type or "").upper().strip()
    return form.startswith("10-K") or form.startswith("10K")


def _non_empty_line(lines: list[str], start: int, *, max_scan: int = 4) -> tuple[int | None, str | None]:
    idx = start
    while idx < len(lines) and idx < start + max_scan:
        if lines[idx].strip():
            return idx, lines[idx]
        idx += 1
    return None, None


def _lettered_item(item_id: str) -> bool:
    return bool(re.search(r"\d+[A-Z]$", item_id))


def _safe_slug(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_-]+", "_", value).strip("_") or "unknown"


def _weak_heading_letter(item_id: str, heading_line: str) -> bool:
    if not _lettered_item(item_id):
        return False
    m = ITEM_LINESTART_PATTERN.match(heading_line or "")
    if not m:
        return True
    detected = f"{m.group('num')}{m.group('let') or ''}".upper()
    return not detected.endswith(item_id[-1])


def _expected_part_for_item(item: dict) -> str | None:
    canonical = item.get("canonical_item")
    if isinstance(canonical, str) and ":" in canonical:
        part = canonical.split(":", 1)[0].upper()
        if part in {"I", "II", "III", "IV"}:
            return part
    item_key = item.get("item")
    if isinstance(item_key, str) and ":" in item_key:
        part = item_key.split(":", 1)[0].upper()
        if part in {"I", "II", "III", "IV"}:
            return part
    item_id = item.get("item_id")
    if isinstance(item_id, str):
        return _default_part_for_item_id(item_id)
    return None


def _looks_like_cross_ref(prefix: str) -> bool:
    if not prefix.strip():
        return False
    if re.search(r"(?i)\bsee\b|\brefer\b|\bas discussed\b|\bunder\b", prefix):
        return True
    return bool(re.search(r"[A-Za-z0-9]", prefix))


def _format_suspicious_items_line(items: list[dict], *, max_items: int = 12) -> str:
    parts: list[str] = []
    for entry in items:
        flags = entry.get("flags") or []
        item_key = entry.get("item") or "UNKNOWN"
        if flags:
            parts.append(f"{item_key} ({','.join(flags)})")
        else:
            parts.append(item_key)
    trimmed = parts[:max_items]
    remainder = len(parts) - len(trimmed)
    line = "SUSPICIOUS_ITEMS: " + "; ".join(trimmed)
    if remainder > 0:
        line += f"; ... +{remainder} more"
    return line


def _write_sample_file(
    path: Path,
    *,
    filing_meta: dict[str, str],
    flagged_items: list[dict],
    full_text: str,
) -> None:
    lines: list[str] = []
    lines.append(_format_suspicious_items_line(flagged_items))
    lines.append("")
    lines.append(
        "FILING_META: "
        f"doc_id={filing_meta.get('doc_id','')} "
        f"cik={filing_meta.get('cik','')} "
        f"accession={filing_meta.get('accession','')} "
        f"form={filing_meta.get('form_type','')} "
        f"filing_date={filing_meta.get('filing_date','')} "
        f"period_end={filing_meta.get('period_end','')}"
    )
    lines.append("")
    lines.append("EXTRACTED_ITEMS:")
    lines.append("")
    for entry in flagged_items:
        item_key = entry.get("item") or "UNKNOWN"
        flags = entry.get("flags") or []
        lines.append(f"---- ITEM {item_key} ({','.join(flags)}) ----")
        heading = entry.get("heading_line") or ""
        lines.append(f"heading_line: {heading}")
        lines.append(f"heading_index: {entry.get('heading_index')}")
        lines.append(f"heading_offset: {entry.get('heading_offset')}")
        lines.append("")
        item_text = entry.get("full_text") or ""
        lines.append(item_text.strip())
        lines.append("")
    lines.append("FULL_FILING_TEXT:")
    lines.append("")
    lines.append(full_text or "")
    path.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Flag suspicious item boundaries in 10-K extraction output."
    )
    parser.add_argument(
        "--parquet-dir",
        type=Path,
        default=DEFAULT_PARQUET_DIR,
        help="Directory containing parquet batch files.",
    )
    parser.add_argument(
        "--out-path",
        type=Path,
        default=DEFAULT_OUT_PATH,
        help="CSV output path for suspicious boundary rows.",
    )
    parser.add_argument(
        "--report-path",
        type=Path,
        default=DEFAULT_REPORT_PATH,
        help="Text report output path.",
    )
    parser.add_argument(
        "--samples-dir",
        type=Path,
        default=DEFAULT_SAMPLES_DIR,
        help="Directory for per-filing sample text files.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Parquet batch size for scanning filings.",
    )
    parser.add_argument(
        "--max-files",
        type=int,
        default=0,
        help="Optional cap on number of parquet files to scan (0 = no cap).",
    )
    parser.add_argument(
        "--max-examples",
        type=int,
        default=25,
        help="Max examples to include in the report.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    parquet_dir = args.parquet_dir
    if not parquet_dir.exists():
        raise SystemExit(f"parquet-dir not found: {parquet_dir}")

    files = sorted(parquet_dir.glob("*_batch_*.parquet"))
    if args.max_files and args.max_files > 0:
        files = files[: args.max_files]
    if not files:
        raise SystemExit(f"No parquet batch files found in {parquet_dir}")

    args.out_path.parent.mkdir(parents=True, exist_ok=True)
    args.report_path.parent.mkdir(parents=True, exist_ok=True)
    args.samples_dir.mkdir(parents=True, exist_ok=True)

    out_headers = [
        "doc_id",
        "cik",
        "accession",
        "filing_date",
        "period_end",
        "item_part",
        "item_id",
        "item",
        "canonical_item",
        "exists_by_regime",
        "item_status",
        "heading_line",
        "heading_index",
        "heading_offset",
        "prev_line",
        "next_line",
        "flags",
    ]

    total_filings = 0
    total_items = 0
    total_flagged = 0
    flags_count = Counter()
    examples_by_flag: dict[str, list[dict[str, str]]] = defaultdict(list)

    want_cols = [
        "doc_id",
        "cik",
        "accession_number",
        "document_type_filename",
        "file_date_filename",
        "full_text",
    ]

    with args.out_path.open("w", newline="", encoding="utf-8") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=out_headers)
        writer.writeheader()

        for file_path in files:
            pf = pq.ParquetFile(file_path)
            available = set(pf.schema.names)
            columns = [c for c in want_cols if c in available]
            for batch in pf.iter_batches(batch_size=args.batch_size, columns=columns):
                tbl = pa.Table.from_batches([batch])
                df = pl.from_arrow(tbl)
                for row in df.iter_rows(named=True):
                    form_type = row.get("document_type_filename")
                    if not _is_10k(form_type):
                        continue

                    total_filings += 1
                    text = row.get("full_text") or ""
                    if not text:
                        continue

                    hdr = parse_header(text)
                    filing_date = _parse_date(hdr.get("header_filing_date_str")) or _parse_date(
                        row.get("file_date_filename")
                    )
                    period_end = _parse_date(hdr.get("header_period_end_str"))

                    items = extract_filing_items(
                        text,
                        form_type=form_type,
                        filing_date=filing_date,
                        period_end=period_end,
                        regime=True,
                        diagnostics=True,
                    )
                    if not items:
                        continue

                    lines = text.splitlines()
                    total_items += len(items)
                    flagged_items_for_doc: list[dict[str, str | int | None]] = []

                    for item in items:
                        heading_line = item.get("_heading_line") or ""
                        heading_idx = item.get("_heading_line_index")
                        heading_offset = item.get("_heading_offset")
                        item_id = str(item.get("item_id") or "")
                        item_part = item.get("item_part")

                        flags: list[str] = []
                        if heading_idx is None or heading_idx < 0:
                            flags.append("missing_heading_line")
                        else:
                            if _looks_like_toc_heading_line(lines, heading_idx):
                                flags.append("toc_like_heading")
                            if heading_offset and _looks_like_cross_ref(
                                heading_line[: heading_offset]
                            ):
                                flags.append("cross_ref_prefix")

                        if heading_offset and heading_offset > 0:
                            flags.append("midline_heading")

                        if _weak_heading_letter(item_id, heading_line):
                            flags.append("weak_letter_heading")

                        if heading_idx is not None and heading_idx >= 0:
                            next_idx, next_line = _non_empty_line(lines, heading_idx + 1)
                            if next_line:
                                if re.match(r"^\s*[ABC]\s*[\.\):\-]", next_line):
                                    if item_id in {"1", "7", "9"} or _lettered_item(item_id):
                                        flags.append("split_letter_line")

                        expected_part = _expected_part_for_item(item)
                        if expected_part and not item_part:
                            flags.append("missing_part")
                        if item_part and expected_part and item_part != expected_part:
                            flags.append("part_mismatch")

                        if TOC_DOT_LEADER_PATTERN.search(heading_line):
                            flags.append("dot_leader_heading")

                        if not flags:
                            continue

                        flags = sorted(set(flags))
                        total_flagged += 1
                        for flag in flags:
                            flags_count[flag] += 1

                        prev_line = ""
                        next_line = ""
                        if heading_idx is not None and heading_idx >= 0:
                            if heading_idx > 0:
                                prev_line = lines[heading_idx - 1].strip()
                            if heading_idx + 1 < len(lines):
                                next_line = lines[heading_idx + 1].strip()

                        flagged_items_for_doc.append(
                            {
                                "item": item.get("item") or "",
                                "item_part": item_part or "",
                                "item_id": item_id,
                                "flags": flags,
                                "heading_line": heading_line.strip(),
                                "heading_index": heading_idx if heading_idx is not None else "",
                                "heading_offset": heading_offset if heading_offset is not None else "",
                                "full_text": item.get("full_text") or "",
                            }
                        )

                        writer.writerow(
                            {
                                "doc_id": row.get("doc_id") or "",
                                "cik": row.get("cik") or "",
                                "accession": row.get("accession_number") or "",
                                "filing_date": filing_date.isoformat() if filing_date else "",
                                "period_end": period_end.isoformat() if period_end else "",
                                "item_part": item_part or "",
                                "item_id": item_id,
                                "item": item.get("item") or "",
                                "canonical_item": item.get("canonical_item") or "",
                                "exists_by_regime": item.get("exists_by_regime"),
                                "item_status": item.get("item_status") or "",
                                "heading_line": heading_line.strip(),
                                "heading_index": heading_idx if heading_idx is not None else "",
                                "heading_offset": heading_offset if heading_offset is not None else "",
                                "prev_line": prev_line,
                                "next_line": next_line,
                                "flags": ";".join(flags),
                            }
                        )

                        for flag in flags:
                            if len(examples_by_flag[flag]) < 3:
                                examples_by_flag[flag].append(
                                    {
                                        "doc_id": str(row.get("doc_id") or ""),
                                        "cik": str(row.get("cik") or ""),
                                        "accession": str(row.get("accession_number") or ""),
                                        "item_id": item_id,
                                        "item_part": str(item_part or ""),
                                        "heading": heading_line.strip(),
                                        "prev": prev_line,
                                        "next": next_line,
                                    }
                                )

                    if flagged_items_for_doc:
                        doc_id = str(row.get("doc_id") or "")
                        accession = str(row.get("accession_number") or "")
                        safe_id = _safe_slug(doc_id or accession)
                        sample_path = args.samples_dir / f"{safe_id}.txt"
                        _write_sample_file(
                            sample_path,
                            filing_meta={
                                "doc_id": doc_id,
                                "cik": str(row.get("cik") or ""),
                                "accession": accession,
                                "form_type": str(form_type or ""),
                                "filing_date": filing_date.isoformat() if filing_date else "",
                                "period_end": period_end.isoformat() if period_end else "",
                            },
                            flagged_items=flagged_items_for_doc,
                            full_text=text,
                        )

    lines_out: list[str] = []
    lines_out.append("Suspicious boundary diagnostics (v3)")
    lines_out.append(f"Parquet dir: {parquet_dir}")
    lines_out.append(f"Total filings processed: {total_filings}")
    lines_out.append(f"Total items extracted: {total_items}")
    lines_out.append(f"Total flagged items: {total_flagged}")
    lines_out.append("")
    lines_out.append("Flags (counts):")
    for flag, count in flags_count.most_common():
        lines_out.append(f"  {flag}: {count}")
    lines_out.append("")
    lines_out.append("Examples by flag:")
    if examples_by_flag:
        for flag, examples in examples_by_flag.items():
            lines_out.append(f"  {flag}:")
            for ex in examples:
                lines_out.append(
                    "    "
                    f"doc_id={ex['doc_id']} cik={ex['cik']} accession={ex['accession']} "
                    f"part={ex['item_part']} item_id={ex['item_id']} "
                    f"heading=\"{ex['heading']}\""
                )
                if ex["prev"]:
                    lines_out.append(f"      prev=\"{ex['prev']}\"")
                if ex["next"]:
                    lines_out.append(f"      next=\"{ex['next']}\"")
    else:
        lines_out.append("  (no flagged examples captured)")

    report = "\n".join(lines_out)
    print(report)
    args.report_path.write_text(report, encoding="utf-8")
    print(f"\nReport written to {args.report_path}")


if __name__ == "__main__":
    main()
