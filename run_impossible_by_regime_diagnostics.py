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
    PART_LINESTART_PATTERN,
    TOC_DOT_LEADER_PATTERN,
    _normalize_item_id,
    _pageish_line,
    extract_filing_items,
    parse_header,
)


DEFAULT_PARQUET_DIR = Path(
    r"C:\Users\erik9\Documents\SEC_Data\Data\Sample_Filings\parquet_batches"
)

DEFAULT_ITEMS_DIR = Path("results/impossible_by_regime_items_v2")
DEFAULT_FILINGS_DIR = Path("results/impossible_by_regime_filings_v2")
DEFAULT_MANIFEST_PATH = Path("results/impossible_by_regime_manifest_v2.csv")
DEFAULT_REPORT_PATH = Path("results/impossible_by_regime_diagnostics_v2.txt")

EARLY_ADOPTION_DATES = {
    "1A": date(2005, 12, 1),
    "1B": date(2005, 12, 1),
    "1C": date(2023, 12, 15),  # fiscal-year-end trigger
    "9C": date(2022, 1, 10),
    "16": date(2016, 6, 9),
}


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


def _safe_slug(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_-]+", "_", value).strip("_")


def _looks_like_toc(lines: list[str], idx: int) -> bool:
    line = lines[idx]
    if TOC_DOT_LEADER_PATTERN.search(line) or re.search(r"\s+\d{1,4}\s*$", line):
        return True
    j = idx + 1
    max_scan = min(len(lines), idx + 5)
    while j < max_scan and lines[j].strip() == "":
        j += 1
    if j < len(lines) and _pageish_line(lines[j]) and len(line.strip()) <= 160:
        return True
    return False


def _is_glued_heading(line: str) -> bool:
    m = re.search(r"(?i)\bITEM\s+\d{1,2}[A-Z]?", line)
    if not m:
        return False
    end = m.end()
    if end >= len(line):
        return False
    return line[end].isalpha()


def _heading_evidence(line: str | None, lines: list[str], idx: int | None) -> str:
    if not line:
        return "missing_heading"
    if re.search(r"(?i)\bITEM\s+\d{1,2}\s*\([A-Z0-9]\)", line):
        return "subitem"
    if _is_glued_heading(line):
        return "glued_heading"
    if idx is not None and _looks_like_toc(lines, idx):
        return "toc_line"
    if re.search(r"(?i)^\s*(see|refer to|as discussed)\b", line.strip()):
        return "cross_ref"
    return "explicit_heading"


def _category_for_item(
    *,
    item_id: str,
    evidence: str,
    filing_date: date | None,
    period_end: date | None,
) -> str:
    if evidence in {"missing_heading", "toc_line", "glued_heading", "subitem", "cross_ref"}:
        return "A"
    threshold = EARLY_ADOPTION_DATES.get(item_id)
    if threshold is None:
        return "B"
    if item_id == "1C":
        compare_date = period_end or filing_date
    else:
        compare_date = filing_date or period_end
    if compare_date and compare_date < threshold:
        return "C"
    return "B"


def _format_date(value: date | None) -> str:
    return value.isoformat() if value else "NA"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract impossible-by-regime items and produce diagnostics."
    )
    parser.add_argument(
        "--parquet-dir",
        type=Path,
        default=DEFAULT_PARQUET_DIR,
        help="Directory containing parquet batch files.",
    )
    parser.add_argument(
        "--items-dir",
        type=Path,
        default=DEFAULT_ITEMS_DIR,
        help="Directory to write extracted item text files.",
    )
    parser.add_argument(
        "--filings-dir",
        type=Path,
        default=DEFAULT_FILINGS_DIR,
        help="Directory to write extracted filing text files.",
    )
    parser.add_argument(
        "--manifest-path",
        type=Path,
        default=DEFAULT_MANIFEST_PATH,
        help="CSV manifest path for extracted items.",
    )
    parser.add_argument(
        "--report-path",
        type=Path,
        default=DEFAULT_REPORT_PATH,
        help="Diagnostics report output path.",
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
        help="Max examples to include in the diagnostics report.",
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

    args.items_dir.mkdir(parents=True, exist_ok=True)
    args.filings_dir.mkdir(parents=True, exist_ok=True)
    args.manifest_path.parent.mkdir(parents=True, exist_ok=True)
    args.report_path.parent.mkdir(parents=True, exist_ok=True)

    manifest_headers = [
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
        "category",
        "evidence_type",
        "heading_part",
        "heading",
        "item_file",
        "filing_file",
    ]

    total_filings = 0
    total_impossible = 0
    items_by_id = Counter()
    items_by_part_year = Counter()
    category_counts = Counter()
    evidence_counts = Counter()
    examples: list[dict[str, str]] = []
    examples_per_item: dict[str, int] = defaultdict(int)

    written_filings: set[str] = set()
    item_counts_by_doc: dict[str, int] = defaultdict(int)

    want_cols = [
        "doc_id",
        "cik",
        "accession_number",
        "document_type_filename",
        "file_date_filename",
        "full_text",
    ]

    with args.manifest_path.open("w", newline="", encoding="utf-8") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=manifest_headers)
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
                    )
                    if not items:
                        continue

                    lines = text.splitlines()
                    heading_map: dict[str, list[tuple[int, str | None, str]]] = defaultdict(list)
                    current_part = None
                    for idx, line in enumerate(lines):
                        m_part = PART_LINESTART_PATTERN.match(line)
                        if m_part is not None:
                            current_part = m_part.group("part").upper()
                        m = ITEM_LINESTART_PATTERN.match(line)
                        if not m:
                            continue
                        item_id = _normalize_item_id(m.group("num"), m.group("let"))
                        if not item_id:
                            continue
                        heading_map[item_id].append((idx, current_part, line))

                    for item in items:
                        if item.get("exists_by_regime") is not False:
                            continue

                        total_impossible += 1
                        item_id = str(item.get("item_id") or "")
                        item_part = item.get("item_part")
                        item_key = str(item.get("item") or "")

                        items_by_id[item_id] += 1
                        year = None
                        ref_date = filing_date or period_end
                        if ref_date is not None:
                            year = ref_date.year
                        items_by_part_year[(item_part or "NONE", year, item_id)] += 1

                        heading_idx = None
                        heading_part = None
                        heading_line = None
                        candidates = heading_map.get(item_id) or []
                        if candidates:
                            heading_idx, heading_part, heading_line = candidates[0]
                        heading_snippet = ""
                        if heading_line:
                            heading_snippet = re.sub(r"\s+", " ", heading_line.strip())
                            if len(heading_snippet) > 220:
                                heading_snippet = heading_snippet[:220] + "..."

                        evidence = _heading_evidence(heading_line, lines, heading_idx)
                        category = _category_for_item(
                            item_id=item_id,
                            evidence=evidence,
                            filing_date=filing_date,
                            period_end=period_end,
                        )

                        category_counts[category] += 1
                        evidence_counts[evidence] += 1

                        if len(examples) < args.max_examples and examples_per_item[item_id] < 3:
                            examples.append(
                                {
                                    "category": category,
                                    "evidence": evidence,
                                    "item_id": item_id,
                                    "item_part": str(item_part or ""),
                                    "year": str(year) if year is not None else "UNK",
                                    "doc_id": str(row.get("doc_id") or ""),
                                    "cik": str(row.get("cik") or ""),
                                    "accession": str(row.get("accession_number") or ""),
                                    "heading": heading_snippet,
                                }
                            )
                            examples_per_item[item_id] += 1

                        doc_id = str(row.get("doc_id") or "")
                        accession = str(row.get("accession_number") or "")
                        safe_doc = _safe_slug(doc_id or accession or "unknown")
                        item_counts_by_doc[safe_doc] += 1
                        item_index = item_counts_by_doc[safe_doc]
                        item_file = (
                            f"{safe_doc}_{item_part or 'NONE'}_{item_id}_{item_index}.txt"
                        )
                        item_path = args.items_dir / item_file

                        item_header = [
                            f"doc_id: {doc_id}",
                            f"cik: {row.get('cik') or ''}",
                            f"accession: {accession}",
                            f"filing_date: {_format_date(filing_date)}",
                            f"period_end: {_format_date(period_end)}",
                            f"item_part: {item_part or ''}",
                            f"item_id: {item_id}",
                            f"item: {item_key}",
                            f"canonical_item: {item.get('canonical_item') or ''}",
                            f"exists_by_regime: {item.get('exists_by_regime')}",
                            f"item_status: {item.get('item_status') or ''}",
                            f"category: {category}",
                            f"evidence_type: {evidence}",
                            f"heading_part: {heading_part or ''}",
                            f"heading: {heading_snippet}",
                            "",
                        ]
                        item_body = (item.get("full_text") or "").strip()
                        item_path.write_text(
                            "\n".join(item_header) + item_body,
                            encoding="utf-8",
                        )

                        filing_file = f"{safe_doc}.txt"
                        filing_path = args.filings_dir / filing_file
                        if filing_file not in written_filings:
                            filing_path.write_text(text, encoding="utf-8")
                            written_filings.add(filing_file)

                        writer.writerow(
                            {
                                "doc_id": doc_id,
                                "cik": row.get("cik") or "",
                                "accession": accession,
                                "filing_date": _format_date(filing_date),
                                "period_end": _format_date(period_end),
                                "item_part": item_part or "",
                                "item_id": item_id,
                                "item": item_key,
                                "canonical_item": item.get("canonical_item") or "",
                                "exists_by_regime": item.get("exists_by_regime"),
                                "item_status": item.get("item_status") or "",
                                "category": category,
                                "evidence_type": evidence,
                                "heading_part": heading_part or "",
                                "heading": heading_snippet,
                                "item_file": str(item_path),
                                "filing_file": str(filing_path),
                            }
                        )

    lines: list[str] = []
    lines.append("Impossible-by-regime diagnostics (v2)")
    lines.append(f"Parquet dir: {parquet_dir}")
    lines.append(f"Total filings processed: {total_filings}")
    lines.append(f"Total impossible items: {total_impossible}")
    lines.append("")
    lines.append("By item_id:")
    for item_id, count in items_by_id.most_common():
        lines.append(f"  item_id {item_id}: {count}")
    lines.append("")
    lines.append("By item_part/year/item_id (top 30):")
    for (part, year, item_id), count in sorted(
        items_by_part_year.items(),
        key=lambda kv: (kv[0][1] or 0, kv[0][0], kv[0][2]),
    )[:30]:
        year_str = str(year) if year is not None else "UNK"
        lines.append(f"  {year_str} part={part} item_id={item_id}: {count}")
    lines.append("")
    lines.append("By category:")
    for category, count in category_counts.most_common():
        lines.append(f"  {category}: {count}")
    lines.append("")
    lines.append("By evidence type:")
    for evidence, count in evidence_counts.most_common():
        lines.append(f"  {evidence}: {count}")
    lines.append("")
    lines.append("Examples:")
    if examples:
        for ex in examples:
            lines.append(
                "  "
                f"category={ex['category']} evidence={ex['evidence']} "
                f"item_id={ex['item_id']} part={ex['item_part']} year={ex['year']} "
                f"doc_id={ex['doc_id']} cik={ex['cik']} accession={ex['accession']} "
                f"heading=\"{ex['heading']}\""
            )
    else:
        lines.append("  (no examples captured)")

    report = "\n".join(lines)
    print(report)
    args.report_path.write_text(report, encoding="utf-8")
    print(f"\nReport written to {args.report_path}")


if __name__ == "__main__":
    main()
