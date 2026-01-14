from __future__ import annotations

import argparse
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
    _annotate_items_with_regime,
    extract_filing_items,
    parse_header,
    ITEM_LINESTART_PATTERN,
    PART_LINESTART_PATTERN,
    _normalize_item_id,
)


DEFAULT_PARQUET_DIR = Path(
    r"C:\Users\erik9\Documents\SEC_Data\Data\Sample_Filings\parquet_batches"
)


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


def _format_date(value: date | None) -> str:
    return value.isoformat() if value else "NA"


def _find_item14_headings(text: str, max_samples: int = 2) -> list[tuple[str | None, str]]:
    lines = text.splitlines()
    current_part = None
    found: list[tuple[str | None, str]] = []
    for line in lines:
        m_part = PART_LINESTART_PATTERN.match(line)
        if m_part is not None:
            current_part = m_part.group("part").upper()
        m = ITEM_LINESTART_PATTERN.match(line)
        if not m:
            continue
        item_id = _normalize_item_id(m.group("num"), m.group("let"))
        if item_id != "14":
            continue
        snippet = re.sub(r"\\s+", " ", line.strip())
        if len(snippet) > 200:
            snippet = snippet[:200] + "..."
        found.append((current_part, snippet))
        if len(found) >= max_samples:
            return found

    if found:
        return found

    m = re.search(r"(?i)\\bITEM\\s+14\\b.{0,200}", text)
    if m:
        snippet = re.sub(r"\\s+", " ", m.group(0).strip())
        return [(None, snippet)]
    return []


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark baseline vs regime-aware 10-K item extraction."
    )
    parser.add_argument(
        "--parquet-dir",
        type=Path,
        default=DEFAULT_PARQUET_DIR,
        help="Directory containing parquet batch files (default: sample parquet_batches).",
    )
    parser.add_argument(
        "--out-path",
        type=Path,
        default=Path("results/item_regime_benchmark.txt"),
        help="Path to write the comparison report.",
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
        default=10,
        help="Max examples of flagged items to print in the report.",
    )
    parser.add_argument(
        "--modern-since",
        type=str,
        default="2018-01-01",
        help="Date threshold for 'modern filings' (YYYY-MM-DD).",
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

    modern_since = _parse_date(args.modern_since)
    if modern_since is None:
        raise SystemExit(f"Invalid --modern-since date: {args.modern_since}")

    total_filings = 0
    total_items_baseline = 0
    total_items_regime = 0

    modern_filings = 0
    modern_items_baseline = 0
    modern_items_regime = 0

    impossible_baseline_by_item = Counter()
    impossible_baseline_by_canonical = Counter()
    impossible_regime_by_item = Counter()
    item14_breakdown = Counter()
    item14_examples: list[dict[str, str]] = []
    item14_examples_per_year: dict[int | None, int] = defaultdict(int)

    examples: list[dict[str, str]] = []

    want_cols = [
        "doc_id",
        "cik",
        "accession_number",
        "document_type_filename",
        "file_date_filename",
        "full_text",
    ]

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

                baseline_items = extract_filing_items(
                    text,
                    form_type=form_type,
                    filing_date=filing_date,
                    period_end=period_end,
                    regime=False,
                )
                regime_items = extract_filing_items(
                    text,
                    form_type=form_type,
                    filing_date=filing_date,
                    period_end=period_end,
                    regime=True,
                )

                total_items_baseline += len(baseline_items)
                total_items_regime += len(regime_items)

                _annotate_items_with_regime(
                    baseline_items,
                    form_type=form_type,
                    filing_date=filing_date,
                    period_end=period_end,
                    enable_regime=True,
                )

                for item in baseline_items:
                    exists = item.get("exists_by_regime")
                    canonical = item.get("canonical_item")
                    status = item.get("item_status")
                    reason = None
                    if exists is False:
                        reason = "nonexistent"
                        impossible_baseline_by_item[item.get("item_id")] += 1
                        if canonical:
                            impossible_baseline_by_canonical[canonical] += 1
                    elif canonical and "VOTING_RESULTS_LEGACY" in canonical:
                        reason = "legacy"
                    elif status == "reserved":
                        reason = "reserved"

                    if reason and len(examples) < args.max_examples:
                        examples.append(
                            {
                                "doc_id": str(row.get("doc_id") or ""),
                                "cik": str(row.get("cik") or ""),
                                "accession": str(row.get("accession_number") or ""),
                                "item": str(item.get("item_id") or ""),
                                "canonical": str(canonical or ""),
                                "status": str(status or ""),
                                "exists": str(exists),
                                "reason": reason,
                                "filing_date": _format_date(filing_date),
                                "period_end": _format_date(period_end),
                            }
                        )

                for item in regime_items:
                    if item.get("exists_by_regime") is False:
                        impossible_regime_by_item[item.get("item_id")] += 1
                        if item.get("item_id") == "14":
                            part = item.get("item_part") or "NONE"
                            year = filing_date.year if filing_date else None
                            item14_breakdown[(part, year)] += 1
                            if (
                                item14_examples_per_year.get(year, 0) < 3
                                and len(item14_examples) < 30
                            ):
                                headings = _find_item14_headings(text, max_samples=2)
                                for heading_part, snippet in headings:
                                    item14_examples.append(
                                        {
                                            "year": str(year) if year is not None else "UNK",
                                            "item_part": part,
                                            "heading_part": heading_part or "",
                                            "doc_id": str(row.get("doc_id") or ""),
                                            "cik": str(row.get("cik") or ""),
                                            "accession": str(row.get("accession_number") or ""),
                                            "heading": snippet,
                                        }
                                    )
                                item14_examples_per_year[year] = (
                                    item14_examples_per_year.get(year, 0) + 1
                                )

                if filing_date and filing_date >= modern_since:
                    modern_filings += 1
                    modern_items_baseline += len(baseline_items)
                    modern_items_regime += len(regime_items)

    avg_items_baseline = total_items_baseline / total_filings if total_filings else 0.0
    avg_items_regime = total_items_regime / total_filings if total_filings else 0.0
    modern_avg_baseline = (
        modern_items_baseline / modern_filings if modern_filings else 0.0
    )
    modern_avg_regime = modern_items_regime / modern_filings if modern_filings else 0.0

    lines: list[str] = []
    lines.append("10-K Item Regime Benchmark")
    lines.append(f"Parquet dir: {parquet_dir}")
    lines.append(f"Total filings processed: {total_filings}")
    lines.append(
        "Total extracted items: "
        f"baseline={total_items_baseline} "
        f"regime-aware={total_items_regime}"
    )
    lines.append(
        "Items per filing: "
        f"baseline={avg_items_baseline:.2f} "
        f"regime-aware={avg_items_regime:.2f}"
    )
    lines.append("")
    lines.append("Impossible by regime (baseline items, exists_by_regime=false):")
    total_impossible = sum(impossible_baseline_by_item.values())
    lines.append(f"  total={total_impossible}")
    if impossible_baseline_by_item:
        for item_id, count in impossible_baseline_by_item.most_common(10):
            lines.append(f"  item_id {item_id}: {count}")
    if impossible_baseline_by_canonical:
        lines.append("  by canonical (top 10):")
        for canonical, count in impossible_baseline_by_canonical.most_common(10):
            lines.append(f"    {canonical}: {count}")

    lines.append("")
    lines.append("Impossible by regime (regime-aware items):")
    total_impossible_regime = sum(impossible_regime_by_item.values())
    lines.append(f"  total={total_impossible_regime}")
    if impossible_regime_by_item:
        for item_id, count in impossible_regime_by_item.most_common(10):
            lines.append(f"  item_id {item_id}: {count}")

    lines.append("")
    lines.append("Representative baseline examples (nonexistent/semantic mismatch):")
    if examples:
        for ex in examples:
            lines.append(
                "  "
                f"doc_id={ex['doc_id']} "
                f"cik={ex['cik']} "
                f"accession={ex['accession']} "
                f"item={ex['item']} "
                f"canonical={ex['canonical']} "
                f"status={ex['status']} "
                f"exists_by_regime={ex['exists']} "
                f"reason={ex['reason']} "
                f"filing_date={ex['filing_date']} "
                f"period_end={ex['period_end']}"
            )
    else:
        lines.append("  (no examples captured)")

    lines.append("")
    lines.append(
        "Modern filings coverage "
        f"(filing_date >= {modern_since.isoformat()}): "
        f"filings={modern_filings} "
        f"items/filing baseline={modern_avg_baseline:.2f} "
        f"regime-aware={modern_avg_regime:.2f}"
    )

    lines.append("")
    lines.append("Remaining Item 14 impossible-by-regime breakdown (regime-aware):")
    if item14_breakdown:
        for (part, year), count in sorted(
            item14_breakdown.items(),
            key=lambda kv: (kv[0][1] or 0, kv[0][0]),
        ):
            year_str = str(year) if year is not None else "UNK"
            lines.append(f"  {year_str} part={part}: {count}")
    else:
        lines.append("  (none)")

    lines.append("")
    lines.append("Item 14 heading samples (regime-aware impossible cases):")
    if item14_examples:
        for ex in item14_examples:
            lines.append(
                "  "
                f"year={ex['year']} part={ex['item_part']} heading_part={ex['heading_part']} "
                f"doc_id={ex['doc_id']} cik={ex['cik']} accession={ex['accession']} "
                f"heading=\"{ex['heading']}\""
            )
    else:
        lines.append("  (no samples captured)")

    report = "\n".join(lines)
    print(report)

    out_path = args.out_path
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(report, encoding="utf-8")

    print(f"\nReport written to {out_path}")


if __name__ == "__main__":
    main()
