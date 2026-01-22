from __future__ import annotations

import argparse
import csv
import re
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path

import polars as pl
import pyarrow as pa
import pyarrow.parquet as pq

from thesis_pkg.core.sec.filing_text import (
    ITEM_LINESTART_PATTERN,
    TOC_DOT_LEADER_PATTERN,
    _default_part_for_item_id,
    _looks_like_toc_heading_line,
    _normalize_newlines,
    _repair_wrapped_headings,
    _strip_edgar_metadata,
    extract_filing_items,
    parse_header,
)


ITEM_MENTION_PATTERN = re.compile(r"\bITEM\s+(?:\d+|[IVXLCDM]+)[A-Z]?\b", re.IGNORECASE)

DEFAULT_PARQUET_DIR = Path(
    r"C:\Users\erik9\Documents\SEC_Data\Data\Sample_Filings\parquet_batches"
)
DEFAULT_OUT_PATH = Path("results/suspicious_boundaries_v3_pre.csv")
DEFAULT_REPORT_PATH = Path("results/suspicious_boundaries_report_v3_pre.txt")
DEFAULT_SAMPLES_DIR = Path("results/Suspicious_Filings_Demo")
DEFAULT_CSV_PATH = Path("results/suspicious_boundaries_v5.csv")
DEFAULT_ROOT_CAUSES_PATH = Path("results/suspicious_root_causes_examples.txt")


@dataclass(frozen=True)
class DiagnosticsConfig:
    parquet_dir: Path = DEFAULT_PARQUET_DIR
    out_path: Path = DEFAULT_OUT_PATH
    report_path: Path = DEFAULT_REPORT_PATH
    samples_dir: Path = DEFAULT_SAMPLES_DIR
    batch_size: int = 8
    max_files: int = 0
    max_examples: int = 25


@dataclass(frozen=True)
class RegressionConfig:
    csv_path: Path = DEFAULT_CSV_PATH
    parquet_dir: Path = DEFAULT_PARQUET_DIR
    sample_per_flag: int = 3
    max_files: int = 0


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


def _compound_item_heading(heading_line: str) -> bool:
    if not heading_line:
        return False
    return len(ITEM_MENTION_PATTERN.findall(heading_line)) >= 2


def _split_flags(flag_field: str | None) -> list[str]:
    if not flag_field:
        return []
    return [flag.strip() for flag in flag_field.split(";") if flag.strip()]


def _normalize_body_lines(text: str) -> list[str]:
    body = _repair_wrapped_headings(_strip_edgar_metadata(_normalize_newlines(text)))
    return body.splitlines()


def load_root_causes_text(path: Path = DEFAULT_ROOT_CAUSES_PATH) -> str:
    if path.exists():
        return path.read_text(encoding="utf-8")
    archive_path = path.parent / "archive" / path.name
    if archive_path.exists():
        return archive_path.read_text(encoding="utf-8")
    raise FileNotFoundError(f"Root causes file not found at {path} or {archive_path}")


def run_boundary_diagnostics(config: DiagnosticsConfig) -> dict[str, int]:
    parquet_dir = config.parquet_dir
    if not parquet_dir.exists():
        raise SystemExit(f"parquet-dir not found: {parquet_dir}")

    files = sorted(parquet_dir.glob("*_batch_*.parquet"))
    if config.max_files and config.max_files > 0:
        files = files[: config.max_files]
    if not files:
        raise SystemExit(f"No parquet batch files found in {parquet_dir}")

    config.out_path.parent.mkdir(parents=True, exist_ok=True)
    config.report_path.parent.mkdir(parents=True, exist_ok=True)
    config.samples_dir.mkdir(parents=True, exist_ok=True)

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

    with config.out_path.open("w", newline="", encoding="utf-8") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=out_headers)
        writer.writeheader()

        for file_path in files:
            pf = pq.ParquetFile(file_path)
            available = set(pf.schema.names)
            columns = [c for c in want_cols if c in available]
            for batch in pf.iter_batches(batch_size=config.batch_size, columns=columns):
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

                    lines = _normalize_body_lines(text)
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
                            if _compound_item_heading(heading_line):
                                flags.append("compound_item_heading")

                        if heading_offset and heading_offset > 0:
                            flags.append("midline_heading")

                        if _weak_heading_letter(item_id, heading_line):
                            flags.append("weak_letter_heading")

                        if heading_idx is not None and heading_idx >= 0:
                            _, next_line = _non_empty_line(lines, heading_idx + 1)
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
                        sample_path = config.samples_dir / f"{safe_id}.txt"
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
    lines_out.append("Suspicious boundary diagnostics (unified)")
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
    config.report_path.write_text(report, encoding="utf-8")
    print(f"\nReport written to {config.report_path}")

    return {
        "total_filings": total_filings,
        "total_items": total_items,
        "total_flagged": total_flagged,
    }


def run_boundary_regression(config: RegressionConfig) -> dict[str, int]:
    csv_path = config.csv_path
    if not csv_path.exists():
        archive_path = csv_path.parent / "archive" / csv_path.name
        if archive_path.exists():
            csv_path = archive_path
    parquet_dir = config.parquet_dir

    if not csv_path.exists():
        raise SystemExit(f"csv-path not found: {csv_path}")
    if not parquet_dir.exists():
        raise SystemExit(f"parquet-dir not found: {parquet_dir}")

    baseline = pl.read_csv(
        csv_path,
        schema_overrides={
            "item_id": pl.Utf8,
            "item_part": pl.Utf8,
            "item": pl.Utf8,
            "canonical_item": pl.Utf8,
            "exists_by_regime": pl.Utf8,
            "item_status": pl.Utf8,
            "heading_index": pl.Utf8,
            "heading_offset": pl.Utf8,
            "flags": pl.Utf8,
        },
        infer_schema_length=50_000,
    )
    doc_ids = set(baseline.get_column("doc_id").to_list())
    before_counts: Counter[str] = Counter()
    for flag_field in baseline.get_column("flags").to_list():
        for flag in _split_flags(flag_field):
            before_counts[flag] += 1

    files = sorted(parquet_dir.glob("*_batch_*.parquet"))
    if config.max_files and config.max_files > 0:
        files = files[: config.max_files]
    if not files:
        raise SystemExit(f"No parquet batch files found in {parquet_dir}")

    after_counts: Counter[str] = Counter()
    examples_by_flag: dict[str, list[dict[str, str]]] = defaultdict(list)
    total_filings = 0
    total_items = 0
    total_flagged = 0
    seen_doc_ids: set[str] = set()

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
        for batch in pf.iter_batches(batch_size=8, columns=columns):
            tbl = pa.Table.from_batches([batch])
            df = pl.from_arrow(tbl)
            for row in df.iter_rows(named=True):
                doc_id = str(row.get("doc_id") or "")
                if doc_id not in doc_ids:
                    continue
                seen_doc_ids.add(doc_id)
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

                lines = _normalize_body_lines(text)
                total_items += len(items)

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
                        if _compound_item_heading(heading_line):
                            flags.append("compound_item_heading")

                    if heading_offset and heading_offset > 0:
                        flags.append("midline_heading")

                    if _weak_heading_letter(item_id, heading_line):
                        flags.append("weak_letter_heading")

                    if heading_idx is not None and heading_idx >= 0:
                        _, next_line = _non_empty_line(lines, heading_idx + 1)
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
                        after_counts[flag] += 1
                        if len(examples_by_flag[flag]) < config.sample_per_flag:
                            examples_by_flag[flag].append(
                                {
                                    "doc_id": doc_id,
                                    "item": str(item.get("item") or ""),
                                    "heading": heading_line.strip(),
                                }
                            )

    missing_doc_ids = doc_ids - seen_doc_ids

    lines_out: list[str] = []
    lines_out.append("Suspicious boundary regression (unified)")
    lines_out.append(f"CSV path: {csv_path}")
    lines_out.append(f"Parquet dir: {parquet_dir}")
    lines_out.append(f"Total filings processed: {total_filings}")
    lines_out.append(f"Total items extracted: {total_items}")
    lines_out.append(f"Total flagged items: {total_flagged}")
    if missing_doc_ids:
        lines_out.append(f"Doc IDs missing in parquet: {len(missing_doc_ids)}")
    lines_out.append("")
    lines_out.append("Flag counts (before -> after):")
    all_flags = set(before_counts) | set(after_counts)
    for flag in sorted(all_flags, key=lambda f: (-after_counts.get(f, 0), f)):
        lines_out.append(
            f"  {flag}: {before_counts.get(flag, 0)} -> {after_counts.get(flag, 0)}"
        )
    lines_out.append("")
    lines_out.append("Remaining examples by flag:")
    if examples_by_flag:
        for flag, examples in examples_by_flag.items():
            lines_out.append(f"  {flag}:")
            for ex in examples:
                lines_out.append(
                    f"    doc_id={ex['doc_id']} item={ex['item']} heading=\"{ex['heading']}\""
                )
    else:
        lines_out.append("  (no remaining flagged examples captured)")

    print("\n".join(lines_out))

    return {
        "total_filings": total_filings,
        "total_items": total_items,
        "total_flagged": total_flagged,
    }


def _safe_slug(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_-]+", "_", value).strip("_") or "unknown"


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


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Unified diagnostics for suspicious item boundaries."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    scan = subparsers.add_parser("scan", help="Scan parquet batches and write flagged CSV/report.")
    scan.add_argument(
        "--parquet-dir",
        type=Path,
        default=DEFAULT_PARQUET_DIR,
        help="Directory containing parquet batch files.",
    )
    scan.add_argument(
        "--out-path",
        type=Path,
        default=DEFAULT_OUT_PATH,
        help="CSV output path for suspicious boundary rows.",
    )
    scan.add_argument(
        "--report-path",
        type=Path,
        default=DEFAULT_REPORT_PATH,
        help="Text report output path.",
    )
    scan.add_argument(
        "--samples-dir",
        type=Path,
        default=DEFAULT_SAMPLES_DIR,
        help="Directory for per-filing sample text files.",
    )
    scan.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Parquet batch size for scanning filings.",
    )
    scan.add_argument(
        "--max-files",
        type=int,
        default=0,
        help="Optional cap on number of parquet files to scan (0 = no cap).",
    )
    scan.add_argument(
        "--max-examples",
        type=int,
        default=25,
        help="Max examples to include in the report.",
    )

    regress = subparsers.add_parser(
        "regress", help="Re-run diagnostics for suspicious_boundaries_v5.csv."
    )
    regress.add_argument(
        "--csv-path",
        type=Path,
        default=DEFAULT_CSV_PATH,
        help="Path to suspicious_boundaries_v5.csv.",
    )
    regress.add_argument(
        "--parquet-dir",
        type=Path,
        default=DEFAULT_PARQUET_DIR,
        help="Directory containing parquet batch files.",
    )
    regress.add_argument(
        "--sample-per-flag",
        type=int,
        default=3,
        help="Number of remaining examples to print per flag.",
    )
    regress.add_argument(
        "--max-files",
        type=int,
        default=0,
        help="Optional cap on parquet files scanned (0 = no cap).",
    )

    return parser


def main(argv: list[str] | None = None) -> None:
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.command == "scan":
        config = DiagnosticsConfig(
            parquet_dir=args.parquet_dir,
            out_path=args.out_path,
            report_path=args.report_path,
            samples_dir=args.samples_dir,
            batch_size=args.batch_size,
            max_files=args.max_files,
            max_examples=args.max_examples,
        )
        run_boundary_diagnostics(config)
        return

    if args.command == "regress":
        config = RegressionConfig(
            csv_path=args.csv_path,
            parquet_dir=args.parquet_dir,
            sample_per_flag=args.sample_per_flag,
            max_files=args.max_files,
        )
        run_boundary_regression(config)
        return

    raise SystemExit(f"Unknown command: {args.command}")


__all__ = [
    "DiagnosticsConfig",
    "RegressionConfig",
    "DEFAULT_ROOT_CAUSES_PATH",
    "load_root_causes_text",
    "run_boundary_diagnostics",
    "run_boundary_regression",
    "main",
]
