from __future__ import annotations

import argparse
import csv
import re
from collections import Counter, defaultdict
from pathlib import Path


def _parse_benchmark(path: Path) -> dict[str, str]:
    data: dict[str, str] = {}
    if not path.exists():
        return data
    text = path.read_text(encoding="utf-8")
    m = re.search(r"Total filings processed:\s*(\d+)", text)
    if m:
        data["filings"] = m.group(1)
    m = re.search(
        r"Total extracted items:\s*baseline=(\d+)\s*regime-aware=(\d+)", text
    )
    if m:
        data["total_baseline"] = m.group(1)
        data["total_regime"] = m.group(2)
    m = re.search(
        r"Items per filing:\s*baseline=([\d\.]+)\s*regime-aware=([\d\.]+)", text
    )
    if m:
        data["items_per_baseline"] = m.group(1)
        data["items_per_regime"] = m.group(2)
    m = re.search(
        r"Modern filings coverage .*: filings=(\d+)\s*items/filing baseline=([\d\.]+)\s*regime-aware=([\d\.]+)",
        text,
    )
    if m:
        data["modern_filings"] = m.group(1)
        data["modern_items_per_baseline"] = m.group(2)
        data["modern_items_per_regime"] = m.group(3)
    return data


def _parse_impossible_report(path: Path) -> tuple[Counter, list[str]]:
    counts = Counter()
    year_lines: list[str] = []
    if not path.exists():
        return counts, year_lines
    text = path.read_text(encoding="utf-8").splitlines()
    in_items = False
    in_years = False
    for line in text:
        if line.startswith("By item_id:"):
            in_items = True
            in_years = False
            continue
        if line.startswith("By item_part/year/item_id"):
            in_items = False
            in_years = True
            continue
        if in_items:
            m = re.search(r"item_id\s+(\S+):\s*(\d+)", line)
            if m:
                counts[m.group(1)] += int(m.group(2))
            continue
        if in_years:
            if not line.strip():
                in_years = False
                continue
            if line.strip().startswith("By category"):
                in_years = False
                continue
            if line.strip().startswith("("):
                continue
            if line.strip():
                year_lines.append(line.strip())
    return counts, year_lines


def _parse_suspicious_csv(path: Path) -> tuple[Counter, Counter]:
    item_counts = Counter()
    flag_counts = Counter()
    if not path.exists():
        return item_counts, flag_counts
    csv.field_size_limit(10_000_000)
    with path.open("r", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            item_id = row.get("item_id") or ""
            if item_id:
                item_counts[item_id] += 1
            flags = (row.get("flags") or "").split(";")
            for flag in flags:
                if flag:
                    flag_counts[flag] += 1
    return item_counts, flag_counts


def _diff_counter(a: Counter, b: Counter) -> Counter:
    out = Counter()
    keys = set(a) | set(b)
    for key in keys:
        out[key] = b.get(key, 0) - a.get(key, 0)
    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare v3 pre/post benchmark outputs.")
    parser.add_argument(
        "--pre-benchmark",
        type=Path,
        default=Path("results/item_regime_benchmark_v3_pre.txt"),
    )
    parser.add_argument(
        "--post-benchmark",
        type=Path,
        default=Path("results/item_regime_benchmark_v3_post.txt"),
    )
    parser.add_argument(
        "--pre-impossible",
        type=Path,
        default=Path("results/impossible_by_regime_diagnostics_v3_pre.txt"),
    )
    parser.add_argument(
        "--post-impossible",
        type=Path,
        default=Path("results/impossible_by_regime_diagnostics_v3_post.txt"),
    )
    parser.add_argument(
        "--pre-suspicious",
        type=Path,
        default=Path("results/suspicious_boundaries_v3_pre.csv"),
    )
    parser.add_argument(
        "--post-suspicious",
        type=Path,
        default=Path("results/suspicious_boundaries_v3_post.csv"),
    )
    parser.add_argument(
        "--out-path",
        type=Path,
        default=Path("results/item_regime_comparison_v3.txt"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    pre_bench = _parse_benchmark(args.pre_benchmark)
    post_bench = _parse_benchmark(args.post_benchmark)
    pre_impossible, pre_years = _parse_impossible_report(args.pre_impossible)
    post_impossible, post_years = _parse_impossible_report(args.post_impossible)
    pre_suspicious, pre_flags = _parse_suspicious_csv(args.pre_suspicious)
    post_suspicious, post_flags = _parse_suspicious_csv(args.post_suspicious)

    impossible_delta = _diff_counter(pre_impossible, post_impossible)
    suspicious_delta = _diff_counter(pre_suspicious, post_suspicious)
    flag_delta = _diff_counter(pre_flags, post_flags)

    lines: list[str] = []
    lines.append("10-K item extraction comparison (v3 pre vs post)")
    lines.append("")
    lines.append("Benchmark summary:")
    lines.append(
        f"  filings: pre={pre_bench.get('filings','NA')} post={post_bench.get('filings','NA')}"
    )
    lines.append(
        "  total items (regime): "
        f"pre={pre_bench.get('total_regime','NA')} "
        f"post={post_bench.get('total_regime','NA')}"
    )
    lines.append(
        "  items/filing (regime): "
        f"pre={pre_bench.get('items_per_regime','NA')} "
        f"post={post_bench.get('items_per_regime','NA')}"
    )
    lines.append(
        "  modern items/filing (regime): "
        f"pre={pre_bench.get('modern_items_per_regime','NA')} "
        f"post={post_bench.get('modern_items_per_regime','NA')}"
    )

    lines.append("")
    lines.append("Impossible-by-regime counts by item_id (post):")
    for item_id, count in post_impossible.most_common():
        delta = impossible_delta.get(item_id, 0)
        sign = "+" if delta >= 0 else ""
        lines.append(f"  item_id {item_id}: {count} (delta {sign}{delta})")

    lines.append("")
    lines.append("Impossible-by-regime by year/part/item (post):")
    if post_years:
        for line in post_years:
            lines.append(f"  {line}")
    else:
        lines.append("  (no year/part breakdown found)")

    lines.append("")
    lines.append("Suspicious-boundary flags (post counts, delta from pre):")
    for flag, count in post_flags.most_common():
        delta = flag_delta.get(flag, 0)
        sign = "+" if delta >= 0 else ""
        lines.append(f"  {flag}: {count} (delta {sign}{delta})")

    lines.append("")
    lines.append("Regression section:")
    try:
        pre_items = float(pre_bench.get("items_per_regime", "0") or 0)
        post_items = float(post_bench.get("items_per_regime", "0") or 0)
        lines.append(f"  items/filing delta: {post_items - pre_items:+.2f}")
    except Exception:
        lines.append("  items/filing delta: NA")

    lines.append("  item_ids with biggest suspicious-boundary decreases:")
    for item_id, delta in sorted(
        suspicious_delta.items(), key=lambda kv: kv[1]
    )[:5]:
        if delta < 0:
            lines.append(f"    {item_id}: {delta}")
    if all(delta >= 0 for delta in suspicious_delta.values()):
        lines.append("    (no decreases recorded)")

    lines.append("  item_ids with biggest suspicious-boundary increases:")
    for item_id, delta in sorted(
        suspicious_delta.items(), key=lambda kv: kv[1], reverse=True
    )[:5]:
        if delta > 0:
            lines.append(f"    {item_id}: +{delta}")

    report = "\n".join(lines)
    print(report)
    args.out_path.parent.mkdir(parents=True, exist_ok=True)
    args.out_path.write_text(report, encoding="utf-8")
    print(f"\nReport written to {args.out_path}")


if __name__ == "__main__":
    main()
