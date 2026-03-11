from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import polars as pl


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_EARLIER_ROOT = ROOT / "full_data_run" / "derived_data"
DEFAULT_NEW_ROOT = ROOT / "full_data_run" / "derived_data-20260308T133329Z-1-004" / "derived_data"
DEFAULT_REPORT_PATH = ROOT / "reports" / "pipeline_run_comparison_20260308.md"


@dataclass(frozen=True)
class FilePair:
    label: str
    earlier_path: Path
    new_path: Path


@dataclass(frozen=True)
class KeyChoice:
    columns: tuple[str, ...]
    earlier_rows: int
    earlier_unique: int
    new_rows: int
    new_unique: int


def _human_bytes(size: int | None) -> str:
    if size is None:
        return "n/a"
    units = ["B", "KB", "MB", "GB", "TB"]
    value = float(size)
    for unit in units:
        if value < 1024 or unit == units[-1]:
            return f"{value:.2f} {unit}"
        value /= 1024.0
    return f"{size} B"


def _safe_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=True, default=str)


def _file_snapshot(path: Path) -> dict[str, Any]:
    snapshot: dict[str, Any] = {
        "path": str(path),
        "exists": path.exists(),
    }
    if not path.exists():
        return snapshot

    stat = path.stat()
    metadata = pl.read_parquet_metadata(path)
    schema = pl.read_parquet_schema(path)
    row_count = int(pl.scan_parquet(path).select(pl.len()).collect().item())
    snapshot.update(
        {
            "size_bytes": stat.st_size,
            "size_human": _human_bytes(stat.st_size),
            "modified_utc": datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc).isoformat(),
            "metadata_keys": sorted(metadata.keys()),
            "schema": {name: str(dtype) for name, dtype in schema.items()},
            "row_count": row_count,
        }
    )
    return snapshot


def _bucket_filter_expr(key_columns: tuple[str, ...], bucket_count: int, bucket_idx: int) -> pl.Expr:
    return (
        pl.struct([pl.col(name) for name in key_columns])
        .hash()
        .mod(bucket_count)
        .eq(bucket_idx)
    )


def _row_hash_expr(columns: tuple[str, ...], alias: str) -> pl.Expr:
    if not columns:
        return pl.lit(0, dtype=pl.UInt64).alias(alias)
    return pl.struct([pl.col(name) for name in columns]).hash().alias(alias)


def _row_count_and_unique(path: Path, key_columns: tuple[str, ...]) -> tuple[int, int]:
    out = (
        pl.scan_parquet(path)
        .select(
            pl.len().alias("rows"),
            pl.struct([pl.col(name) for name in key_columns]).n_unique().alias("unique_rows"),
        )
        .collect()
        .row(0, named=True)
    )
    return int(out["rows"]), int(out["unique_rows"])


def _collect(lf: pl.LazyFrame) -> pl.DataFrame:
    try:
        return lf.collect(engine="streaming")
    except TypeError:
        return lf.collect()


def _pick_bucket_count(max_size_bytes: int) -> int:
    if max_size_bytes >= 3_000_000_000:
        return 256
    if max_size_bytes >= 1_000_000_000:
        return 128
    if max_size_bytes >= 250_000_000:
        return 64
    return 32


def _key_candidates(label: str, shared_columns: set[str]) -> list[tuple[str, ...]]:
    if label == "canonical_link":
        candidates = [
            (
                "cik_10",
                "gvkey",
                "kypermno",
                "liid",
                "link_start",
                "link_end",
                "cik_start",
                "cik_end",
                "linktype",
                "linkprim",
                "link_source",
                "source_priority",
                "is_sparse_fallback",
            ),
            (
                "cik_10",
                "gvkey",
                "kypermno",
                "liid",
                "valid_start",
                "valid_end",
            ),
            (
                "cik_10",
                "gvkey",
                "kypermno",
                "liid",
                "valid_start",
                "valid_end",
                "link_source",
            ),
        ]
    else:
        candidates = [
            ("KYPERMNO", "CALDT"),
            ("KYPERMNO", "CALDT", "GVKEY"),
            ("KYPERMNO", "CALDT", "KYGVKEY_final"),
            ("KYPERMNO", "CALDT", "LIID"),
        ]

    out: list[tuple[str, ...]] = []
    for candidate in candidates:
        usable = tuple(name for name in candidate if name in shared_columns)
        if usable and usable not in out:
            out.append(usable)
    return out


def _choose_key(label: str, shared_columns: set[str], earlier_path: Path, new_path: Path) -> KeyChoice | None:
    for candidate in _key_candidates(label, shared_columns):
        earlier_rows, earlier_unique = _row_count_and_unique(earlier_path, candidate)
        new_rows, new_unique = _row_count_and_unique(new_path, candidate)
        if earlier_rows == earlier_unique and new_rows == new_unique:
            return KeyChoice(
                columns=candidate,
                earlier_rows=earlier_rows,
                earlier_unique=earlier_unique,
                new_rows=new_rows,
                new_unique=new_unique,
            )
    return None


def _multiset_hash_fallback(
    earlier_path: Path,
    new_path: Path,
    shared_columns: tuple[str, ...],
    *,
    bucket_count: int,
    sample_limit: int = 10,
) -> dict[str, Any]:
    samples_earlier: list[dict[str, Any]] = []
    samples_new: list[dict[str, Any]] = []
    only_earlier = 0
    only_new = 0

    for bucket_idx in range(bucket_count):
        earlier_hashes = (
            pl.scan_parquet(earlier_path)
            .select(_row_hash_expr(shared_columns, "_row_hash"))
            .filter(pl.col("_row_hash").mod(bucket_count) == bucket_idx)
            .group_by("_row_hash")
            .agg(pl.len().alias("_count"))
        )
        new_hashes = (
            pl.scan_parquet(new_path)
            .select(_row_hash_expr(shared_columns, "_row_hash"))
            .filter(pl.col("_row_hash").mod(bucket_count) == bucket_idx)
            .group_by("_row_hash")
            .agg(pl.len().alias("_count"))
        )

        earlier_only_bucket = (
            earlier_hashes.join(new_hashes, on="_row_hash", how="left", suffix="_new")
            .with_columns(pl.col("_count_new").fill_null(0))
            .filter(pl.col("_count") > pl.col("_count_new"))
            .select(
                pl.col("_row_hash"),
                (pl.col("_count") - pl.col("_count_new")).alias("_delta"),
            )
        )
        new_only_bucket = (
            new_hashes.join(earlier_hashes, on="_row_hash", how="left", suffix="_earlier")
            .with_columns(pl.col("_count_earlier").fill_null(0))
            .filter(pl.col("_count") > pl.col("_count_earlier"))
            .select(
                pl.col("_row_hash"),
                (pl.col("_count") - pl.col("_count_earlier")).alias("_delta"),
            )
        )

        earlier_summary = earlier_only_bucket.select(pl.col("_delta").sum().fill_null(0)).collect().item()
        new_summary = new_only_bucket.select(pl.col("_delta").sum().fill_null(0)).collect().item()
        only_earlier += int(earlier_summary)
        only_new += int(new_summary)

        if len(samples_earlier) < sample_limit:
            room = sample_limit - len(samples_earlier)
            samples_earlier.extend(earlier_only_bucket.limit(room).collect().to_dicts())
        if len(samples_new) < sample_limit:
            room = sample_limit - len(samples_new)
            samples_new.extend(new_only_bucket.limit(room).collect().to_dicts())

    return {
        "method": "shared_row_hash_multiset_fallback",
        "bucket_count": bucket_count,
        "only_in_earlier": only_earlier,
        "only_in_new": only_new,
        "samples_only_in_earlier": samples_earlier,
        "samples_only_in_new": samples_new,
    }


def _compare_with_key(
    earlier_path: Path,
    new_path: Path,
    key_columns: tuple[str, ...],
    value_columns: tuple[str, ...],
    *,
    bucket_count: int,
    sample_limit: int = 10,
) -> dict[str, Any]:
    earlier_keys = pl.scan_parquet(earlier_path).select(list(key_columns))
    new_keys = pl.scan_parquet(new_path).select(list(key_columns))

    earlier_only = earlier_keys.join(new_keys, on=list(key_columns), how="anti")
    new_only = new_keys.join(earlier_keys, on=list(key_columns), how="anti")

    only_in_earlier = int(_collect(earlier_only.select(pl.len())).item())
    only_in_new = int(_collect(new_only.select(pl.len())).item())
    key_samples_earlier = _collect(earlier_only.limit(sample_limit)).to_dicts()
    key_samples_new = _collect(new_only.limit(sample_limit)).to_dicts()

    payload_mismatch_rows = 0
    payload_samples: list[dict[str, Any]] = []
    column_mismatch_counts: dict[str, int] = {}

    if value_columns:
        earlier_hash = pl.scan_parquet(earlier_path).select([*key_columns, _row_hash_expr(value_columns, "_row_hash")])
        new_hash = pl.scan_parquet(new_path).select([*key_columns, _row_hash_expr(value_columns, "_row_hash")])
        payload_hash_mismatches = (
            earlier_hash.join(new_hash, on=list(key_columns), how="inner", suffix="_new")
            .filter(pl.col("_row_hash") != pl.col("_row_hash_new"))
            .select(list(key_columns))
        )
        payload_mismatch_rows = int(_collect(payload_hash_mismatches.select(pl.len())).item())
        sample_mismatch_keys = _collect(payload_hash_mismatches.limit(sample_limit)).to_dicts()

        if sample_mismatch_keys:
            sample_keys_lf = pl.DataFrame(sample_mismatch_keys).lazy()
            earlier_sample = (
                pl.scan_parquet(earlier_path)
                .join(sample_keys_lf, on=list(key_columns), how="inner")
                .select([*key_columns, *value_columns])
            )
            new_sample = (
                pl.scan_parquet(new_path)
                .join(sample_keys_lf, on=list(key_columns), how="inner")
                .select([*key_columns, *value_columns])
            )
            mismatch_names_expr = [
                pl.when(pl.col(name).eq_missing(pl.col(f"{name}_new")).not_())
                .then(pl.lit(name))
                .otherwise(None)
                for name in value_columns
            ]
            sample_joined = (
                earlier_sample.join(new_sample, on=list(key_columns), how="inner", suffix="_new")
                .with_columns(pl.concat_list(mismatch_names_expr).list.drop_nulls().alias("_mismatch_columns"))
                .filter(pl.col("_mismatch_columns").list.len() > 0)
                .select([*key_columns, "_mismatch_columns"])
            )
            payload_samples = _collect(sample_joined).to_dicts()

        # Exact per-column counts are cheap for the small canonical-link parquet.
        if max(earlier_path.stat().st_size, new_path.stat().st_size) <= 250_000_000 and value_columns:
            mismatch_exprs = [
                pl.col(name).eq_missing(pl.col(f"{name}_new")).not_().alias(f"_mismatch__{name}")
                for name in value_columns
            ]
            exact_joined = (
                pl.scan_parquet(earlier_path)
                .select([*key_columns, *value_columns])
                .join(
                    pl.scan_parquet(new_path).select([*key_columns, *value_columns]),
                    on=list(key_columns),
                    how="inner",
                    suffix="_new",
                )
                .with_columns(mismatch_exprs)
            )
            aggregates = _collect(
                exact_joined.select(
                    [
                        pl.col(f"_mismatch__{name}").sum().fill_null(0).alias(name)
                        for name in value_columns
                    ]
                )
            ).row(0, named=True)
            column_mismatch_counts = {
                name: int(count)
                for name, count in aggregates.items()
                if int(count) > 0
            }
        elif payload_samples:
            sample_counts: dict[str, int] = {}
            for sample in payload_samples:
                for name in sample.get("_mismatch_columns", []):
                    sample_counts[name] = sample_counts.get(name, 0) + 1
            column_mismatch_counts = sample_counts

    return {
        "method": "direct_key_compare_with_hash_payload_check",
        "bucket_count": bucket_count,
        "key_columns": list(key_columns),
        "value_columns": list(value_columns),
        "only_in_earlier": only_in_earlier,
        "only_in_new": only_in_new,
        "payload_mismatch_rows": payload_mismatch_rows,
        "column_mismatch_counts": {k: v for k, v in column_mismatch_counts.items() if v > 0},
        "samples_only_in_earlier": key_samples_earlier,
        "samples_only_in_new": key_samples_new,
        "payload_mismatch_samples": payload_samples,
    }


def compare_pair(pair: FilePair) -> dict[str, Any]:
    earlier = _file_snapshot(pair.earlier_path)
    new = _file_snapshot(pair.new_path)
    result: dict[str, Any] = {
        "label": pair.label,
        "earlier": earlier,
        "new": new,
    }
    if not earlier["exists"] or not new["exists"]:
        result["status"] = "missing_file"
        return result

    earlier_columns = set(earlier["schema"])
    new_columns = set(new["schema"])
    shared_columns = tuple(sorted(earlier_columns & new_columns))
    added_columns = sorted(new_columns - earlier_columns)
    removed_columns = sorted(earlier_columns - new_columns)
    result["shared_columns"] = list(shared_columns)
    result["added_columns"] = added_columns
    result["removed_columns"] = removed_columns

    max_size = max(int(earlier["size_bytes"]), int(new["size_bytes"]))
    bucket_count = _pick_bucket_count(max_size)

    key_choice = _choose_key(pair.label, set(shared_columns), pair.earlier_path, pair.new_path)
    if key_choice is None:
        fallback = _multiset_hash_fallback(
            pair.earlier_path,
            pair.new_path,
            shared_columns,
            bucket_count=bucket_count,
        )
        result["comparison"] = fallback
        result["comparison_summary"] = {
            "shared_content_unchanged": fallback["only_in_earlier"] == 0 and fallback["only_in_new"] == 0,
            "used_fallback": True,
        }
        return result

    value_columns = tuple(name for name in shared_columns if name not in key_choice.columns)
    comparison = _compare_with_key(
        pair.earlier_path,
        pair.new_path,
        key_choice.columns,
        value_columns,
        bucket_count=bucket_count,
    )
    result["key_choice"] = {
        "columns": list(key_choice.columns),
        "earlier_rows": key_choice.earlier_rows,
        "earlier_unique": key_choice.earlier_unique,
        "new_rows": key_choice.new_rows,
        "new_unique": key_choice.new_unique,
    }
    result["comparison"] = comparison

    shared_content_unchanged = (
        comparison["only_in_earlier"] == 0
        and comparison["only_in_new"] == 0
        and comparison["payload_mismatch_rows"] == 0
    )
    result["comparison_summary"] = {
        "shared_content_unchanged": shared_content_unchanged,
        "used_fallback": False,
    }
    return result


def _top_changed_columns(column_mismatch_counts: dict[str, int], limit: int = 12) -> list[tuple[str, int]]:
    return sorted(column_mismatch_counts.items(), key=lambda item: (-item[1], item[0]))[:limit]


def _classification(result: dict[str, Any]) -> str:
    earlier = result["earlier"]
    new = result["new"]
    if not earlier["exists"] or not new["exists"]:
        return "missing file"

    added_columns = result.get("added_columns", [])
    removed_columns = result.get("removed_columns", [])
    same_rows = earlier["row_count"] == new["row_count"]
    comparison = result.get("comparison", {})
    used_fallback = result.get("comparison_summary", {}).get("used_fallback", False)

    if used_fallback:
        only_earlier = comparison.get("only_in_earlier", 0)
        only_new = comparison.get("only_in_new", 0)
        if only_earlier == 0 and only_new == 0:
            if added_columns or removed_columns:
                return "schema/output-shape differences only"
            return "no shared-row payload difference detected; remaining difference likely storage/layout"
        return "true payload/content differences"

    if (
        comparison.get("only_in_earlier", 0) == 0
        and comparison.get("only_in_new", 0) == 0
        and comparison.get("payload_mismatch_rows", 0) == 0
    ):
        if added_columns or removed_columns:
            return "schema/output-shape differences only"
        if same_rows and earlier["size_bytes"] != new["size_bytes"]:
            return "storage/compression/layout differences only"
        return "no detectable shared payload difference"
    return "true payload/content differences"


def _likely_cause_notes(result: dict[str, Any]) -> list[str]:
    label = result["label"]
    comparison = result.get("comparison", {})
    added_columns = result.get("added_columns", [])
    removed_columns = result.get("removed_columns", [])
    notes: list[str] = []

    if label == "canonical_link":
        notes.append(
            "This parquet is produced by `build_or_reuse_ccm_daily_stage(...)` via `build_canonical_link_table(...)` in `src/thesis_pkg/pipelines/ccm_pipeline.py` and `src/thesis_pkg/core/ccm/canonical_links.py`."
        )
        if added_columns or removed_columns:
            notes.append(
                "A column-set change here points first to canonical-link output schema evolution or a renamed artifact contract, not necessarily a different linked row universe."
            )
        top_changed = [name for name, _ in _top_changed_columns(comparison.get("column_mismatch_counts", {}), limit=8)]
        if any(name in {"valid_start", "valid_end", "cik_start", "cik_end"} for name in top_changed):
            notes.append(
                "Differences in `valid_start`/`valid_end` or `cik_start`/`cik_end` are consistent with changes in the CIK-history window policy and interval intersection logic in `canonical_links.py`."
            )
        if any(name in {"link_rank_effective", "row_quality_tier", "link_quality"} for name in top_changed):
            notes.append(
                "Differences in rank/quality fields are consistent with altered canonical-link tie-breaking rather than a raw row-count drop."
            )
        if result.get("comparison_summary", {}).get("shared_content_unchanged"):
            notes.append(
                "If shared columns match exactly, the file-name change from `canonical_link_table_after_startdate_change.parquet` to `canonical_link_table.parquet` is most likely a naming/output-contract cleanup."
            )

    if label == "final_flagged_data_compdesc_added":
        notes.append(
            "This parquet is downstream of canonical-link construction: `build_or_reuse_ccm_daily_stage(...)` builds the canonical table, `attach_ccm_links(...)` applies date-valid link selection, `merge_histories(...)` adds security/company histories, and `attach_company_description(...)` appends descriptive fields."
        )
        top_changed = [name for name, _ in _top_changed_columns(comparison.get("column_mismatch_counts", {}), limit=12)]
        linkish = {
            "KYGVKEY_ccm",
            "KYGVKEY_final",
            "LIID",
            "valid_link",
            "is_canonical_link",
            "LINKPRIM",
            "LINKTYPE",
            "link_rank",
        }
        if any(name in linkish for name in top_changed):
            notes.append(
                "When changed columns are concentrated in link-assignment fields, the most likely cause is a canonical-link or CCM attach-path change rather than storage layout."
            )
        if any(name.startswith("HIST_") or name in {"HTPCI", "HEXCNTRY", "HSCUSIP", "HISIN", "HSIC", "HNAICS", "HGSUBIND"} for name in top_changed):
            notes.append(
                "Changes concentrated in history-derived columns point to the downstream `merge_histories(...)` as-of joins rather than the raw price panel."
            )
        if result.get("comparison_summary", {}).get("shared_content_unchanged"):
            notes.append(
                "If shared columns match and row counts match, a smaller parquet on disk is explained by storage/layout choices, not by fewer output rows."
            )

    if not notes:
        notes.append("No specific pipeline-cause heuristic fired beyond the raw comparison results.")
    return notes


def _render_pair_report(result: dict[str, Any]) -> str:
    earlier = result["earlier"]
    new = result["new"]
    comparison = result.get("comparison", {})
    summary = result.get("comparison_summary", {})
    lines: list[str] = []
    lines.append(f"## {result['label']}")
    lines.append("")
    lines.append("### File checks")
    lines.append("")
    lines.append(
        f"- Earlier: `{earlier['path']}`"
        + (f" | size `{earlier['size_human']}` | rows `{earlier['row_count']}` | modified `{earlier['modified_utc']}`" if earlier["exists"] else " | missing")
    )
    lines.append(
        f"- New: `{new['path']}`"
        + (f" | size `{new['size_human']}` | rows `{new['row_count']}` | modified `{new['modified_utc']}`" if new["exists"] else " | missing")
    )
    if earlier["exists"] and new["exists"]:
        lines.append(f"- Earlier parquet metadata keys: `{', '.join(earlier['metadata_keys']) or '(none)'}`")
        lines.append(f"- New parquet metadata keys: `{', '.join(new['metadata_keys']) or '(none)'}`")
    lines.append("")

    if result.get("status") == "missing_file":
        lines.append("Comparison stopped because at least one file is missing.")
        lines.append("")
        return "\n".join(lines)

    lines.append("### Schema")
    lines.append("")
    lines.append(f"- Shared columns: `{len(result['shared_columns'])}`")
    lines.append(f"- Added columns in new: `{_safe_json(result['added_columns'])}`")
    lines.append(f"- Removed columns in new: `{_safe_json(result['removed_columns'])}`")
    dtype_changes = []
    earlier_schema = earlier["schema"]
    new_schema = new["schema"]
    for name in result["shared_columns"]:
        if earlier_schema[name] != new_schema[name]:
            dtype_changes.append(
                {
                    "column": name,
                    "earlier": earlier_schema[name],
                    "new": new_schema[name],
                }
            )
    lines.append(f"- Shared-column dtype changes: `{_safe_json(dtype_changes[:20])}`")
    lines.append("")

    lines.append("### Comparison")
    lines.append("")
    if "key_choice" in result:
        lines.append(f"- Chosen diagnostic key: `{_safe_json(result['key_choice']['columns'])}`")
        lines.append(
            f"- Key uniqueness check: earlier `{result['key_choice']['earlier_unique']}/{result['key_choice']['earlier_rows']}`, new `{result['key_choice']['new_unique']}/{result['key_choice']['new_rows']}`"
        )
    else:
        lines.append("- No safe unique key detected across shared columns; used shared-row hash multiset fallback.")
    lines.append(f"- Classification: `{_classification(result)}`")
    if "only_in_earlier" in comparison:
        lines.append(f"- Keys/rows only in earlier: `{comparison['only_in_earlier']}`")
    if "only_in_new" in comparison:
        lines.append(f"- Keys/rows only in new: `{comparison['only_in_new']}`")
    if "payload_mismatch_rows" in comparison:
        lines.append(f"- Matched keys with shared-column payload differences: `{comparison['payload_mismatch_rows']}`")
    lines.append(f"- Shared-column content unchanged: `{summary.get('shared_content_unchanged')}`")
    top_changed = _top_changed_columns(comparison.get("column_mismatch_counts", {}))
    lines.append(f"- Top changed shared columns: `{_safe_json(top_changed)}`")
    lines.append("")

    if comparison.get("samples_only_in_earlier"):
        lines.append(f"- Sample keys only in earlier: `{_safe_json(comparison['samples_only_in_earlier'][:5])}`")
    if comparison.get("samples_only_in_new"):
        lines.append(f"- Sample keys only in new: `{_safe_json(comparison['samples_only_in_new'][:5])}`")
    if comparison.get("payload_mismatch_samples"):
        lines.append(f"- Sample matched keys with changed columns: `{_safe_json(comparison['payload_mismatch_samples'][:5])}`")
    if comparison.get("samples_only_in_earlier") or comparison.get("samples_only_in_new") or comparison.get("payload_mismatch_samples"):
        lines.append("")

    lines.append("### Likely Causes")
    lines.append("")
    for note in _likely_cause_notes(result):
        lines.append(f"- {note}")
    lines.append("")

    return "\n".join(lines)


def build_report(results: list[dict[str, Any]], *, earlier_root: Path, new_root: Path) -> str:
    generated_at = datetime.now(timezone.utc).isoformat()
    lines = [
        "# Pipeline Run Comparison Report",
        "",
        f"- Generated at: `{generated_at}`",
        f"- Earlier root: `{earlier_root}`",
        f"- New root: `{new_root}`",
        "- Execution model: lazy Polars scans, projected columns, and bucketed key/hash diagnostics to avoid fully materializing both large parquets at once.",
        "- Important: a smaller parquet on disk does not by itself imply fewer rows or less payload; compression, row-group layout, and column-set changes can reduce file size while content stays the same.",
        "",
    ]
    for result in results:
        lines.append(_render_pair_report(result))

    lines.append("## Overall Summary")
    lines.append("")
    for result in results:
        lines.append(
            f"- `{result['label']}`: `{_classification(result)}`"
        )
    lines.append("")
    return "\n".join(lines)


def run(earlier_root: Path, new_root: Path, report_path: Path) -> list[dict[str, Any]]:
    pairs = [
        FilePair(
            label="canonical_link",
            earlier_path=earlier_root / "canonical_link_table_after_startdate_change.parquet",
            new_path=new_root / "canonical_link_table.parquet",
        ),
        FilePair(
            label="final_flagged_data_compdesc_added",
            earlier_path=earlier_root / "final_flagged_data_compdesc_added.parquet",
            new_path=new_root / "final_flagged_data_compdesc_added.parquet",
        ),
    ]

    results = [compare_pair(pair) for pair in pairs]
    report_text = build_report(results, earlier_root=earlier_root, new_root=new_root)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(report_text, encoding="utf-8")
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare two local pipeline runs with RAM-safe Polars scans.")
    parser.add_argument("--earlier-root", type=Path, default=DEFAULT_EARLIER_ROOT)
    parser.add_argument("--new-root", type=Path, default=DEFAULT_NEW_ROOT)
    parser.add_argument("--report-path", type=Path, default=DEFAULT_REPORT_PATH)
    args = parser.parse_args()

    results = run(args.earlier_root, args.new_root, args.report_path)
    print(json.dumps({"report_path": str(args.report_path), "results": results}, ensure_ascii=True, default=str, indent=2))


if __name__ == "__main__":
    main()
