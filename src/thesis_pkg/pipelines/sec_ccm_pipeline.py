from __future__ import annotations

import datetime as dt
import json
from pathlib import Path
import time
import uuid
from typing import Any

import polars as pl

from thesis_pkg.core.ccm.sec_ccm_contracts import (
    MatchReasonCode,
    SecCcmJoinSpecV1,
    SecCcmJoinSpecV2,
    normalize_sec_ccm_join_spec,
)
from thesis_pkg.core.ccm.sec_ccm_premerge import (
    align_doc_dates_phase_b,
    apply_phase_b_reason_codes,
    build_match_status_doc,
    build_unmatched_diagnostics_doc,
    normalize_sec_filings_phase_a,
    resolve_links_phase_a,
    join_daily_phase_b,
)
from thesis_pkg.core.ccm.transforms import apply_concept_filter_flags_doc


_FORM_COLUMNS = ("form_type", "document_type_filename", "SRCTYPE")


def _iso_utc(ts: dt.datetime) -> str:
    return ts.astimezone(dt.timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def _write_lazy_parquet(lf: pl.LazyFrame, path: Path, compression: str = "zstd") -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    lf.sink_parquet(path, compression=compression)
    return path


def _write_text(path: Path, text: str) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def _write_json(path: Path, payload: dict[str, Any]) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return path


def _count_parquet_rows(path: Path) -> int:
    return int(pl.scan_parquet(path).select(pl.len().alias("n_rows")).collect().item())


def _assert_unique_doc_id(lf: pl.LazyFrame, label: str) -> None:
    dup = (
        lf.group_by("doc_id")
        .agg(pl.len().alias("n_rows"))
        .filter(pl.col("n_rows") > 1)
        .limit(5)
        .collect()
    )
    if dup.height > 0:
        samples = dup.to_dicts()
        raise ValueError(f"{label} has non-unique doc_id values; sample duplicates: {samples}")


def _step_events_to_frame(run_id: str, events: list[dict[str, Any]]) -> pl.DataFrame:
    if not events:
        return pl.DataFrame(
            schema={
                "run_id": pl.Utf8,
                "step_order": pl.Int32,
                "step_name": pl.Utf8,
                "started_at_utc": pl.Utf8,
                "finished_at_utc": pl.Utf8,
                "duration_ms": pl.Int64,
                "artifact_key": pl.Utf8,
                "artifact_path": pl.Utf8,
                "rows_out": pl.Int64,
                "notes": pl.Utf8,
            }
        )

    return (
        pl.DataFrame(events)
        .with_columns(
            pl.lit(run_id).alias("run_id"),
            pl.col("step_order").cast(pl.Int32, strict=False),
            pl.col("step_name").cast(pl.Utf8, strict=False),
            pl.col("started_at_utc").cast(pl.Utf8, strict=False),
            pl.col("finished_at_utc").cast(pl.Utf8, strict=False),
            pl.col("duration_ms").cast(pl.Int64, strict=False),
            pl.col("artifact_key").cast(pl.Utf8, strict=False),
            pl.col("artifact_path").cast(pl.Utf8, strict=False),
            pl.col("rows_out").cast(pl.Int64, strict=False),
            pl.col("notes").cast(pl.Utf8, strict=False),
        )
        .select(
            "run_id",
            "step_order",
            "step_name",
            "started_at_utc",
            "finished_at_utc",
            "duration_ms",
            "artifact_key",
            "artifact_path",
            "rows_out",
            "notes",
        )
    )


def _fmt_markdown_value(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        if value != value:  # NaN
            return ""
        return f"{value:.6g}"
    if isinstance(value, dt.datetime):
        return _iso_utc(value)
    if isinstance(value, dt.date):
        return value.isoformat()
    return str(value).replace("|", "\\|")


def _to_markdown_table(df: pl.DataFrame, *, max_rows: int = 25) -> str:
    if df.height == 0:
        return "_No rows._"
    clipped = df.head(max_rows)
    cols = clipped.columns
    header = "| " + " | ".join(cols) + " |"
    divider = "| " + " | ".join(["---"] * len(cols)) + " |"
    rows = []
    for row in clipped.iter_rows(named=True):
        rows.append("| " + " | ".join(_fmt_markdown_value(row[col]) for col in cols) + " |")
    out = "\n".join([header, divider, *rows])
    if df.height > max_rows:
        out += f"\n\n_Truncated to first {max_rows} rows._"
    return out


def _build_summary_metrics(final_doc_path: Path) -> dict[str, object]:
    final_doc = pl.scan_parquet(final_doc_path)
    schema = final_doc.collect_schema()
    lag_rejected_expr = (
        pl.col("daily_lag_gate_rejected").cast(pl.Int64).sum()
        if "daily_lag_gate_rejected" in schema
        else pl.lit(0, dtype=pl.Int64)
    ).alias("n_daily_lag_gate_rejected")
    base = final_doc.select(
        pl.len().cast(pl.Int64).alias("n_docs_total"),
        (pl.col("match_reason_code") == pl.lit(MatchReasonCode.OK.value)).cast(pl.Int64).sum().alias("n_docs_matched"),
        pl.col("has_acceptance_datetime").cast(pl.Int64).sum().alias("n_docs_with_acceptance_datetime"),
        pl.col("aligned_caldt").is_not_null().cast(pl.Int64).sum().alias("n_docs_with_aligned_caldt"),
        pl.col("alignment_lag_days").min().alias("alignment_lag_days_min"),
        pl.col("alignment_lag_days").max().alias("alignment_lag_days_max"),
        pl.col("alignment_lag_days").mean().alias("alignment_lag_days_mean"),
        lag_rejected_expr,
    ).collect().row(0, named=True)

    n_total = int(base["n_docs_total"] or 0)
    n_matched = int(base["n_docs_matched"] or 0)
    n_unmatched = n_total - n_matched
    n_acceptance = int(base["n_docs_with_acceptance_datetime"] or 0)
    n_aligned = int(base["n_docs_with_aligned_caldt"] or 0)
    n_lag_rejected = int(base["n_daily_lag_gate_rejected"] or 0)
    return {
        "n_docs_total": n_total,
        "n_docs_matched": n_matched,
        "n_docs_unmatched": n_unmatched,
        "matched_rate": (float(n_matched) / float(n_total)) if n_total > 0 else 0.0,
        "unmatched_rate": (float(n_unmatched) / float(n_total)) if n_total > 0 else 0.0,
        "n_docs_with_acceptance_datetime": n_acceptance,
        "acceptance_datetime_coverage_rate": (float(n_acceptance) / float(n_total)) if n_total > 0 else 0.0,
        "n_docs_with_aligned_caldt": n_aligned,
        "aligned_rate": (float(n_aligned) / float(n_total)) if n_total > 0 else 0.0,
        "alignment_lag_days_min": base["alignment_lag_days_min"],
        "alignment_lag_days_max": base["alignment_lag_days_max"],
        "alignment_lag_days_mean": base["alignment_lag_days_mean"],
        "n_daily_lag_gate_rejected": n_lag_rejected,
        "daily_lag_gate_rejected_rate": (float(n_lag_rejected) / float(n_total)) if n_total > 0 else 0.0,
    }


def _build_reason_count_tables(match_status_path: Path) -> dict[str, pl.DataFrame]:
    match_status = pl.scan_parquet(match_status_path)
    return {
        "match_reason_counts": match_status.group_by("match_reason_code").agg(pl.len().alias("n_docs")).sort(
            "n_docs", descending=True
        ).collect(),
        "phase_a_reason_counts": match_status.group_by("phase_a_reason_code").agg(pl.len().alias("n_docs")).sort(
            "n_docs", descending=True
        ).collect(),
        "phase_b_reason_counts": match_status.drop_nulls(subset=["phase_b_reason_code"]).group_by(
            "phase_b_reason_code"
        ).agg(pl.len().alias("n_docs")).sort("n_docs", descending=True).collect(),
    }


def _build_unmatched_tables(unmatched_diag_path: Path) -> dict[str, pl.DataFrame]:
    unmatched = pl.scan_parquet(unmatched_diag_path)
    schema = unmatched.collect_schema()
    form_col = next((col for col in _FORM_COLUMNS if col in schema), None)

    group_cols = ["filing_year"]
    if form_col is not None:
        group_cols.append(form_col)

    unmatched_rate_by_year_form = (
        unmatched.with_columns(pl.col("filing_date").dt.year().alias("filing_year"))
        .group_by(group_cols)
        .agg(pl.len().alias("n_unmatched_docs"))
        .sort("n_unmatched_docs", descending=True)
        .collect()
    )

    top_unmatched_cik = (
        unmatched.group_by("cik_10")
        .agg(pl.len().alias("n_unmatched_docs"))
        .sort("n_unmatched_docs", descending=True)
        .head(25)
        .collect()
    )

    diag_signals = unmatched.select(
        pl.len().alias("n_unmatched_docs"),
        pl.col("diag_cik_in_link_universe").cast(pl.Int64).sum().alias("n_cik_in_link_universe"),
        pl.col("has_other_filings_matched_for_cik").cast(pl.Int64).sum().alias("n_has_other_filings_matched_for_cik"),
        pl.col("diag_date_before_key_coverage").cast(pl.Int64).sum().alias("n_before_key_coverage"),
        pl.col("diag_date_after_key_coverage").cast(pl.Int64).sum().alias("n_after_key_coverage"),
    ).collect().row(0, named=True)

    diag_signals_df = pl.DataFrame(
        {
            "metric": list(diag_signals.keys()),
            "value": [diag_signals[key] for key in diag_signals.keys()],
        }
    )
    return {
        "unmatched_by_year_form": unmatched_rate_by_year_form,
        "top_unmatched_cik": top_unmatched_cik,
        "diag_signals": diag_signals_df,
    }


def _build_acceptance_coverage_by_year(match_status_path: Path) -> pl.DataFrame:
    return (
        pl.scan_parquet(match_status_path)
        .with_columns(pl.col("filing_date").dt.year().alias("filing_year"))
        .group_by("filing_year")
        .agg(
            pl.len().alias("n_docs"),
            pl.col("has_acceptance_datetime").cast(pl.Int64).sum().alias("n_with_acceptance_datetime"),
        )
        .with_columns(
            (
                pl.when(pl.col("n_docs") > 0)
                .then(pl.col("n_with_acceptance_datetime") / pl.col("n_docs"))
                .otherwise(pl.lit(0.0))
            ).alias("acceptance_datetime_coverage_rate")
        )
        .sort("filing_year")
        .collect()
    )


def _build_dag_mermaid(step_durations_ms: dict[str, int], daily_join_enabled: bool) -> str:
    def _dur(step_name: str) -> str:
        return f"{step_durations_ms.get(step_name, 0)} ms"

    daily_label = "Phase B Daily Join (disabled)"
    if daily_join_enabled:
        daily_label = f"Phase B Daily Join ({_dur('phase_b_daily_join')})"

    lines = [
        "%% Auto-generated by run_sec_ccm_premerge_pipeline",
        "graph TD",
        f'  A["Phase A Normalize ({_dur("phase_a_normalize")})"]',
        f'  B["Phase A Resolve Links ({_dur("phase_a_resolve_links")})"]',
        f'  C["Write sec_ccm_links_doc.parquet ({_dur("write_sec_ccm_links_doc")})"]',
        f'  D["Phase B Align Trading Date ({_dur("phase_b_align")})"]',
        f'  E["{daily_label}"]',
        f'  F["Phase B Reason + Filter Flags ({_dur("phase_b_reason_and_filter_flags")})"]',
        f'  G["Write final_flagged_data.parquet ({_dur("write_final_flagged_data")})"]',
        f'  H["Write match status + partitions ({_dur("write_status_and_partitions")})"]',
        f'  I["Write unmatched diagnostics ({_dur("write_unmatched_diagnostics")})"]',
        f'  J["Write clean outputs + allowlists ({_dur("write_clean_and_allowlists")})"]',
        f'  K["Write join spec ({_dur("write_join_spec")})"]',
        "  A --> B --> C --> D --> E --> F --> G --> H --> I --> J --> K",
        "  H --> L[sec_ccm_match_status.parquet]",
        "  H --> M[sec_ccm_matched_filings.parquet]",
        "  H --> N[sec_ccm_unmatched_filings.parquet]",
        "  I --> O[sec_ccm_unmatched_diagnostics.parquet]",
        "  J --> P[sec_ccm_matched_clean.parquet]",
        "  J --> Q[sec_ccm_analysis_doc_ids.parquet]",
    ]
    return "\n".join(lines) + "\n"


def _build_dag_dot(step_durations_ms: dict[str, int], daily_join_enabled: bool) -> str:
    def _dur(step_name: str) -> str:
        return f"{step_durations_ms.get(step_name, 0)} ms"

    daily_label = "Phase B Daily Join\\n(disabled)"
    if daily_join_enabled:
        daily_label = f"Phase B Daily Join\\n({_dur('phase_b_daily_join')})"

    lines = [
        "digraph sec_ccm_premerge_run {",
        '  rankdir=LR;',
        '  node [shape=box, fontsize=10];',
        f'  A [label="Phase A Normalize\\n({_dur("phase_a_normalize")})"];',
        f'  B [label="Phase A Resolve Links\\n({_dur("phase_a_resolve_links")})"];',
        f'  C [label="Write sec_ccm_links_doc\\n({_dur("write_sec_ccm_links_doc")})"];',
        f'  D [label="Phase B Align\\n({_dur("phase_b_align")})"];',
        f'  E [label="{daily_label}"];',
        f'  F [label="Phase B Reason + Filters\\n({_dur("phase_b_reason_and_filter_flags")})"];',
        f'  G [label="Write final_flagged_data\\n({_dur("write_final_flagged_data")})"];',
        f'  H [label="Write status + partitions\\n({_dur("write_status_and_partitions")})"];',
        f'  I [label="Write unmatched diagnostics\\n({_dur("write_unmatched_diagnostics")})"];',
        f'  J [label="Write clean + allowlists\\n({_dur("write_clean_and_allowlists")})"];',
        f'  K [label="Write join spec\\n({_dur("write_join_spec")})"];',
        "  A -> B -> C -> D -> E -> F -> G -> H -> I -> J -> K;",
        "}",
    ]
    return "\n".join(lines) + "\n"


def _build_markdown_report(
    run_id: str,
    started_at_utc: dt.datetime,
    finished_at_utc: dt.datetime,
    join_spec: SecCcmJoinSpecV2,
    summary: dict[str, object],
    steps_df: pl.DataFrame,
    paths: dict[str, Path],
    reason_tables: dict[str, pl.DataFrame],
    unmatched_tables: dict[str, pl.DataFrame],
    acceptance_by_year: pl.DataFrame,
) -> str:
    artifact_df = pl.DataFrame(
        {
            "artifact_key": list(paths.keys()),
            "artifact_path": [str(path) for path in paths.values()],
            "rows_out": [
                _count_parquet_rows(path)
                if path.suffix.lower() == ".parquet" and path.exists()
                else None
                for path in paths.values()
            ],
        }
    ).sort("artifact_key")

    summary_df = pl.DataFrame(
        [
            {"metric": key, "value": _fmt_markdown_value(summary[key])}
            for key in summary.keys()
        ]
    )

    lines = [
        f"# SEC-CCM Pre-Merge Run Report ({run_id})",
        "",
        "## Run Metadata",
        "",
        f"- started_at_utc: `{_iso_utc(started_at_utc)}`",
        f"- finished_at_utc: `{_iso_utc(finished_at_utc)}`",
        f"- phase_b_alignment_mode: `{join_spec.phase_b_alignment_mode.value}`",
        f"- phase_b_daily_join_mode: `{join_spec.phase_b_daily_join_mode.value}`",
        f"- daily_join_enabled: `{join_spec.daily_join_enabled}`",
        f"- daily_join_source: `{join_spec.daily_join_source}`",
        "",
        "## Summary Metrics",
        "",
        _to_markdown_table(summary_df, max_rows=100),
        "",
        "## Step Performance",
        "",
        _to_markdown_table(steps_df.sort("step_order"), max_rows=200),
        "",
        "## Artifact Outputs",
        "",
        _to_markdown_table(artifact_df, max_rows=200),
        "",
        "## Match Reason Counts",
        "",
        "### Final Match Reason",
        "",
        _to_markdown_table(reason_tables["match_reason_counts"]),
        "",
        "### Phase A Reason",
        "",
        _to_markdown_table(reason_tables["phase_a_reason_counts"]),
        "",
        "### Phase B Reason",
        "",
        _to_markdown_table(reason_tables["phase_b_reason_counts"]),
        "",
        "## Unmatched Diagnostics",
        "",
        "### Signal Counts",
        "",
        _to_markdown_table(unmatched_tables["diag_signals"]),
        "",
        "### Unmatched by Year/Form",
        "",
        _to_markdown_table(unmatched_tables["unmatched_by_year_form"], max_rows=40),
        "",
        "### Top Unmatched CIK",
        "",
        _to_markdown_table(unmatched_tables["top_unmatched_cik"], max_rows=25),
        "",
        "## Acceptance Datetime Coverage by Year",
        "",
        _to_markdown_table(acceptance_by_year, max_rows=60),
        "",
    ]
    return "\n".join(lines)


def run_sec_ccm_premerge_pipeline(
    sec_filings_lf: pl.LazyFrame,
    link_universe_lf: pl.LazyFrame,
    trading_calendar_lf: pl.LazyFrame,
    output_dir: Path,
    *,
    daily_lf: pl.LazyFrame | None = None,
    join_spec: SecCcmJoinSpecV1 | SecCcmJoinSpecV2 = SecCcmJoinSpecV1(),
    emit_run_report: bool = True,
) -> dict[str, Path]:
    """
    Run two-phase SEC-CCM pre-merge at doc grain and persist canonical artifacts.
    Optionally writes automatic run diagnostics/performance artifacts.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    join_spec_v2 = normalize_sec_ccm_join_spec(join_spec)

    run_started_utc = dt.datetime.now(dt.timezone.utc)
    run_started_perf = time.perf_counter()
    run_id = f"sec_ccm_{run_started_utc.strftime('%Y%m%dT%H%M%SZ')}_{uuid.uuid4().hex[:8]}"
    step_started_utc = run_started_utc
    step_started_perf = run_started_perf
    step_events: list[dict[str, Any]] = []
    step_order = 0

    def _record_step(
        step_name: str,
        *,
        artifact_key: str | None = None,
        artifact_path: Path | None = None,
        rows_out: int | None = None,
        notes: str | None = None,
    ) -> None:
        nonlocal step_started_utc, step_started_perf, step_order
        step_finished_utc = dt.datetime.now(dt.timezone.utc)
        step_finished_perf = time.perf_counter()
        step_order += 1
        step_events.append(
            {
                "step_order": step_order,
                "step_name": step_name,
                "started_at_utc": _iso_utc(step_started_utc),
                "finished_at_utc": _iso_utc(step_finished_utc),
                "duration_ms": int(round((step_finished_perf - step_started_perf) * 1000.0)),
                "artifact_key": artifact_key,
                "artifact_path": str(artifact_path) if artifact_path is not None else None,
                "rows_out": rows_out,
                "notes": notes,
            }
        )
        step_started_utc = step_finished_utc
        step_started_perf = step_finished_perf

    phase_a_norm = normalize_sec_filings_phase_a(sec_filings_lf)
    _assert_unique_doc_id(phase_a_norm, "sec_filings")
    _record_step("phase_a_normalize")

    phase_a_links = resolve_links_phase_a(phase_a_norm, link_universe_lf)
    _assert_unique_doc_id(phase_a_links, "Phase A links")
    _record_step("phase_a_resolve_links")

    paths: dict[str, Path] = {}
    paths["sec_ccm_links_doc"] = _write_lazy_parquet(
        phase_a_links,
        output_dir / "sec_ccm_links_doc.parquet",
    )
    _record_step(
        "write_sec_ccm_links_doc",
        artifact_key="sec_ccm_links_doc",
        artifact_path=paths["sec_ccm_links_doc"],
        rows_out=_count_parquet_rows(paths["sec_ccm_links_doc"]),
    )

    phase_b_aligned = align_doc_dates_phase_b(phase_a_links, trading_calendar_lf, join_spec_v2)
    _record_step("phase_b_align")
    if join_spec_v2.daily_join_enabled:
        if daily_lf is None:
            raise ValueError("daily_lf is required when join_spec_v2.daily_join_enabled=True")
        phase_b_joined = join_daily_phase_b(phase_b_aligned, daily_lf, join_spec_v2)
        diagnostics_daily_lf = daily_lf
        _record_step("phase_b_daily_join")
    else:
        phase_b_joined = join_daily_phase_b(
            phase_b_aligned,
            pl.DataFrame({"KYPERMNO": [], "CALDT": []}).lazy(),
            join_spec_v2,
        )
        diagnostics_daily_lf = trading_calendar_lf
        _record_step("phase_b_daily_join", notes="daily_join_enabled=False")

    final_doc = apply_phase_b_reason_codes(phase_a_links, phase_b_joined, join_spec_v2)
    if join_spec_v2.daily_join_enabled:
        final_doc = apply_concept_filter_flags_doc(final_doc)
    else:
        # In no-daily mode we preserve doc rows and leave filter pass flags as explicit False.
        final_doc = final_doc.with_columns(
            pl.lit(False).alias("filter_price_pass"),
            pl.lit(False).alias("filter_common_stock_pass"),
            pl.lit(False).alias("filter_major_exchange_pass"),
            pl.lit(False).alias("filter_liquidity_pass"),
            pl.lit(False).alias("filter_non_microcap_pass"),
            pl.lit(False).alias("passes_all_filters"),
        )
    _assert_unique_doc_id(final_doc, "final doc output")
    _record_step("phase_b_reason_and_filter_flags")

    paths["final_flagged_data"] = _write_lazy_parquet(final_doc, output_dir / "final_flagged_data.parquet")
    _record_step(
        "write_final_flagged_data",
        artifact_key="final_flagged_data",
        artifact_path=paths["final_flagged_data"],
        rows_out=_count_parquet_rows(paths["final_flagged_data"]),
    )

    match_status = build_match_status_doc(final_doc)
    paths["sec_ccm_match_status"] = _write_lazy_parquet(
        match_status,
        output_dir / "sec_ccm_match_status.parquet",
    )

    matched = final_doc.filter(pl.col("match_reason_code") == pl.lit(MatchReasonCode.OK.value))
    unmatched = final_doc.filter(pl.col("match_reason_code") != pl.lit(MatchReasonCode.OK.value))
    paths["sec_ccm_matched_filings"] = _write_lazy_parquet(
        matched,
        output_dir / "sec_ccm_matched_filings.parquet",
    )
    paths["sec_ccm_unmatched_filings"] = _write_lazy_parquet(
        unmatched,
        output_dir / "sec_ccm_unmatched_filings.parquet",
    )
    _record_step(
        "write_status_and_partitions",
        rows_out=_count_parquet_rows(paths["sec_ccm_match_status"]),
        notes="sec_ccm_match_status + matched + unmatched",
    )

    unmatched_diagnostics = build_unmatched_diagnostics_doc(final_doc, link_universe_lf, diagnostics_daily_lf)
    paths["sec_ccm_unmatched_diagnostics"] = _write_lazy_parquet(
        unmatched_diagnostics,
        output_dir / "sec_ccm_unmatched_diagnostics.parquet",
    )
    _record_step(
        "write_unmatched_diagnostics",
        artifact_key="sec_ccm_unmatched_diagnostics",
        artifact_path=paths["sec_ccm_unmatched_diagnostics"],
        rows_out=_count_parquet_rows(paths["sec_ccm_unmatched_diagnostics"]),
    )

    matched_clean = matched
    paths["sec_ccm_matched_clean"] = _write_lazy_parquet(
        matched_clean,
        output_dir / "sec_ccm_matched_clean.parquet",
    )
    paths["sec_ccm_matched_clean_filtered"] = _write_lazy_parquet(
        matched_clean.filter(pl.col("passes_all_filters").fill_null(False)),
        output_dir / "sec_ccm_matched_clean_filtered.parquet",
    )

    paths["sec_ccm_analysis_doc_ids"] = _write_lazy_parquet(
        match_status.filter(pl.col("match_flag"))
        .select(pl.col("doc_id").cast(pl.Utf8))
        .unique(),
        output_dir / "sec_ccm_analysis_doc_ids.parquet",
    )
    paths["sec_ccm_diagnostic_doc_ids"] = _write_lazy_parquet(
        match_status.filter(pl.col("match_flag").not_())
        .select(pl.col("doc_id").cast(pl.Utf8))
        .unique(),
        output_dir / "sec_ccm_diagnostic_doc_ids.parquet",
    )
    _record_step(
        "write_clean_and_allowlists",
        notes="matched_clean + matched_clean_filtered + analysis/diagnostic allowlists",
    )

    paths["sec_ccm_join_spec_v1"] = join_spec_v2.write_json(output_dir / "sec_ccm_join_spec_v1.json")
    _record_step(
        "write_join_spec",
        artifact_key="sec_ccm_join_spec_v1",
        artifact_path=paths["sec_ccm_join_spec_v1"],
    )

    if emit_run_report:
        steps_df = _step_events_to_frame(run_id, step_events)
        paths["sec_ccm_run_steps"] = _write_lazy_parquet(
            steps_df.lazy(),
            output_dir / "sec_ccm_run_steps.parquet",
        )

        step_duration_lookup = {
            row["step_name"]: int(row["duration_ms"])
            for row in steps_df.iter_rows(named=True)
        }
        paths["sec_ccm_run_dag_mermaid"] = _write_text(
            output_dir / "sec_ccm_run_dag.mmd",
            _build_dag_mermaid(step_duration_lookup, join_spec_v2.daily_join_enabled),
        )
        paths["sec_ccm_run_dag_dot"] = _write_text(
            output_dir / "sec_ccm_run_dag.dot",
            _build_dag_dot(step_duration_lookup, join_spec_v2.daily_join_enabled),
        )

        summary = _build_summary_metrics(paths["final_flagged_data"])
        print(
            "[sec_ccm] daily lag gate "
            f"mode={join_spec_v2.phase_b_daily_join_mode.value} "
            f"threshold_days={join_spec_v2.daily_join_max_forward_lag_days} "
            f"rejected_rows={int(summary.get('n_daily_lag_gate_rejected', 0) or 0)}"
        )
        reason_tables = _build_reason_count_tables(paths["sec_ccm_match_status"])
        unmatched_tables = _build_unmatched_tables(paths["sec_ccm_unmatched_diagnostics"])
        acceptance_by_year = _build_acceptance_coverage_by_year(paths["sec_ccm_match_status"])
        run_finished_utc = dt.datetime.now(dt.timezone.utc)
        run_finished_perf = time.perf_counter()
        manifest_payload: dict[str, object] = {
            "run_id": run_id,
            "pipeline_name": "sec_ccm_premerge",
            "started_at_utc": _iso_utc(run_started_utc),
            "finished_at_utc": _iso_utc(run_finished_utc),
            "duration_ms": int(round((run_finished_perf - run_started_perf) * 1000.0)),
            "join_spec": join_spec_v2.to_dict(),
            "summary": summary,
            "slowest_steps": (
                steps_df.sort("duration_ms", descending=True)
                .head(10)
                .select("step_name", "duration_ms", "artifact_key", "rows_out")
                .to_dicts()
            ),
            "artifacts": {key: str(path) for key, path in sorted(paths.items())},
        }
        paths["sec_ccm_run_manifest"] = _write_json(
            output_dir / "sec_ccm_run_manifest.json",
            manifest_payload,
        )

        report_markdown = _build_markdown_report(
            run_id,
            run_started_utc,
            run_finished_utc,
            join_spec_v2,
            summary,
            steps_df,
            paths,
            reason_tables,
            unmatched_tables,
            acceptance_by_year,
        )
        paths["sec_ccm_run_report"] = _write_text(
            output_dir / "sec_ccm_run_report.md",
            report_markdown,
        )
    return paths
