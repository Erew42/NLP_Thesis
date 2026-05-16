from __future__ import annotations

import argparse
import gc
import json
import sys
import time
from collections.abc import Callable
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import polars as pl
from polars.testing import assert_frame_equal

SRC_ROOT = Path(__file__).resolve().parent
REPO_ROOT = SRC_ROOT.parent
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from thesis_refinitiv import bridge
from thesis_pkg.pipelines.refinitiv import authority


TARGET_STATIC_CHECKS: tuple[tuple[Path, str, tuple[str, ...]], ...] = (
    (
        SRC_ROOT / "rust" / "lm2011_rust" / "src" / "refinitiv_bridge.rs",
        "refinitiv_bridge_resolution_frame_rows_from_columns",
        ("refinitiv_bridge_py_dict_rows_from_column_values",),
    ),
    (
        SRC_ROOT / "rust" / "lm2011_rust" / "src" / "refinitiv_bridge.rs",
        "refinitiv_bridge_ownership_universe_handoff_columns",
        ("refinitiv_bridge_py_dict_rows_from_column_values",),
    ),
    (
        SRC_ROOT / "rust" / "lm2011_rust" / "src" / "refinitiv_authority.rs",
        "refinitiv_authority_candidate_metric_record_columns",
        (
            "py_dict_rows_from_column_values",
            "py_dict_from_column_row",
            "request_rows",
            "result_rows",
        ),
    ),
    (
        SRC_ROOT / "rust" / "lm2011_rust" / "src" / "refinitiv_authority.rs",
        "refinitiv_authority_final_panel_rows_columns",
        ("py_dict_rows_from_column_values", "py_dict_from_column_row"),
    ),
    (
        SRC_ROOT / "rust" / "lm2011_rust" / "src" / "refinitiv_authority.rs",
        "refinitiv_authority_review_required_rows_columns",
        ("py_dict_rows_from_column_values", "py_dict_from_column_row"),
    ),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--step1-root",
        type=Path,
        default=REPO_ROOT / "full_data_run" / "refinitiv_step1",
    )
    parser.add_argument(
        "--sample-root",
        type=Path,
        default=REPO_ROOT / "full_data_run" / "sample_5pct_seed42",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=SRC_ROOT / "benchmark_results" / "refinitiv_column_native_benchmark.json",
    )
    parser.add_argument("--subset-permnos", type=int, default=750)
    parser.add_argument(
        "--skip-full-candidate",
        action="store_true",
        help="Skip full Rust candidate-metrics parity against persisted step1 artifacts.",
    )
    parser.add_argument(
        "--only",
        choices=("all", "candidate-rust-subset", "candidate-python-subset"),
        default="all",
        help="Run only one candidate-metrics subset mode for external memory probes.",
    )
    return parser.parse_args()


def resolve_existing(path: Path) -> Path:
    path = path if path.is_absolute() else REPO_ROOT / path
    if not path.exists():
        raise FileNotFoundError(path)
    return path


def function_body(text: str, function_name: str) -> str:
    marker = f"fn {function_name}"
    start = text.find(marker)
    if start < 0:
        raise ValueError(f"missing Rust function: {function_name}")
    brace_start = text.find("{", start)
    if brace_start < 0:
        raise ValueError(f"missing Rust function body: {function_name}")
    depth = 0
    for idx in range(brace_start, len(text)):
        char = text[idx]
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return text[brace_start : idx + 1]
    raise ValueError(f"unterminated Rust function body: {function_name}")


def static_checks() -> dict[str, Any]:
    checks: dict[str, Any] = {}
    for path, function_name, forbidden_terms in TARGET_STATIC_CHECKS:
        body = function_body(path.read_text(encoding="utf-8"), function_name)
        hits = [term for term in forbidden_terms if term in body]
        checks[function_name] = {
            "path": str(path.relative_to(REPO_ROOT)),
            "forbidden_terms": list(forbidden_terms),
            "forbidden_hits": hits,
            "passed": not hits,
        }
    if not all(check["passed"] for check in checks.values()):
        raise AssertionError(f"static column-native checks failed: {checks}")
    return checks


def timed(label: str, fn: Callable[[], Any], results: dict[str, Any]) -> Any:
    gc.collect()
    start = time.perf_counter()
    out = fn()
    seconds = time.perf_counter() - start
    results.setdefault("benchmarks", {})[label] = {"seconds": seconds}
    print(f"{label}: {seconds:.3f}s", flush=True)
    return out


@contextmanager
def rust_enabled(module: Any):
    original = module._lm2011_rust
    if original is None:
        raise RuntimeError(f"Rust extension is not available for {module.__name__}")
    try:
        module._lm2011_rust = original
        yield
    finally:
        module._lm2011_rust = original


@contextmanager
def rust_disabled(module: Any):
    original = module._lm2011_rust
    try:
        module._lm2011_rust = None
        yield
    finally:
        module._lm2011_rust = original


def canonical_frame(df: pl.DataFrame, columns: list[str], sort_keys: list[str]) -> pl.DataFrame:
    out = df.select(columns)
    keys = [key for key in sort_keys if key in out.columns]
    if keys:
        out = out.sort(keys)
    return out


def compare_frames(
    left: pl.DataFrame,
    right: pl.DataFrame,
    *,
    columns: list[str] | tuple[str, ...],
    sort_keys: list[str],
    abs_tol: float = 1e-8,
) -> None:
    column_list = list(columns)
    assert_frame_equal(
        canonical_frame(left, column_list, sort_keys),
        canonical_frame(right, column_list, sort_keys),
        check_row_order=False,
        check_column_order=True,
        check_dtypes=True,
        check_exact=False,
        rel_tol=1e-7,
        abs_tol=abs_tol,
    )


def normalize_candidate_meta(meta: dict[tuple[str, str], dict[str, Any]]) -> dict[str, dict[str, Any]]:
    normalized: dict[str, dict[str, Any]] = {}
    for key, row in meta.items():
        out = dict(row)
        out.pop("request_rows", None)
        out.pop("result_rows", None)
        observations = []
        for obs_key, values in out.get("observation_value_sets", {}).items():
            date_value, category = obs_key
            observations.append(
                (
                    date_value.isoformat() if hasattr(date_value, "isoformat") else str(date_value),
                    str(category),
                    sorted("None" if value is None else f"{float(value):.17g}" for value in values),
                )
            )
        out["observation_value_sets"] = sorted(observations)
        unique_rows = []
        for date_value, category, value in out.get("unique_row_set", set()):
            unique_rows.append(
                (
                    date_value.isoformat() if hasattr(date_value, "isoformat") else str(date_value),
                    str(category),
                    "None" if value is None else f"{float(value):.17g}",
                )
            )
        out["unique_row_set"] = sorted(unique_rows)
        normalized[f"{key[0]}|{key[1]}"] = out
    return normalized


def compare_candidate_outputs(left: tuple[Any, ...], right: tuple[Any, ...]) -> None:
    compare_frames(
        left[0],
        right[0],
        columns=list(authority.CANDIDATE_METRIC_COLUMNS),
        sort_keys=["KYPERMNO", "candidate_ric"],
    )
    if normalize_candidate_meta(left[1]) != normalize_candidate_meta(right[1]):
        raise AssertionError("candidate_meta mismatch after excluding source row payloads")
    if {key: set(value) for key, value in left[2].items()} != {
        key: set(value) for key, value in right[2].items()
    }:
        raise AssertionError("permno date-set mismatch")
    if {key: set(value) for key, value in left[3].items()} != {
        key: set(value) for key, value in right[3].items()
    }:
        raise AssertionError("permno row-set mismatch")


def filter_permnos(df: pl.DataFrame, permnos: list[str]) -> pl.DataFrame:
    if "KYPERMNO" not in df.columns:
        return df
    return df.filter(pl.col("KYPERMNO").cast(pl.Utf8, strict=False).is_in(permnos))


def write_results(results: dict[str, Any], output_json: Path) -> None:
    output_json = output_json if output_json.is_absolute() else REPO_ROOT / output_json
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(results, indent=2, sort_keys=True, default=str), encoding="utf-8")
    print(json.dumps(results, indent=2, sort_keys=True, default=str), flush=True)


def artifact_paths(step1_root: Path) -> dict[str, Path]:
    ownership_universe = step1_root / "ownership_universe_common_stock"
    ownership_authority = step1_root / "ownership_authority_common_stock"
    paths = {
        "lookup_extended": step1_root / "refinitiv_ric_lookup_handoff_common_stock_extended.parquet",
        "resolution": step1_root / "refinitiv_ric_resolution_common_stock.parquet",
        "handoff": ownership_universe / "refinitiv_ownership_universe_handoff_common_stock.parquet",
        "row_summary": ownership_universe / "refinitiv_ownership_universe_row_summary.parquet",
        "results": ownership_universe / "refinitiv_ownership_universe_results.parquet",
        "candidate_metrics": ownership_authority / "refinitiv_permno_ownership_candidate_metrics.parquet",
        "alias_diagnostics": ownership_authority / "refinitiv_permno_ownership_alias_diagnostics.parquet",
        "authority_decisions": ownership_authority / "refinitiv_permno_ownership_authority_decisions.parquet",
        "authority_exceptions": ownership_authority / "refinitiv_permno_ownership_authority_exceptions.parquet",
        "review_required": ownership_authority / "refinitiv_permno_ownership_review_required.parquet",
        "ticker_candidates": ownership_authority / "refinitiv_permno_ownership_ticker_candidates.parquet",
        "final_panel": ownership_authority / "refinitiv_permno_date_ownership_panel.parquet",
    }
    return {key: resolve_existing(value) for key, value in paths.items()}


def main() -> None:
    args = parse_args()
    step1_root = resolve_existing(args.step1_root)
    sample_root = resolve_existing(args.sample_root)
    paths = artifact_paths(step1_root)

    results: dict[str, Any] = {
        "step1_root": str(step1_root),
        "sample_root": str(sample_root),
        "sample_refinitiv_artifact_count": sum(1 for _ in sample_root.rglob("*refinitiv*")),
        "static_checks": static_checks(),
        "benchmarks": {},
        "parity": {},
        "row_counts": {},
    }

    if args.only != "all":
        row_summary_df = pl.read_parquet(paths["row_summary"])
        normalized_row_summary = authority._normalize_row_summary_df(row_summary_df)
        selected_permnos = (
            normalized_row_summary.get_column("KYPERMNO")
            .cast(pl.Utf8, strict=False)
            .drop_nulls()
            .unique(maintain_order=True)
            .head(args.subset_permnos)
            .to_list()
        )
        subset_row_summary = filter_permnos(normalized_row_summary, selected_permnos)
        ownership_results_subset = (
            pl.scan_parquet(paths["results"])
            .filter(pl.col("KYPERMNO").cast(pl.Utf8, strict=False).is_in(selected_permnos))
            .collect()
        )
        subset_results = authority._normalize_results_df(ownership_results_subset)
        results["row_counts"].update(
            {
                "ownership_results_subset_loaded": ownership_results_subset.height,
                "ownership_row_summary": row_summary_df.height,
                "subset_permnos": len(selected_permnos),
                "subset_results": subset_results.height,
                "subset_row_summary": subset_row_summary.height,
            }
        )
        if args.only == "candidate-rust-subset":
            with rust_enabled(authority):
                out = timed(
                    "authority_candidate_metrics_rust_subset",
                    lambda: authority._build_candidate_metrics_from_frames(
                        subset_row_summary,
                        subset_results,
                    ),
                    results,
                )
            if any("request_rows" in meta or "result_rows" in meta for meta in out[1].values()):
                raise AssertionError("column-native candidate metrics returned reconstructed source rows")
        else:
            with rust_disabled(authority):
                out = timed(
                    "authority_candidate_metrics_python_subset",
                    lambda: authority._build_candidate_metrics_from_frames(
                        subset_row_summary,
                        subset_results,
                    ),
                    results,
                )
        results["row_counts"]["candidate_metrics"] = out[0].height
        results["parity"]["single_mode"] = args.only
        write_results(results, args.output_json)
        return

    lookup_df = pl.read_parquet(paths["lookup_extended"])
    persisted_resolution_df = pl.read_parquet(paths["resolution"])
    persisted_handoff_df = pl.read_parquet(paths["handoff"])

    results["row_counts"].update(
        {
            "lookup_extended": lookup_df.height,
            "resolution": persisted_resolution_df.height,
            "ownership_handoff": persisted_handoff_df.height,
        }
    )

    with rust_enabled(bridge):
        resolution_rust = timed(
            "resolution_frame_rust_full",
            lambda: bridge.build_refinitiv_step1_resolution_frame(lookup_df),
            results,
        )
    with rust_disabled(bridge):
        resolution_python = timed(
            "resolution_frame_python_full",
            lambda: bridge.build_refinitiv_step1_resolution_frame(lookup_df),
            results,
        )
    compare_frames(
        resolution_rust,
        resolution_python,
        columns=list(bridge.RIC_LOOKUP_RESOLUTION_OUTPUT_COLUMNS),
        sort_keys=["bridge_row_id"],
    )
    compare_frames(
        resolution_rust,
        persisted_resolution_df,
        columns=list(bridge.RIC_LOOKUP_RESOLUTION_OUTPUT_COLUMNS),
        sort_keys=["bridge_row_id"],
    )
    results["parity"]["resolution_frame"] = "rust_vs_python_and_persisted"
    del lookup_df, resolution_rust, resolution_python
    gc.collect()

    with rust_enabled(bridge):
        handoff_rust, handoff_summary = timed(
            "ownership_universe_handoff_rust_full",
            lambda: bridge.build_refinitiv_step1_ownership_universe_handoff(
                persisted_resolution_df,
                include_ticker_fallback=False,
            ),
            results,
        )
    with rust_disabled(bridge):
        handoff_python, _ = timed(
            "ownership_universe_handoff_python_full",
            lambda: bridge.build_refinitiv_step1_ownership_universe_handoff(
                persisted_resolution_df,
                include_ticker_fallback=False,
            ),
            results,
        )
    compare_frames(
        handoff_rust,
        handoff_python,
        columns=list(bridge.OWNERSHIP_UNIVERSE_HANDOFF_COLUMNS),
        sort_keys=["bridge_row_id", "ownership_lookup_role", "lookup_input", "lookup_input_source"],
    )
    compare_frames(
        handoff_rust,
        persisted_handoff_df,
        columns=list(bridge.OWNERSHIP_UNIVERSE_HANDOFF_COLUMNS),
        sort_keys=["bridge_row_id", "ownership_lookup_role", "lookup_input", "lookup_input_source"],
    )
    results["parity"]["ownership_universe_handoff"] = {
        "status": "rust_vs_python_and_persisted",
        "summary": handoff_summary,
    }
    del persisted_handoff_df, handoff_rust, handoff_python
    gc.collect()

    row_summary_df = pl.read_parquet(paths["row_summary"])
    results["row_counts"].update(
        {
            "ownership_row_summary": row_summary_df.height,
        }
    )

    normalized_row_summary = authority._normalize_row_summary_df(row_summary_df)
    del row_summary_df
    gc.collect()

    if not args.skip_full_candidate:
        ownership_results_df = pl.read_parquet(paths["results"])
        results["row_counts"]["ownership_results"] = ownership_results_df.height
        normalized_results = authority._normalize_results_df(ownership_results_df)
        del ownership_results_df
        gc.collect()
        with rust_enabled(authority):
            candidate_full = timed(
                "authority_candidate_metrics_rust_full",
                lambda: authority._build_candidate_metrics_from_frames(
                    normalized_row_summary,
                    normalized_results,
                ),
                results,
            )
        persisted_candidate_metrics = pl.read_parquet(paths["candidate_metrics"])
        compare_frames(
            candidate_full[0],
            persisted_candidate_metrics,
            columns=list(authority.CANDIDATE_METRIC_COLUMNS),
            sort_keys=["KYPERMNO", "candidate_ric"],
        )
        results["parity"]["authority_candidate_metrics_full"] = "rust_vs_persisted"
        results["row_counts"]["candidate_metrics"] = candidate_full[0].height
        del candidate_full
        gc.collect()

    selected_permnos = (
        normalized_row_summary.get_column("KYPERMNO")
        .cast(pl.Utf8, strict=False)
        .drop_nulls()
        .unique(maintain_order=True)
        .head(args.subset_permnos)
        .to_list()
    )
    if not selected_permnos:
        raise RuntimeError("No KYPERMNO values available for subset probe")
    subset_resolution = filter_permnos(persisted_resolution_df, selected_permnos)
    subset_row_summary = filter_permnos(normalized_row_summary, selected_permnos)
    if args.skip_full_candidate:
        ownership_results_subset = (
            pl.scan_parquet(paths["results"])
            .filter(pl.col("KYPERMNO").cast(pl.Utf8, strict=False).is_in(selected_permnos))
            .collect()
        )
        subset_results = authority._normalize_results_df(ownership_results_subset)
        results["row_counts"]["ownership_results_subset_loaded"] = ownership_results_subset.height
        del ownership_results_subset
    else:
        subset_results = filter_permnos(normalized_results, selected_permnos)
    results["row_counts"].update(
        {
            "subset_permnos": len(selected_permnos),
            "subset_resolution": subset_resolution.height,
            "subset_row_summary": subset_row_summary.height,
            "subset_results": subset_results.height,
        }
    )

    with rust_enabled(authority):
        candidate_subset_rust = timed(
            "authority_candidate_metrics_rust_subset",
            lambda: authority._build_candidate_metrics_from_frames(
                subset_row_summary,
                subset_results,
            ),
            results,
        )
    if any("request_rows" in meta or "result_rows" in meta for meta in candidate_subset_rust[1].values()):
        raise AssertionError("column-native candidate metrics returned reconstructed source rows")
    with rust_disabled(authority):
        candidate_subset_python = timed(
            "authority_candidate_metrics_python_subset",
            lambda: authority._build_candidate_metrics_from_frames(
                subset_row_summary,
                subset_results,
            ),
            results,
        )
    compare_candidate_outputs(candidate_subset_rust, candidate_subset_python)
    results["parity"]["authority_candidate_metrics_subset"] = "rust_vs_python"
    del candidate_subset_rust, candidate_subset_python
    gc.collect()

    with rust_enabled(authority):
        authority_tables_rust, _ = timed(
            "authority_tables_rust_subset",
            lambda: authority.build_refinitiv_step1_ownership_authority_tables(
                subset_resolution,
                subset_results,
                subset_row_summary,
            ),
            results,
        )
    with rust_disabled(authority):
        authority_tables_python, _ = timed(
            "authority_tables_python_subset",
            lambda: authority.build_refinitiv_step1_ownership_authority_tables(
                subset_resolution,
                subset_results,
                subset_row_summary,
            ),
            results,
        )

    table_specs: dict[str, tuple[tuple[str, ...], list[str], float]] = {
        "candidate_metrics": (authority.CANDIDATE_METRIC_COLUMNS, ["KYPERMNO", "candidate_ric"], 1e-8),
        "alias_diagnostics": (
            authority.ALIAS_DIAGNOSTIC_COLUMNS,
            ["KYPERMNO", "left_candidate_ric", "right_candidate_ric"],
            1e-8,
        ),
        "authority_decisions": (authority.AUTHORITY_DECISION_COLUMNS, ["KYPERMNO"], 1e-8),
        "authority_exceptions": (
            authority.AUTHORITY_EXCEPTION_COLUMNS,
            ["KYPERMNO", "authoritative_component_id", "authority_window_start_date"],
            1e-8,
        ),
        "review_required": (authority.REVIEW_REQUIRED_COLUMNS, ["KYPERMNO"], 1e-8),
        "ticker_candidates": (authority.TICKER_CANDIDATE_COLUMNS, ["KYPERMNO", "candidate_ric"], 1e-8),
        "final_panel": (
            authority.FINAL_PANEL_COLUMNS,
            ["KYPERMNO", "returned_date", "returned_category", "source_candidate_ric"],
            1e-8,
        ),
    }
    persisted_table_paths = {
        "candidate_metrics": paths["candidate_metrics"],
        "alias_diagnostics": paths["alias_diagnostics"],
        "authority_decisions": paths["authority_decisions"],
        "authority_exceptions": paths["authority_exceptions"],
        "review_required": paths["review_required"],
        "ticker_candidates": paths["ticker_candidates"],
        "final_panel": paths["final_panel"],
    }
    for table_name, (columns, sort_keys, abs_tol) in table_specs.items():
        compare_frames(
            authority_tables_rust[table_name],
            authority_tables_python[table_name],
            columns=columns,
            sort_keys=sort_keys,
            abs_tol=abs_tol,
        )
        persisted_subset = filter_permnos(pl.read_parquet(persisted_table_paths[table_name]), selected_permnos)
        compare_frames(
            authority_tables_rust[table_name],
            persisted_subset,
            columns=columns,
            sort_keys=sort_keys,
            abs_tol=abs_tol,
        )
    results["parity"]["authority_tables_subset"] = "rust_vs_python_and_persisted_subset"
    results["row_counts"].update(
        {f"authority_subset_{name}": table.height for name, table in authority_tables_rust.items()}
    )

    rust_seconds = results["benchmarks"]["authority_candidate_metrics_rust_subset"]["seconds"]
    python_seconds = results["benchmarks"]["authority_candidate_metrics_python_subset"]["seconds"]
    results["candidate_metrics_subset_speedup"] = python_seconds / rust_seconds if rust_seconds > 0 else None

    write_results(results, args.output_json)


if __name__ == "__main__":
    main()
