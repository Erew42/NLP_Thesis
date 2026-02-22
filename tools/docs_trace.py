from __future__ import annotations

import argparse
import csv
import datetime as dt
import html
import json
import random
import shutil
import sys
import time
from collections import Counter
from contextlib import contextmanager
from pathlib import Path
from typing import Any


DEFAULT_TRACE_SCOPE = "both"
DEFAULT_TRACE_SEED = 42
DEFAULT_BEHAVIOR_PAGE_REL = Path("docs/reference/behavior_evidence.md")
DEFAULT_PUBLISH_DIR_REL = Path("docs/assets/behavior")
DEFAULT_DOCS_DIR_REL = Path("docs")
DEFAULT_OUTWARD_API_MANIFEST_REL = Path("docs_metadata/outward_api.json")
DEFAULT_IMPORT_EVIDENCE_MANIFEST_REL = Path("docs_metadata/import_evidence.json")
DEFAULT_MAX_PUBLISH_SIZE_MB = 5.0
DEFAULT_MAX_PREVIEW_ROWS = 500
DEFAULT_MAX_PUBLISH_ROWS = 10_000

PUBLISH_STATUS_PUBLISHED = "published"
PUBLISH_STATUS_TRUNCATED = "truncated"
PUBLISH_STATUS_OMITTED = "omitted_size_limit"
PUBLISH_NOTE_TRUNCATED = "TRUNCATED_FOR_PUBLISHING"
PUBLISH_NOTE_OMITTED = "OMITTED_SIZE_LIMIT"
PUBLISH_NOTE_MISSING_ALLOWED = "MISSING_CANONICAL_ALLOWED"

ARTIFACT_FIELDNAMES = [
    "workload",
    "artifact_key",
    "canonical_path",
    "published_path",
    "published_doc_path",
    "preview_path",
    "preview_doc_path",
    "publish_status",
    "publish_note",
    "size_bytes",
]


def _ensure_src_on_sys_path(repo_root: Path) -> None:
    src_dir = repo_root / "src"
    if src_dir.exists():
        src_str = str(src_dir)
        if src_str not in sys.path:
            sys.path.insert(0, src_str)


def _iso_utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat()


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in fieldnames})


def _rel_path(repo_root: Path, path: Path) -> str:
    if path.is_absolute():
        try:
            return path.resolve().relative_to(repo_root.resolve()).as_posix()
        except ValueError:
            return path.resolve().as_posix()
    return path.as_posix()


def _is_within(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
        return True
    except ValueError:
        return False


def _resolve_repo_path(repo_root: Path, path_value: str) -> Path:
    path = Path(path_value)
    return path if path.is_absolute() else (repo_root / path)


def _doc_relative_path(path: Path, docs_dir: Path) -> str:
    if _is_within(path, docs_dir):
        return path.resolve().relative_to(docs_dir.resolve()).as_posix()

    parts = list(path.resolve().parts)
    doc_indexes = [idx for idx, part in enumerate(parts) if part.lower() == "docs"]
    if not doc_indexes:
        return ""
    start = doc_indexes[-1] + 1
    if start >= len(parts):
        return ""
    return Path(*parts[start:]).as_posix()


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        return {}
    return payload


def _md_asset_publish_rel_path(rel_path: Path) -> Path:
    if rel_path.suffix.lower() != ".md":
        return rel_path
    return rel_path.with_name(f"{rel_path.name}.txt")


def _is_tabular_path(path: Path) -> bool:
    return path.suffix.lower() in {".csv", ".parquet"}


def _read_tabular(path: Path):
    import polars as pl

    suffix = path.suffix.lower()
    if suffix == ".csv":
        return pl.read_csv(path)
    if suffix == ".parquet":
        return pl.read_parquet(path)
    raise ValueError(f"Unsupported tabular artifact type: {path}")


def _write_tabular(df, path: Path) -> None:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        df.write_csv(path)
        return
    if suffix == ".parquet":
        df.write_parquet(path)
        return
    raise ValueError(f"Unsupported tabular artifact type: {path}")


def _render_preview_html(path: Path, *, artifact_key: str, max_preview_rows: int) -> str:
    df = _read_tabular(path).head(max_preview_rows)
    columns = list(df.columns)
    header = "".join(f"<th>{html.escape(col)}</th>" for col in columns)
    rows: list[str] = []
    for row in df.iter_rows(named=True):
        cells = "".join(
            f"<td>{html.escape('' if row[col] is None else str(row[col]))}</td>"
            for col in columns
        )
        rows.append(f"<tr>{cells}</tr>")
    body = "\n".join(rows) if rows else "<tr><td colspan='1'><em>No rows.</em></td></tr>"
    return (
        "<!doctype html>\n"
        "<html lang='en'>\n"
        "<head>\n"
        "  <meta charset='utf-8'>\n"
        f"  <title>Preview: {html.escape(artifact_key)}</title>\n"
        "  <style>body{font-family:Arial,sans-serif;margin:1rem;}table{border-collapse:collapse;width:100%;}"
        "th,td{border:1px solid #ccc;padding:0.35rem;font-size:0.85rem;text-align:left;vertical-align:top;}"
        "thead{background:#f5f5f5;}code{background:#f0f0f0;padding:0.1rem 0.2rem;}</style>\n"
        "</head>\n"
        "<body>\n"
        f"<h1>{html.escape(artifact_key)}</h1>\n"
        f"<p>Preview rows shown: <code>{max_preview_rows}</code> (or fewer if source is smaller).</p>\n"
        "<table>\n"
        f"<thead><tr>{header}</tr></thead>\n"
        f"<tbody>{body}</tbody>\n"
        "</table>\n"
        "</body>\n"
        "</html>\n"
    )


def _symbol_evidence_map(import_evidence: dict[str, Any]) -> tuple[dict[str, set[str]], set[str]]:
    module_symbols: dict[str, set[str]] = {}
    modules_with_symbol_evidence: set[str] = set()
    modules_payload = import_evidence.get("modules", {})
    if not isinstance(modules_payload, dict):
        return module_symbols, modules_with_symbol_evidence

    for module_name, payload in modules_payload.items():
        if not isinstance(payload, dict):
            continue
        imported_by_symbols = payload.get("imported_by_symbols", [])
        if not isinstance(imported_by_symbols, list) or not imported_by_symbols:
            continue
        modules_with_symbol_evidence.add(str(module_name))
        module_symbols[str(module_name)] = {
            str(row.get("symbol"))
            for row in imported_by_symbols
            if isinstance(row, dict) and row.get("symbol")
        }
    return module_symbols, modules_with_symbol_evidence


def _build_outward_classifier(
    *,
    outward_api_manifest: Path,
    import_evidence_manifest: Path,
) -> tuple[set[str], dict[str, set[str]], set[str]]:
    """Load outward-facing module/symbol evidence from extractor metadata.

    WHY (TODO by Erik): The function-touch split should follow import evidence,
    not just underscore naming, so behavior evidence reflects externally used API.
    """
    outward_api = _load_json(outward_api_manifest)
    import_evidence = _load_json(import_evidence_manifest)

    outward_modules: set[str] = set()
    all_outward = outward_api.get("all_outward_modules", [])
    if isinstance(all_outward, list):
        outward_modules.update(str(value) for value in all_outward)
    modules_payload = outward_api.get("modules", {})
    if isinstance(modules_payload, dict):
        outward_modules.update(str(module_name) for module_name in modules_payload.keys())

    module_symbols, modules_with_symbol_evidence = _symbol_evidence_map(import_evidence)
    return outward_modules, module_symbols, modules_with_symbol_evidence


def _is_outward_facing_function(
    function_key: str,
    *,
    outward_modules: set[str],
    module_symbols: dict[str, set[str]],
    modules_with_symbol_evidence: set[str],
) -> bool:
    """Classify function touches using module + symbol import evidence."""
    module_name, sep, function_name = function_key.partition(":")
    if not sep or not module_name or not function_name:
        return False
    if function_name.startswith("<") and function_name.endswith(">"):
        return False
    if function_name.startswith("_"):
        return False
    if module_name not in outward_modules:
        return False
    if module_name in modules_with_symbol_evidence:
        return function_name in module_symbols.get(module_name, set())
    return True


def _copy_publish_artifacts(
    *,
    repo_root: Path,
    out_dir: Path,
    docs_dir: Path,
    publish_dir: Path,
    artifacts: list[dict[str, Any]],
    overwrite: bool,
    max_publish_size_mb: float,
    max_preview_rows: int,
    max_publish_rows: int,
    allow_missing_canonical: bool,
) -> list[dict[str, Any]]:
    """Copy/cap artifacts for MkDocs publishing and emit publish metadata.

    WHY (TODO by Erik): Keep canonical machine artifacts unchanged in
    `docs_metadata/behavior`, while producing publish-safe assets under docs/.
    """
    if publish_dir.exists() and overwrite:
        shutil.rmtree(publish_dir)
    publish_dir.mkdir(parents=True, exist_ok=True)
    max_publish_size_bytes = int(max_publish_size_mb * 1024 * 1024)

    published_rows: list[dict[str, Any]] = []
    for row in artifacts:
        canonical = str(row.get("canonical_path", ""))
        if not canonical:
            continue
        canonical_abs = _resolve_repo_path(repo_root, canonical)
        row_copy = dict(row)
        row_copy["published_path"] = ""
        row_copy["published_doc_path"] = ""
        row_copy["preview_path"] = ""
        row_copy["preview_doc_path"] = ""
        row_copy["publish_status"] = PUBLISH_STATUS_OMITTED
        row_copy["publish_note"] = ""
        if not canonical_abs.exists() or not canonical_abs.is_file():
            if not allow_missing_canonical:
                raise FileNotFoundError(f"Missing canonical artifact: {canonical_abs}")
            row_copy["publish_note"] = PUBLISH_NOTE_MISSING_ALLOWED
            published_rows.append(row_copy)
            continue

        if _is_within(canonical_abs, out_dir):
            rel_in_trace = canonical_abs.resolve().relative_to(out_dir.resolve())
        else:
            workload = str(row.get("workload", "misc")) or "misc"
            rel_in_trace = Path(workload) / canonical_abs.name

        rel_for_publish = _md_asset_publish_rel_path(rel_in_trace)
        published_abs = publish_dir / rel_for_publish
        published_abs.parent.mkdir(parents=True, exist_ok=True)
        size_bytes = int(canonical_abs.stat().st_size)
        row_copy["size_bytes"] = size_bytes
        is_tabular = _is_tabular_path(canonical_abs)
        if is_tabular and size_bytes > max_publish_size_bytes:
            tabular_df = _read_tabular(canonical_abs).head(max_publish_rows)
            _write_tabular(tabular_df, published_abs)
            row_copy["publish_status"] = PUBLISH_STATUS_TRUNCATED
            row_copy["publish_note"] = PUBLISH_NOTE_TRUNCATED
        elif (not is_tabular) and size_bytes > max_publish_size_bytes:
            row_copy["publish_status"] = PUBLISH_STATUS_OMITTED
            row_copy["publish_note"] = PUBLISH_NOTE_OMITTED
            published_rows.append(row_copy)
            continue
        else:
            shutil.copy2(canonical_abs, published_abs)
            row_copy["publish_status"] = PUBLISH_STATUS_PUBLISHED

        row_copy["published_path"] = _rel_path(repo_root, published_abs)
        row_copy["published_doc_path"] = _doc_relative_path(published_abs, docs_dir)
        if is_tabular:
            preview_abs = published_abs.with_suffix(published_abs.suffix + ".preview.html")
            preview_abs.write_text(
                _render_preview_html(
                    canonical_abs,
                    artifact_key=str(row.get("artifact_key", "")),
                    max_preview_rows=max_preview_rows,
                ),
                encoding="utf-8",
            )
            row_copy["preview_path"] = _rel_path(repo_root, preview_abs)
            row_copy["preview_doc_path"] = _doc_relative_path(preview_abs, docs_dir)
        published_rows.append(row_copy)

    return published_rows


class _CallProfiler:
    def __init__(self) -> None:
        self.module_counts: Counter[str] = Counter()
        self.function_counts: Counter[str] = Counter()

    def __call__(self, frame, event: str, arg) -> None:  # type: ignore[no-untyped-def]
        if event != "call":
            return
        module_name = str(frame.f_globals.get("__name__", ""))
        if not module_name.startswith("thesis_pkg"):
            return
        func_name = str(frame.f_code.co_name)
        if func_name.startswith("<") and func_name.endswith(">"):
            return
        self.module_counts[module_name] += 1
        self.function_counts[f"{module_name}:{func_name}"] += 1


@contextmanager
def _profile_calls() -> Any:
    profiler = _CallProfiler()
    prior = sys.getprofile()
    sys.setprofile(profiler)
    try:
        yield profiler
    finally:
        sys.setprofile(prior)


def _build_sec_ccm_inputs() -> dict[str, Any]:
    import datetime as pydt

    import polars as pl

    sec = pl.DataFrame(
        {
            "doc_id": ["d1", "d2", "d3"],
            "cik_10": ["0000000001", "0000000002", "bad"],
            "filing_date": [pydt.date(2024, 1, 2), pydt.date(2024, 1, 4), pydt.date(2024, 1, 2)],
            "document_type_filename": ["10-K", "10-K", "10-K"],
        }
    )
    link_universe = pl.DataFrame(
        {
            "cik_10": ["0000000001", "0000000002"],
            "gvkey": ["1000", "2000"],
            "kypermno": [1, 2],
            "link_rank": [0, 0],
            "link_quality": [1.0, 1.0],
        }
    )
    trading_calendar = pl.DataFrame({"CALDT": [pydt.date(2024, 1, 3), pydt.date(2024, 1, 4)]})
    daily = pl.DataFrame(
        {
            "KYPERMNO": [1],
            "CALDT": [pydt.date(2024, 1, 3)],
            "RET": [0.01],
            "RETX": [0.01],
            "PRC": [10.0],
            "BIDLO": [9.5],
            "ASKHI": [10.5],
            "SHRCD": [10],
            "EXCHCD": [1],
            "VOL": [1000.0],
            "TCAP": [100_000_000.0],
        }
    )
    return {
        "sec": sec,
        "link_universe": link_universe,
        "trading_calendar": trading_calendar,
        "daily": daily,
    }


def _run_sec_ccm_workload(repo_root: Path, output_dir: Path) -> dict[str, Any]:
    _ensure_src_on_sys_path(repo_root)
    from thesis_pkg.core.ccm.sec_ccm_contracts import SecCcmJoinSpecV1
    from thesis_pkg.pipelines.sec_ccm_pipeline import run_sec_ccm_premerge_pipeline

    inputs = _build_sec_ccm_inputs()
    output_dir.mkdir(parents=True, exist_ok=True)

    started = _iso_utc_now()
    t0 = time.perf_counter()
    paths = run_sec_ccm_premerge_pipeline(
        inputs["sec"].lazy(),
        inputs["link_universe"].lazy(),
        inputs["trading_calendar"].lazy(),
        output_dir,
        daily_lf=inputs["daily"].lazy(),
        join_spec=SecCcmJoinSpecV1(required_daily_non_null_features=("RET",)),
        emit_run_report=True,
    )
    duration_ms = int(round((time.perf_counter() - t0) * 1000.0))
    finished = _iso_utc_now()

    artifacts: list[dict[str, Any]] = []
    for key, value in sorted(paths.items()):
        artifact_path = Path(value)
        artifacts.append(
            {
                "workload": "sec_ccm",
                "artifact_key": key,
                "canonical_path": _rel_path(repo_root, artifact_path),
                "size_bytes": artifact_path.stat().st_size if artifact_path.exists() else 0,
            }
        )

    step_rows: list[dict[str, Any]] = []
    steps_path = paths.get("sec_ccm_run_steps")
    if steps_path is not None:
        import polars as pl

        steps_df = pl.read_parquet(steps_path).sort("step_order")
        for row in steps_df.iter_rows(named=True):
            step_rows.append(
                {
                    "workload": "sec_ccm",
                    "step_name": row.get("step_name"),
                    "duration_ms": int(row.get("duration_ms") or 0),
                    "artifact_key": row.get("artifact_key") or "",
                    "rows_out": row.get("rows_out") if row.get("rows_out") is not None else "",
                    "notes": row.get("notes") or "",
                }
            )

    summary: dict[str, Any] = {}
    manifest_path = paths.get("sec_ccm_run_manifest")
    if manifest_path is not None and Path(manifest_path).exists():
        manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
        if isinstance(manifest, dict):
            summary = manifest.get("summary", {}) if isinstance(manifest.get("summary", {}), dict) else {}

    return {
        "name": "sec_ccm",
        "status": "ok",
        "started_at_utc": started,
        "finished_at_utc": finished,
        "duration_ms": duration_ms,
        "summary": summary,
        "artifacts": artifacts,
        "steps": step_rows,
    }


def _build_boundary_input_parquet(parquet_dir: Path, fixture_text: str) -> Path:
    import polars as pl

    parquet_dir.mkdir(parents=True, exist_ok=True)
    df = pl.DataFrame(
        [
            {
                "doc_id": "0000000001:000000000000000001",
                "cik": "0000000001",
                "accession_number": "0000000000-00-000001",
                "document_type_filename": "10-Q",
                "file_date_filename": "20200131",
                "full_text": fixture_text,
            }
        ]
    )
    out_path = parquet_dir / "2020.parquet"
    df.write_parquet(out_path)
    return out_path


def _run_boundary_workload(repo_root: Path, output_dir: Path, seed: int) -> dict[str, Any]:
    _ensure_src_on_sys_path(repo_root)
    from thesis_pkg.core.sec.suspicious_boundary_diagnostics import (
        DiagnosticsConfig,
        run_boundary_diagnostics,
    )

    fixture_path = repo_root / "tests" / "fixtures" / "legacy_simple_10q.txt"
    if not fixture_path.exists():
        raise FileNotFoundError(f"Boundary fixture not found: {fixture_path}")
    fixture_text = fixture_path.read_text(encoding="utf-8")

    data_dir = output_dir / "input"
    _build_boundary_input_parquet(data_dir, fixture_text)

    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / "suspicious.csv"
    report_path = output_dir / "report.txt"
    samples_dir = output_dir / "samples"
    manifest_items_path = output_dir / "manifest_items.csv"
    manifest_filings_path = output_dir / "manifest_filings.csv"

    started = _iso_utc_now()
    t0 = time.perf_counter()
    result = run_boundary_diagnostics(
        DiagnosticsConfig(
            parquet_dir=data_dir,
            out_path=out_path,
            report_path=report_path,
            samples_dir=samples_dir,
            batch_size=8,
            max_files=0,
            max_examples=5,
            enable_embedded_verifier=True,
            emit_manifest=True,
            manifest_items_path=manifest_items_path,
            manifest_filings_path=manifest_filings_path,
            sample_pass=0,
            sample_seed=seed,
            core_items=("1", "2"),
            target_set=None,
            emit_html=False,
            html_out=output_dir / "html",
            html_scope="sample",
            extraction_regime="legacy",
            diagnostics_regime="legacy",
        )
    )
    duration_ms = int(round((time.perf_counter() - t0) * 1000.0))
    finished = _iso_utc_now()

    artifacts: list[dict[str, Any]] = []
    for key, path in [
        ("boundary_suspicious_csv", out_path),
        ("boundary_report_txt", report_path),
        ("boundary_manifest_items_csv", manifest_items_path),
        ("boundary_manifest_filings_csv", manifest_filings_path),
    ]:
        artifacts.append(
            {
                "workload": "boundary",
                "artifact_key": key,
                "canonical_path": _rel_path(repo_root, path),
                "size_bytes": path.stat().st_size if path.exists() else 0,
            }
        )

    step_rows = [
        {
            "workload": "boundary",
            "step_name": "run_boundary_diagnostics",
            "duration_ms": duration_ms,
            "artifact_key": "",
            "rows_out": "",
            "notes": "fixture-driven yearly parquet replay",
        }
    ]

    return {
        "name": "boundary",
        "status": "ok",
        "started_at_utc": started,
        "finished_at_utc": finished,
        "duration_ms": duration_ms,
        "summary": result if isinstance(result, dict) else {},
        "artifacts": artifacts,
        "steps": step_rows,
    }


def _render_placeholder() -> str:
    return (
        "# Behavior Evidence\n\n"
        "No trace artifacts are currently available.\n\n"
        "Run `python tools/docs_trace.py all` to generate this page from dynamic traces.\n"
    )


def _render_behavior_markdown(manifest: dict[str, Any]) -> str:
    workloads = manifest.get("workloads", [])
    top_modules = manifest.get("top_modules", [])
    top_outward_functions = manifest.get("top_outward_functions", [])
    top_internal_functions = manifest.get("top_internal_functions", [])
    artifacts = manifest.get("artifacts", [])

    lines: list[str] = [
        "# Behavior Evidence",
        "",
        "Auto-generated trace summary from reproducible docs tooling workloads.",
        "",
        "## Run Summary",
        "",
        f"- Generated at (UTC): `{manifest.get('generated_at_utc', '')}`",
        f"- Trace scope: `{manifest.get('trace_scope', '')}`",
        f"- Seed: `{manifest.get('seed', '')}`",
        "",
        "## Workloads",
        "",
    ]
    if isinstance(workloads, list) and workloads:
        for row in workloads:
            if not isinstance(row, dict):
                continue
            lines.append(
                f"- `{row.get('name', '')}`: status=`{row.get('status', '')}`, "
                f"duration_ms=`{row.get('duration_ms', '')}`"
            )
    else:
        lines.append("- No workload records found.")

    lines.extend(["", "## Top Module Touches", ""])
    if isinstance(top_modules, list) and top_modules:
        for row in top_modules[:20]:
            if not isinstance(row, dict):
                continue
            lines.append(f"- `{row.get('module', '')}`: `{row.get('call_count', 0)}` calls")
    else:
        lines.append("- No module touch data found.")

    lines.extend(["", "## Top Outward-Facing Function Touches", ""])
    if isinstance(top_outward_functions, list) and top_outward_functions:
        for row in top_outward_functions[:20]:
            if not isinstance(row, dict):
                continue
            lines.append(f"- `{row.get('function', '')}`: `{row.get('call_count', 0)}` calls")
    else:
        lines.append("- No outward-facing function touch data found.")

    lines.extend(["", "## Top Internal Function Touches", ""])
    if isinstance(top_internal_functions, list) and top_internal_functions:
        for row in top_internal_functions[:20]:
            if not isinstance(row, dict):
                continue
            lines.append(f"- `{row.get('function', '')}`: `{row.get('call_count', 0)}` calls")
    else:
        lines.append("- No internal function touch data found.")

    lines.extend(["", "## Artifact Manifest", ""])
    if isinstance(artifacts, list) and artifacts:
        for row in artifacts[:300]:
            if not isinstance(row, dict):
                continue
            key = str(row.get("artifact_key", ""))
            publish_status = str(row.get("publish_status", ""))
            publish_note = str(row.get("publish_note", ""))
            preview_doc_path = str(row.get("preview_doc_path", ""))
            published_doc_path = str(row.get("published_doc_path", ""))
            parts = [f"`{key}`", f"status=`{publish_status}`"]
            if preview_doc_path:
                parts.append(f"[Preview](../{preview_doc_path})")
            if published_doc_path:
                parts.append(f"[Download](../{published_doc_path})")
            if publish_note:
                parts.append(f"note=`{publish_note}`")
            if not preview_doc_path and not published_doc_path:
                parts.append("_No published artifact link available._")
            lines.append(f"- {' | '.join(parts)}")
    else:
        lines.append("- No artifact references found.")

    lines.append("")
    return "\n".join(lines)


def run_traces(
    *,
    repo_root: Path,
    out_dir: Path,
    publish_dir: Path,
    docs_dir: Path,
    trace_scope: str,
    seed: int,
    overwrite: bool,
    outward_api_manifest: Path,
    import_evidence_manifest: Path,
    max_publish_size_mb: float,
    max_preview_rows: int,
    max_publish_rows: int,
    allow_missing_canonical: bool,
) -> dict[str, Any]:
    """Run deterministic trace workloads and produce docs behavior evidence.

    WHY (TODO by Erik): This binds reproducible runtime evidence to docs while
    separating canonical machine artifacts from publish-safe site assets.
    """
    repo_root = repo_root.resolve()
    out_dir = out_dir.resolve()
    publish_dir = publish_dir.resolve()
    docs_dir = docs_dir.resolve()
    random.seed(seed)

    run_manifest_path = out_dir / "run_manifest.json"
    if run_manifest_path.exists() and not overwrite:
        raise FileExistsError(
            f"Trace outputs already exist at {out_dir}. Pass --overwrite to replace them."
        )

    outward_modules, module_symbols, modules_with_symbol_evidence = _build_outward_classifier(
        outward_api_manifest=outward_api_manifest,
        import_evidence_manifest=import_evidence_manifest,
    )

    out_dir.mkdir(parents=True, exist_ok=True)

    workload_names: list[str]
    if trace_scope == "sec_ccm":
        workload_names = ["sec_ccm"]
    elif trace_scope == "boundary":
        workload_names = ["boundary"]
    else:
        workload_names = ["sec_ccm", "boundary"]

    module_counts: Counter[str] = Counter()
    function_counts: Counter[str] = Counter()
    workload_results: list[dict[str, Any]] = []
    artifact_rows: list[dict[str, Any]] = []
    step_rows: list[dict[str, Any]] = []

    for workload in workload_names:
        with _profile_calls() as profiler:
            if workload == "sec_ccm":
                result = _run_sec_ccm_workload(repo_root, out_dir / "sec_ccm")
            else:
                result = _run_boundary_workload(repo_root, out_dir / "boundary", seed)
        module_counts.update(profiler.module_counts)
        function_counts.update(profiler.function_counts)
        workload_results.append(result)
        artifact_rows.extend(result.get("artifacts", []))
        step_rows.extend(result.get("steps", []))

    artifact_rows = sorted(
        artifact_rows,
        key=lambda row: (str(row.get("workload", "")), str(row.get("artifact_key", ""))),
    )
    step_rows = sorted(
        step_rows,
        key=lambda row: (str(row.get("workload", "")), str(row.get("step_name", ""))),
    )
    module_rows = [
        {"module": module_name, "call_count": count}
        for module_name, count in sorted(
            module_counts.items(),
            key=lambda item: (-item[1], item[0]),
        )
    ]
    function_rows = [
        {"function": func_name, "call_count": count}
        for func_name, count in sorted(
            function_counts.items(),
            key=lambda item: (-item[1], item[0]),
        )
    ]
    outward_function_rows = [
        row
        for row in function_rows
        if _is_outward_facing_function(
            str(row["function"]),
            outward_modules=outward_modules,
            module_symbols=module_symbols,
            modules_with_symbol_evidence=modules_with_symbol_evidence,
        )
    ]
    internal_function_rows = [
        row
        for row in function_rows
        if not _is_outward_facing_function(
            str(row["function"]),
            outward_modules=outward_modules,
            module_symbols=module_symbols,
            modules_with_symbol_evidence=modules_with_symbol_evidence,
        )
    ]

    artifact_manifest_path = out_dir / "artifact_manifest.csv"
    step_timings_path = out_dir / "step_timings.csv"
    module_touch_path = out_dir / "module_touch_counts.csv"
    function_touch_path = out_dir / "function_call_counts.csv"

    _write_csv(
        artifact_manifest_path,
        ARTIFACT_FIELDNAMES,
        artifact_rows,
    )
    _write_csv(
        step_timings_path,
        ["workload", "step_name", "duration_ms", "artifact_key", "rows_out", "notes"],
        step_rows,
    )
    _write_csv(module_touch_path, ["module", "call_count"], module_rows)
    _write_csv(function_touch_path, ["function", "call_count"], function_rows)

    artifacts_for_publish = list(artifact_rows)
    artifacts_for_publish.extend(
        [
            {
                "workload": "trace",
                "artifact_key": "artifact_manifest_csv",
                "canonical_path": _rel_path(repo_root, artifact_manifest_path),
            },
            {
                "workload": "trace",
                "artifact_key": "step_timings_csv",
                "canonical_path": _rel_path(repo_root, step_timings_path),
            },
            {
                "workload": "trace",
                "artifact_key": "module_touch_counts_csv",
                "canonical_path": _rel_path(repo_root, module_touch_path),
            },
            {
                "workload": "trace",
                "artifact_key": "function_call_counts_csv",
                "canonical_path": _rel_path(repo_root, function_touch_path),
            },
        ]
    )

    artifacts_for_manifest = _copy_publish_artifacts(
        repo_root=repo_root,
        out_dir=out_dir,
        docs_dir=docs_dir,
        publish_dir=publish_dir,
        artifacts=artifacts_for_publish,
        overwrite=overwrite,
        max_publish_size_mb=max_publish_size_mb,
        max_preview_rows=max_preview_rows,
        max_publish_rows=max_publish_rows,
        allow_missing_canonical=allow_missing_canonical,
    )

    # Rewrite artifact manifest with published-path columns now populated.
    _write_csv(
        artifact_manifest_path,
        ARTIFACT_FIELDNAMES,
        artifacts_for_manifest,
    )
    published_artifact_manifest = publish_dir / "artifact_manifest.csv"
    published_artifact_manifest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(artifact_manifest_path, published_artifact_manifest)

    manifest = {
        "schema_version": "v2",
        "generated_at_utc": _iso_utc_now(),
        "trace_scope": trace_scope,
        "seed": seed,
        "trace_generation": {
            "allow_missing_canonical": allow_missing_canonical,
            "max_publish_size_mb": max_publish_size_mb,
            "max_preview_rows": max_preview_rows,
            "max_publish_rows": max_publish_rows,
            "docs_dir": _rel_path(repo_root, docs_dir),
        },
        "workloads": workload_results,
        "top_modules": module_rows[:100],
        "top_functions": function_rows[:100],
        "top_outward_functions": outward_function_rows[:100],
        "top_internal_functions": internal_function_rows[:100],
        "artifacts": artifacts_for_manifest,
    }
    _write_json(run_manifest_path, manifest)
    return manifest


def render_behavior_page(*, repo_root: Path, out_dir: Path, behavior_page: Path) -> Path:
    repo_root = repo_root.resolve()
    out_dir = out_dir.resolve()
    behavior_page = behavior_page.resolve()

    run_manifest_path = out_dir / "run_manifest.json"
    if run_manifest_path.exists():
        payload = _load_json(run_manifest_path)
        content = _render_behavior_markdown(payload)
    else:
        content = _render_placeholder()

    behavior_page.parent.mkdir(parents=True, exist_ok=True)
    behavior_page.write_text(content, encoding="utf-8")
    return behavior_page


def _build_parser() -> argparse.ArgumentParser:
    def _add_common_args(target: argparse.ArgumentParser) -> None:
        target.add_argument("--repo-root", type=Path, default=Path("."))
        target.add_argument("--out-dir", type=Path, default=Path("docs_metadata/behavior"))
        target.add_argument("--publish-dir", type=Path, default=DEFAULT_PUBLISH_DIR_REL)
        target.add_argument("--docs-dir", type=Path, default=DEFAULT_DOCS_DIR_REL)
        target.add_argument("--behavior-page", type=Path, default=DEFAULT_BEHAVIOR_PAGE_REL)
        target.add_argument("--outward-api-manifest", type=Path, default=DEFAULT_OUTWARD_API_MANIFEST_REL)
        target.add_argument(
            "--import-evidence-manifest",
            type=Path,
            default=DEFAULT_IMPORT_EVIDENCE_MANIFEST_REL,
        )
        target.add_argument(
            "--trace-scope",
            type=str,
            choices=("sec_ccm", "boundary", "both"),
            default=DEFAULT_TRACE_SCOPE,
        )
        target.add_argument("--seed", type=int, default=DEFAULT_TRACE_SEED)
        target.add_argument("--max-publish-size-mb", type=float, default=DEFAULT_MAX_PUBLISH_SIZE_MB)
        target.add_argument("--max-preview-rows", type=int, default=DEFAULT_MAX_PREVIEW_ROWS)
        target.add_argument("--max-publish-rows", type=int, default=DEFAULT_MAX_PUBLISH_ROWS)
        target.add_argument("--allow-missing-canonical", action="store_true")
        target.add_argument("--overwrite", action="store_true")

    parser = argparse.ArgumentParser(description="Generate dynamic behavior-evidence artifacts for docs.")
    subparsers = parser.add_subparsers(dest="command", required=True)
    run_parser = subparsers.add_parser("run", help="Generate trace artifacts and run manifest.")
    render_parser = subparsers.add_parser("render", help="Render behavior evidence markdown from run manifest.")
    all_parser = subparsers.add_parser("all", help="Run trace generation and render behavior page.")
    _add_common_args(run_parser)
    _add_common_args(render_parser)
    _add_common_args(all_parser)
    return parser


def _resolve_path(repo_root: Path, value: Path) -> Path:
    return value if value.is_absolute() else (repo_root / value)


def main() -> int:
    parser = _build_parser()
    args = parser.parse_args()

    repo_root = args.repo_root.resolve()
    out_dir = _resolve_path(repo_root, args.out_dir)
    publish_dir = _resolve_path(repo_root, args.publish_dir)
    docs_dir = _resolve_path(repo_root, args.docs_dir)
    behavior_page = _resolve_path(repo_root, args.behavior_page)
    outward_api_manifest = _resolve_path(repo_root, args.outward_api_manifest)
    import_evidence_manifest = _resolve_path(repo_root, args.import_evidence_manifest)

    if args.command == "run":
        manifest = run_traces(
            repo_root=repo_root,
            out_dir=out_dir,
            publish_dir=publish_dir,
            docs_dir=docs_dir,
            trace_scope=args.trace_scope,
            seed=int(args.seed),
            overwrite=bool(args.overwrite),
            outward_api_manifest=outward_api_manifest,
            import_evidence_manifest=import_evidence_manifest,
            max_publish_size_mb=float(args.max_publish_size_mb),
            max_preview_rows=int(args.max_preview_rows),
            max_publish_rows=int(args.max_publish_rows),
            allow_missing_canonical=bool(args.allow_missing_canonical),
        )
        print(
            f"Wrote trace manifest to {out_dir / 'run_manifest.json'} "
            f"(workloads={len(manifest.get('workloads', []))})"
        )
        return 0

    if args.command == "render":
        path = render_behavior_page(repo_root=repo_root, out_dir=out_dir, behavior_page=behavior_page)
        print(f"Wrote behavior evidence page to {path}")
        return 0

    if args.command == "all":
        manifest = run_traces(
            repo_root=repo_root,
            out_dir=out_dir,
            publish_dir=publish_dir,
            docs_dir=docs_dir,
            trace_scope=args.trace_scope,
            seed=int(args.seed),
            overwrite=bool(args.overwrite),
            outward_api_manifest=outward_api_manifest,
            import_evidence_manifest=import_evidence_manifest,
            max_publish_size_mb=float(args.max_publish_size_mb),
            max_preview_rows=int(args.max_preview_rows),
            max_publish_rows=int(args.max_publish_rows),
            allow_missing_canonical=bool(args.allow_missing_canonical),
        )
        path = render_behavior_page(repo_root=repo_root, out_dir=out_dir, behavior_page=behavior_page)
        print(
            f"Wrote trace manifest to {out_dir / 'run_manifest.json'} "
            f"(workloads={len(manifest.get('workloads', []))}); "
            f"rendered {path}"
        )
        return 0

    parser.error(f"Unsupported command: {args.command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
