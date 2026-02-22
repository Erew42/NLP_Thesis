from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import polars as pl
import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools import docs_trace


def _read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        return list(reader)


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def test_render_behavior_page_writes_placeholder_without_manifest(tmp_path: Path) -> None:
    out_dir = tmp_path / "docs_metadata" / "behavior"
    behavior_page = tmp_path / "docs" / "reference" / "behavior_evidence.md"
    written = docs_trace.render_behavior_page(
        repo_root=REPO_ROOT,
        out_dir=out_dir,
        behavior_page=behavior_page,
    )
    assert written == behavior_page.resolve()
    content = behavior_page.read_text(encoding="utf-8")
    assert "No trace artifacts are currently available." in content


def test_outward_symbol_classifier_uses_module_and_symbol_evidence(tmp_path: Path) -> None:
    outward_api_path = tmp_path / "outward_api.json"
    import_evidence_path = tmp_path / "import_evidence.json"

    _write_json(
        outward_api_path,
        {
            "all_outward_modules": ["thesis_pkg.mod_a", "thesis_pkg.mod_b"],
            "modules": {},
        },
    )
    _write_json(
        import_evidence_path,
        {
            "modules": {
                "thesis_pkg.mod_a": {
                    "imported_by_symbols": [{"symbol": "public_func"}],
                },
                "thesis_pkg.mod_b": {
                    "imported_by_symbols": [],
                },
            }
        },
    )

    outward_modules, module_symbols, modules_with_symbol_evidence = docs_trace._build_outward_classifier(
        outward_api_manifest=outward_api_path,
        import_evidence_manifest=import_evidence_path,
    )

    assert docs_trace._is_outward_facing_function(
        "thesis_pkg.mod_a:public_func",
        outward_modules=outward_modules,
        module_symbols=module_symbols,
        modules_with_symbol_evidence=modules_with_symbol_evidence,
    )
    assert not docs_trace._is_outward_facing_function(
        "thesis_pkg.mod_a:other_func",
        outward_modules=outward_modules,
        module_symbols=module_symbols,
        modules_with_symbol_evidence=modules_with_symbol_evidence,
    )
    assert docs_trace._is_outward_facing_function(
        "thesis_pkg.mod_b:any_public_name",
        outward_modules=outward_modules,
        module_symbols=module_symbols,
        modules_with_symbol_evidence=modules_with_symbol_evidence,
    )
    assert not docs_trace._is_outward_facing_function(
        "thesis_pkg.mod_b:_internal",
        outward_modules=outward_modules,
        module_symbols=module_symbols,
        modules_with_symbol_evidence=modules_with_symbol_evidence,
    )


def test_copy_publish_artifacts_handles_size_caps_previews_and_md_rename(tmp_path: Path) -> None:
    repo_root = tmp_path
    out_dir = repo_root / "docs_metadata" / "behavior"
    docs_dir = repo_root / "docs_custom"
    publish_dir = docs_dir / "assets" / "behavior"
    sec_dir = out_dir / "sec_ccm"
    sec_dir.mkdir(parents=True, exist_ok=True)
    docs_dir.mkdir(parents=True, exist_ok=True)

    # Oversized tabular artifact should be truncated deterministically.
    df = pl.DataFrame({"x": list(range(2000)), "y": list(range(2000))})
    csv_path = sec_dir / "big.csv"
    df.write_csv(csv_path)

    # Markdown artifact should be published as .md.txt to avoid MkDocs nav warnings.
    md_path = sec_dir / "sec_ccm_run_report.md"
    md_path.write_text("# report\n", encoding="utf-8")

    # Oversized non-tabular artifact should be omitted from published assets.
    blob_path = sec_dir / "large.bin"
    blob_path.write_bytes(b"x" * 8192)

    rows = docs_trace._copy_publish_artifacts(
        repo_root=repo_root,
        out_dir=out_dir,
        docs_dir=docs_dir,
        publish_dir=publish_dir,
        artifacts=[
            {"workload": "sec_ccm", "artifact_key": "csv", "canonical_path": "docs_metadata/behavior/sec_ccm/big.csv"},
            {
                "workload": "sec_ccm",
                "artifact_key": "run_report",
                "canonical_path": "docs_metadata/behavior/sec_ccm/sec_ccm_run_report.md",
            },
            {
                "workload": "sec_ccm",
                "artifact_key": "blob",
                "canonical_path": "docs_metadata/behavior/sec_ccm/large.bin",
            },
        ],
        overwrite=True,
        max_publish_size_mb=0.001,
        max_preview_rows=25,
        max_publish_rows=100,
        allow_missing_canonical=False,
    )

    by_key = {row["artifact_key"]: row for row in rows}
    csv_row = by_key["csv"]
    md_row = by_key["run_report"]
    blob_row = by_key["blob"]

    assert csv_row["publish_status"] == docs_trace.PUBLISH_STATUS_TRUNCATED
    assert csv_row["publish_note"] == docs_trace.PUBLISH_NOTE_TRUNCATED
    assert csv_row["published_doc_path"].startswith("assets/behavior/")
    assert csv_row["preview_doc_path"].endswith(".preview.html")
    truncated_df = pl.read_csv(repo_root / csv_row["published_path"])
    assert truncated_df.height == 100

    assert md_row["publish_status"] == docs_trace.PUBLISH_STATUS_PUBLISHED
    assert md_row["published_doc_path"].endswith(".md.txt")
    assert (repo_root / md_row["published_path"]).exists()

    assert blob_row["publish_status"] == docs_trace.PUBLISH_STATUS_OMITTED
    assert blob_row["publish_note"] == docs_trace.PUBLISH_NOTE_OMITTED
    assert blob_row["published_path"] == ""


def test_copy_publish_artifacts_missing_canonical_policy(tmp_path: Path) -> None:
    repo_root = tmp_path
    out_dir = repo_root / "docs_metadata" / "behavior"
    docs_dir = repo_root / "docs"
    publish_dir = docs_dir / "assets" / "behavior"
    out_dir.mkdir(parents=True, exist_ok=True)

    with pytest.raises(FileNotFoundError):
        docs_trace._copy_publish_artifacts(
            repo_root=repo_root,
            out_dir=out_dir,
            docs_dir=docs_dir,
            publish_dir=publish_dir,
            artifacts=[
                {"workload": "sec_ccm", "artifact_key": "missing", "canonical_path": "docs_metadata/behavior/missing.csv"}
            ],
            overwrite=True,
            max_publish_size_mb=5.0,
            max_preview_rows=10,
            max_publish_rows=10,
            allow_missing_canonical=False,
        )

    rows = docs_trace._copy_publish_artifacts(
        repo_root=repo_root,
        out_dir=out_dir,
        docs_dir=docs_dir,
        publish_dir=publish_dir,
        artifacts=[
            {"workload": "sec_ccm", "artifact_key": "missing", "canonical_path": "docs_metadata/behavior/missing.csv"}
        ],
        overwrite=True,
        max_publish_size_mb=5.0,
        max_preview_rows=10,
        max_publish_rows=10,
        allow_missing_canonical=True,
    )
    assert len(rows) == 1
    assert rows[0]["publish_status"] == docs_trace.PUBLISH_STATUS_OMITTED
    assert rows[0]["publish_note"] == docs_trace.PUBLISH_NOTE_MISSING_ALLOWED


def test_run_traces_boundary_generates_publish_assets_and_behavior_page(tmp_path: Path) -> None:
    out_dir = tmp_path / "docs_metadata" / "behavior"
    docs_dir = tmp_path / "docs_custom"
    publish_dir = docs_dir / "assets" / "behavior"
    behavior_page = docs_dir / "reference" / "behavior_evidence.md"

    manifest = docs_trace.run_traces(
        repo_root=REPO_ROOT,
        out_dir=out_dir,
        publish_dir=publish_dir,
        docs_dir=docs_dir,
        trace_scope="boundary",
        seed=42,
        overwrite=True,
        outward_api_manifest=REPO_ROOT / "docs_metadata" / "outward_api.json",
        import_evidence_manifest=REPO_ROOT / "docs_metadata" / "import_evidence.json",
        max_publish_size_mb=5.0,
        max_preview_rows=50,
        max_publish_rows=500,
        allow_missing_canonical=False,
    )

    assert manifest["trace_scope"] == "boundary"
    assert len(manifest.get("workloads", [])) == 1
    assert manifest["workloads"][0]["name"] == "boundary"
    assert str(manifest["trace_generation"]["docs_dir"]).replace("\\", "/").endswith("/docs_custom")
    assert manifest["artifacts"]
    assert any(row.get("preview_doc_path") for row in manifest["artifacts"])

    docs_trace.render_behavior_page(
        repo_root=REPO_ROOT,
        out_dir=out_dir,
        behavior_page=behavior_page,
    )
    page_content = behavior_page.read_text(encoding="utf-8")
    assert "## Top Outward-Facing Function Touches" in page_content
    assert "## Top Internal Function Touches" in page_content
    assert "../assets/behavior/" in page_content
    assert "docs_metadata/behavior" not in page_content

    artifact_manifest_path = out_dir / "artifact_manifest.csv"
    rows = _read_csv_rows(artifact_manifest_path)
    assert rows
    assert all("publish_status" in row for row in rows)
