from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools import docs_trace


def _read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        return list(reader)


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


def test_run_traces_both_generates_expected_artifacts_and_render(tmp_path: Path) -> None:
    out_dir = tmp_path / "docs_metadata" / "behavior"
    publish_dir = tmp_path / "docs" / "assets" / "behavior"
    behavior_page = tmp_path / "docs" / "reference" / "behavior_evidence.md"

    manifest = docs_trace.run_traces(
        repo_root=REPO_ROOT,
        out_dir=out_dir,
        publish_dir=publish_dir,
        trace_scope="both",
        seed=42,
        overwrite=True,
    )
    assert manifest["trace_scope"] == "both"
    assert len(manifest.get("workloads", [])) == 2
    workload_names = {row["name"] for row in manifest["workloads"]}
    assert workload_names == {"sec_ccm", "boundary"}

    run_manifest_path = out_dir / "run_manifest.json"
    artifact_manifest_path = out_dir / "artifact_manifest.csv"
    step_timings_path = out_dir / "step_timings.csv"
    module_touch_path = out_dir / "module_touch_counts.csv"
    function_touch_path = out_dir / "function_call_counts.csv"
    published_artifact_manifest = publish_dir / "artifact_manifest.csv"

    assert run_manifest_path.exists()
    assert artifact_manifest_path.exists()
    assert step_timings_path.exists()
    assert module_touch_path.exists()
    assert function_touch_path.exists()
    assert published_artifact_manifest.exists()

    loaded_manifest = json.loads(run_manifest_path.read_text(encoding="utf-8"))
    assert loaded_manifest["trace_scope"] == "both"
    assert isinstance(loaded_manifest.get("artifacts"), list)
    assert loaded_manifest["artifacts"]
    assert any(row.get("published_path") for row in loaded_manifest["artifacts"])
    assert any(row.get("published_doc_path") for row in loaded_manifest["artifacts"])

    module_rows = _read_csv_rows(module_touch_path)
    function_rows = _read_csv_rows(function_touch_path)
    assert module_rows
    assert function_rows
    assert any(row["module"].startswith("thesis_pkg") for row in module_rows)
    assert any(row["function"].startswith("thesis_pkg") for row in function_rows)

    docs_trace.render_behavior_page(
        repo_root=REPO_ROOT,
        out_dir=out_dir,
        behavior_page=behavior_page,
    )
    page_content = behavior_page.read_text(encoding="utf-8")
    assert "## Workloads" in page_content
    assert "`sec_ccm`" in page_content
    assert "`boundary`" in page_content
    assert "## Top Outward-Facing Function Touches" in page_content
    assert "## Top Internal Function Touches" in page_content
    assert "../assets/behavior/" in page_content
    assert "docs_metadata/behavior" not in page_content
