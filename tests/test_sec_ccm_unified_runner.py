from __future__ import annotations

from pathlib import Path


def test_doc_ownership_stage_runs_after_sec_ccm_premerge_block() -> None:
    runner_path = Path("src/thesis_pkg/notebooks_and_scripts/sec_ccm_unified_runner.py")
    source = runner_path.read_text(encoding="utf-8")

    premerge_marker = "sec_ccm_paths: dict[str, Path] | None = None"
    doc_stage_marker = "if RUN_REFINITIV_DOC_OWNERSHIP_LM2011_EXACT_HANDOFF:"
    premerge_run_marker = "sec_ccm_paths = run_sec_ccm_premerge_pipeline("

    assert premerge_marker in source
    assert premerge_run_marker in source
    assert doc_stage_marker in source
    assert source.index(doc_stage_marker) > source.index(premerge_run_marker)
