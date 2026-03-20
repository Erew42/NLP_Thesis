from __future__ import annotations

import json
from pathlib import Path

from pytest import MonkeyPatch

from thesis_pkg.notebooks_and_scripts import sec_ccm_unified_runner as runner


def test_doc_ownership_stage_runs_after_sec_ccm_premerge_block() -> None:
    runner_path = Path("src/thesis_pkg/notebooks_and_scripts/sec_ccm_unified_runner.py")
    source = runner_path.read_text(encoding="utf-8")

    premerge_marker = "sec_ccm_paths = run_sec_ccm_premerge_pipeline("
    doc_stage_marker = "if RUN_REFINITIV_DOC_OWNERSHIP_LM2011_EXACT_HANDOFF:"

    assert premerge_marker in source
    assert doc_stage_marker in source
    assert source.index(doc_stage_marker) > source.index(premerge_marker)


def test_first_existing_path_prefers_existing_candidate(tmp_path: Path) -> None:
    missing = tmp_path / "missing"
    existing = tmp_path / "existing"
    existing.mkdir()

    assert runner._first_existing_path(missing, existing) == existing


def test_resolve_repo_root_honors_env_override(
    monkeypatch: MonkeyPatch,
    tmp_path: Path,
) -> None:
    repo_root = tmp_path / "repo"
    pipeline_path = repo_root / "src" / "thesis_pkg" / "pipeline.py"
    pipeline_path.parent.mkdir(parents=True)
    pipeline_path.write_text("from __future__ import annotations\n", encoding="utf-8")

    monkeypatch.setenv(runner.REPO_ROOT_ENV_VAR, str(repo_root))

    assert runner._resolve_repo_root() == repo_root


def test_resolve_ff48_siccodes_path_honors_env_override(
    monkeypatch: MonkeyPatch,
    tmp_path: Path,
) -> None:
    ff48_path = tmp_path / "FF_Siccodes_48_Industries.txt"
    ff48_path.write_text("placeholder", encoding="utf-8")

    monkeypatch.setenv("SEC_CCM_FF48_SICCODES_PATH", str(ff48_path))

    assert runner._resolve_ff48_siccodes_path(tmp_path) == ff48_path


def test_sec_ccm_unified_runner_notebook_bootstrap_is_valid() -> None:
    notebook_path = Path("src/thesis_pkg/notebooks_and_scripts/sec_ccm_unified_runner.ipynb")
    notebook = json.loads(notebook_path.read_text(encoding="utf-8"))

    assert len(notebook["cells"]) == 3
    bootstrap_cell = "".join(notebook["cells"][1]["source"])
    run_cell = "".join(notebook["cells"][2]["source"])

    assert 'subprocess.check_call(["git", "clone"' in bootstrap_cell
    assert 'SEC_CCM_WORK_ROOT' in bootstrap_cell
    assert "from thesis_pkg.notebooks_and_scripts.sec_ccm_unified_runner import main" in run_cell
    assert "main()" in run_cell
