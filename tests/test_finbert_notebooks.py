from __future__ import annotations

import json
from pathlib import Path


def test_finbert_benchmark_notebook_keeps_benchmark_flow_and_adds_smoke_stages() -> None:
    notebook_path = Path("src/thesis_pkg/notebooks_and_scripts/finbert_benchmark_orchestration.ipynb")
    notebook = json.loads(notebook_path.read_text(encoding="utf-8"))

    source = notebook_path.read_text(encoding="utf-8")
    assert "run_finbert_benchmark" in source
    assert "run_finbert_benchmark_sweep" in source
    assert "run_finbert_sentence_preprocessing" in source
    assert "run_finbert_tokenizer_profile" in source
    assert "SMOKE_PREP_YEAR_FILTER" in source
    assert "RUN_SMOKE_SENTENCE_PREP" in source
    assert "RUN_SMOKE_TOKENIZER_PROFILE" in source
    assert "TOKENIZER_PROFILE_ROW_CAP_PER_BUCKET" in source

    assert len(notebook["cells"]) >= 10


def test_finbert_full_data_staged_workflow_notebook_has_colab_bootstrap_and_split_stages() -> None:
    notebook_path = Path("src/thesis_pkg/notebooks_and_scripts/finbert_full_data_staged_workflow.ipynb")
    notebook = json.loads(notebook_path.read_text(encoding="utf-8"))

    source = "\n".join("".join(cell.get("source", [])) for cell in notebook["cells"])
    assert "drive.mount" in source
    assert "/content/NLP_Thesis" in source
    assert "%cd /content" in source
    assert "%pip install -e .[benchmark]" in source
    assert "sys.path.insert" in source
    assert "RUN_SENTENCE_STAGE" in source
    assert "RUN_TOKENIZER_STAGE" in source
    assert "RUN_MODEL_STAGE" in source
    assert "FinbertSentencePreprocessingRunConfig" in source
    assert "FinbertTokenizerProfileRunConfig" in source
    assert "FinbertSentenceParquetInferenceRunConfig" in source
    assert "run_finbert_sentence_preprocessing" in source
    assert "run_finbert_tokenizer_profile" in source
    assert "run_finbert_sentence_parquet_inference" in source

    assert len(notebook["cells"]) >= 11
