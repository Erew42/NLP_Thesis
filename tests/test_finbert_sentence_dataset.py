from __future__ import annotations

import json
from pathlib import Path

import polars as pl
import pytest

from thesis_pkg.benchmarking.contracts import DEFAULT_FINBERT_AUTHORITY
from thesis_pkg.benchmarking.contracts import FinbertSentencePreprocessingRunConfig
from thesis_pkg.benchmarking.contracts import FinbertSectionUniverseConfig
from thesis_pkg.benchmarking.contracts import SentenceDatasetConfig
from thesis_pkg.benchmarking.finbert_sentence_preprocessing import run_finbert_sentence_preprocessing
from thesis_pkg.benchmarking.sentences import materialize_sentence_benchmark_dataset


class _FakeSpan:
    def __init__(self, text: str) -> None:
        self.text = text


class _FakeDoc:
    def __init__(self, text: str) -> None:
        self.sents = [_FakeSpan(part.strip()) for part in text.split(".") if part.strip()]


class _FakeNLP:
    pipe_names = ("sentencizer",)

    def pipe(self, texts, batch_size: int = 0):
        del batch_size
        for text in texts:
            yield _FakeDoc(text)


def _write_items_year(
    path: Path,
    *,
    year: int,
    doc_id: str | None = None,
    item_texts: dict[str, str] | None = None,
) -> None:
    doc_id = doc_id or f"{year}:doc:001"
    default_text = ("Sentence. " * 80).strip()
    item_texts = item_texts or {}
    path.parent.mkdir(parents=True, exist_ok=True)
    pl.DataFrame(
        {
            "doc_id": [doc_id, doc_id, doc_id],
            "cik_10": ["0000000001", "0000000001", "0000000001"],
            "accession_nodash": [f"{year}00000000000001"] * 3,
            "filing_date": [f"{year}-03-01"] * 3,
            "document_type_filename": ["10-K", "10-K", "10-K"],
            "item_id": ["1", "1A", "7"],
            "canonical_item": ["I:1_BUSINESS", "I:1A_RISK_FACTORS", "II:7_MDA"],
            "item_part": ["PART I", "PART I", "PART II"],
            "item_status": ["active", "active", "active"],
            "exists_by_regime": [True, True, True],
            "filename": [f"{doc_id}_1.htm", f"{doc_id}_1a.htm", f"{doc_id}_7.htm"],
            "full_text": [
                item_texts.get("1", default_text),
                item_texts.get("1A", default_text),
                item_texts.get("7", default_text),
            ],
        }
    ).write_parquet(path)


def _empty_split_audit(sentences_module) -> pl.DataFrame:
    return pl.DataFrame(schema=sentences_module.SENTENCE_SPLIT_AUDIT_SCHEMA)


def _fake_annotate_token_lengths(
    df: pl.DataFrame,
    authority,
    *,
    text_col: str = "sentence_text",
    batch_size: int | None = None,
) -> pl.DataFrame:
    del authority, text_col, batch_size
    return df.with_columns(
        [
            pl.lit(5, dtype=pl.Int32).alias("finbert_token_count_512"),
            pl.lit("short", dtype=pl.Utf8).alias("finbert_token_bucket_512"),
        ]
    )


def test_materialize_sentence_benchmark_dataset_honors_authority_and_compression(
    tmp_path: Path,
    monkeypatch,
) -> None:
    sections_path = tmp_path / "sections.parquet"
    pl.DataFrame(
        {
            "benchmark_row_id": ["doc1:item_1", "doc2:item_7"],
            "doc_id": ["doc1", "doc2"],
            "cik_10": ["0000000001", "0000000002"],
            "accession_nodash": ["000000000100000001", "000000000200000001"],
            "filing_date": [None, None],
            "filing_year": [2000, 2001],
            "benchmark_item_code": ["item_1", "item_7"],
            "benchmark_item_label": ["10-K Item 1", "10-K Item 7"],
            "source_year_file": [2000, 2001],
            "document_type": ["10-K", "10-K"],
            "document_type_raw": ["10-K", "10-K"],
            "document_type_normalized": ["10-K", "10-K"],
            "canonical_item": ["I:1_BUSINESS", "II:7_MDA"],
            "full_text": ["First sentence. Second sentence.", "Third sentence only."],
        }
    ).write_parquet(sections_path)

    from thesis_pkg.benchmarking import sentences

    captured: dict[str, object] = {}

    def _fake_annotate(
        df: pl.DataFrame,
        authority,
        *,
        text_col: str = "full_text",
        batch_size: int | None = None,
    ) -> pl.DataFrame:
        del batch_size
        captured["authority"] = authority
        captured["text_col"] = text_col
        return df.with_columns(
            [
                pl.Series("finbert_token_count_512", [5] * df.height, dtype=pl.Int32),
                pl.Series("finbert_token_bucket_512", ["short"] * df.height, dtype=pl.Utf8),
            ]
        )

    def _fake_write_parquet(self, path: Path, *, compression: str = "zstd", **kwargs) -> None:
        del kwargs
        captured["compression"] = compression
        original_write_parquet(self, path, compression=compression)

    original_write_parquet = pl.DataFrame.write_parquet
    monkeypatch.setattr(sentences, "_build_sentencizer", lambda cfg: _FakeNLP())
    monkeypatch.setattr(sentences, "_sentencizer_version", lambda: "test")
    monkeypatch.setattr(sentences, "annotate_finbert_token_lengths_in_batches", _fake_annotate)
    monkeypatch.setattr(pl.DataFrame, "write_parquet", _fake_write_parquet)

    out_path = tmp_path / "sentences.parquet"
    materialize_sentence_benchmark_dataset(
        sections_path,
        SentenceDatasetConfig(enabled=True, compression="lz4"),
        authority=DEFAULT_FINBERT_AUTHORITY,
        compression="lz4",
        out_path=out_path,
    )

    result = pl.read_parquet(out_path).sort(["benchmark_row_id", "sentence_index"])
    assert result["benchmark_sentence_id"].to_list() == [
        "doc1:item_1:0",
        "doc1:item_1:1",
        "doc2:item_7:0",
    ]
    assert result["cik_10"].to_list() == ["0000000001", "0000000001", "0000000002"]
    assert result["accession_nodash"].to_list() == [
        "000000000100000001",
        "000000000100000001",
        "000000000200000001",
    ]
    assert result["benchmark_item_label"].to_list() == [
        "10-K Item 1",
        "10-K Item 1",
        "10-K Item 7",
    ]
    assert result["source_year_file"].to_list() == [2000, 2000, 2001]
    assert result["document_type_normalized"].to_list() == ["10-K", "10-K", "10-K"]
    assert result["canonical_item"].to_list() == [
        "I:1_BUSINESS",
        "I:1_BUSINESS",
        "II:7_MDA",
    ]
    assert captured["authority"] == DEFAULT_FINBERT_AUTHORITY
    assert captured["text_col"] == "full_text"
    assert captured["compression"] == "lz4"


def test_run_finbert_sentence_preprocessing_writes_by_year_artifacts(
    tmp_path: Path,
    monkeypatch,
) -> None:
    source_dir = tmp_path / "items_analysis"
    _write_items_year(source_dir / "2006.parquet", year=2006)

    from thesis_pkg.benchmarking import finbert_sentence_preprocessing

    from thesis_pkg.benchmarking import sentences

    def _fake_derive(sections_df: pl.DataFrame, sentence_cfg, *, authority):
        del sentence_cfg, authority
        rows: list[dict[str, object]] = []
        for row in sections_df.select(
            [
                "benchmark_row_id",
                "doc_id",
                "cik_10",
                "accession_nodash",
                "filing_date",
                "filing_year",
                "benchmark_item_code",
                "benchmark_item_label",
                "source_year_file",
                "document_type",
                "document_type_raw",
                "document_type_normalized",
                "canonical_item",
            ]
        ).iter_rows(named=True):
            rows.append(
                {
                    "benchmark_sentence_id": f"{row['benchmark_row_id']}:0",
                    "benchmark_row_id": row["benchmark_row_id"],
                    "doc_id": row["doc_id"],
                    "cik_10": row["cik_10"],
                    "accession_nodash": row["accession_nodash"],
                    "filing_date": row["filing_date"],
                    "filing_year": row["filing_year"],
                    "benchmark_item_code": row["benchmark_item_code"],
                    "benchmark_item_label": row["benchmark_item_label"],
                    "source_year_file": row["source_year_file"],
                    "document_type": row["document_type"],
                    "document_type_raw": row["document_type_raw"],
                    "document_type_normalized": row["document_type_normalized"],
                    "canonical_item": row["canonical_item"],
                    "sentence_index": 0,
                    "sentence_text": "stub sentence",
                    "sentence_char_count": 13,
                    "sentencizer_backend": "test",
                    "sentencizer_version": "test",
                    "finbert_token_count_512": 5,
                    "finbert_token_bucket_512": "short",
                }
            )
        return pl.DataFrame(rows), _empty_split_audit(sentences)

    monkeypatch.setattr(finbert_sentence_preprocessing, "_derive_sentence_frame_with_split_audit", _fake_derive)

    artifacts = run_finbert_sentence_preprocessing(
        FinbertSentencePreprocessingRunConfig(
            source_items_dir=source_dir,
            out_root=tmp_path / "runs",
            section_universe=FinbertSectionUniverseConfig(source_items_dir=source_dir),
            run_name="sentence_prep",
        )
    )

    summary = pl.read_parquet(artifacts.yearly_summary_path)
    sentence_dataset = pl.read_parquet(artifacts.sentence_dataset_dir / "2006.parquet")
    manifest = json.loads(artifacts.run_manifest_path.read_text(encoding="utf-8"))

    assert artifacts.oversize_sections_path is not None
    assert artifacts.oversize_sections_path.exists()
    assert summary["sentence_rows"].to_list() == [3]
    assert summary["short_sentence_rows"].to_list() == [3]
    assert summary["oversize_section_rows"].to_list() == [0]
    assert summary["chunked_section_rows"].to_list() == [0]
    assert summary["warning_split_rows"].to_list() == [0]
    assert sentence_dataset.height == 3
    assert "benchmark_item_label" in sentence_dataset.columns
    assert "source_year_file" in sentence_dataset.columns
    assert "document_type_raw" in sentence_dataset.columns
    assert manifest["accepted_universe_contract"]["accepted_unit"] == ["doc_id", "benchmark_item_code"]
    assert manifest["accepted_universe_contract"]["filters"]["raw_form_allowlist"] == ["10-K", "10-K405"]
    assert manifest["accepted_universe_contract"]["dedupe"]["key"] == ["doc_id", "benchmark_item_code"]


def test_run_finbert_sentence_preprocessing_reuses_existing_outputs(
    tmp_path: Path,
    monkeypatch,
) -> None:
    source_dir = tmp_path / "items_analysis"
    _write_items_year(source_dir / "2006.parquet", year=2006)

    from thesis_pkg.benchmarking import finbert_sentence_preprocessing

    call_count = {"derive": 0}

    from thesis_pkg.benchmarking import sentences

    def _fake_derive(sections_df: pl.DataFrame, sentence_cfg, *, authority):
        del sentence_cfg, authority
        call_count["derive"] += 1
        return pl.DataFrame(
            {
                "benchmark_sentence_id": [f"{sections_df['benchmark_row_id'][0]}:0"],
                "benchmark_row_id": [sections_df["benchmark_row_id"][0]],
                "doc_id": [sections_df["doc_id"][0]],
                "cik_10": [sections_df["cik_10"][0]],
                "accession_nodash": [sections_df["accession_nodash"][0]],
                "filing_date": [sections_df["filing_date"][0]],
                "filing_year": [sections_df["filing_year"][0]],
                "benchmark_item_code": [sections_df["benchmark_item_code"][0]],
                "benchmark_item_label": [sections_df["benchmark_item_label"][0]],
                "source_year_file": [sections_df["source_year_file"][0]],
                "document_type": [sections_df["document_type"][0]],
                "document_type_raw": [sections_df["document_type_raw"][0]],
                "document_type_normalized": [sections_df["document_type_normalized"][0]],
                "canonical_item": [sections_df["canonical_item"][0]],
                "sentence_index": [0],
                "sentence_text": ["stub sentence"],
                "sentence_char_count": [13],
                "sentencizer_backend": ["test"],
                "sentencizer_version": ["test"],
                "finbert_token_count_512": [5],
                "finbert_token_bucket_512": ["short"],
            }
        ), _empty_split_audit(sentences)

    monkeypatch.setattr(finbert_sentence_preprocessing, "_derive_sentence_frame_with_split_audit", _fake_derive)

    cfg = FinbertSentencePreprocessingRunConfig(
        source_items_dir=source_dir,
        out_root=tmp_path / "runs",
        section_universe=FinbertSectionUniverseConfig(source_items_dir=source_dir),
        run_name="sentence_prep",
    )

    run_finbert_sentence_preprocessing(cfg)
    run_finbert_sentence_preprocessing(cfg)

    assert call_count["derive"] == 1


def test_run_finbert_sentence_preprocessing_filters_to_target_doc_universe(
    tmp_path: Path,
    monkeypatch,
) -> None:
    source_dir = tmp_path / "items_analysis"
    _write_items_year(source_dir / "2006.parquet", year=2006)
    doc_id_two = "2006:doc:999"
    pl.DataFrame(
        {
            "doc_id": [doc_id_two, doc_id_two, doc_id_two],
            "cik_10": ["0000000999"] * 3,
            "accession_nodash": ["2006000000000999"] * 3,
            "filing_date": ["2006-04-01"] * 3,
            "document_type_filename": ["10-K", "10-K", "10-K"],
            "item_id": ["1", "1A", "7"],
            "canonical_item": ["I:1_BUSINESS", "I:1A_RISK_FACTORS", "II:7_MDA"],
            "item_part": ["PART I", "PART I", "PART II"],
            "item_status": ["active", "active", "active"],
            "exists_by_regime": [True, True, True],
            "filename": [f"{doc_id_two}_1.htm", f"{doc_id_two}_1a.htm", f"{doc_id_two}_7.htm"],
            "full_text": [("Sentence. " * 80).strip()] * 3,
        }
    ).write_parquet(source_dir / "2007.parquet")
    target_path = tmp_path / "target_docs.parquet"
    pl.DataFrame({"doc_id": ["2006:doc:001"]}).write_parquet(target_path)

    from thesis_pkg.benchmarking import finbert_sentence_preprocessing

    from thesis_pkg.benchmarking import sentences

    def _fake_derive(sections_df: pl.DataFrame, sentence_cfg, *, authority):
        del sentence_cfg, authority
        rows: list[dict[str, object]] = []
        for row in sections_df.select(
            [
                "benchmark_row_id",
                "doc_id",
                "cik_10",
                "accession_nodash",
                "filing_date",
                "filing_year",
                "benchmark_item_code",
                "benchmark_item_label",
                "source_year_file",
                "document_type",
                "document_type_raw",
                "document_type_normalized",
                "canonical_item",
            ]
        ).iter_rows(named=True):
            rows.append(
                {
                    "benchmark_sentence_id": f"{row['benchmark_row_id']}:0",
                    "benchmark_row_id": row["benchmark_row_id"],
                    "doc_id": row["doc_id"],
                    "cik_10": row["cik_10"],
                    "accession_nodash": row["accession_nodash"],
                    "filing_date": row["filing_date"],
                    "filing_year": row["filing_year"],
                    "benchmark_item_code": row["benchmark_item_code"],
                    "benchmark_item_label": row["benchmark_item_label"],
                    "source_year_file": row["source_year_file"],
                    "document_type": row["document_type"],
                    "document_type_raw": row["document_type_raw"],
                    "document_type_normalized": row["document_type_normalized"],
                    "canonical_item": row["canonical_item"],
                    "sentence_index": 0,
                    "sentence_text": "stub sentence",
                    "sentence_char_count": 13,
                    "sentencizer_backend": "test",
                    "sentencizer_version": "test",
                    "finbert_token_count_512": 5,
                    "finbert_token_bucket_512": "short",
                }
            )
        return pl.DataFrame(rows), _empty_split_audit(sentences)

    monkeypatch.setattr(finbert_sentence_preprocessing, "_derive_sentence_frame_with_split_audit", _fake_derive)

    artifacts = run_finbert_sentence_preprocessing(
        FinbertSentencePreprocessingRunConfig(
            source_items_dir=source_dir,
            out_root=tmp_path / "runs",
            section_universe=FinbertSectionUniverseConfig(source_items_dir=source_dir),
            target_doc_universe_path=target_path,
            year_filter=(2006,),
            run_name="sentence_prep_targeted",
        )
    )

    sentence_dataset = pl.read_parquet(artifacts.sentence_dataset_dir / "2006.parquet")
    assert sentence_dataset["doc_id"].unique().to_list() == ["2006:doc:001"]


def test_derive_sentence_frame_chunks_large_sections_without_spacy_length_error(monkeypatch) -> None:
    from thesis_pkg.benchmarking import sentences

    huge_text = ("A" * 206_529) * 5
    sections_df = pl.DataFrame(
        {
            "benchmark_row_id": ["doc1:item_1"],
            "doc_id": ["doc1"],
            "cik_10": ["0000000001"],
            "accession_nodash": ["000000000100000001"],
            "filing_date": [None],
            "filing_year": [2006],
            "benchmark_item_code": ["item_1"],
            "benchmark_item_label": ["10-K Item 1"],
            "source_year_file": [2006],
            "document_type": ["10-K"],
            "document_type_raw": ["10-K"],
            "document_type_normalized": ["10-K"],
            "canonical_item": ["I:1_BUSINESS"],
            "full_text": [huge_text],
        }
    )

    monkeypatch.setattr(sentences, "_build_sentencizer", lambda cfg: _FakeNLP())
    monkeypatch.setattr(sentences, "_sentencizer_version", lambda: "test")
    monkeypatch.setattr(sentences, "annotate_finbert_token_lengths_in_batches", _fake_annotate_token_lengths)

    sentence_df, split_audit_df = sentences._derive_sentence_frame_with_split_audit(
        sections_df,
        SentenceDatasetConfig(enabled=True, spacy_batch_size=1),
        authority=DEFAULT_FINBERT_AUTHORITY,
    )

    assert sentence_df.height == 5
    assert split_audit_df["total_chunk_count"].unique().to_list() == [5]
    assert split_audit_df["split_reason"].to_list()[-1] == "end_of_text"
    assert split_audit_df["warning_boundary_used"].to_list() == [True, True, True, True, False]


def test_derive_sentence_frame_prefers_double_newline_and_keeps_sentence_indices_continuous(monkeypatch) -> None:
    from thesis_pkg.benchmarking import sentences

    preferred_text = ("Alpha. " * 35_713) + "\n\n" + ("Beta. " * 4)
    sections_df = pl.DataFrame(
        {
            "benchmark_row_id": ["doc1:item_1"],
            "doc_id": ["doc1"],
            "cik_10": ["0000000001"],
            "accession_nodash": ["000000000100000001"],
            "filing_date": [None],
            "filing_year": [2006],
            "benchmark_item_code": ["item_1"],
            "benchmark_item_label": ["10-K Item 1"],
            "source_year_file": [2006],
            "document_type": ["10-K"],
            "document_type_raw": ["10-K"],
            "document_type_normalized": ["10-K"],
            "canonical_item": ["I:1_BUSINESS"],
            "full_text": [preferred_text],
        }
    )

    monkeypatch.setattr(sentences, "_build_sentencizer", lambda cfg: _FakeNLP())
    monkeypatch.setattr(sentences, "_sentencizer_version", lambda: "test")
    monkeypatch.setattr(sentences, "annotate_finbert_token_lengths_in_batches", _fake_annotate_token_lengths)

    sentence_df, split_audit_df = sentences._derive_sentence_frame_with_split_audit(
        sections_df,
        SentenceDatasetConfig(enabled=True, spacy_batch_size=1),
        authority=DEFAULT_FINBERT_AUTHORITY,
    )

    split_reasons = split_audit_df["split_reason"].to_list()
    assert split_reasons[0] == "double_newline"
    assert split_reasons[-1] == "end_of_text"
    assert sentence_df["sentence_index"].to_list() == list(range(sentence_df.height))
    assert sentence_df["sentence_text"].to_list()[:2] == ["Alpha", "Alpha"]
    assert sentence_df["sentence_text"].to_list()[-2:] == ["Beta", "Beta"]


def test_run_finbert_sentence_preprocessing_logs_fallback_split_reasons_and_warnings(
    tmp_path: Path,
    monkeypatch,
) -> None:
    source_dir = tmp_path / "items_analysis"
    whitespace_text = ("A" * 249_999) + (" " + ("B" * 10)) * 3
    hard_limit_text = "C" * 250_010
    _write_items_year(
        source_dir / "2006.parquet",
        year=2006,
        item_texts={
            "1": whitespace_text,
            "7": hard_limit_text,
        },
    )

    from thesis_pkg.benchmarking import sentences

    monkeypatch.setattr(sentences, "_build_sentencizer", lambda cfg: _FakeNLP())
    monkeypatch.setattr(sentences, "_sentencizer_version", lambda: "test")
    monkeypatch.setattr(sentences, "annotate_finbert_token_lengths_in_batches", _fake_annotate_token_lengths)

    with pytest.warns(RuntimeWarning, match="fallback split boundaries"):
        artifacts = run_finbert_sentence_preprocessing(
            FinbertSentencePreprocessingRunConfig(
                source_items_dir=source_dir,
                out_root=tmp_path / "runs",
                section_universe=FinbertSectionUniverseConfig(source_items_dir=source_dir),
                run_name="sentence_prep_fallback_warning",
            )
        )

    split_audit_df = pl.read_parquet(artifacts.oversize_sections_path)
    summary_df = pl.read_parquet(artifacts.yearly_summary_path)
    manifest = json.loads(artifacts.run_manifest_path.read_text(encoding="utf-8"))

    warning_reasons = split_audit_df.filter(pl.col("warning_boundary_used"))["split_reason"].unique().sort().to_list()
    assert warning_reasons == ["hard_limit_250k", "whitespace"]
    assert summary_df["oversize_section_rows"].to_list() == [2]
    assert summary_df["chunked_section_rows"].to_list() == [2]
    assert summary_df["warning_split_rows"].to_list() == [2]
    assert manifest["counts"]["oversize_section_rows"] == 2
    assert manifest["counts"]["chunked_section_rows"] == 2
    assert manifest["counts"]["warning_split_rows"] == 2
    assert manifest["artifacts"]["oversize_sections_path"] == str(artifacts.oversize_sections_path.resolve())
