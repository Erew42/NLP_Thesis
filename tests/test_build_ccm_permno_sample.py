from __future__ import annotations

import datetime as dt
import importlib.util
import json
from pathlib import Path
import sys

import polars as pl

from thesis_pkg.core.ccm.canonical_links import normalize_canonical_link_table


_SAMPLER_SPEC = importlib.util.spec_from_file_location(
    "build_ccm_permno_sample",
    Path(__file__).resolve().parents[1] / "tools" / "build_ccm_permno_sample.py",
)
assert _SAMPLER_SPEC is not None and _SAMPLER_SPEC.loader is not None
sampler = importlib.util.module_from_spec(_SAMPLER_SPEC)
sys.modules[_SAMPLER_SPEC.name] = sampler
_SAMPLER_SPEC.loader.exec_module(sampler)


def _cfg(
    tmp_path: Path,
    *,
    sample_frac: float = 0.05,
    overlap_target: float = 0.5,
    seed: int = 42,
    start_year: int = 2024,
    end_year: int = 2024,
) -> sampler.SampleConfig:
    full_data_root = tmp_path / "full_data_run"
    return sampler.SampleConfig(
        full_data_root=full_data_root,
        out_root=full_data_root / "sample_out",
        canonical_link_name="canonical_link_table_after_startdate_change.parquet",
        ccm_daily_name="final_flagged_data_compdesc_added.parquet",
        sample_frac=sample_frac,
        overlap_target=overlap_target,
        seed=seed,
        start_year=start_year,
        end_year=end_year,
    )


def _write_parquet(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pl.DataFrame(rows).write_parquet(path)


def _canonical_links(rows: list[dict[str, object]]) -> pl.DataFrame:
    base = {
        "cik_10": "0000000000",
        "gvkey": "0",
        "kypermno": 0,
        "lpermco": None,
        "liid": "01",
        "valid_start": dt.date(1900, 1, 1),
        "valid_end": None,
        "link_start": dt.date(1900, 1, 1),
        "link_end": None,
        "cik_start": None,
        "cik_end": None,
        "linktype": "LC",
        "linkprim": "P",
        "link_rank_raw": None,
        "link_rank_effective": 90,
        "link_quality": 4.0,
        "link_source": "linkhistory",
        "source_priority": 1,
        "row_quality_tier": 10,
        "has_window": True,
        "is_sparse_fallback": False,
    }
    return pl.DataFrame([{**base, **row} for row in rows])


def test_load_sec_filings_drops_blank_and_non_digit_ciks(tmp_path: Path) -> None:
    cfg = _cfg(tmp_path)
    _write_parquet(
        cfg.year_merged_dir / "2024.parquet",
        [
            {"doc_id": "d1", "cik_10": "", "filing_date": dt.date(2024, 1, 1)},
            {"doc_id": "d2", "cik_10": "ABC", "filing_date": dt.date(2024, 1, 2)},
            {"doc_id": "d3", "cik_10": "12345", "filing_date": dt.date(2024, 1, 3)},
            {"doc_id": "d4", "cik_10": "0000123456", "filing_date": dt.date(2024, 1, 4)},
        ],
    )

    filings = sampler._load_sec_filings(cfg).collect().sort("doc_id")

    assert filings["doc_id"].to_list() == ["d3", "d4"]
    assert filings["cik_10"].to_list() == ["0000012345", "0000123456"]


def test_build_filing_doc_link_maps_is_doc_grain_and_date_valid() -> None:
    sec_filings = pl.DataFrame(
        {
            "doc_id": ["d1", "d2"],
            "cik_10": ["0000000001", "0000000001"],
            "filing_date": [dt.date(2024, 1, 15), dt.date(2024, 7, 15)],
        }
    ).lazy()
    links = normalize_canonical_link_table(
        _canonical_links(
            [
                {
                    "cik_10": "0000000001",
                    "gvkey": "1000",
                    "kypermno": 1,
                    "valid_end": dt.date(2024, 3, 31),
                },
                {
                    "cik_10": "0000000001",
                    "gvkey": "1000",
                    "kypermno": 2,
                    "valid_start": dt.date(2024, 4, 1),
                },
            ]
        ).lazy(),
        strict=True,
    )

    long_df, permno_to_doc_ids, linked_doc_count, permno_to_ciks = sampler._build_filing_doc_link_maps(
        sec_filings,
        links,
    )

    assert linked_doc_count == 2
    assert long_df.select(pl.col("cik_10").n_unique()).item() == 1
    assert set(long_df["doc_id"].to_list()) == {"d1", "d2"}
    assert long_df.sort("doc_id")["kypermno"].to_list() == [1, 2]
    assert permno_to_doc_ids == {1: {"d1"}, 2: {"d2"}}
    assert permno_to_ciks == {1: {"0000000001"}, 2: {"0000000001"}}


def test_select_mandatory_anchor_permnos_uses_exact_fallback() -> None:
    year_mask_by_permno = {
        1: 0b001,
        2: 0b001,
        3: 0b001,
        4: 0b010,
        5: 0b100,
    }
    doc_mask_by_permno = {
        1: 0b001,
        2: 0b001,
        3: 0b010,
        4: 0b001,
        5: 0b100,
    }

    result = sampler._select_mandatory_anchor_permnos(
        year_mask_by_permno=year_mask_by_permno,
        doc_mask_by_permno=doc_mask_by_permno,
        target_year_mask=0b111,
        target_doc_count=3,
        budget_k=3,
        years=[2022, 2023, 2024],
    )

    assert result.strategy == "exact_fallback"
    assert result.greedy_count == 4
    assert result.selected_permnos == [3, 4, 5]


def test_sample_tag_and_tagged_output_names() -> None:
    default_cfg = sampler.SampleConfig(
        full_data_root=Path("full_data_run"),
        out_root=Path("full_data_run/sample_5pct_seed42"),
        canonical_link_name="canonical_link_table_after_startdate_change.parquet",
        ccm_daily_name="final_flagged_data_compdesc_added.parquet",
        sample_frac=0.05,
        overlap_target=0.5,
        seed=42,
        start_year=2024,
        end_year=2024,
    )
    custom_cfg = sampler.SampleConfig(
        full_data_root=Path("full_data_run"),
        out_root=Path("full_data_run/sample_10pct_seed7"),
        canonical_link_name="canonical_link_table_after_startdate_change.parquet",
        ccm_daily_name="final_flagged_data_compdesc_added.parquet",
        sample_frac=0.10,
        overlap_target=0.5,
        seed=7,
        start_year=2024,
        end_year=2024,
    )

    assert sampler._sample_tag(default_cfg) == "sample_5pct_seed42"
    assert (
        sampler._tagged_output_name(default_cfg.ccm_daily_name, sampler._sample_tag(default_cfg))
        == "final_flagged_data_compdesc_added.sample_5pct_seed42.parquet"
    )
    assert sampler._sample_tag(custom_cfg) == "sample_10pct_seed7"
    assert (
        sampler._tagged_output_name(custom_cfg.ccm_daily_name, sampler._sample_tag(custom_cfg))
        == "final_flagged_data_compdesc_added.sample_10pct_seed7.parquet"
    )
    assert (
        sampler._tagged_output_name(
            "final_flagged_data_compdesc_added.sample_10pct_seed7.parquet",
            sampler._sample_tag(custom_cfg),
        )
        == "final_flagged_data_compdesc_added.sample_10pct_seed7.parquet"
    )


def test_run_end_to_end_emits_doc_grain_artifacts(tmp_path: Path) -> None:
    cfg = _cfg(tmp_path, sample_frac=0.5, overlap_target=0.5, start_year=2024, end_year=2024)

    _write_parquet(
        cfg.year_merged_dir / "2024.parquet",
        [
            {"doc_id": "d1", "cik_10": "0000000001", "filing_date": dt.date(2024, 1, 10)},
            {"doc_id": "d2", "cik_10": "0000000002", "filing_date": dt.date(2024, 1, 10)},
            {"doc_id": "d3", "cik_10": "0000000003", "filing_date": dt.date(2024, 1, 10)},
        ],
    )
    cfg.canonical_link_path.parent.mkdir(parents=True, exist_ok=True)
    _canonical_links(
        [
            {"cik_10": "0000000001", "gvkey": "1000", "kypermno": 1},
            {"cik_10": "0000000002", "gvkey": "2000", "kypermno": 2},
            {"cik_10": "0000000003", "gvkey": "3000", "kypermno": 3},
            {"cik_10": "0000000004", "gvkey": "4000", "kypermno": 4},
        ]
    ).write_parquet(cfg.canonical_link_path)
    _write_parquet(
        cfg.ccm_daily_path,
        [
            {"KYPERMNO": 1, "CALDT": dt.date(2024, 1, 2), "RET": 0.01},
            {"KYPERMNO": 2, "CALDT": dt.date(2024, 1, 2), "RET": 0.02},
            {"KYPERMNO": 3, "CALDT": dt.date(2024, 1, 2), "RET": 0.03},
            {"KYPERMNO": 4, "CALDT": dt.date(2024, 1, 2), "RET": 0.04},
        ],
    )
    _write_parquet(
        cfg.ccm_parquet_dir / "by_permno.parquet",
        [
            {"KYPERMNO": 1, "value": 10},
            {"KYPERMNO": 2, "value": 20},
            {"KYPERMNO": 3, "value": 30},
            {"KYPERMNO": 4, "value": 40},
        ],
    )
    _write_parquet(
        cfg.ccm_parquet_dir / "by_gvkey.parquet",
        [
            {"gvkey": "1000", "value": 1},
            {"gvkey": "2000", "value": 2},
            {"gvkey": "3000", "value": 3},
            {"gvkey": "4000", "value": 4},
        ],
    )

    sampler.run(cfg)

    report = json.loads((cfg.out_reports_dir / "sample_overlap_report.json").read_text(encoding="utf-8"))
    manifest = json.loads((cfg.out_root / "sample_manifest.json").read_text(encoding="utf-8"))

    assert report["filing_doc_all_count"] == 3
    assert report["filing_doc_linked_count"] == 3
    assert report["filing_doc_linked_target_count"] == 2
    assert report["filing_doc_linked_covered_count"] == 2
    assert report["filing_doc_linked_covered_ratio"] == 2 / 3
    assert report["missing_years"] == []

    assert manifest["config"]["sample_tag"] == "sample_50pct_seed42"
    assert manifest["anchors"]["anchor_selection_strategy"] == "greedy"

    assert (cfg.out_mappings_dir / "filing_doc_to_permno_long.parquet").exists()
    assert (cfg.out_mappings_dir / "filing_doc_to_permno_collapsed.parquet").exists()
    assert (cfg.out_mappings_dir / "filing_cik_to_permno_collapsed.parquet").exists()
    assert not (cfg.out_mappings_dir / "filing_cik_to_permno_long.parquet").exists()

    daily_out = cfg.out_derived_dir / "final_flagged_data_compdesc_added.sample_50pct_seed42.parquet"
    canon_out = cfg.out_derived_dir / "canonical_link_table_after_startdate_change.sample_50pct_seed42.parquet"
    assert daily_out.exists()
    assert canon_out.exists()
    assert pl.read_parquet(daily_out)["KYPERMNO"].to_list() == [1, 2]
