from __future__ import annotations

import argparse
import csv
import json
import math
import random
from collections import defaultdict
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path
from typing import Any

import polars as pl

from thesis_pkg.core.ccm.canonical_links import normalize_canonical_link_table
from thesis_pkg.core.ccm.sec_ccm_premerge import normalize_sec_filings_phase_a


PERMNO_CANDIDATES: tuple[str, ...] = ("KYPERMNO", "LPERMNO", "lpermno", "PERMNO")
GVKEY_CANDIDATES: tuple[str, ...] = ("KYGVKEY", "gvkey", "GVKEY")
REBUILD_REQUIRED_STEMS: tuple[str, ...] = (
    "filingdates",
    "linkhistory",
    "linkfiscalperiodall",
    "companydescription",
    "companyhistory",
    "securityheader",
    "securityheaderhistory",
    "sfz_ds_dly",
    "sfz_dp_dly",
    "sfz_del",
    "sfz_nam",
    "sfz_hdr",
)


@dataclass(frozen=True)
class SampleConfig:
    full_data_root: Path
    out_root: Path
    canonical_link_name: str
    ccm_daily_name: str
    sample_frac: float
    overlap_target: float
    seed: int
    start_year: int
    end_year: int
    compression: str = "zstd"

    @property
    def ccm_parquet_dir(self) -> Path:
        return self.full_data_root / "ccm_parquet_data"

    @property
    def derived_dir(self) -> Path:
        return self.full_data_root / "derived_data"

    @property
    def year_merged_dir(self) -> Path:
        return self.full_data_root / "year_merged"

    @property
    def canonical_link_path(self) -> Path:
        return self.derived_dir / self.canonical_link_name

    @property
    def ccm_daily_path(self) -> Path:
        return self.derived_dir / self.ccm_daily_name

    @property
    def out_ccm_dir(self) -> Path:
        return self.out_root / "ccm_parquet_data"

    @property
    def out_derived_dir(self) -> Path:
        return self.out_root / "derived_data"

    @property
    def out_inventory_dir(self) -> Path:
        return self.out_root / "inventory"

    @property
    def out_mappings_dir(self) -> Path:
        return self.out_root / "mappings"

    @property
    def out_reports_dir(self) -> Path:
        return self.out_root / "reports"

    @property
    def years(self) -> list[int]:
        return list(range(self.start_year, self.end_year + 1))


@dataclass(frozen=True)
class CoverageCandidate:
    permno: int
    year_mask: int
    doc_mask: int
    year_count: int
    doc_count: int


@dataclass(frozen=True)
class AnchorSelectionResult:
    selected_permnos: list[int]
    strategy: str
    greedy_count: int
    pruned_candidate_count: int


def _parse_args() -> SampleConfig:
    parser = argparse.ArgumentParser(
        description="Build overlap-constrained 5%% CCM sample with SEC filing overlap and time-horizon constraints."
    )
    parser.add_argument("--full-data-root", type=Path, default=Path("full_data_run"))
    parser.add_argument("--out-root", type=Path, default=Path("full_data_run/sample_5pct_seed42"))
    parser.add_argument(
        "--canonical-link-name",
        type=str,
        default="canonical_link_table_after_startdate_change.parquet",
    )
    parser.add_argument(
        "--ccm-daily-name",
        type=str,
        default="final_flagged_data_compdesc_added.parquet",
    )
    parser.add_argument("--sample-frac", type=float, default=0.05)
    parser.add_argument("--overlap-target", type=float, default=0.50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--start-year", type=int, default=1990)
    parser.add_argument("--end-year", type=int, default=2024)
    parser.add_argument("--compression", type=str, default="zstd")
    args = parser.parse_args()

    if not (0.0 < args.sample_frac <= 1.0):
        raise ValueError("--sample-frac must be in (0, 1].")
    if not (0.0 <= args.overlap_target <= 1.0):
        raise ValueError("--overlap-target must be in [0, 1].")
    if args.start_year > args.end_year:
        raise ValueError("--start-year must be <= --end-year.")

    return SampleConfig(
        full_data_root=args.full_data_root.resolve(),
        out_root=args.out_root.resolve(),
        canonical_link_name=args.canonical_link_name,
        ccm_daily_name=args.ccm_daily_name,
        sample_frac=args.sample_frac,
        overlap_target=args.overlap_target,
        seed=args.seed,
        start_year=args.start_year,
        end_year=args.end_year,
        compression=args.compression,
    )


def _rows_in_parquet(path: Path) -> int:
    return int(pl.scan_parquet(path).select(pl.len().alias("n")).collect().item())


def _unique_count(path: Path, col_name: str) -> int:
    return int(
        pl.scan_parquet(path)
        .select(pl.col(col_name).drop_nulls().n_unique().alias("n"))
        .collect()
        .item()
    )


def _first_existing(names: list[str], candidates: tuple[str, ...]) -> str | None:
    for c in candidates:
        if c in names:
            return c
    return None


def _discover_stem_paths(root: Path) -> dict[str, list[Path]]:
    stem_paths: dict[str, list[Path]] = defaultdict(list)
    for path in sorted(root.rglob("*.parquet")):
        stem_paths[path.stem].append(path)
    return dict(stem_paths)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _build_pipeline_inventory(cfg: SampleConfig) -> dict[str, Any]:
    stem_paths = _discover_stem_paths(cfg.ccm_parquet_dir)

    rebuild_inputs: list[dict[str, Any]] = []
    for stem in REBUILD_REQUIRED_STEMS:
        candidates = stem_paths.get(stem, [])
        entry: dict[str, Any] = {
            "stem": stem,
            "found": len(candidates) > 0,
            "path": str(candidates[0]) if candidates else None,
            "duplicates": [str(p) for p in candidates[1:]],
        }
        if candidates:
            rel_parent = candidates[0].parent.relative_to(cfg.ccm_parquet_dir)
            entry["source_subdir"] = str(rel_parent.parts[0]) if rel_parent.parts else "."
        else:
            entry["source_subdir"] = None
        rebuild_inputs.append(entry)

    year_files = sorted(
        p for p in cfg.year_merged_dir.glob("*.parquet") if p.stem.isdigit() and len(p.stem) == 4
    )
    reuse_inputs = {
        "ccm_daily_path": str(cfg.ccm_daily_path),
        "ccm_daily_exists": cfg.ccm_daily_path.exists(),
        "canonical_link_path": str(cfg.canonical_link_path),
        "canonical_link_exists": cfg.canonical_link_path.exists(),
    }
    sec_inputs = {
        "year_merged_dir": str(cfg.year_merged_dir),
        "year_merged_exists": cfg.year_merged_dir.exists(),
        "year_files": [str(p) for p in year_files],
        "year_file_count": len(year_files),
    }

    payload = {
        "full_data_root": str(cfg.full_data_root),
        "ccm_parquet_dir": str(cfg.ccm_parquet_dir),
        "rebuild_required_inputs": rebuild_inputs,
        "reuse_inputs": reuse_inputs,
        "sec_inputs": sec_inputs,
    }

    csv_rows: list[dict[str, Any]] = []
    for row in rebuild_inputs:
        csv_rows.append(
            {
                "category": "rebuild_required",
                "name": row["stem"],
                "found": row["found"],
                "path": row["path"],
                "source_subdir": row["source_subdir"],
                "duplicates": ";".join(row["duplicates"]),
            }
        )
    csv_rows.append(
        {
            "category": "reuse",
            "name": "ccm_daily_path",
            "found": reuse_inputs["ccm_daily_exists"],
            "path": reuse_inputs["ccm_daily_path"],
            "source_subdir": None,
            "duplicates": "",
        }
    )
    csv_rows.append(
        {
            "category": "reuse",
            "name": "canonical_link_path",
            "found": reuse_inputs["canonical_link_exists"],
            "path": reuse_inputs["canonical_link_path"],
            "source_subdir": None,
            "duplicates": "",
        }
    )
    for p in year_files:
        csv_rows.append(
            {
                "category": "sec_year_merged",
                "name": p.stem,
                "found": True,
                "path": str(p),
                "source_subdir": None,
                "duplicates": "",
            }
        )

    _write_json(cfg.out_inventory_dir / "pipeline_inputs.json", payload)
    _write_csv(cfg.out_inventory_dir / "pipeline_inputs.csv", csv_rows)
    return payload


def _load_sec_filings(cfg: SampleConfig) -> pl.LazyFrame:
    year_files = sorted(
        p for p in cfg.year_merged_dir.glob("*.parquet") if p.stem.isdigit() and len(p.stem) == 4
    )
    if not year_files:
        raise FileNotFoundError(f"No yearly SEC parquet files found in {cfg.year_merged_dir}")

    sec_lf = pl.scan_parquet([str(p) for p in year_files])
    schema = sec_lf.collect_schema()
    if "doc_id" not in schema:
        raise ValueError("SEC year_merged files missing doc_id column.")
    if "cik_10" in schema:
        cik_expr = pl.col("cik_10").cast(pl.Utf8, strict=False).alias("cik_10")
    elif "cik" in schema:
        cik_expr = pl.col("cik").cast(pl.Utf8, strict=False).alias("cik_10")
    else:
        raise ValueError("SEC year_merged files missing cik_10 and cik columns.")

    if "filing_date" in schema and "file_date_filename" in schema:
        filing_date_expr = pl.coalesce(
            [
                pl.col("filing_date").cast(pl.Date, strict=False),
                pl.col("file_date_filename").cast(pl.Date, strict=False),
            ]
        ).alias("filing_date")
    elif "filing_date" in schema:
        filing_date_expr = pl.col("filing_date").cast(pl.Date, strict=False).alias("filing_date")
    elif "file_date_filename" in schema:
        filing_date_expr = pl.col("file_date_filename").cast(pl.Date, strict=False).alias("filing_date")
    else:
        raise ValueError("SEC year_merged files missing both filing_date and file_date_filename.")

    sec_minimal = sec_lf.select(
        pl.col("doc_id").cast(pl.Utf8, strict=False).alias("doc_id"),
        cik_expr,
        filing_date_expr,
    )
    return (
        normalize_sec_filings_phase_a(sec_minimal)
        .select("doc_id", "cik_10", "filing_date")
        .filter(
            pl.col("doc_id").is_not_null()
            & pl.col("filing_date").is_not_null()
            & pl.col("cik_10").str.contains(r"^\d{10}$").fill_null(False)
        )
    )


def _load_canonical_links(cfg: SampleConfig) -> pl.LazyFrame:
    if not cfg.canonical_link_path.exists():
        raise FileNotFoundError(f"Canonical link file not found: {cfg.canonical_link_path}")
    return normalize_canonical_link_table(pl.scan_parquet(cfg.canonical_link_path), strict=True)


def _build_year_mask_map(cfg: SampleConfig, permno_universe: set[int]) -> dict[int, int]:
    if not cfg.ccm_daily_path.exists():
        raise FileNotFoundError(f"CCM daily file not found: {cfg.ccm_daily_path}")

    year_bits = {year: 1 << idx for idx, year in enumerate(cfg.years)}
    year_pairs = (
        pl.scan_parquet(cfg.ccm_daily_path)
        .select(
            pl.col("KYPERMNO").cast(pl.Int32, strict=False).alias("kypermno"),
            pl.col("CALDT").cast(pl.Date, strict=False).dt.year().alias("year"),
        )
        .drop_nulls(subset=["kypermno", "year"])
        .filter(
            (pl.col("kypermno") > 0)
            & (pl.col("year") >= cfg.start_year)
            & (pl.col("year") <= cfg.end_year)
        )
        .unique(subset=["kypermno", "year"])
        .collect()
    )

    by_permno: dict[int, int] = defaultdict(int)
    for row in year_pairs.iter_rows(named=True):
        permno = int(row["kypermno"])
        year = int(row["year"])
        if permno in permno_universe:
            by_permno[permno] |= year_bits[year]
    return dict(by_permno)


def _build_filing_doc_link_maps(
    sec_filings_lf: pl.LazyFrame,
    links_lf: pl.LazyFrame,
) -> tuple[pl.DataFrame, dict[int, set[str]], int, dict[int, set[str]]]:
    date_valid_expr = (
        (pl.col("valid_start").is_null() | (pl.col("filing_date") >= pl.col("valid_start")))
        & (pl.col("valid_end").is_null() | (pl.col("filing_date") <= pl.col("valid_end")))
    )
    long_df = (
        sec_filings_lf.join(
            links_lf.select("cik_10", "gvkey", "kypermno", "valid_start", "valid_end"),
            on="cik_10",
            how="inner",
        )
        .filter(date_valid_expr & (pl.col("kypermno") > 0))
        .select("doc_id", "cik_10", "filing_date", "gvkey", "kypermno")
        .unique(subset=["doc_id", "gvkey", "kypermno"])
        .sort(["doc_id", "kypermno", "gvkey"], nulls_last=True)
        .collect()
    )

    permno_to_doc_ids: dict[int, set[str]] = defaultdict(set)
    permno_to_ciks: dict[int, set[str]] = defaultdict(set)
    for row in long_df.iter_rows(named=True):
        permno = int(row["kypermno"])
        permno_to_doc_ids[permno].add(str(row["doc_id"]))
        permno_to_ciks[permno].add(str(row["cik_10"]))

    linked_doc_count = (
        int(long_df.select(pl.col("doc_id").n_unique().alias("n")).item()) if long_df.height else 0
    )
    return long_df, dict(permno_to_doc_ids), linked_doc_count, dict(permno_to_ciks)


def _build_doc_mask_map(permno_to_doc_ids: dict[int, set[str]]) -> tuple[dict[int, int], list[str]]:
    ordered_doc_ids = sorted({doc_id for doc_ids in permno_to_doc_ids.values() for doc_id in doc_ids})
    doc_index = {doc_id: idx for idx, doc_id in enumerate(ordered_doc_ids)}

    doc_mask_by_permno: dict[int, int] = {}
    for permno, doc_ids in permno_to_doc_ids.items():
        mask = 0
        for doc_id in doc_ids:
            mask |= 1 << doc_index[doc_id]
        if mask:
            doc_mask_by_permno[permno] = mask
    return doc_mask_by_permno, ordered_doc_ids


def _prune_candidates(
    year_mask_by_permno: dict[int, int],
    doc_mask_by_permno: dict[int, int],
) -> list[CoverageCandidate]:
    best_by_coverage: dict[tuple[int, int], int] = {}
    all_permnos = set(year_mask_by_permno) | set(doc_mask_by_permno)
    for permno in all_permnos:
        year_mask = year_mask_by_permno.get(permno, 0)
        doc_mask = doc_mask_by_permno.get(permno, 0)
        if year_mask == 0 and doc_mask == 0:
            continue
        coverage = (year_mask, doc_mask)
        previous = best_by_coverage.get(coverage)
        if previous is None or permno < previous:
            best_by_coverage[coverage] = permno

    candidates = [
        CoverageCandidate(
            permno=permno,
            year_mask=year_mask,
            doc_mask=doc_mask,
            year_count=year_mask.bit_count(),
            doc_count=doc_mask.bit_count(),
        )
        for (year_mask, doc_mask), permno in best_by_coverage.items()
    ]
    candidates.sort(key=lambda candidate: (-candidate.year_count, -candidate.doc_count, candidate.permno))

    pruned: list[CoverageCandidate] = []
    for candidate in candidates:
        dominated = False
        for kept in pruned:
            if (
                (candidate.year_mask | kept.year_mask) == kept.year_mask
                and (candidate.doc_mask | kept.doc_mask) == kept.doc_mask
            ):
                dominated = True
                break
        if not dominated:
            pruned.append(candidate)
    return pruned


def _combined_greedy_candidates(
    candidates: list[CoverageCandidate],
    target_year_mask: int,
    target_doc_count: int,
) -> tuple[list[int], int, int]:
    uncovered_year_mask = target_year_mask
    covered_doc_mask = 0
    remaining = list(candidates)
    selected: list[int] = []

    while uncovered_year_mask != 0 or covered_doc_mask.bit_count() < target_doc_count:
        docs_remaining = max(target_doc_count - covered_doc_mask.bit_count(), 0)
        best_index: int | None = None
        best_score = 0
        for index, candidate in enumerate(remaining):
            new_year_count = (candidate.year_mask & uncovered_year_mask).bit_count()
            new_doc_count = (candidate.doc_mask & ~covered_doc_mask).bit_count()
            score = new_year_count * (target_doc_count + 1) + min(new_doc_count, docs_remaining)
            if score > best_score:
                best_index = index
                best_score = score
                continue
            if score == best_score and score > 0 and best_index is not None:
                current_best = remaining[best_index]
                if candidate.permno < current_best.permno:
                    best_index = index

        if best_index is None or best_score <= 0:
            break

        chosen = remaining.pop(best_index)
        selected.append(chosen.permno)
        uncovered_year_mask &= ~chosen.year_mask
        covered_doc_mask |= chosen.doc_mask

    return selected, target_year_mask & ~uncovered_year_mask, covered_doc_mask


def _ceil_div(numerator: int, denominator: int) -> int:
    return (numerator + denominator - 1) // denominator


def _find_exact_feasible_subset(
    candidates: list[CoverageCandidate],
    target_year_mask: int,
    target_doc_count: int,
    budget_k: int,
) -> list[int] | None:
    ordered = sorted(
        candidates,
        key=lambda candidate: (
            -(
                candidate.year_count * (target_doc_count + 1)
                + min(candidate.doc_count, target_doc_count)
            ),
            -candidate.year_count,
            -candidate.doc_count,
            candidate.permno,
        ),
    )
    n_candidates = len(ordered)
    suffix_year_union = [0] * (n_candidates + 1)
    suffix_doc_union = [0] * (n_candidates + 1)
    suffix_max_year = [0] * (n_candidates + 1)
    suffix_max_doc = [0] * (n_candidates + 1)
    for idx in range(n_candidates - 1, -1, -1):
        candidate = ordered[idx]
        suffix_year_union[idx] = suffix_year_union[idx + 1] | candidate.year_mask
        suffix_doc_union[idx] = suffix_doc_union[idx + 1] | candidate.doc_mask
        suffix_max_year[idx] = max(suffix_max_year[idx + 1], candidate.year_count)
        suffix_max_doc[idx] = max(suffix_max_doc[idx + 1], candidate.doc_count)

    def dfs(
        idx: int,
        selected: list[int],
        covered_year_mask: int,
        covered_doc_mask: int,
    ) -> list[int] | None:
        if covered_year_mask == target_year_mask and covered_doc_mask.bit_count() >= target_doc_count:
            return list(selected)
        if idx >= n_candidates or len(selected) >= budget_k:
            return None
        if (covered_year_mask | suffix_year_union[idx]) != target_year_mask:
            return None
        if (covered_doc_mask | suffix_doc_union[idx]).bit_count() < target_doc_count:
            return None

        uncovered_years = (target_year_mask & ~covered_year_mask).bit_count()
        docs_remaining = max(target_doc_count - covered_doc_mask.bit_count(), 0)
        lower_bounds: list[int] = []
        if uncovered_years > 0:
            max_year_gain = suffix_max_year[idx]
            if max_year_gain <= 0:
                return None
            lower_bounds.append(_ceil_div(uncovered_years, max_year_gain))
        if docs_remaining > 0:
            max_doc_gain = suffix_max_doc[idx]
            if max_doc_gain <= 0:
                return None
            lower_bounds.append(_ceil_div(docs_remaining, max_doc_gain))
        min_needed = max(lower_bounds) if lower_bounds else 0
        if len(selected) + min_needed > budget_k:
            return None

        candidate = ordered[idx]
        new_year_mask = covered_year_mask | candidate.year_mask
        new_doc_mask = covered_doc_mask | candidate.doc_mask
        if new_year_mask != covered_year_mask or new_doc_mask != covered_doc_mask:
            selected.append(candidate.permno)
            found = dfs(idx + 1, selected, new_year_mask, new_doc_mask)
            if found is not None:
                return found
            selected.pop()

        return dfs(idx + 1, selected, covered_year_mask, covered_doc_mask)

    return dfs(0, [], 0, 0)


def _mask_to_years(mask: int, years: list[int]) -> list[int]:
    return [year for idx, year in enumerate(years) if mask & (1 << idx)]


def _select_mandatory_anchor_permnos(
    year_mask_by_permno: dict[int, int],
    doc_mask_by_permno: dict[int, int],
    target_year_mask: int,
    target_doc_count: int,
    budget_k: int,
    years: list[int],
) -> AnchorSelectionResult:
    candidates = _prune_candidates(year_mask_by_permno, doc_mask_by_permno)
    if target_year_mask == 0 and target_doc_count <= 0:
        return AnchorSelectionResult([], "greedy", 0, len(candidates))

    greedy_selected, greedy_year_mask, greedy_doc_mask = _combined_greedy_candidates(
        candidates,
        target_year_mask,
        target_doc_count,
    )
    greedy_doc_count = greedy_doc_mask.bit_count()
    if (
        greedy_year_mask == target_year_mask
        and greedy_doc_count >= target_doc_count
        and len(greedy_selected) <= budget_k
    ):
        return AnchorSelectionResult(
            selected_permnos=sorted(greedy_selected),
            strategy="greedy",
            greedy_count=len(greedy_selected),
            pruned_candidate_count=len(candidates),
        )

    fallback_selected = _find_exact_feasible_subset(
        candidates,
        target_year_mask=target_year_mask,
        target_doc_count=target_doc_count,
        budget_k=budget_k,
    )
    if fallback_selected is not None:
        return AnchorSelectionResult(
            selected_permnos=sorted(fallback_selected),
            strategy="exact_fallback",
            greedy_count=len(greedy_selected),
            pruned_candidate_count=len(candidates),
        )

    uncovered_years = _mask_to_years(target_year_mask & ~greedy_year_mask, years)
    uncovered_docs = max(target_doc_count - greedy_doc_count, 0)
    raise ValueError(
        "Unable to satisfy mandatory anchor constraints within sample budget. "
        f"k={budget_k} greedy_anchor_count={len(greedy_selected)} "
        f"pruned_candidate_count={len(candidates)} uncovered_years={uncovered_years} "
        f"uncovered_linked_docs={uncovered_docs}"
    )


def _choose_sampled_permnos(
    cfg: SampleConfig,
    permno_universe: list[int],
    mandatory_anchor_permnos: list[int],
) -> list[int]:
    k = max(1, math.ceil(len(permno_universe) * cfg.sample_frac))
    anchors = set(mandatory_anchor_permnos)
    if len(anchors) > k:
        raise ValueError(
            "Mandatory anchor set exceeds sample size. "
            f"anchors={len(anchors)} sample_size={k}. Increase --sample-frac."
        )

    rng = random.Random(cfg.seed)
    remaining = sorted(set(permno_universe) - anchors)
    n_fill = k - len(anchors)
    fill = rng.sample(remaining, n_fill) if n_fill > 0 else []
    sampled = sorted(anchors | set(fill))
    return sampled


def _year_coverage_from_daily(cfg: SampleConfig, sampled_permnos: list[int]) -> dict[int, int]:
    sampled_daily = (
        pl.scan_parquet(cfg.ccm_daily_path)
        .filter(pl.col("KYPERMNO").cast(pl.Int32, strict=False).is_in(sampled_permnos))
        .select(
            pl.col("KYPERMNO").cast(pl.Int32, strict=False).alias("kypermno"),
            pl.col("CALDT").cast(pl.Date, strict=False).dt.year().alias("year"),
        )
        .drop_nulls(subset=["kypermno", "year"])
        .filter((pl.col("year") >= cfg.start_year) & (pl.col("year") <= cfg.end_year))
        .group_by("year")
        .agg(pl.col("kypermno").n_unique().alias("n_permno"))
        .collect()
    )

    out = {int(y): 0 for y in cfg.years}
    for row in sampled_daily.iter_rows(named=True):
        out[int(row["year"])] = int(row["n_permno"])
    return out


def _build_file_sampling_filter(
    schema: pl.Schema,
    sampled_permnos: list[int],
    sampled_gvkeys: list[str],
) -> tuple[pl.Expr | None, str | None, str | None]:
    names = list(schema.names())
    permno_col = _first_existing(names, PERMNO_CANDIDATES)
    gvkey_col = _first_existing(names, GVKEY_CANDIDATES)
    if permno_col is None and gvkey_col is None:
        return None, None, None

    if permno_col is not None and gvkey_col is None:
        filter_expr = pl.col(permno_col).cast(pl.Int32, strict=False).is_in(sampled_permnos)
        return filter_expr, permno_col, None
    if permno_col is None and gvkey_col is not None:
        filter_expr = pl.col(gvkey_col).cast(pl.Utf8, strict=False).is_in(sampled_gvkeys)
        return filter_expr, None, gvkey_col

    perm_expr = pl.col(permno_col).cast(pl.Int32, strict=False)
    gvkey_expr = pl.col(gvkey_col).cast(pl.Utf8, strict=False)
    filter_expr = perm_expr.is_in(sampled_permnos) | (
        perm_expr.is_null() & gvkey_expr.is_in(sampled_gvkeys)
    )
    return filter_expr, permno_col, gvkey_col


def _write_filing_mapping_outputs(cfg: SampleConfig, long_df: pl.DataFrame) -> None:
    cfg.out_mappings_dir.mkdir(parents=True, exist_ok=True)
    long_parquet = cfg.out_mappings_dir / "filing_doc_to_permno_long.parquet"
    long_csv = cfg.out_mappings_dir / "filing_doc_to_permno_long.csv"
    long_df.write_parquet(long_parquet, compression=cfg.compression)
    long_df.write_csv(long_csv)

    doc_collapsed = (
        long_df.group_by("doc_id", "cik_10", "filing_date")
        .agg(
            pl.len().alias("n_long_rows"),
            pl.col("gvkey").drop_nulls().n_unique().alias("n_gvkeys"),
            pl.col("kypermno").filter(pl.col("kypermno") > 0).n_unique().alias("n_permnos_pos"),
            pl.col("kypermno")
            .filter(pl.col("kypermno") > 0)
            .sort()
            .first()
            .alias("best_permno"),
        )
        .with_columns(
            pl.col("best_permno").is_not_null().alias("has_pos_permno"),
            pl.col("best_permno").fill_null(0).cast(pl.Int32).alias("permno"),
        )
        .select(
            "doc_id",
            "cik_10",
            "filing_date",
            "permno",
            "best_permno",
            "has_pos_permno",
            "n_permnos_pos",
            "n_gvkeys",
            "n_long_rows",
        )
        .sort(["filing_date", "doc_id"])
    )
    doc_collapsed.write_parquet(
        cfg.out_mappings_dir / "filing_doc_to_permno_collapsed.parquet",
        compression=cfg.compression,
    )
    doc_collapsed.write_csv(cfg.out_mappings_dir / "filing_doc_to_permno_collapsed.csv")

    cik_collapsed = (
        long_df.group_by("cik_10")
        .agg(
            pl.col("doc_id").n_unique().alias("n_docs"),
            pl.len().alias("n_long_rows"),
            pl.col("gvkey").drop_nulls().n_unique().alias("n_gvkeys"),
            pl.col("kypermno").filter(pl.col("kypermno") > 0).n_unique().alias("n_permnos_pos"),
            pl.col("kypermno")
            .filter(pl.col("kypermno") > 0)
            .sort()
            .first()
            .alias("best_permno"),
        )
        .with_columns(
            pl.col("best_permno").is_not_null().alias("has_pos_permno"),
            pl.col("best_permno").fill_null(0).cast(pl.Int32).alias("permno"),
        )
        .select(
            "cik_10",
            "n_docs",
            "permno",
            "best_permno",
            "has_pos_permno",
            "n_permnos_pos",
            "n_gvkeys",
            "n_long_rows",
        )
        .sort("cik_10")
    )
    cik_collapsed.write_parquet(
        cfg.out_mappings_dir / "filing_cik_to_permno_collapsed.parquet",
        compression=cfg.compression,
    )
    cik_collapsed.write_csv(cfg.out_mappings_dir / "filing_cik_to_permno_collapsed.csv")


def _sample_ccm_files(
    cfg: SampleConfig,
    sampled_permnos: list[int],
    sampled_gvkeys: list[str],
) -> list[dict[str, Any]]:
    stats_rows: list[dict[str, Any]] = []
    included_files: list[str] = []
    excluded_files: list[str] = []
    empty_files: list[str] = []

    for src in sorted(cfg.ccm_parquet_dir.rglob("*.parquet")):
        schema = pl.scan_parquet(src).collect_schema()
        filter_expr, permno_col, gvkey_col = _build_file_sampling_filter(
            schema,
            sampled_permnos=sampled_permnos,
            sampled_gvkeys=sampled_gvkeys,
        )
        rel = src.relative_to(cfg.ccm_parquet_dir)

        if filter_expr is None:
            excluded_files.append(str(rel))
            stats_rows.append(
                {
                    "file": str(rel),
                    "sampled": False,
                    "reason": "no_permno_or_gvkey",
                    "permno_col": None,
                    "gvkey_col": None,
                    "input_rows": _rows_in_parquet(src),
                    "output_rows": None,
                    "input_permno_nunique": None,
                    "output_permno_nunique": None,
                    "input_gvkey_nunique": None,
                    "output_gvkey_nunique": None,
                }
            )
            continue

        dst = cfg.out_ccm_dir / rel
        dst.parent.mkdir(parents=True, exist_ok=True)

        sampled_lf = pl.scan_parquet(src).filter(filter_expr)
        sampled_lf.sink_parquet(dst, compression=cfg.compression)

        in_rows = _rows_in_parquet(src)
        out_rows = _rows_in_parquet(dst)
        in_permno = _unique_count(src, permno_col) if permno_col is not None else None
        out_permno = _unique_count(dst, permno_col) if permno_col is not None else None
        in_gvkey = _unique_count(src, gvkey_col) if gvkey_col is not None else None
        out_gvkey = _unique_count(dst, gvkey_col) if gvkey_col is not None else None

        included_files.append(str(rel))
        if out_rows == 0:
            empty_files.append(str(rel))

        stats_rows.append(
            {
                "file": str(rel),
                "sampled": True,
                "reason": "id_filtered",
                "permno_col": permno_col,
                "gvkey_col": gvkey_col,
                "input_rows": in_rows,
                "output_rows": out_rows,
                "input_permno_nunique": in_permno,
                "output_permno_nunique": out_permno,
                "input_gvkey_nunique": in_gvkey,
                "output_gvkey_nunique": out_gvkey,
            }
        )

    cfg.out_reports_dir.mkdir(parents=True, exist_ok=True)
    (cfg.out_reports_dir / "included_files.txt").write_text(
        "\n".join(included_files) + ("\n" if included_files else ""),
        encoding="utf-8",
    )
    (cfg.out_reports_dir / "excluded_files.txt").write_text(
        "\n".join(excluded_files) + ("\n" if excluded_files else ""),
        encoding="utf-8",
    )
    (cfg.out_reports_dir / "empty_files.txt").write_text(
        "\n".join(empty_files) + ("\n" if empty_files else ""),
        encoding="utf-8",
    )
    _write_csv(cfg.out_reports_dir / "sample_file_stats.csv", stats_rows)
    return stats_rows


def _format_pct_token(sample_frac: float) -> str:
    pct = Decimal(str(sample_frac)) * Decimal("100")
    token = format(pct, "f").rstrip("0").rstrip(".")
    if not token:
        token = "0"
    return token.replace(".", "p")


def _sample_tag(cfg: SampleConfig) -> str:
    return f"sample_{_format_pct_token(cfg.sample_frac)}pct_seed{cfg.seed}"


def _tagged_output_name(base_name: str, sample_tag: str) -> str:
    path = Path(base_name)
    suffix = path.suffix
    stem = path.stem if suffix else path.name
    if stem.endswith(sample_tag):
        return path.name
    return f"{stem}.{sample_tag}{suffix}"


def _write_overlap_report(
    cfg: SampleConfig,
    sampled_permnos: list[int],
    sampled_gvkeys: list[str],
    doc_long_df: pl.DataFrame,
    filing_doc_all_count: int,
    filing_cik_all_count: int,
    linked_doc_target_count: int,
    year_coverage: dict[int, int],
) -> dict[str, Any]:
    linked_doc_count = (
        int(doc_long_df.select(pl.col("doc_id").n_unique().alias("n")).item()) if doc_long_df.height else 0
    )
    linked_cik_count = (
        int(doc_long_df.select(pl.col("cik_10").n_unique().alias("n")).item()) if doc_long_df.height else 0
    )

    covered_linked_docs = (
        int(
            doc_long_df.lazy()
            .filter(pl.col("kypermno").is_in(sampled_permnos))
            .select(pl.col("doc_id").n_unique().alias("n"))
            .collect()
            .item()
        )
        if doc_long_df.height
        else 0
    )
    covered_linked_ciks = (
        int(
            doc_long_df.lazy()
            .filter(pl.col("kypermno").is_in(sampled_permnos))
            .select(pl.col("cik_10").n_unique().alias("n"))
            .collect()
            .item()
        )
        if doc_long_df.height
        else 0
    )
    linked_doc_ratio = (covered_linked_docs / linked_doc_count) if linked_doc_count > 0 else 0.0
    all_doc_ratio = (covered_linked_docs / filing_doc_all_count) if filing_doc_all_count > 0 else 0.0
    linked_cik_ratio = (covered_linked_ciks / linked_cik_count) if linked_cik_count > 0 else 0.0
    all_cik_ratio = (covered_linked_ciks / filing_cik_all_count) if filing_cik_all_count > 0 else 0.0

    missing_years = [y for y in cfg.years if year_coverage.get(y, 0) <= 0]
    if linked_doc_ratio < cfg.overlap_target:
        raise ValueError(
            "Linked filing doc overlap target not met. "
            f"observed={linked_doc_ratio:.6f} target={cfg.overlap_target:.6f}"
        )
    if missing_years:
        raise ValueError(
            f"Sampled CCM daily year coverage missing years: {missing_years}"
        )

    payload = {
        "sampled_permno_count": len(sampled_permnos),
        "sampled_gvkey_count": len(sampled_gvkeys),
        "sample_fraction_target": cfg.sample_frac,
        "filing_overlap_target_linked_ratio": cfg.overlap_target,
        "filing_doc_all_count": filing_doc_all_count,
        "filing_doc_linked_count": linked_doc_count,
        "filing_doc_linked_target_count": linked_doc_target_count,
        "filing_doc_linked_covered_count": covered_linked_docs,
        "filing_doc_linked_covered_ratio": linked_doc_ratio,
        "filing_doc_all_covered_count": covered_linked_docs,
        "filing_doc_all_covered_ratio": all_doc_ratio,
        "filing_cik_all_count": filing_cik_all_count,
        "filing_cik_linked_count": linked_cik_count,
        "filing_cik_linked_covered_count": covered_linked_ciks,
        "filing_cik_linked_covered_ratio": linked_cik_ratio,
        "filing_cik_all_covered_count": covered_linked_ciks,
        "filing_cik_all_covered_ratio": all_cik_ratio,
        "year_coverage_start": cfg.start_year,
        "year_coverage_end": cfg.end_year,
        "yearly_ccm_permno_counts": year_coverage,
        "missing_years": missing_years,
    }
    _write_json(cfg.out_reports_dir / "sample_overlap_report.json", payload)

    csv_rows = [
        {"metric": "sampled_permno_count", "value": len(sampled_permnos)},
        {"metric": "sampled_gvkey_count", "value": len(sampled_gvkeys)},
        {"metric": "filing_doc_all_count", "value": filing_doc_all_count},
        {"metric": "filing_doc_linked_count", "value": linked_doc_count},
        {"metric": "filing_doc_linked_target_count", "value": linked_doc_target_count},
        {"metric": "filing_doc_linked_covered_count", "value": covered_linked_docs},
        {"metric": "filing_doc_linked_covered_ratio", "value": linked_doc_ratio},
        {"metric": "filing_doc_all_covered_count", "value": covered_linked_docs},
        {"metric": "filing_doc_all_covered_ratio", "value": all_doc_ratio},
        {"metric": "filing_cik_all_count", "value": filing_cik_all_count},
        {"metric": "filing_cik_linked_count", "value": linked_cik_count},
        {"metric": "filing_cik_linked_covered_count", "value": covered_linked_ciks},
        {"metric": "filing_cik_linked_covered_ratio", "value": linked_cik_ratio},
        {"metric": "filing_cik_all_covered_count", "value": covered_linked_ciks},
        {"metric": "filing_cik_all_covered_ratio", "value": all_cik_ratio},
    ]
    for year in cfg.years:
        csv_rows.append({"metric": f"year_{year}_permno_count", "value": year_coverage.get(year, 0)})
    _write_csv(cfg.out_reports_dir / "sample_overlap_report.csv", csv_rows)
    return payload


def _sample_derived_outputs(
    cfg: SampleConfig,
    sampled_permnos: list[int],
    sampled_gvkeys: list[str],
) -> dict[str, str]:
    cfg.out_derived_dir.mkdir(parents=True, exist_ok=True)

    sample_tag = _sample_tag(cfg)
    daily_out = cfg.out_derived_dir / _tagged_output_name(cfg.ccm_daily_name, sample_tag)
    (
        pl.scan_parquet(cfg.ccm_daily_path)
        .filter(pl.col("KYPERMNO").cast(pl.Int32, strict=False).is_in(sampled_permnos))
        .sink_parquet(daily_out, compression=cfg.compression)
    )

    canon_out = cfg.out_derived_dir / _tagged_output_name(cfg.canonical_link_name, sample_tag)
    (
        pl.scan_parquet(cfg.canonical_link_path)
        .with_columns(
            pl.col("kypermno").cast(pl.Int32, strict=False).alias("kypermno"),
            pl.col("gvkey").cast(pl.Utf8, strict=False).alias("gvkey"),
        )
        .filter(
            pl.col("kypermno").is_in(sampled_permnos)
            | (pl.col("kypermno").is_null() & pl.col("gvkey").is_in(sampled_gvkeys))
        )
        .sink_parquet(canon_out, compression=cfg.compression)
    )

    return {
        "daily_sample_path": str(daily_out),
        "canonical_sample_path": str(canon_out),
    }


def run(cfg: SampleConfig) -> None:
    cfg.out_root.mkdir(parents=True, exist_ok=True)
    cfg.out_inventory_dir.mkdir(parents=True, exist_ok=True)
    cfg.out_reports_dir.mkdir(parents=True, exist_ok=True)

    inventory = _build_pipeline_inventory(cfg)
    if not inventory["reuse_inputs"]["ccm_daily_exists"]:
        raise FileNotFoundError(f"Missing CCM daily parquet: {cfg.ccm_daily_path}")
    if not inventory["reuse_inputs"]["canonical_link_exists"]:
        raise FileNotFoundError(f"Missing canonical link parquet: {cfg.canonical_link_path}")

    links_lf = _load_canonical_links(cfg)
    permno_universe = (
        links_lf.filter(pl.col("kypermno") > 0)
        .select(pl.col("kypermno").unique().sort())
        .collect()
        .get_column("kypermno")
        .to_list()
    )
    if not permno_universe:
        raise ValueError("No positive permno values found in canonical link table.")
    permno_universe_set = set(int(x) for x in permno_universe)

    sec_filings_lf = _load_sec_filings(cfg)
    sec_counts = (
        sec_filings_lf.select(
            pl.len().alias("filing_doc_all_count"),
            pl.col("cik_10").n_unique().alias("filing_cik_all_count"),
        )
        .collect()
        .row(0, named=True)
    )
    filing_doc_all_count = int(sec_counts["filing_doc_all_count"])
    filing_cik_all_count = int(sec_counts["filing_cik_all_count"])

    filing_long, permno_to_doc_ids, linked_doc_count, permno_to_ciks = _build_filing_doc_link_maps(
        sec_filings_lf,
        links_lf,
    )
    linked_target_count = int(math.ceil(linked_doc_count * cfg.overlap_target))

    year_mask_by_permno = _build_year_mask_map(cfg, permno_universe_set)
    doc_mask_by_permno, _ = _build_doc_mask_map(permno_to_doc_ids)
    target_year_mask = (1 << len(cfg.years)) - 1
    k = max(1, math.ceil(len(permno_universe) * cfg.sample_frac))
    anchor_result = _select_mandatory_anchor_permnos(
        year_mask_by_permno=year_mask_by_permno,
        doc_mask_by_permno=doc_mask_by_permno,
        target_year_mask=target_year_mask,
        target_doc_count=linked_target_count,
        budget_k=k,
        years=cfg.years,
    )
    sampled_permnos = _choose_sampled_permnos(
        cfg,
        permno_universe=permno_universe,
        mandatory_anchor_permnos=anchor_result.selected_permnos,
    )

    sampled_gvkeys = (
        links_lf.filter(pl.col("kypermno").is_in(sampled_permnos))
        .select(pl.col("gvkey").drop_nulls().unique().sort())
        .collect()
        .get_column("gvkey")
        .to_list()
    )

    _write_filing_mapping_outputs(cfg, filing_long)
    derived_paths = _sample_derived_outputs(cfg, sampled_permnos, sampled_gvkeys)
    file_stats = _sample_ccm_files(cfg, sampled_permnos, sampled_gvkeys)

    year_coverage = _year_coverage_from_daily(cfg, sampled_permnos)
    overlap_payload = _write_overlap_report(
        cfg=cfg,
        sampled_permnos=sampled_permnos,
        sampled_gvkeys=sampled_gvkeys,
        doc_long_df=filing_long,
        filing_doc_all_count=filing_doc_all_count,
        filing_cik_all_count=filing_cik_all_count,
        linked_doc_target_count=linked_target_count,
        year_coverage=year_coverage,
    )

    manifest = {
        "config": {
            "full_data_root": str(cfg.full_data_root),
            "out_root": str(cfg.out_root),
            "sample_frac": cfg.sample_frac,
            "sample_tag": _sample_tag(cfg),
            "overlap_target": cfg.overlap_target,
            "seed": cfg.seed,
            "start_year": cfg.start_year,
            "end_year": cfg.end_year,
            "canonical_link_name": cfg.canonical_link_name,
            "ccm_daily_name": cfg.ccm_daily_name,
            "compression": cfg.compression,
        },
        "counts": {
            "permno_universe_count": len(permno_universe),
            "sampled_permno_count": len(sampled_permnos),
            "sampled_gvkey_count": len(sampled_gvkeys),
            "mandatory_anchor_permno_count": len(anchor_result.selected_permnos),
            "anchor_greedy_count": anchor_result.greedy_count,
            "anchor_candidate_count": anchor_result.pruned_candidate_count,
            "linked_filing_doc_count": linked_doc_count,
            "linked_filing_doc_target_count": linked_target_count,
            "linked_filing_cik_count": int(
                filing_long.select(pl.col("cik_10").n_unique().alias("n")).item() if filing_long.height else 0
            ),
        },
        "anchors": {
            "mandatory_anchor_permnos": anchor_result.selected_permnos,
            "anchor_selection_strategy": anchor_result.strategy,
            "anchor_permno_to_ciks": {
                str(permno): sorted(permno_to_ciks.get(permno, set()))
                for permno in anchor_result.selected_permnos
            },
        },
        "derived_outputs": derived_paths,
        "overlap_report": overlap_payload,
        "sample_file_stats_path": str(cfg.out_reports_dir / "sample_file_stats.csv"),
        "sampled_ccm_files": int(sum(1 for row in file_stats if bool(row.get("sampled")))),
        "excluded_ccm_files": int(sum(1 for row in file_stats if not bool(row.get("sampled")))),
    }
    _write_json(cfg.out_root / "sample_manifest.json", manifest)

    print(
        json.dumps(
            {
                "out_root": str(cfg.out_root),
                "sampled_permnos": len(sampled_permnos),
                "sampled_gvkeys": len(sampled_gvkeys),
                "linked_overlap_ratio": overlap_payload["filing_doc_linked_covered_ratio"],
                "missing_years": overlap_payload["missing_years"],
                "anchor_strategy": anchor_result.strategy,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    run(_parse_args())
