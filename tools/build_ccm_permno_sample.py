from __future__ import annotations

import argparse
import csv
import json
import math
import random
from collections import defaultdict
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any

import polars as pl


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


def _normalize_cik10_expr(col_name: str) -> pl.Expr:
    return (
        pl.col(col_name)
        .cast(pl.Utf8, strict=False)
        .str.replace_all(r"\D+", "")
        .str.zfill(10)
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


def _load_filing_ciks(cfg: SampleConfig) -> pl.DataFrame:
    year_files = sorted(
        p for p in cfg.year_merged_dir.glob("*.parquet") if p.stem.isdigit() and len(p.stem) == 4
    )
    if not year_files:
        raise FileNotFoundError(f"No yearly SEC parquet files found in {cfg.year_merged_dir}")

    sec_lf = pl.scan_parquet([str(p) for p in year_files])
    schema = sec_lf.collect_schema()
    if "cik_10" in schema:
        cik_expr = _normalize_cik10_expr("cik_10").alias("cik_10")
    elif "cik" in schema:
        cik_expr = _normalize_cik10_expr("cik").alias("cik_10")
    else:
        raise ValueError("SEC year_merged files missing cik_10 and cik columns.")

    ciks = (
        sec_lf.select(cik_expr)
        .drop_nulls(subset=["cik_10"])
        .filter(pl.col("cik_10").str.len_chars() == 10)
        .unique(subset=["cik_10"])
        .sort("cik_10")
        .collect()
    )
    return ciks


def _load_canonical_links(cfg: SampleConfig) -> pl.LazyFrame:
    if not cfg.canonical_link_path.exists():
        raise FileNotFoundError(f"Canonical link file not found: {cfg.canonical_link_path}")

    return (
        pl.scan_parquet(cfg.canonical_link_path)
        .with_columns(
            _normalize_cik10_expr("cik_10").alias("cik_10"),
            pl.col("gvkey").cast(pl.Utf8, strict=False).alias("gvkey"),
            pl.col("kypermno").cast(pl.Int32, strict=False).alias("kypermno"),
            pl.col("valid_start").cast(pl.Date, strict=False).alias("valid_start"),
            pl.col("valid_end").cast(pl.Date, strict=False).alias("valid_end"),
        )
        .filter(pl.col("cik_10").str.len_chars() == 10)
    )


def _build_year_anchor_map(cfg: SampleConfig, permno_universe: set[int]) -> dict[int, set[int]]:
    if not cfg.ccm_daily_path.exists():
        raise FileNotFoundError(f"CCM daily file not found: {cfg.ccm_daily_path}")

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

    by_permno: dict[int, set[int]] = defaultdict(set)
    for row in year_pairs.iter_rows(named=True):
        permno = int(row["kypermno"])
        year = int(row["year"])
        if permno in permno_universe:
            by_permno[permno].add(year)
    return dict(by_permno)


def _greedy_year_anchors(target_years: set[int], by_permno: dict[int, set[int]]) -> list[int]:
    uncovered = set(target_years)
    selected: list[int] = []

    while uncovered:
        best_permno: int | None = None
        best_gain = 0
        for permno, years in by_permno.items():
            gain = len(years & uncovered)
            if gain > best_gain or (gain == best_gain and gain > 0 and best_permno is not None and permno < best_permno):
                best_permno = permno
                best_gain = gain
            elif gain > 0 and best_permno is None:
                best_permno = permno
                best_gain = gain

        if best_permno is None or best_gain <= 0:
            raise ValueError(
                f"Cannot cover target years. Uncovered years: {sorted(uncovered)}"
            )
        selected.append(best_permno)
        uncovered -= by_permno.get(best_permno, set())
    return selected


def _build_filing_link_maps(
    filing_ciks: pl.DataFrame, links_lf: pl.LazyFrame
) -> tuple[pl.DataFrame, dict[int, set[str]], int]:
    link_pairs = (
        links_lf.select("cik_10", "gvkey", "kypermno")
        .unique(subset=["cik_10", "gvkey", "kypermno"])
        .collect()
    )

    long_df = (
        filing_ciks.lazy()
        .join(link_pairs.lazy(), on="cik_10", how="left")
        .sort(["cik_10", "kypermno", "gvkey"], nulls_last=True)
        .collect()
    )

    linked = long_df.filter(pl.col("kypermno") > 0)
    permno_to_ciks: dict[int, set[str]] = defaultdict(set)
    for row in linked.select("cik_10", "kypermno").iter_rows(named=True):
        permno_to_ciks[int(row["kypermno"])].add(str(row["cik_10"]))

    linked_cik_count = int(
        linked.select(pl.col("cik_10").n_unique().alias("n")).item()
    )
    return long_df, dict(permno_to_ciks), linked_cik_count


def _greedy_filing_anchors(
    permno_to_ciks: dict[int, set[str]],
    linked_target_count: int,
) -> tuple[list[int], set[str]]:
    if linked_target_count <= 0:
        return [], set()

    covered: set[str] = set()
    selected: list[int] = []
    available = dict(permno_to_ciks)

    while len(covered) < linked_target_count:
        best_permno: int | None = None
        best_gain = 0
        for permno, ciks in available.items():
            gain = len(ciks - covered)
            if gain > best_gain or (gain == best_gain and gain > 0 and best_permno is not None and permno < best_permno):
                best_permno = permno
                best_gain = gain
            elif gain > 0 and best_permno is None:
                best_permno = permno
                best_gain = gain

        if best_permno is None or best_gain <= 0:
            raise ValueError(
                "Unable to satisfy filing overlap target with available permno->cik links."
            )
        selected.append(best_permno)
        covered |= available.get(best_permno, set())
        available.pop(best_permno, None)

    return selected, covered


def _choose_sampled_permnos(
    cfg: SampleConfig,
    permno_universe: list[int],
    year_anchor_permnos: list[int],
    filing_anchor_permnos: list[int],
) -> list[int]:
    k = max(1, math.ceil(len(permno_universe) * cfg.sample_frac))
    anchors = set(year_anchor_permnos) | set(filing_anchor_permnos)
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
    long_parquet = cfg.out_mappings_dir / "filing_cik_to_permno_long.parquet"
    long_csv = cfg.out_mappings_dir / "filing_cik_to_permno_long.csv"
    long_df.write_parquet(long_parquet, compression=cfg.compression)
    long_df.write_csv(long_csv)

    collapsed = (
        long_df.group_by("cik_10")
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
            "cik_10",
            "permno",
            "best_permno",
            "has_pos_permno",
            "n_permnos_pos",
            "n_gvkeys",
            "n_long_rows",
        )
        .sort("cik_10")
    )
    collapsed_parquet = cfg.out_mappings_dir / "filing_cik_to_permno_collapsed.parquet"
    collapsed_csv = cfg.out_mappings_dir / "filing_cik_to_permno_collapsed.csv"
    collapsed.write_parquet(collapsed_parquet, compression=cfg.compression)
    collapsed.write_csv(collapsed_csv)


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


def _write_overlap_report(
    cfg: SampleConfig,
    sampled_permnos: list[int],
    sampled_gvkeys: list[str],
    filing_long: pl.DataFrame,
    linked_target_count: int,
    year_coverage: dict[int, int],
) -> dict[str, Any]:
    n_filing_all = int(filing_long.select(pl.col("cik_10").n_unique()).item())
    linked_ciks = filing_long.filter(pl.col("kypermno") > 0).select("cik_10").unique()
    linked_cik_count = linked_ciks.height

    covered_linked = (
        filing_long.lazy()
        .filter((pl.col("kypermno") > 0) & pl.col("kypermno").is_in(sampled_permnos))
        .select("cik_10")
        .unique(subset=["cik_10"])
        .collect()
        .height
    )
    covered_all = (
        filing_long.lazy()
        .filter(pl.col("kypermno").is_in(sampled_permnos))
        .select("cik_10")
        .unique(subset=["cik_10"])
        .collect()
        .height
    )
    linked_ratio = (covered_linked / linked_cik_count) if linked_cik_count > 0 else 0.0
    all_ratio = (covered_all / n_filing_all) if n_filing_all > 0 else 0.0

    missing_years = [y for y in cfg.years if year_coverage.get(y, 0) <= 0]
    if linked_ratio < cfg.overlap_target:
        raise ValueError(
            "Linked filing overlap target not met. "
            f"observed={linked_ratio:.6f} target={cfg.overlap_target:.6f}"
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
        "filing_cik_all_count": n_filing_all,
        "filing_cik_linked_count": linked_cik_count,
        "filing_cik_linked_target_count": linked_target_count,
        "filing_cik_linked_covered_count": covered_linked,
        "filing_cik_linked_covered_ratio": linked_ratio,
        "filing_cik_all_covered_count": covered_all,
        "filing_cik_all_covered_ratio": all_ratio,
        "year_coverage_start": cfg.start_year,
        "year_coverage_end": cfg.end_year,
        "yearly_ccm_permno_counts": year_coverage,
        "missing_years": missing_years,
    }
    _write_json(cfg.out_reports_dir / "sample_overlap_report.json", payload)

    csv_rows = [
        {"metric": "sampled_permno_count", "value": len(sampled_permnos)},
        {"metric": "sampled_gvkey_count", "value": len(sampled_gvkeys)},
        {"metric": "filing_cik_all_count", "value": n_filing_all},
        {"metric": "filing_cik_linked_count", "value": linked_cik_count},
        {"metric": "filing_cik_linked_target_count", "value": linked_target_count},
        {"metric": "filing_cik_linked_covered_count", "value": covered_linked},
        {"metric": "filing_cik_linked_covered_ratio", "value": linked_ratio},
        {"metric": "filing_cik_all_covered_count", "value": covered_all},
        {"metric": "filing_cik_all_covered_ratio", "value": all_ratio},
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

    daily_out = cfg.out_derived_dir / "final_flagged_data_compdesc_added.sample_5pct_seed42.parquet"
    (
        pl.scan_parquet(cfg.ccm_daily_path)
        .filter(pl.col("KYPERMNO").cast(pl.Int32, strict=False).is_in(sampled_permnos))
        .sink_parquet(daily_out, compression=cfg.compression)
    )

    canon_out = (
        cfg.out_derived_dir
        / "canonical_link_table_after_startdate_change.sample_5pct_seed42.parquet"
    )
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

    filing_ciks = _load_filing_ciks(cfg)
    filing_long, permno_to_ciks, linked_cik_count = _build_filing_link_maps(filing_ciks, links_lf)
    linked_target_count = int(math.ceil(linked_cik_count * cfg.overlap_target))

    year_map = _build_year_anchor_map(cfg, permno_universe_set)
    year_anchor_permnos = _greedy_year_anchors(set(cfg.years), year_map)
    filing_anchor_permnos, _ = _greedy_filing_anchors(permno_to_ciks, linked_target_count)
    sampled_permnos = _choose_sampled_permnos(
        cfg,
        permno_universe=permno_universe,
        year_anchor_permnos=year_anchor_permnos,
        filing_anchor_permnos=filing_anchor_permnos,
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
        filing_long=filing_long,
        linked_target_count=linked_target_count,
        year_coverage=year_coverage,
    )

    manifest = {
        "config": {
            "full_data_root": str(cfg.full_data_root),
            "out_root": str(cfg.out_root),
            "sample_frac": cfg.sample_frac,
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
            "year_anchor_permno_count": len(year_anchor_permnos),
            "filing_anchor_permno_count": len(filing_anchor_permnos),
            "linked_filing_cik_count": linked_cik_count,
            "linked_filing_target_count": linked_target_count,
        },
        "anchors": {
            "year_anchor_permnos": year_anchor_permnos,
            "filing_anchor_permnos": filing_anchor_permnos,
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
                "linked_overlap_ratio": overlap_payload["filing_cik_linked_covered_ratio"],
                "missing_years": overlap_payload["missing_years"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    run(_parse_args())
