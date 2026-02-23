from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import polars as pl


@dataclass(frozen=True)
class SampleConfig:
    linkdata_dir: Path
    output_dir: Path
    seed: int
    target_overlap_pairs: int
    target_history_only_pairs: int
    target_fiscal_only_pairs: int
    max_fiscal_rows_per_pair: int
    max_history_rows_per_pair: int
    force_large_mismatch_pairs: int
    min_per_overlap_stratum: int
    min_per_history_stratum: int


def _decade_label(year: int | None) -> str:
    if year is None:
        return "unknown"
    return f"{(year // 10) * 10}s"


def _row_is_sparse_fiscal(row: dict[str, object]) -> bool:
    return (
        row.get("liid") is None
        and row.get("linktype") is None
        and row.get("linkprim") is None
        and row.get("linkdt") is None
        and row.get("linkenddt") is None
    )


def _sample_by_strata(
    rows: list[dict[str, object]],
    *,
    target: int,
    strata_keys: tuple[str, ...],
    rng: random.Random,
    min_per_stratum: int,
) -> list[dict[str, object]]:
    if target <= 0 or not rows:
        return []

    groups: dict[tuple[object, ...], list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        groups[tuple(row.get(k) for k in strata_keys)].append(row)
    for values in groups.values():
        rng.shuffle(values)

    selected: list[dict[str, object]] = []
    for key in sorted(groups, key=lambda x: str(x)):
        take = min(min_per_stratum, len(groups[key]), max(target - len(selected), 0))
        if take > 0:
            selected.extend(groups[key][:take])
            del groups[key][:take]
        if len(selected) >= target:
            return selected[:target]

    while len(selected) < target:
        active = [key for key, values in groups.items() if values]
        if not active:
            break
        weights = [len(groups[key]) for key in active]
        picked = rng.choices(active, weights=weights, k=1)[0]
        selected.append(groups[picked].pop())

    return selected[:target]


def _read_pair_profiles(linkdata_dir: Path) -> pl.DataFrame:
    history = (
        pl.scan_parquet(str(linkdata_dir / "linkhistory.parquet"))
        .select(
            pl.col("KYGVKEY").cast(pl.Int32).alias("KYGVKEY"),
            pl.col("LPERMNO").cast(pl.Int32).alias("permno"),
            pl.col("LINKDT").cast(pl.Date).alias("h_start"),
            pl.col("LINKENDDT").cast(pl.Date).alias("h_end"),
        )
        .drop_nulls(subset=["KYGVKEY", "permno"])
        .group_by(["KYGVKEY", "permno"])
        .agg(
            pl.col("h_start").min().alias("h_min_start"),
            pl.col("h_end").max().alias("h_max_end"),
            pl.len().cast(pl.Int32).alias("h_rows"),
        )
    )

    fiscal = (
        pl.scan_parquet(str(linkdata_dir / "linkfiscalperiodall.parquet"))
        .select(
            pl.col("KYGVKEY").cast(pl.Int32).alias("KYGVKEY"),
            pl.col("lpermno").cast(pl.Int32).alias("permno"),
            pl.coalesce([pl.col("linkdt"), pl.col("FiscalPeriodCRSPStartDt")]).cast(pl.Date).alias("f_start"),
            pl.coalesce([pl.col("linkenddt"), pl.col("FiscalPeriodCRSPEndDt")]).cast(pl.Date).alias("f_end"),
            (
                pl.col("liid").is_null()
                & pl.col("linktype").is_null()
                & pl.col("linkprim").is_null()
                & pl.col("linkdt").is_null()
                & pl.col("linkenddt").is_null()
            ).alias("is_sparse"),
        )
        .drop_nulls(subset=["KYGVKEY", "permno"])
        .group_by(["KYGVKEY", "permno"])
        .agg(
            pl.col("f_start").min().alias("f_min_start"),
            pl.col("f_end").max().alias("f_max_end"),
            pl.len().cast(pl.Int32).alias("f_rows"),
            pl.col("is_sparse").sum().cast(pl.Int32).alias("f_sparse_rows"),
        )
    )

    all_pairs = pl.concat(
        [
            history.select("KYGVKEY", "permno"),
            fiscal.select("KYGVKEY", "permno"),
        ],
        how="vertical_relaxed",
    ).unique()

    prof = (
        all_pairs.join(history, on=["KYGVKEY", "permno"], how="left")
        .join(fiscal, on=["KYGVKEY", "permno"], how="left")
        .with_columns(
            pl.when(pl.col("h_rows").is_not_null() & pl.col("f_rows").is_not_null())
            .then(pl.lit("overlap"))
            .when(pl.col("h_rows").is_not_null())
            .then(pl.lit("history_only"))
            .otherwise(pl.lit("fiscal_only"))
            .alias("pair_class"),
            (pl.col("permno") > 0).alias("is_tradable"),
            pl.coalesce([pl.col("h_min_start"), pl.col("f_min_start")]).dt.year().alias("anchor_year"),
        )
        .with_columns(
            pl.when(pl.col("h_min_start").is_not_null() & pl.col("f_min_start").is_not_null())
            .then((pl.col("h_min_start") - pl.col("f_min_start")).dt.total_days().abs())
            .otherwise(pl.lit(None, dtype=pl.Int64))
            .alias("start_delta_days"),
            pl.when(pl.col("h_max_end").is_not_null() & pl.col("f_max_end").is_not_null())
            .then((pl.col("h_max_end") - pl.col("f_max_end")).dt.total_days().abs())
            .otherwise(pl.lit(None, dtype=pl.Int64))
            .alias("end_delta_days"),
        )
        .with_columns(
            pl.max_horizontal(["start_delta_days", "end_delta_days"]).alias("max_boundary_delta_days"),
        )
        .with_columns(
            (pl.col("max_boundary_delta_days") > 365).fill_null(False).alias("is_large_mismatch"),
        )
        .collect()
    )

    return prof.with_columns(
        pl.when(pl.col("anchor_year").is_null())
        .then(pl.lit("unknown"))
        .otherwise((((pl.col("anchor_year") // 10) * 10).cast(pl.Int32).cast(pl.Utf8) + pl.lit("s")))
        .alias("anchor_decade")
    )


def _pick_sample_pairs(profile: pl.DataFrame, cfg: SampleConfig) -> pl.DataFrame:
    rng = random.Random(cfg.seed)
    rows = profile.to_dicts()

    overlap = [row for row in rows if row["pair_class"] == "overlap"]
    history_only = [row for row in rows if row["pair_class"] == "history_only"]
    fiscal_only = [row for row in rows if row["pair_class"] == "fiscal_only"]

    overlap_mismatch = [row for row in overlap if bool(row.get("is_large_mismatch"))]
    overlap_regular = [row for row in overlap if not bool(row.get("is_large_mismatch"))]

    forced_mismatch_target = min(
        cfg.force_large_mismatch_pairs,
        cfg.target_overlap_pairs,
        len(overlap_mismatch),
    )
    forced_mismatch = _sample_by_strata(
        overlap_mismatch,
        target=forced_mismatch_target,
        strata_keys=("is_tradable", "anchor_decade"),
        rng=rng,
        min_per_stratum=1,
    )

    forced_keys = {(row["KYGVKEY"], row["permno"]) for row in forced_mismatch}
    overlap_regular = [
        row for row in overlap_regular if (row["KYGVKEY"], row["permno"]) not in forced_keys
    ]
    overlap_rest = _sample_by_strata(
        overlap_regular,
        target=max(cfg.target_overlap_pairs - len(forced_mismatch), 0),
        strata_keys=("is_tradable", "anchor_decade"),
        rng=rng,
        min_per_stratum=cfg.min_per_overlap_stratum,
    )

    picked_overlap = forced_mismatch + overlap_rest
    picked_history = _sample_by_strata(
        history_only,
        target=cfg.target_history_only_pairs,
        strata_keys=("is_tradable", "anchor_decade"),
        rng=rng,
        min_per_stratum=cfg.min_per_history_stratum,
    )
    picked_fiscal = _sample_by_strata(
        fiscal_only,
        target=cfg.target_fiscal_only_pairs,
        strata_keys=("is_tradable", "anchor_decade"),
        rng=rng,
        min_per_stratum=1,
    )

    picked = picked_overlap + picked_history + picked_fiscal
    out = pl.DataFrame(picked).unique(subset=["KYGVKEY", "permno"])
    return out.sort(["pair_class", "KYGVKEY", "permno"])


def _sample_history_rows(cfg: SampleConfig, sample_pairs: pl.DataFrame) -> pl.DataFrame:
    pair_lf = sample_pairs.select(["KYGVKEY", "permno"]).lazy()
    hist = (
        pl.scan_parquet(str(cfg.linkdata_dir / "linkhistory.parquet"))
        .with_columns(
            pl.col("KYGVKEY").cast(pl.Int32),
            pl.col("LPERMNO").cast(pl.Int32),
        )
        .join(pair_lf, left_on=["KYGVKEY", "LPERMNO"], right_on=["KYGVKEY", "permno"], how="inner")
        .collect()
        .unique()
    )

    if cfg.max_history_rows_per_pair > 0:
        hist = (
            hist.sort(["KYGVKEY", "LPERMNO", "LINKDT", "LINKENDDT"])
            .group_by(["KYGVKEY", "LPERMNO"], maintain_order=True)
            .head(cfg.max_history_rows_per_pair)
        )
    return hist


def _sample_fiscal_rows(cfg: SampleConfig, sample_pairs: pl.DataFrame) -> pl.DataFrame:
    pair_lf = sample_pairs.select(["KYGVKEY", "permno"]).lazy()
    fiscal_cols = list(
        pl.scan_parquet(str(cfg.linkdata_dir / "linkfiscalperiodall.parquet")).collect_schema().names()
    )

    fiscal = (
        pl.scan_parquet(str(cfg.linkdata_dir / "linkfiscalperiodall.parquet"))
        .with_columns(
            pl.col("KYGVKEY").cast(pl.Int32),
            pl.col("lpermno").cast(pl.Int32),
        )
        .join(pair_lf, left_on=["KYGVKEY", "lpermno"], right_on=["KYGVKEY", "permno"], how="inner")
        .collect()
    )

    dedupe_cols = [
        "KYGVKEY",
        "lpermno",
        "lpermco",
        "liid",
        "linktype",
        "linkprim",
        "linkrank",
        "linkdt",
        "linkenddt",
        "FiscalPeriodCRSPStartDt",
        "FiscalPeriodCRSPEndDt",
    ]
    fiscal = fiscal.unique(subset=dedupe_cols)

    fiscal = fiscal.with_columns(
        pl.coalesce([pl.col("linkdt"), pl.col("FiscalPeriodCRSPStartDt")]).alias("_link_start"),
        pl.coalesce([pl.col("linkenddt"), pl.col("FiscalPeriodCRSPEndDt")]).alias("_link_end"),
        (
            pl.col("liid").is_null()
            & pl.col("linktype").is_null()
            & pl.col("linkprim").is_null()
            & pl.col("linkdt").is_null()
            & pl.col("linkenddt").is_null()
        ).alias("_is_sparse"),
    )

    if cfg.max_fiscal_rows_per_pair > 0:
        core = (
            fiscal
            .sort(
                ["KYGVKEY", "lpermno", "_is_sparse", "linkrank", "_link_start", "_link_end"],
                descending=[False, False, False, False, False, False],
                nulls_last=True,
            )
            .group_by(["KYGVKEY", "lpermno"], maintain_order=True)
            .head(cfg.max_fiscal_rows_per_pair)
        )
    else:
        core = fiscal

    sparse_first = (
        fiscal.filter(pl.col("_is_sparse"))
        .sort(["KYGVKEY", "lpermno", "_link_start", "_link_end"], nulls_last=True)
        .group_by(["KYGVKEY", "lpermno"], maintain_order=True)
        .head(1)
    )

    return (
        pl.concat([core, sparse_first], how="vertical_relaxed")
        .unique()
        .select([col for col in fiscal_cols if col in core.columns])
    )


def _sample_company_tables(cfg: SampleConfig, sample_pairs: pl.DataFrame) -> tuple[pl.DataFrame, pl.DataFrame]:
    gvkeys = sample_pairs.select("KYGVKEY").unique().lazy()
    company_history = (
        pl.scan_parquet(str(cfg.linkdata_dir / "companyhistory.parquet"))
        .with_columns(pl.col("KYGVKEY").cast(pl.Int32))
        .join(gvkeys, on="KYGVKEY", how="inner")
        .collect()
    )
    company_description = (
        pl.scan_parquet(str(cfg.linkdata_dir / "companydescription.parquet"))
        .with_columns(pl.col("KYGVKEY").cast(pl.Int32))
        .join(gvkeys, on="KYGVKEY", how="inner")
        .collect()
    )
    return company_history, company_description


def _build_manifest(
    cfg: SampleConfig,
    pair_profile: pl.DataFrame,
    sample_pairs: pl.DataFrame,
    linkhistory_sample: pl.DataFrame,
    linkfiscal_sample: pl.DataFrame,
    companyhistory_sample: pl.DataFrame,
    companydescription_sample: pl.DataFrame,
) -> dict[str, object]:
    pair_counts_full = (
        pair_profile.group_by("pair_class")
        .agg(pl.len().alias("n"))
        .sort("pair_class")
        .to_dicts()
    )
    pair_counts_sample = (
        sample_pairs.group_by("pair_class")
        .agg(pl.len().alias("n"))
        .sort("pair_class")
        .to_dicts()
    )
    decade_sample = (
        sample_pairs.group_by(["pair_class", "anchor_decade"])
        .agg(pl.len().alias("n"))
        .sort(["pair_class", "n"], descending=[False, True])
        .to_dicts()
    )

    hist_pairs = (
        linkhistory_sample.select(
            pl.col("KYGVKEY").cast(pl.Int32).alias("KYGVKEY"),
            pl.col("LPERMNO").cast(pl.Int32).alias("permno"),
        )
        .unique()
    )
    fisc_pairs = (
        linkfiscal_sample.select(
            pl.col("KYGVKEY").cast(pl.Int32).alias("KYGVKEY"),
            pl.col("lpermno").cast(pl.Int32).alias("permno"),
        )
        .unique()
    )
    overlap_pairs_in_sample = (
        hist_pairs.join(fisc_pairs, on=["KYGVKEY", "permno"], how="inner").select(pl.len()).item()
    )

    sparse_rows_in_fiscal_sample = (
        linkfiscal_sample.select(
            (
                pl.col("liid").is_null()
                & pl.col("linktype").is_null()
                & pl.col("linkprim").is_null()
                & pl.col("linkdt").is_null()
                & pl.col("linkenddt").is_null()
            ).sum()
        ).item()
    )

    return {
        "config": {
            "linkdata_dir": str(cfg.linkdata_dir),
            "output_dir": str(cfg.output_dir),
            "seed": cfg.seed,
            "target_overlap_pairs": cfg.target_overlap_pairs,
            "target_history_only_pairs": cfg.target_history_only_pairs,
            "target_fiscal_only_pairs": cfg.target_fiscal_only_pairs,
            "max_fiscal_rows_per_pair": cfg.max_fiscal_rows_per_pair,
            "max_history_rows_per_pair": cfg.max_history_rows_per_pair,
            "force_large_mismatch_pairs": cfg.force_large_mismatch_pairs,
            "min_per_overlap_stratum": cfg.min_per_overlap_stratum,
            "min_per_history_stratum": cfg.min_per_history_stratum,
        },
        "full_pair_counts": pair_counts_full,
        "sample_pair_counts": pair_counts_sample,
        "sample_pair_decade_breakdown": decade_sample,
        "sample_sizes": {
            "pair_rows": sample_pairs.height,
            "linkhistory_rows": linkhistory_sample.height,
            "linkfiscalperiodall_rows": linkfiscal_sample.height,
            "companyhistory_rows": companyhistory_sample.height,
            "companydescription_rows": companydescription_sample.height,
            "sample_pair_overlap_between_link_tables": overlap_pairs_in_sample,
            "sample_sparse_fiscal_rows": sparse_rows_in_fiscal_sample,
        },
    }


def _parse_args() -> SampleConfig:
    parser = argparse.ArgumentParser(
        description=(
            "Build deterministic, stratified LinkData parquet samples with "
            "cross-source ID overlap and time coverage."
        )
    )
    parser.add_argument(
        "--linkdata-dir",
        type=Path,
        default=Path(r"C:\Users\erik9\Documents\SEC_Data\Data\original data parquet\LinkData"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Defaults to <linkdata-dir>/samples/stratified_overlap_time_v1",
    )
    parser.add_argument("--seed", type=int, default=20260223)
    parser.add_argument("--target-overlap-pairs", type=int, default=4000)
    parser.add_argument("--target-history-only-pairs", type=int, default=800)
    parser.add_argument("--target-fiscal-only-pairs", type=int, default=600)
    parser.add_argument("--max-fiscal-rows-per-pair", type=int, default=8)
    parser.add_argument("--max-history-rows-per-pair", type=int, default=0)
    parser.add_argument("--force-large-mismatch-pairs", type=int, default=250)
    parser.add_argument("--min-per-overlap-stratum", type=int, default=20)
    parser.add_argument("--min-per-history-stratum", type=int, default=10)

    args = parser.parse_args()
    linkdata_dir = args.linkdata_dir
    output_dir = (
        args.output_dir
        if args.output_dir is not None
        else linkdata_dir / "samples" / "stratified_overlap_time_v1"
    )
    return SampleConfig(
        linkdata_dir=linkdata_dir,
        output_dir=output_dir,
        seed=args.seed,
        target_overlap_pairs=args.target_overlap_pairs,
        target_history_only_pairs=args.target_history_only_pairs,
        target_fiscal_only_pairs=args.target_fiscal_only_pairs,
        max_fiscal_rows_per_pair=args.max_fiscal_rows_per_pair,
        max_history_rows_per_pair=args.max_history_rows_per_pair,
        force_large_mismatch_pairs=args.force_large_mismatch_pairs,
        min_per_overlap_stratum=args.min_per_overlap_stratum,
        min_per_history_stratum=args.min_per_history_stratum,
    )


def main() -> None:
    cfg = _parse_args()
    cfg.output_dir.mkdir(parents=True, exist_ok=True)

    pair_profile = _read_pair_profiles(cfg.linkdata_dir)
    sample_pairs = _pick_sample_pairs(pair_profile, cfg)
    linkhistory_sample = _sample_history_rows(cfg, sample_pairs)
    linkfiscal_sample = _sample_fiscal_rows(cfg, sample_pairs)
    companyhistory_sample, companydescription_sample = _sample_company_tables(cfg, sample_pairs)

    sample_pairs.write_parquet(cfg.output_dir / "sample_pairs.parquet")
    linkhistory_sample.write_parquet(cfg.output_dir / "linkhistory_sample.parquet")
    linkfiscal_sample.write_parquet(cfg.output_dir / "linkfiscalperiodall_sample.parquet")
    companyhistory_sample.write_parquet(cfg.output_dir / "companyhistory_sample.parquet")
    companydescription_sample.write_parquet(cfg.output_dir / "companydescription_sample.parquet")

    manifest = _build_manifest(
        cfg=cfg,
        pair_profile=pair_profile,
        sample_pairs=sample_pairs,
        linkhistory_sample=linkhistory_sample,
        linkfiscal_sample=linkfiscal_sample,
        companyhistory_sample=companyhistory_sample,
        companydescription_sample=companydescription_sample,
    )
    (cfg.output_dir / "sample_manifest.json").write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )

    print(f"sample output dir: {cfg.output_dir}")
    print(json.dumps(manifest["sample_sizes"], indent=2))


if __name__ == "__main__":
    main()
