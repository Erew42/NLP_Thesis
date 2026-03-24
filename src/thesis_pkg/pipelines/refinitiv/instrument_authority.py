from __future__ import annotations

from pathlib import Path

import polars as pl

from thesis_pkg.pipelines.refinitiv_bridge_pipeline import (
    _cast_df_to_schema,
    _read_resolution_artifact_parquet,
)


INSTRUMENT_AUTHORITY_COLUMNS: tuple[str, ...] = (
    "bridge_row_id",
    "KYPERMNO",
    "gvkey_int",
    "gvkey_source_column",
    "first_seen_caldt",
    "last_seen_caldt",
    "effective_collection_ric",
    "effective_collection_ric_source",
    "effective_resolution_status",
    "authority_eligible",
    "authority_exclusion_reason",
)

_GVKEY_SOURCE_CANDIDATES: tuple[str, ...] = (
    "gvkey",
    "KYGVKEY_final",
    "KYGVKEY",
    "KYGVKEY_ccm",
)


def _instrument_authority_schema() -> dict[str, pl.DataType]:
    return {
        "bridge_row_id": pl.Utf8,
        "KYPERMNO": pl.Int32,
        "gvkey_int": pl.Int32,
        "gvkey_source_column": pl.Utf8,
        "first_seen_caldt": pl.Date,
        "last_seen_caldt": pl.Date,
        "effective_collection_ric": pl.Utf8,
        "effective_collection_ric_source": pl.Utf8,
        "effective_resolution_status": pl.Utf8,
        "authority_eligible": pl.Boolean,
        "authority_exclusion_reason": pl.Utf8,
    }


def _resolve_first_existing(schema: pl.Schema, candidates: tuple[str, ...], label: str) -> str:
    for candidate in candidates:
        if candidate in schema:
            return candidate
    raise ValueError(f"{label} missing any of expected columns: {list(candidates)}")


def _read_bridge_artifact_parquet(parquet_path: Path | str) -> pl.DataFrame:
    parquet_path = Path(parquet_path)
    if not parquet_path.exists():
        raise FileNotFoundError(f"bridge artifact not found: {parquet_path}")
    return pl.read_parquet(parquet_path)


def build_refinitiv_step1_instrument_authority_frame(
    bridge_df: pl.DataFrame,
    resolution_df: pl.DataFrame,
) -> pl.DataFrame:
    bridge_schema = bridge_df.schema
    bridge_required = ("bridge_row_id", "first_seen_caldt", "last_seen_caldt")
    missing_bridge = [name for name in bridge_required if name not in bridge_schema]
    if missing_bridge:
        raise ValueError(f"bridge_df missing required columns: {missing_bridge}")

    resolution_required = (
        "bridge_row_id",
        "effective_collection_ric",
        "effective_collection_ric_source",
        "effective_resolution_status",
    )
    missing_resolution = [name for name in resolution_required if name not in resolution_df.columns]
    if missing_resolution:
        raise ValueError(f"resolution_df missing required columns: {missing_resolution}")

    gvkey_source_column = _resolve_first_existing(
        bridge_schema,
        _GVKEY_SOURCE_CANDIDATES,
        "refinitiv bridge artifact for instrument authority",
    )

    bridge_cols = [
        pl.col("bridge_row_id").cast(pl.Utf8, strict=False).alias("bridge_row_id"),
        (
            pl.col("KYPERMNO").cast(pl.Int32, strict=False).alias("KYPERMNO")
            if "KYPERMNO" in bridge_schema
            else pl.lit(None, dtype=pl.Int32).alias("KYPERMNO")
        ),
        pl.col(gvkey_source_column).cast(pl.Int32, strict=False).alias("gvkey_int"),
        pl.lit(gvkey_source_column, dtype=pl.Utf8).alias("gvkey_source_column"),
        pl.col("first_seen_caldt").cast(pl.Date, strict=False).alias("first_seen_caldt"),
        pl.col("last_seen_caldt").cast(pl.Date, strict=False).alias("last_seen_caldt"),
    ]
    bridge_selected = (
        bridge_df.select(bridge_cols)
        .unique(subset=["bridge_row_id"], keep="first", maintain_order=True)
    )

    resolution_selected = resolution_df.select(
        pl.col("bridge_row_id").cast(pl.Utf8, strict=False).alias("bridge_row_id"),
        pl.col("effective_collection_ric").cast(pl.Utf8, strict=False).alias("effective_collection_ric"),
        pl.col("effective_collection_ric_source").cast(pl.Utf8, strict=False).alias("effective_collection_ric_source"),
        pl.col("effective_resolution_status").cast(pl.Utf8, strict=False).alias("effective_resolution_status"),
    ).unique(subset=["bridge_row_id"], keep="first", maintain_order=True)

    authority = resolution_selected.join(bridge_selected, on="bridge_row_id", how="left").with_columns(
        pl.when(pl.col("gvkey_int").is_not_null() & pl.col("effective_collection_ric").is_not_null())
        .then(pl.lit(True))
        .otherwise(pl.lit(False))
        .alias("authority_eligible"),
        pl.when(pl.col("gvkey_int").is_null() & pl.col("effective_collection_ric").is_null())
        .then(pl.lit("missing_gvkey_int_and_effective_collection_ric"))
        .when(pl.col("gvkey_int").is_null())
        .then(pl.lit("missing_gvkey_int"))
        .when(pl.col("effective_collection_ric").is_null())
        .then(pl.lit("missing_effective_collection_ric"))
        .otherwise(pl.lit(None, dtype=pl.Utf8))
        .alias("authority_exclusion_reason"),
    )

    return _cast_df_to_schema(
        authority.select(INSTRUMENT_AUTHORITY_COLUMNS),
        _instrument_authority_schema(),
    ).select(INSTRUMENT_AUTHORITY_COLUMNS)


def run_refinitiv_step1_instrument_authority_pipeline(
    *,
    bridge_artifact_path: Path | str,
    resolution_artifact_path: Path | str,
    output_dir: Path | str,
) -> dict[str, Path]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    bridge_df = _read_bridge_artifact_parquet(bridge_artifact_path)
    resolution_df = _read_resolution_artifact_parquet(resolution_artifact_path)
    authority_df = build_refinitiv_step1_instrument_authority_frame(bridge_df, resolution_df)

    parquet_path = output_dir / "refinitiv_instrument_authority_common_stock.parquet"
    authority_df.write_parquet(parquet_path, compression="zstd")
    return {"refinitiv_instrument_authority_common_stock_parquet": parquet_path}


__all__ = [
    "INSTRUMENT_AUTHORITY_COLUMNS",
    "build_refinitiv_step1_instrument_authority_frame",
    "run_refinitiv_step1_instrument_authority_pipeline",
]
