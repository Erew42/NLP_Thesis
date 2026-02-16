from __future__ import annotations

from pathlib import Path

import polars as pl

from thesis_pkg.core.ccm.transforms import apply_concept_filter_flags_doc, filter_us_common_major_exchange


def clean_us_common_major_exchange_panel(
    panel_lf: pl.LazyFrame,
    *,
    refresh_concept_flags: bool = True,
) -> pl.LazyFrame:
    """
    Keep only U.S. common stocks on major exchanges as a cleaning step.

    When ``refresh_concept_flags`` is True, concept-filter bits are recomputed
    from canonical panel columns before applying the status-bit filter.
    """
    prepared = apply_concept_filter_flags_doc(panel_lf) if refresh_concept_flags else panel_lf
    return filter_us_common_major_exchange(prepared)


def clean_us_common_major_exchange_parquet(
    input_path: Path | str,
    output_path: Path | str,
    *,
    refresh_concept_flags: bool = True,
    compression: str = "zstd",
) -> Path:
    """Read parquet lazily, run the U.S.-common-major-exchange cleaner, and write parquet."""
    input_parquet = Path(input_path)
    output_parquet = Path(output_path)
    cleaned = clean_us_common_major_exchange_panel(
        pl.scan_parquet(input_parquet),
        refresh_concept_flags=refresh_concept_flags,
    )
    output_parquet.parent.mkdir(parents=True, exist_ok=True)
    cleaned.sink_parquet(output_parquet, compression=compression)
    return output_parquet

