import datetime as dt

import polars as pl

from thesis_pkg.cleaning import clean_us_common_major_exchange_panel, clean_us_common_major_exchange_parquet


def test_clean_us_common_major_exchange_panel_refreshes_flags_before_filter():
    base = pl.DataFrame(
        {
            "KYPERMNO": [1, 2, 3],
            "CALDT": [dt.date(2024, 1, 2), dt.date(2024, 1, 2), dt.date(2024, 1, 2)],
            "PRC": [10.0, 10.0, 10.0],
            "SHRCD": [10, 12, 11],
            "EXCHCD": [1, 1, 4],
            "VOL": [100.0, 100.0, 100.0],
            "TCAP": [100_000_000.0, 100_000_000.0, 100_000_000.0],
            "data_status": [0, 0, 0],
        }
    )

    out = clean_us_common_major_exchange_panel(base.lazy(), refresh_concept_flags=True).collect()
    assert out.select("KYPERMNO").to_series().to_list() == [1]


def test_clean_us_common_major_exchange_panel_without_refresh_uses_existing_bits():
    base = pl.DataFrame(
        {
            "KYPERMNO": [1, 2],
            "CALDT": [dt.date(2024, 1, 2), dt.date(2024, 1, 2)],
            "PRC": [10.0, 10.0],
            "SHRCD": [10, 10],
            "EXCHCD": [1, 1],
            "VOL": [100.0, 100.0],
            "TCAP": [100_000_000.0, 100_000_000.0],
            "data_status": [0, 0],
        }
    )

    out = clean_us_common_major_exchange_panel(base.lazy(), refresh_concept_flags=False).collect()
    assert out.height == 0


def test_clean_us_common_major_exchange_parquet_roundtrip(tmp_path):
    in_path = tmp_path / "in.parquet"
    out_path = tmp_path / "out.parquet"

    pl.DataFrame(
        {
            "KYPERMNO": [1, 2],
            "CALDT": [dt.date(2024, 1, 2), dt.date(2024, 1, 2)],
            "PRC": [10.0, 10.0],
            "SHRCD": [10, 12],
            "EXCHCD": [1, 1],
            "VOL": [100.0, 100.0],
            "TCAP": [100_000_000.0, 100_000_000.0],
            "data_status": [0, 0],
        }
    ).write_parquet(in_path)

    written = clean_us_common_major_exchange_parquet(in_path, out_path, refresh_concept_flags=True)
    assert written == out_path
    assert out_path.exists()

    out = pl.read_parquet(out_path)
    assert out.select("KYPERMNO").to_series().to_list() == [1]

