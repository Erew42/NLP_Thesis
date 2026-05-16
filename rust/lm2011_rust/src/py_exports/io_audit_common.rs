// Common audit, hashing, markdown, parquet, and streaming-I/O entrypoints.
use crate::*;
use pyo3::wrap_pyfunction;

pub(crate) fn register(m: &pyo3::Bound<'_, pyo3::types::PyModule>) -> pyo3::PyResult<()> {
    m.add_function(wrap_pyfunction!(sec_ccm_markdown_value, m)?)?;
    m.add_function(wrap_pyfunction!(sec_ccm_markdown_table, m)?)?;
    m.add_function(wrap_pyfunction!(sec_ccm_markdown_table_columns, m)?)?;
    m.add_function(wrap_pyfunction!(sec_no_item_aggregate_rows, m)?)?;
    m.add_function(wrap_pyfunction!(sec_no_item_aggregate_columns, m)?)?;
    m.add_function(wrap_pyfunction!(normalize_binary_label_value, m)?)?;
    m.add_function(wrap_pyfunction!(sha256_hex_value, m)?)?;
    m.add_function(wrap_pyfunction!(sha256_first_u64_value, m)?)?;
    m.add_function(wrap_pyfunction!(sha256_hex_values, m)?)?;
    m.add_function(wrap_pyfunction!(stable_string_fingerprint_values, m)?)?;
    m.add_function(wrap_pyfunction!(semantic_guard_mismatches_value, m)?)?;
    m.add_function(wrap_pyfunction!(sha256_file_hex_value, m)?)?;
    m.add_function(wrap_pyfunction!(parquet_magic_probe, m)?)?;
    m.add_function(wrap_pyfunction!(copy_file_stream, m)?)?;
    m.add_function(wrap_pyfunction!(parquet_stream_selected_doc_indices, m)?)?;
    m.add_function(wrap_pyfunction!(lm2011_window_row_index_pairs, m)?)?;
    Ok(())
}
