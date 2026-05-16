mod finbert;
mod io_audit_common;
mod lm2011_analysis;
mod lm2011_core;
mod refinitiv_lseg;
mod sec;
mod text_processing;

pub(crate) fn register_all(m: &pyo3::Bound<'_, pyo3::types::PyModule>) -> pyo3::PyResult<()> {
    lm2011_core::register(m)?;
    sec::register(m)?;
    refinitiv_lseg::register(m)?;
    text_processing::register(m)?;
    finbert::register(m)?;
    lm2011_analysis::register(m)?;
    io_audit_common::register(m)?;
    Ok(())
}
