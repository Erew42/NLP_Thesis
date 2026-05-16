use pyo3::prelude::*;

mod audit_summaries;
mod common;
mod doc_ownership;
mod finbert_allocation;
mod finbert_confusion;
mod finbert_sampling;
mod finbert_tail;
mod html_audit;
mod io_parquet;
mod item_cleaning;
mod lm2011_extension_tables;
mod lm2011_features;
mod lm2011_inputs;
mod lm2011_regressions;
mod lm2011_validation;
mod lseg_api_rows;
mod lseg_ops;
mod multisurface_audit;
mod refinitiv_analyst;
mod refinitiv_authority;
mod refinitiv_bridge;
mod refinitiv_excel;
mod sec_extraction;
mod sentence_cleaning;
mod sentence_quality_api;

pub(crate) use audit_summaries::*;
pub(crate) use common::*;
pub(crate) use doc_ownership::*;
pub(crate) use finbert_allocation::*;
pub(crate) use finbert_confusion::*;
pub(crate) use finbert_sampling::*;
pub(crate) use finbert_tail::*;
pub(crate) use html_audit::*;
pub(crate) use io_parquet::*;
pub(crate) use item_cleaning::*;
pub(crate) use lm2011_extension_tables::*;
pub(crate) use lm2011_features::*;
pub(crate) use lm2011_inputs::*;
pub(crate) use lm2011_regressions::*;
pub(crate) use lm2011_validation::*;
pub(crate) use lseg_api_rows::*;
pub(crate) use lseg_ops::*;
pub(crate) use multisurface_audit::*;
pub(crate) use refinitiv_analyst::*;
pub(crate) use refinitiv_authority::*;
pub(crate) use refinitiv_bridge::*;
pub(crate) use refinitiv_excel::*;
pub(crate) use sec_extraction::*;
pub(crate) use sentence_quality_api::*;

mod py_exports;

#[pymodule]
fn _lm2011_rust(m: &Bound<'_, PyModule>) -> PyResult<()> {
    py_exports::register_all(m)
}
