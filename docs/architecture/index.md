# Architecture

This section groups the durable design notes for `thesis_pkg`. It complements the generated API reference by explaining how the SEC extraction stack, CCM merge logic, and pipeline entry points fit together.

## Recommended Reading Order

1. [Thesis Pipeline Architecture](FINAL_THESIS_PIPELINE.md)
2. [Master Execution Index](00_master_index.md)
3. [Full Pipeline Architecture](FULL_PIPELINE_ARCHITECTURE.md)

## Package Entry Points

- [Package Root](16___init__.md)
- [API Facade](14_api.md)
- [Pipeline Facade](17_pipeline.md)
- [Filing Text Facade](18_filing_text.md)
- [Settings](01_settings.md)

## Pipelines and IO

- [SEC Pipeline](25_sec_pipeline.md)
- [CCM Pipeline](22_ccm_pipeline.md)
- [SEC-CCM Pipeline](19_sec_ccm_pipeline.md)
- [Parquet IO](26_parquet.md)

## CCM Merge Stack

- [Canonical Links](24_canonical_links.md)
- [Transforms](23_transforms.md)
- [SEC-CCM Contracts](21_sec_ccm_contracts.md)
- [SEC-CCM Pre-Merge](20_sec_ccm_premerge.md)
- [CCM Cleaning](13_ccm_cleaning.md)

## SEC Extraction Stack

- [Patterns](32_patterns.md)
- [Heuristics](29_heuristics.md)
- [Embedded Headings](08_embedded_headings.md)
- [Extraction Utilities](33_extraction_utils.md)
- [Extraction Engine](28_extraction.md)
- [Regime Logic](31_regime.md)
- [Parquet Streaming](06_parquet_stream.md)
- [HTML Audit](07_html_audit.md)
- [Boundary Diagnostics](05_suspicious_boundary_diagnostics.md)
- [Utilities](30_utilities.md)

## Context

- [Design Choices](98_design_choices.md)
- [Literature Mapping](99_literature_mapping.md)
