---
description: Workflow to document code pipeline
---

# Workflow: Code-Driven Architecture & Parameter Mapping

## Phase 1: AST Extraction & DAG Generation
1. **Execute Native Tooling**: Run `python extract_ast.py` and `python tools/docs_trace.py` against `src/thesis_pkg/`.
2. **Parse Execution Flow**: Build a topological sort of the execution graph, anchored by the entry points in `src/thesis_pkg/pipelines/` (e.g., `sec_pipeline.py`, `ccm_pipeline.py`, `sec_ccm_pipeline.py`).
3. **Scaffold**: Generate `docs/architecture/00_master_index.md` listing the execution sequence strictly as it exists in the AST.

## Phase 2: Granular Module Documentation (Iterative Map-Reduce)
For every Python file identified in the DAG, spawn a sub-task to generate `docs/architecture/XX_[module_name].md`. Extract information strictly from the source code:

### A. Signatures & Parameters
1. List all classes and functions with their exact type hints.
2. Generate a "Parameter Configuration" table for every callable. Columns must include: Parameter Name, Type, Default Value, and Code-Derived Purpose. Highlight any optional parameters.

### B. Execution Logic & Transformations
1. Translate the exact Polars/LazyFrame operations (joins, group_bys, `with_columns` vectorizations) into sequential English steps. 
2. Specify the exact column names being mutated or generated at each step.

### C. Hardcoded Assumptions & Boundary Conditions
1. Scan the AST for hardcoded logic: regex patterns in `core/sec/`, lag day limits in `core/ccm/`, and default thresholds (e.g., line lengths for sparsity checks).
2. Document the exact conditional logic used for error handling and routing (e.g., the assignment of discrete `match_reason_code` values).

## Phase 3: Design Choice Extraction
Synthesize the mechanics from Phase 2 to explicitly state the design choices embedded in the code:
1. **Join Mechanics**: Document the temporal alignment strategy (e.g., how `filing_date` is evaluated against `trading_calendar_lf` using `PhaseBAlignmentMode`).
2. **Textual Boundaries**: Detail how Table of Contents (TOC) masking and cross-reference rejections are mathematically enforced prior to extraction.

## Phase 4: Literature Mapping (Appendix)
Only after the code is fully documented, generate `docs/architecture/99_literature_mapping.md`. 
1. Cross-reference the documented text extraction boundaries against the requirements of Cohen et al. (2020) for isolating document-level textual changes.
2. Cross-reference any tokenization or text cleaning steps against the requirements for Loughran & McDonald (2011) sentiment dictionaries.
3. Explicitly flag any implementation details in the code that mathematically or logically diverge from the methodologies of the target papers.

## Phase 5: Compilation
1. Concatenate all `XX_[module_name].md` files and the appendix into `docs/architecture/FULL_PIPELINE_ARCHITECTURE.md`.