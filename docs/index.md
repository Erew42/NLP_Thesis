# NLP Thesis Documentation

This site documents the `thesis_pkg` codebase that supports SEC filing extraction, CRSP/Compustat linking, SEC-CCM pre-merge logic, and downstream thesis analysis workflows.

## Start Here

- [Architecture](architecture/index.md) for system maps, pipeline flows, and module-level design notes.
- [Reference](reference/index.md) for generated API pages and behavior evidence artifacts.
- [Notes](other/index.md) for contracts, reviews, and supporting operational documentation.
- [Decisions](decisions/index.md) for stable architectural choices and ADR-style records.

## Primary Workflows

1. Market-data alignment and CCM preparation.
2. SEC filing text normalization and item extraction.
3. Master merge plus `data_status` flagging.
4. Analysis-ready outputs and diagnostics.

## Docs Workflow

Prefer the repository wrappers over direct `mkdocs` commands so metadata refresh and Windows UTF-8 handling stay consistent.

```bash
python tools/docs_pipeline.py extract
python tools/docs_pipeline.py scaffold
python tools/docs_pipeline.py check
python tools/docs_pipeline.py build
```
