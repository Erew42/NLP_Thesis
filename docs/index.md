# Master Thesis Documentation

This site documents the `thesis_pkg` codebase that supports SEC filing extraction, CRSP/Compustat linking, SEC-CCM pre-merge logic, and downstream thesis analysis workflows.

## Start Here

- [Architecture](architecture/index.md) for the tracked architecture overview and source-area map.
- [Reference](reference/index.md) for generated API pages and behavior evidence artifacts.
- [Decisions](decisions/index.md) for stable architectural choices and ADR-style records.
- [Docstring Audit Report](docstring_audit_report.md) for tracked docstring coverage notes.

The reference and behavior-evidence sections are generated or refreshed locally
by the docs workflow. If those pages are missing in a fresh checkout, run the
workflow below before building the site.

## Primary Workflows

1. Market-data alignment and CCM preparation.
2. SEC filing text normalization and item extraction.
3. Master merge plus `data_status` flagging.
4. Analysis-ready outputs and diagnostics.

## Docs Workflow

Prefer the repository wrapper over direct `mkdocs` commands so metadata refresh
and Windows UTF-8 handling stay consistent.

```bash
python tools/docs_pipeline.py all
```

Use the individual `extract`, `scaffold`, `check`, and `build` subcommands when
debugging a specific docs stage. `check` runs a MkDocs build by default.
