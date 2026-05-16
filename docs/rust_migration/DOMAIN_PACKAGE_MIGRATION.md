# Domain Package Migration Note

`src_rust_migration` now exposes repo-internal domain package boundaries beside
the compatibility `thesis_pkg` package:

- `thesis_core`: shared thesis infrastructure, including status-bitmask
  primitives.
- `thesis_sec`: SEC filing text and item-extraction domain boundary.
- `thesis_lm2011`: LM2011 replication and extension boundary.
- `thesis_refinitiv`: thesis-specific Refinitiv ownership, authority, bridge,
  analyst, and document-handoff boundary.
- `thesis_native`: neutral PyO3 extension boundary for `_lm2011_rust`.

Existing scripts and runners can continue to import from `thesis_pkg`.
The new packages are intentionally additive for the migration copy, so runner
behavior and the single Rust extension build remain stable while future imports
can move toward domain-owned namespaces.

## Native Extension Boundary

The Rust extension is built and imported as:

```text
thesis_native._lm2011_rust
```

Internal migration-copy fast paths import `_lm2011_rust` from
`thesis_native`. The historical `thesis_pkg.core.sec` package re-exports the
same module and aliases `thesis_pkg.core.sec._lm2011_rust` in `sys.modules` as
a compatibility shim, so legacy imports continue to receive the same native
module object and keep the existing fallback behavior.

## Refinitiv Bridge Boundary

`thesis_refinitiv.bridge` owns the Refinitiv bridge implementation in the
migration copy. Internal Refinitiv modules import bridge helpers and reducers
from that domain boundary. The historical
`thesis_pkg.pipelines.refinitiv_bridge_pipeline` module is a `sys.modules`
compatibility alias to `thesis_refinitiv.bridge`, preserving public/private
symbol access, monkeypatch identity, Rust metrics, and fallback behavior for
legacy callers.
The narrower legacy module `thesis_pkg.pipelines.refinitiv.bridge` is also a
`sys.modules` compatibility alias to the same implementation.

## LSEG Client Extraction Boundary

Reusable LSEG client code is grouped under:

```text
thesis_refinitiv.lseg_client
```

The extraction-ready subset is:

- `api_common`: generic error policy, retry, JSON log, and atomic output
  helpers.
- `batching`: request-item, batch-definition, hashing, grouping, chunking, and
  split helpers.
- `ledger`: request ledger state, retry/defer/fatal states, and resume
  compatibility checks.
- `provider`: provider/session errors, header sanitization, response coercion,
  and LSEG availability checks.
- `stage_audit`: stage/fetch manifest helpers.

These modules now own the reusable LSEG client implementation in the migration
copy. The historical
`thesis_pkg.pipelines.refinitiv.lseg_api_common`,
`lseg_batching`, `lseg_ledger`, `lseg_provider`, and `lseg_stage_audit`
modules are compatibility shims that re-export from
`thesis_refinitiv.lseg_client`. They alias the new module objects directly so
legacy tests and monkeypatch hooks still see private compatibility symbols while
the implementation has a single owner.

The reusable client boundary imports the shared Rust extension from
`thesis_native`, the neutral native package boundary. Legacy
`thesis_pkg.core.sec._lm2011_rust` imports remain supported only through the
compatibility shim described above.

The current `thesis_pkg.pipelines.refinitiv.lseg_api_execution` module is not
yet part of the extraction-ready client boundary because it still contains one
domain policy hook for mixed zero/positive ownership universe responses. Before
extracting a standalone LSEG API package, pass that policy in from the
thesis-specific Refinitiv ownership runner or convert it to a generic callback.

The standalone package candidate should not include thesis concepts such as
`KYPERMNO`, `doc_id`, SEC filing identifiers, LM2011 sample logic, ownership
authority decisions, analyst-panel construction, or document-level Refinitiv
handoff rules. Those stay in `thesis_refinitiv`, `thesis_sec`, and
`thesis_lm2011`.

## Deferred Domain Facades

The following domain boundaries are intentionally deferred facades in this
readiness snapshot:

- `thesis_refinitiv.authority`: re-exports
  `thesis_pkg.pipelines.refinitiv.authority`. Moving it safely requires a
  broader lazy-export pass because authority participates in legacy package
  initialization and downstream document-ownership assembly.
- `thesis_refinitiv.ownership`: re-exports
  `thesis_pkg.pipelines.refinitiv.ownership`. The handoff builders it exposes
  are owned by `thesis_refinitiv.bridge`, but the module still combines those
  bridge entrypoints with legacy ownership API runners.
- `thesis_core.status`: re-exports the legacy status-bitmask primitives without
  changing `DataStatus` identity.
- `thesis_sec`: remains a boundary placeholder while SEC extraction still uses
  Python/Cython-backed item-boundary code under `thesis_pkg.core.sec`.
- `thesis_lm2011`: remains a boundary placeholder until the LM2011 runners and
  extension pipeline can move as a unit without changing runner behavior.

## Native Binary Audit

The only Python-loadable `_lm2011_rust*.pyd` package artifact should live under
`thesis_native`. Cargo may still create `_lm2011_rust.dll` build-cache outputs
under `rust/lm2011_rust/target`, and `thesis_pkg.core.sec.extraction_fast` is a
separate Cython SEC extraction accelerator rather than a stale Rust extension
copy.
