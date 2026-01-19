# Agent Instructions



\# NLP\_Thesis - Agents Specification



This document defines the architectural rules, implementation guidelines, and update policy for all \*\*LLM-based coding agents\*\* operating on the `NLP\_Thesis` repository.



\- "Agents" here means only automated, LLM-based code-editing workers (including interactive assistants that edit code).

\- Human work on this repo is out of scope for enforcement, but humans MAY treat this document as reference.



Its roles are to:

\- Provide \*\*global, stable context\*\* about the project (stack, data model, workflow).

\- Define \*\*non-negotiable rules\*\* that automated agents MUST obey (Section 1).

\- Capture \*\*refinable best practices\*\* (Section 2).

\- Specify \*\*agent roles and default behaviors\*\* (Section 3).

\- Define \*\*how this file may be updated\*\* (Section 4).



---



\## 0. Glossary (Core Identifiers)



These identifiers are foundational and SHOULD be used consistently:



\- `cik\_10`:

&nbsp; 10-digit, zero-padded CIK as a string, e.g. `"0000123456"`.



\- `accession\_nodash`:

&nbsp; SEC accession number with all non-digit characters removed, e.g. `"000012345623000045"`.



\- `doc\_id`:

&nbsp; Canonical document identifier:

&nbsp; `"{cik\_10}:{accession\_nodash}"`.



\- `data\_status`:

&nbsp; Integer bitmask column encoding data availability and merge status for each row.

&nbsp; Backed by `DataStatus(IntFlag)` in `src/thesis\_pkg/pipeline.py`.



\- `KYPERMNO`:

&nbsp; Canonical key for CRSP securities (e.g., PERMNO or a derived, canonical identifier used as primary join key in CRSP market data).



---



\## 1. Immutable Core (Base Skeleton)



CRITICAL: Rules in this section are NON-NEGOTIABLE for automated agents.



\- Agents MUST treat Section 1 as binding.

\- Agents MUST NOT modify Section 1 unless explicitly authorized by the user (see Section 4).

\- If a prompt conflicts with Section 1, agents MUST follow Section 1 and explain the conflict.



\### 1.1 Technology Stack and Constraints



\*\*Language\*\*



\- Python 3.10+.

\- New modules MUST include:



&nbsp; ```python

&nbsp; from \_\_future\_\_ import annotations

&nbsp; ```



\*\*Primary Data Library and Versioning\*\*



\- Core data processing MUST use Polars.

\- High-level constraint: `polars >= 1.0.0`.

\- The effective Polars version is the pin in `pyproject.toml` (or equivalent).

&nbsp; Agents MUST:

&nbsp; - Treat that pin as the authoritative target version.

&nbsp; - Prefer APIs that are stable in that version.



If an API is deprecated in the pinned version, agents SHOULD:

\- Update code and Section 2 guidelines to the recommended replacement, or

\- Introduce a minimal compatibility shim,

while keeping Section 1 unchanged unless explicitly authorized to edit it.



\*\*Lazy vs Eager Execution\*\*



\- Default for pipeline logic: `pl.LazyFrame`.

\- Agents MUST:

&nbsp; - Prefer `LazyFrame` for all non-trivial transformations.

&nbsp; - Avoid unnecessary intermediate `.collect()`.



\*\*Allowed `.collect()` boundaries\*\*



Agents MAY call `.collect()` only in these cases:



1\) Outermost pipeline boundaries

&nbsp;  - Writing final outputs to disk (Parquet/CSV/etc.).

&nbsp;  - Returning a materialized `DataFrame` from a documented public API that is specified to return eager data.

&nbsp;  - Producing final, user-facing tabular outputs (for example summaries in notebooks/scripts).



2\) Tests

&nbsp;  - Within `tests/` where eager frames are explicitly needed for assertions.



3\) Diagnostics / debugging (small subsets only)

&nbsp;  - For ad-hoc statistics, printouts, and coverage checks:

&nbsp;    - Collect small slices or samples (SHOULD be limited to about 5,000 rows or fewer and mindful of column widths).

&nbsp;    - These `.collect()` calls MUST NOT remain in core pipeline code in their debug form; they MUST either be removed or converted into well-documented diagnostics that are clearly bounded and justified.



\*\*Pandas Usage\*\*



\- Core pipeline MUST NOT rely on pandas.

\- Pandas MAY be used only in thin interface layers, for example:

&nbsp; - Plotting / visualization,

&nbsp; - Export to Excel or other tools requiring pandas,

&nbsp; - Very light-weight interactive inspection.



Constraints:



\- Before converting Polars to pandas, agents MUST:

&nbsp; - Ensure or enforce that row count is <= 50,000, and

&nbsp; - If possible, ensure approximate in-memory size is <= ~500 MB.

\- If approximate size cannot be reasonably estimated, agents:

&nbsp; - SHOULD avoid the conversion, but

&nbsp; - MAY proceed if the task appears impossible to complete without conversion and the prompt strongly implies that conversion is required.

&nbsp; - In such cases, agents MUST:

&nbsp;   - Explicitly state that a potentially heavy conversion was performed,

&nbsp;   - Describe why it was necessary.



\*\*Testing Framework\*\*



\- Testing MUST use `pytest`.

\- New or modified core logic MUST be accompanied by appropriate tests (see Section 2.4).



---



\### 1.2 Core Workflow and Goals



The project implements a structured data pipeline with the following conceptual stages:



1\) Market Data Merge

&nbsp;  - Merge CRSP market data and Compustat fundamentals into a financial base table keyed primarily by `KYPERMNO` and other canonical identifiers.

&nbsp;  - Ensure temporal alignment and ID consistency.



2\) Filing Text Processing

&nbsp;  - Load, clean, and normalize raw SEC filing text data.

&nbsp;  - Construct identifiers (`cik\_10`, `accession\_nodash`, `doc\_id`) that allow robust linking to the financial base table.



3\) Master Merge and Flagging

&nbsp;  - Merge financial base table and filing-level text data.

&nbsp;  - Populate the `data\_status` bitmask to encode:

&nbsp;    - Data availability,

&nbsp;    - Merge successes/failures,

&nbsp;    - Provenance and quality of key fields.



4\) Analysis

&nbsp;  - Run NLP and econometric analyses on the final merged dataset:

&nbsp;    - Text similarity/sentiment/topic measures.

&nbsp;    - Asset-pricing / event-study regressions.



Agents MUST preserve this high-level workflow.

Refactors MUST NOT change the conceptual order or meaning of these stages without explicit user instruction.



---



\### 1.3 Architectural Invariant: `DataStatus` Bitmask



We track data provenance, availability, and merge results via a single integer bitmask:



\- Enum: `DataStatus(IntFlag)` in `src/thesis\_pkg/pipeline.py`.

\- Column: `data\_status` in relevant tables/frames.



\*\*Type and Null Policy\*\*



\- `data\_status` MUST be an integer type (canonical: `pl.UInt64` via `STATUS\_DTYPE` in `pipeline.py`).

\- Default value MUST be `0` (no flags set).

\- `data\_status` MUST NOT contain nulls; use `0` instead of null to represent "no flags / no information".



\*\*Usage Rules\*\*



\- Agents MUST NOT introduce separate boolean availability columns (for example `has\_ret`, `has\_prc`) that duplicate `DataStatus`.

\- Flags MUST only be represented as bits in `data\_status`.

\- Flags are assumed to be independent bits; flag values MUST NOT overlap.



\*\*Helpers\*\*



\- Use `\_ensure\_data\_status(lf)` (in `src/thesis\_pkg/pipeline.py`) before performing any bitwise operations on `data\_status` for a `LazyFrame`. This ensures:

&nbsp; - The column exists,

&nbsp; - Has the correct dtype,

&nbsp; - Has a defined default value.

\- Use `\_flag\_if(expr, flag)` to set or update `data\_status` based on boolean expressions.



\*\*Canonical Example\*\*



\- Reference implementation and pattern:

&nbsp; - `src/thesis\_pkg/pipeline.py`, especially:

&nbsp;   - `DataStatus` enum definition,

&nbsp;   - `\_ensure\_data\_status(...)`,

&nbsp;   - Call sites where `\_flag\_if(...)` sets flags after joins or quality checks.



\*\*Bitwise Semantics\*\*



\- Setting a flag (Python):



&nbsp; ```python

&nbsp; row\_status = row\_status | DataStatus.HAS\_RET

&nbsp; ```



\- Testing a flag (Python):



&nbsp; ```python

&nbsp; if row\_status \& DataStatus.HAS\_RET:

&nbsp;     ...

&nbsp; ```



\- Testing a flag (Polars expression):



&nbsp; ```python

&nbsp; (pl.col("data\_status") \& pl.lit(DataStatus.HAS\_RET)).neq(0)

&nbsp; ```



Agents MUST preserve the semantics and integrity of `DataStatus`.



---



\### 1.4 Handling Violations of Core Rules



If an agent detects that the existing code violates a Section 1 rule (for example heavy pandas use in core pipeline, `data\_status` missing or nullable):



\- The agent MUST:

&nbsp; - Treat Section 1 as binding.

&nbsp; - NOT extend or deepen the violating pattern.

\- When possible within the current task scope:

&nbsp; - Prefer to align new/changed code to Section 1 (for example use Polars LazyFrame, fix `data\_status` handling).

\- If fixing the violation is out of scope for the current task:

&nbsp; - Leave the violating code unchanged,

&nbsp; - Explicitly describe the violation and recommend a follow-up refactor in the final output.



If a user prompt explicitly instructs behavior that conflicts with Section 1 (for example "rewrite this core pipeline to use pandas only"):



\- The agent MUST refuse the conflicting part and explain that Section 1 rules are non-negotiable unless explicitly authorized to modify Section 1.



---



\## 2. Implementation Guidelines (Mutable)



Rules in this section are best practices.

Agents SHOULD follow them but MAY propose updates if they find better patterns, as long as they remain consistent with Section 1.



\### 2.1 SEC Text Processing (`filing\_text.py`)



\*\*Input\*\*



\- Raw SEC filings provided as monthly or yearly ZIP archives containing plain text files.

\- Each file corresponds to a single filing.



\*\*Identifier Construction\*\*



\- Compute `cik\_10`, `accession\_nodash`, and `doc\_id` using canonical helpers.

\- `doc\_id = f"{cik\_10}:{accession\_nodash}"`.



Agents SHOULD reuse existing helpers for CIK/accession parsing and `doc\_id` construction instead of re-implementing logic.



\*\*Processing Pattern\*\*



\- Prefer streaming:

&nbsp; - Iterate over ZIP contents, process files one-by-one or in bounded batches.

&nbsp; - Avoid extracting full archives to disk unless strictly necessary for performance or tooling.

\- Normalize text:

&nbsp; - Enforce UTF-8 decoding.

&nbsp; - Apply standardized cleaning and normalization steps as implemented in existing helpers.



\### 2.2 Merging Logic (CRSP / Compustat / SEC)



\*\*RAM Safety\*\*



\- Use `pl.LazyFrame` for merges and heavy transformations.

\- Avoid operations that produce very wide intermediates or near-cross-joins.

\- Chunked processing or `join\_asof` SHOULD be used where data volume is large.



\*\*Join Discipline\*\*



\- Time-series joins SHOULD use `join\_asof` where appropriate:

&nbsp; - Ensure sort order and consistent dtypes on join keys.

&nbsp; - Explicitly specify strategy ("backward", "forward", "nearest") as required.

\- Identifier hierarchy (for example `KYPERMNO` / PERMNO / PERMCO / GVKEY) MUST be explicit in join logic and helpers.



\*\*Schema Alignment\*\*



Before vertical concatenation:



\- Use `.collect\_schema()` to determine reference schema.

\- Use a helper such as `\_align\_to\_schema(df, schema)` (see `merge\_histories` in `src/thesis\_pkg/pipeline.py`) to:

&nbsp; - Ensure identical column names and order across frames,

&nbsp; - Ensure compatible dtypes (no implicit downcasting; explicit `cast` if types change),

&nbsp; - Ensure consistent nullability:

&nbsp;   - If a reference column is non-nullable, missing values MUST be filled with an appropriate default before concat.



Agents SHOULD extend existing merge helpers instead of scattering ad-hoc joins across the codebase.



\### 2.3 Coding Standards



\*\*Type Hints\*\*



\- All public functions MUST have fully typed parameters and return types.

\- Internal helpers SHOULD also be typed to keep `mypy` output clean.



\*\*Dates and Times\*\*



\- In Python:

&nbsp; - Use `datetime.date` for date-only values.

&nbsp; - Use `datetime.datetime` for timestamps (for example filing timestamps).

\- In Polars:

&nbsp; - Use proper `Date` / `Datetime` dtypes; avoid storing dates as raw integers.

\- Use `\_coerce\_date` or the designated helpers to parse/filter dates safely.



\*\*Path Handling\*\*



\- Use `pathlib.Path` for all local filesystem interactions.

\- Core pipeline assumes local POSIX/NT-like filesystems (including, for example, mounted drives in Colab).

\- Agents MUST NOT treat remote URIs (S3, GCS, HTTP) as `Path` objects in core pipeline logic.

&nbsp; - Any future remote-storage support MUST be implemented via dedicated IO helpers/modules and clearly separated.



\*\*Imports and Style\*\*



\- Import order:

&nbsp; 1. Standard library,

&nbsp; 2. Third-party (Polars, PyArrow, etc.),

&nbsp; 3. Local modules (`thesis\_pkg.\*`).

\- Avoid wildcard imports.

\- Keep functions cohesive and reasonably small.

\- Prefer explicitness over cleverness in core pipeline logic.

\- Comment or log non-obvious decisions, especially filters, join keys, and flag semantics.



\### 2.4 Testing Guidelines



\*\*Test Types\*\*



\- Unit tests:

&nbsp; - Target small, cohesive functions.

&nbsp; - Use minimal synthetic data (dozens to low hundreds of rows).



\- Integration / pipeline tests:

&nbsp; - Cover join behavior, flagging logic (`DataStatus`), and end-to-end consistency.

&nbsp; - Datasets MAY be larger but SHOULD stay at thousands of rows, not tens of thousands, by default.



Larger regression datasets (tens of thousands of rows or more), if needed, SHOULD:

\- Live in dedicated performance/regression tests, and

\- NOT be run by default in the main test suite.



\*\*Filesystem / I/O\*\*



\- Use `tmp\_path` for any test that touches the filesystem.

\- Tests MUST NOT write into the project's source or data directories.



\*\*Polars DataFrame Testing\*\*



\- When asserting equality:

&nbsp; - Prefer `df1.frame\_equal(df2)` or a similar robust comparison, or

&nbsp; - Compare key columns explicitly, sorted by stable keys.

\- Avoid brittle assumptions about column order unless order is semantically important.



\*\*DataStatus Sanity Checks\*\*



\- Whenever a new `DataStatus` flag is introduced or semantics change:

&nbsp; - Add tests that assert:

&nbsp;   - The flag is set when the condition holds.

&nbsp;   - The flag is not set when the condition does not hold.

\- Use bitwise checks (`\&`) in tests, in Python or Polars expressions, consistent with Section 1.3.



---



\## 3. Agents and Roles



This section describes typical roles. An implementation MAY combine roles in one agent, but MUST still follow the constraints.



\*\*Scope\*\*



\- These roles and rules apply only to LLM-based coding/analysis agents acting on this repository.



\### 3.1 Roles



\- Planner / Reviewer Agent

&nbsp; - Analyzes, inspects, and explains code, data flows, and architecture.

&nbsp; - Decomposes high-level requests into concrete tasks.

&nbsp; - Produces plans, diagnostics, and recommendations.

&nbsp; - DOES NOT modify code in this role.



\- Code Writer Agent

&nbsp; - Implements, modifies, or refactors code under `src/thesis\_pkg/`.

&nbsp; - May perform lightweight planning needed to complete its task.

&nbsp; - MUST obey Section 1 and SHOULD follow Section 2.



\- Test Runner / QA Agent

&nbsp; - Runs `pytest` or subsets of the test suite.

&nbsp; - Summarizes failures, coverage gaps, and potential new tests.

&nbsp; - Does not modify code directly.



\- Refactor / Cleanup Agent

&nbsp; - Performs structural improvements (naming, decomposition, minor performance tweaks) while preserving behavior.

&nbsp; - Must not change public interfaces without coordination/justification.



A single LLM instance may act as multiple roles in sequence (for example first Planner, then Code Writer), but MUST respect the behavioral constraints of each role at each step.



\### 3.2 Default Behavior Based on Prompts



If the prompt:



\- Contains only analytic verbs such as:

&nbsp; - "analyze", "inspect", "explain", "review", "diagnose"

&nbsp; - and does NOT explicitly ask to "modify", "refactor", "implement", "fix", "add", "update", "change"

&nbsp; 

&nbsp; -> The agent MUST treat this as a read-only request and act as a Planner/Reviewer:

&nbsp; - It MUST NOT modify code.

&nbsp; - It MAY propose concrete edits, refactors, or new functions in its response.

&nbsp; - If the prompt is ambiguous, the agent SHOULD default to read-only behavior and may ask the user whether it should apply changes after presenting its analysis.



\- Contains explicit modification verbs, such as:

&nbsp; - "implement", "refactor", "rewrite", "fix", "add", "update", "change", or phrases like:

&nbsp;   - "refactor it for inclusion in `pipeline.py`"

&nbsp;   - "apply your suggested changes"

&nbsp; 

&nbsp; -> The agent SHOULD behave as a Code Writer Agent (with lightweight planning as needed) and is allowed to modify code, subject to:

&nbsp; - All Section 1 constraints,

&nbsp; - Relevant Section 2 guidelines.



If a prompt both asks to analyze and to refactor (for example "Explain this code section and refactor it for inclusion in `pipeline.py`"):



\- The agent SHOULD:

&nbsp; - First conceptually act as Planner (explain, identify issues),

&nbsp; - Then act as Code Writer (propose and/or apply refactor), in one or more steps as appropriate.



If role is not explicitly named in the prompt but code changes are requested:



\- The agent MUST treat itself as a Code Writer Agent and follow all associated constraints.



If a user explicitly writes that code MUST NOT be modified (for example "do NOT modify the code, only analyze"), this instruction overrides any other cues, and the agent MUST remain in read-only Planner/Reviewer mode.



---



\## 4. Meta-Instructions: Updating `agents.md`



`agents.md` is a living document. Automated agents MAY update it, but with strict rules:



\### 4.1 Protection of Section 1



\- Section 1 ("Immutable Core") is immutable by default.

\- Automated agents MUST NOT modify Section 1 unless:

&nbsp; - The prompt explicitly references Section 1 and

&nbsp; - Explicitly authorizes a change with an action/permission verb such as:

&nbsp;   - "update", "modify", "relax", "change", "you are allowed to...", "you have permission to...", etc.

\- If it is unclear whether the user has granted permission to modify Section 1, the agent MUST:

&nbsp; - Ask the user for clarification before changing Section 1, or

&nbsp; - In fully automated, non-interactive contexts, assume no permission and NOT modify Section 1.



When proposing a Section 1 change (without actually editing it):



\- The agent MUST include in its output:

&nbsp; - A short description of the proposed change,

&nbsp; - The rationale (for example performance, correctness, upstream changes),

&nbsp; - Expected impact on existing code (including migration steps),

&nbsp; - A suggestion to record the change (for example as a GitHub issue or PR description).



When an agent actually edits Section 1 (with explicit permission):



\- It MUST state in its final response, with a short description:

&nbsp; - "I have updated agents.md Section 1 to reflect: \[short description]."



\### 4.2 Updating Sections 2 and 3



\- Sections 2 and 3 MAY be updated by agents without special permission IF:

&nbsp; - The changes reflect actual code evolution (for example new helpers, new modules),

&nbsp; - The changes remain consistent with all Section 1 rules.

\- Typical updates include:

&nbsp; - Adding/removing modules from examples,

&nbsp; - Refining recommended patterns (for example new Polars APIs),

&nbsp; - Clarifying roles and typical workflows.



When updating these sections:



\- Agents SHOULD briefly mention in their final output:

&nbsp; - "I have updated agents.md to reflect new guidelines for \[topic]."



\### 4.3 Consistency Obligation



If `agents.md` and the actual code diverge:



\- For execution and changes, agents MUST prioritize the actual code behavior and Section 1 constraints.

\- They SHOULD propose updates to `agents.md` to restore consistency.

\- They SHOULD not silently treat conflicting guidelines in Section 2 as binding if the codebase has clearly moved on; instead, they:

&nbsp; - Follow Section 1,

&nbsp; - Align with current code,

&nbsp; - Suggest a documentation update.



---



