---
description: SEC Cython Optimization Pipeline
---

# Workflow: SEC Cython Optimization Pipeline (`/optimize-cython`)

## Objective
Execute a strictly verified, mathematically isomorphic optimization of the SEC parsing pipeline. The agent will transition through auditing, fuzzing, profiling, mutating, and validating states, enforcing the Plan-Act-Reflect protocol at critical mutations.

## Phase 1: Static Parity Audit
- **Action:** Perform a strict static Control Flow Graph (CFG) and AST comparison between `src/thesis_pkg/core/sec/extraction.py` and `src/thesis_pkg/core/sec/extraction_fast.pyx`.
- **Constraint:** Mathematically map all conditional branches returning `None` or empty lists to guarantee the Cython implementation replicates this graceful degradation. Verify that zero implicit C-level exceptions (e.g., `IndexError` via bypassed bounds checking) exist.
- **Output:** Generate `ast_parity_audit.md`. Do not halt execution; proceed immediately to Phase 2.

## Phase 2: Property-Based Test Synthesis
- **Action:** Scaffold a formal test harness in `tests/test_extraction_fuzz.py` utilizing the `hypothesis` library.
- **Constraint:** The test suite must concurrently route identical input permutations through `_scan_item_boundaries_py` and `scan_item_boundaries_fast`, asserting absolute equality on the returned lists of `_ItemBoundary` dataclasses. Synthesize string distributions mimicking flattened SEC EDGAR artifacts (e.g., tabular whitespace voids, hard-wrapped prose, orphaned pagination).
- **Output:** Save the generated test script. Proceed to Phase 3.

## Phase 3: Diagnostic Profiling & Strategy
- **Action:** Execute the generated `hypothesis` suite and run `line_profiler` on `extraction_fast.pyx`. Isolate Python C-API calls, GIL acquisitions, and `PyObject*` heap allocations.
- **Optimization Heuristic:** Formulate refactoring strategies to bypass the Python runtime for strings containing zero extraction value (e.g., evaluating a Zero-Allocation Heuristic Pre-Filter using pure C `const char*` array scanning).
- **Output:** Generate `optimization_proposal.md` detailing the profiling baseline and Big-O complexities of the proposed mutations.
- **🛑 REVIEW GATE 1 (INTERACTIVE PAUSE):** Halt all execution. Suspend the workflow and open an interactive terminal loop. Await human interrogation of the proposed algorithms. Do not proceed to Phase 4 until explicit authorization is granted.

## Phase 4: C-Level Mutation
- **Action:** Upon authorization, refactor `extraction_fast.pyx` applying the approved heuristic. Aggressively type local variables (`cdef int`, `cdef Py_ssize_t`) and strictly utilize continuous memory views (`cdef long long[:] starts_mv`).
- **Constraint:** Mathematical parity of the CFG must be preserved.
- **Output:** Output a unified `.patch` diff of the modifications. 
- **🛑 REVIEW GATE 2 (ACT MODE PAUSE):** Halt execution. Await explicit human approval to apply the `.patch` to the filesystem.

## Phase 5: Differential Validation & Automatic Rollback
- **Action:** Compile the extension (`python setup.py build_ext --inplace`).
- **Validation Execution:** 1. Rerun the `hypothesis` fuzzing suite.
  2. Execute `run_boundary_comparison` within `src/thesis_pkg/core/sec/suspicious_boundary_diagnostics.py`, asserting that the `delta` dictionaries for `status_counts` and `flag_counts` are mathematically zero.
- **Rollback Protocol:** If any assertion fails, immediately execute `git restore src/thesis_pkg/core/sec/extraction_fast.pyx src/thesis_pkg/core/sec/extraction_fast.c` to purge the state. Output the failure trace.
- **Success Protocol:** If successful, generate `optimization_delta_report.md` quantifying the computational speedup ratio.