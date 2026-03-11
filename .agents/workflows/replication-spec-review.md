---
description: Checkpointed paper-by-paper audit of the LM2011 replication spec and authority map using a scaffolded low-context workflow.
---

# LM2011 Replication Paper Audit

Audit the LM2011 replication documentation paper by paper with strict context control and resumable outputs.

## Inputs
Primary:
- `specs_drafts/lm2011_replication_spec.yaml`
- `docs/replication/lm2011_step1_authority_map.md`

Optional:
- `docs/replication/lm2011_repo_grounding_map.md`
- `docs/replication/lm2011_step1_evidence_packet_manifest.md`
- `docs/replication/lm2011_step1_evidence_packet.pdf`

Source dir:
- `core_paper/Loughran_McDonald_2011`

## Principles
- LM2011 main paper and Internet Appendix are top-level authorities.
- Other papers are delegated/supporting only where LM2011 relies on them.
- Audit one paper at a time.
- Save progress after each paper.
- Do not rewrite the YAML spec unless explicitly asked.
- Prefer exact local evidence over assumptions.
- Use screenshot capture only when text extraction is weak or tables/equations need visual confirmation.

## Outputs
Create or update:
- `docs/replication/paper_audits/00_source_registry.md`
- `docs/replication/paper_audits/01_master_scaffold.md`
- one audit file per paper in `docs/replication/paper_audits/`
- `docs/replication/paper_audits/90_cross_paper_gap_analysis.md`
- `docs/replication/paper_audits/95_spec_revision_backlog.md`

## Phase A — Source registry
Read the spec, authority map, optional supporting docs if present, and local source folder.

Build `00_source_registry.md` with:
- `source_id`
- `file_name`
- `normalized_citation`
- `document_type`
- `current_authority_status`
- `claimed_roles_in_current_docs`
- `currently_supported_steps`
- `priority_for_audit`
- `is_local_file_present`
- `notes`

Use only these authority classes:
- `main_replication_authority`
- `delegated_implementation_authority`
- `benchmark_or_estimation_authority`
- `label_or_data_provenance_authority`
- `background_only`
- `uncertain`

Default audit order:
1. LM2011 main paper
2. LM2011 Internet Appendix
3. Fama and French (2001)
4. Griffin (2003)
5. Fama and French (1997)
6. Fama and MacBeth (1973)
7. Newey and West (1987)
8. Carhart (1997)
9. Doyle, Ge, and McVay (2007)
10. remaining materially relevant local sources

## Phase B — Master scaffold
Create `01_master_scaffold.md`.

For each materially relevant source include:
- `source_id`
- `normalized_citation`
- `file_name`
- `current_authority_status`
- `claimed_by_current_docs_for_steps`
- `provisional_review_priority`

Checklist:
- [ ] Correctness of imported claims
- [ ] Completeness for supported steps
- [ ] Caveats / exclusions / footnotes
- [ ] Authority overextension
- [ ] Underuse
- [ ] Formula / timing / parameter details
- [ ] Data / label provenance
- [ ] Misleading or unnecessary information
- [ ] Cross-paper dependencies
- [ ] Recommended authority status

Fields to fill later:
- `claimed_current_role`
- `currently_imported_information`
- `is_that_information_correct`
- `important_information_missing_from_this_paper`
- `information_currently_overstated_or_misleading`
- `is_the_paper_used_for_too_many_steps`
- `is_the_paper_used_for_too_few_steps`
- `exact_locators_to_verify`
- `actually_supported_steps_after_review`
- `recommended_authority_status_after_review`
- `recommended_changes_to_spec_or_map`
- `cross_paper_dependencies`
- `open_questions`

## Phase C — Per-paper audit
Audit one paper at a time using only:
1. the paper,
2. relevant YAML blocks,
3. relevant authority-map excerpts,
4. that paper’s scaffold section,
5. minimal cross-paper context only if needed.

Create one audit file per paper.

Each audit file must contain:
1. paper summary
2. current claims imported from docs
3. correctness check (`correct`, `partly correct`, `unsupported`, `unclear`)
4. completeness check
5. overextension / underuse check
6. misleading or unnecessary information check
7. cross-paper dependency check
8. authority status recommendation
9. recommended changes and open questions

Allowed final statuses:
- `retain_as_primary`
- `retain_as_delegated`
- `downgrade_to_supporting`
- `downgrade_to_background`
- `upgrade_if_missing_in_current_docs`
- `uncertain_needs_followup`

After each paper:
- save the audit file,
- update its section in `01_master_scaffold.md`,
- update `00_source_registry.md` if authority status changed,
- resume safely on reruns.

## Phase D — Cross-paper reducer
After all material papers are audited, create `90_cross_paper_gap_analysis.md`.

It must identify:
- contradictions across papers or current docs,
- materially missing information,
- overused or underused papers,
- missing-paper recommendations,
- misleading current information,
- unresolved authority questions,
- implementation decisions still not pinned down.

## Phase E — Spec revision backlog
Create `95_spec_revision_backlog.md`.

For each backlog item include:
- `priority`
- `type` (`correctness`, `missing`, `misleading`, `authority_reassignment`, `new_source_needed`, `followup_decision`)
- `affected_steps`
- `affected_docs`
- `short_description`
- `source_basis`
- `suggested_action`

This is a backlog only, not a spec rewrite.

## Decision rules
`correct_and_sufficient` only if the information is supported by the right paper, assigned to the right authority, and adequate for the relevant step.

`missing` if the paper materially supports a step but relevant definitions, caveats, exclusions, delegated sources, equations, timing rules, or provenance notes are absent.

`misleading` if the information is attributed to the wrong paper, stated too strongly, based on a non-equivalent later substitute, or likely to steer implementation incorrectly.

`overextended` if a paper is used for more steps or a stronger authority role than the evidence supports.

## Special focus
Audit LM2011 main paper and Internet Appendix first, especially for:
- filing universe
- sample construction
- 10-K / 10-K405
- amended filing exclusions
- parsing and preprocessing
- MD&A scope
- dictionary construction
- term weighting
- variable definitions
- external dataset descriptions
- econometric references

For delegated sources, verify rather than assume roles:
- FF2001 → accounting definitions / timing
- FF1997 → industry definitions
- Griffin → event timing rationale
- Fama-MacBeth / Newey-West → estimation
- Carhart / FF1993 → factor benchmarks
- Doyle et al. → label provenance
- data-artifact docs → provenance only unless stronger evidence exists

## Completion condition
Complete only when:
- `00_source_registry.md` exists
- `01_master_scaffold.md` exists
- all material sources have audit files
- `90_cross_paper_gap_analysis.md` exists
- `95_spec_revision_backlog.md` exists

If anything is missing, continue from the latest checkpoint.