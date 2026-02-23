---
description: Workflow to run premerge pipeline and validate the rejected rate.
---

# Validate Premerge Pipeline

1. Execute the premerge pipeline command:
   ```bash
   run_sec_ccm_premerge_pipeline
   ```
2. Audit the resulting `sec_ccm_run_report.md` file.
3. Check the `daily_lag_gate_rejected_rate` value.
4. **Halt execution** if `daily_lag_gate_rejected_rate` > 0.05.
