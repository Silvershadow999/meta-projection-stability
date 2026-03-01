# Metrics & Boundary Catalog (v1)

This catalog standardizes interpretation of decisions, statuses, and explicit safety boundaries.

References:
- src/meta_projection_stability/types.py (EventType, Severity, BoundarySignal, TelemetryEvent)
- src/meta_projection_stability/state.py (RunState boundary snapshots)
- docs/telemetry_contract.md
- docs/data_dictionary.md

---

## 1) Boundary signals (names + intent)

### REVIEW
- Intent: escalate to human review; system should avoid irreversible actions.
- Typical severity: warning
- Allowed when: inputs are suspicious, out-of-distribution, integrity signals degrade, or metrics exceed warning thresholds.
- Expected behavior: continue producing telemetry; prefer conservative decisions.

### REFUSE
- Intent: decline to proceed with the requested action; safe non-cooperation.
- Typical severity: error (or warning if soft refusal)
- Allowed when: policy conflict, unsafe request, strong tamper suspicion, or unacceptable risk profile.
- Expected behavior: output structured refusal rationale (details in boundary.details).

### EMERGENCY_STOP
- Intent: immediate hard stop of the run / action pipeline.
- Typical severity: critical
- Allowed when: catastrophic safety boundary is violated, or irreversible harm is detected (or simulated).
- Expected behavior: terminate run ASAP; ensure run_end and boundary event are emitted if possible.

### Notes on naming
- Boundary names must be stable strings (do not rename casually).
- Adding new boundary names requires documenting them here and updating any expectations/tests.

---

## 2) Decision taxonomy (adapter outputs)

These are model-level outputs (not necessarily telemetry event types).

### CONTINUE
- Meaning: stay in nominal operation.
- Expected telemetry: STEP events + METRIC summary.

### BLOCK_AND_REFLECT
- Meaning: pause / block action and request reflection/verification (soft guardrail).
- Often pairs with REVIEW boundary.

### EMERGENCY_RESET
- Meaning: reset to safe baseline state and enter cooldown (if implemented).
- Often pairs with EMERGENCY_STOP or critical boundary depending on the design.

### AXIOM_ZERO_LOCK (if present)
- Meaning: non-recoverable lock due to irreversible violation; capability collapses toward zero.
- Must be extremely rare; should be protected by tests and explicit review.

---

## 3) Status taxonomy (adapter statuses)

### nominal
- risk below recovery threshold; stable regime.

### transitioning
- in hysteresis band; caution; flip-risk should remain bounded under benign perturbations.

### cooldown
- post-reset stabilization window.

### critical
- beyond critical threshold; safety action required.

---

## 4) Severity mapping (recommended)

- debug: step-level internals (high volume)
- info: lifecycle events (run_start/run_end), metric summaries
- warning: REVIEW, soft boundary triggers
- error: REFUSE or failed invariant that does not imply catastrophe
- critical: EMERGENCY_STOP, catastrophic invariant violation

---

## 5) Review triggers (governance)

Human review is required when:
- any CI gate is modified (workflows/validators)
- golden files change
- robustness_expectations change
- boundary taxonomy changes (new boundary names or changed semantics)
- telemetry schema changes (non-additive or meaning changes)

---

## 6) Scenario mapping (recommended)

- baseline: should generally produce no boundary triggers under benign conditions.
- adversarial_min: may intentionally trigger REVIEW early (demo boundary), but must remain telemetry-valid.

