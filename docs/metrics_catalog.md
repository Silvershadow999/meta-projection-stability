# Metrics & Boundary Catalog (v1)

This catalog standardizes interpretation of decisions, statuses, and explicit safety boundaries.

References:
- src/meta_projection_stability/types.py
- src/meta_projection_stability/state.py
- docs/telemetry_contract.md
- docs/data_dictionary.md

---

## 1) Boundary signals (names + intent)

### REVIEW
- Intent: escalate to human review; avoid irreversible actions.
- Typical severity: warning

### REFUSE
- Intent: decline to proceed; safe non-cooperation.
- Typical severity: error (or warning for soft refusal)

### EMERGENCY_STOP
- Intent: immediate hard stop.
- Typical severity: critical

Notes:
- Boundary names are stable strings; do not rename casually.
- Adding new boundary names requires documenting them here.

---

## 2) Decision taxonomy (adapter outputs)
- CONTINUE: nominal operation
- BLOCK_AND_REFLECT: pause/block and request verification (often pairs with REVIEW)
- EMERGENCY_RESET: reset to safe baseline (often pairs with critical boundary)
- AXIOM_ZERO_LOCK (if present): non-recoverable lock due to irreversible violation (rare; requires review)

---

## 3) Status taxonomy (adapter statuses)
- nominal
- transitioning
- cooldown
- critical

---

## 4) Severity mapping (recommended)
- debug: step internals
- info: lifecycle + metric summaries
- warning: REVIEW / soft boundaries
- error: REFUSE / non-catastrophic invariant failures
- critical: EMERGENCY_STOP / catastrophic violations

---

## 5) Review triggers (governance)
Review required when:
- CI gates or validators change
- goldens change
- robustness_expectations change
- boundary taxonomy changes
- telemetry schema changes beyond additive evolution

---

## 6) Scenario mapping (recommended)
- baseline: should not trigger boundaries under benign conditions
- adversarial_min: may intentionally trigger REVIEW early, but must remain telemetry-valid
