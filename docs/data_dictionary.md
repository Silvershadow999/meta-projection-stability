# Data Dictionary (v1)

This document defines the meaning and expected ranges of key fields and metrics.

References:
- docs/telemetry_contract.md
- src/meta_projection_stability/types.py
- scripts/validate_results.py
- scripts/validate_robustness.py
- scripts/validate_robustness_goldens.py

---

## A) results.jsonl (TelemetryEvent)

### Core identifiers
- run_id (string)
  - Meaning: stable identifier for a single run.
  - Expected: non-empty.
- scenario_id (string)
  - Meaning: scenario manifest identifier (maps to scenarios/<id>.json).
  - Expected: non-empty; must exist in manifests for gated runs.
- step (int|null)
  - Meaning: logical step index for STEP events and per-step boundary events.
  - Expected: null or >= 0.

### Event classification
- event_type (string)
  - Meaning: event class in the telemetry stream.
  - Expected: one of {run_start, run_end, step, metric, boundary, note}.
- severity (string)
  - Meaning: log severity for structured triage.
  - Expected: one of {debug, info, warning, error, critical}.
- message (string)
  - Meaning: short label for humans; not used as a machine contract.

### Timing
- ts_utc_s (float)
  - Meaning: unix timestamp (UTC seconds).
  - Expected: monotonically increasing per run (weak constraint).

### payload.provenance (object)
Common fields:
- git_commit (string|null): commit hash (best effort)
- git_dirty (bool|null): whether workspace had uncommitted changes (best effort)

Validated by: scripts/validate_results.py (presence and basic sanity).

---

## B) results.jsonl metrics (METRIC event)

These are runner-specific but expected keys include:
- duration_s (float)
  - Meaning: wall-clock runtime of a run.
  - Range: >= 0.
- n_steps (float/int)
  - Meaning: number of skeleton/model steps executed.
  - Range: >= 0.
- step_events (float/int)
  - Meaning: count of emitted STEP events.
  - Range: >= 0.
- boundary_events (float/int)
  - Meaning: count of emitted BOUNDARY events.
  - Range: >= 0.

Validated by: scripts/validate_results.py (presence of >=1 metric event; step event presence if n_steps>0).

---

## C) Boundary signals (BOUNDARY event)

boundary.name (string)
- Meaning: explicit safety boundary identifier.
- Examples: REVIEW, REFUSE, EMERGENCY_STOP.
- Expected: non-empty.

boundary.triggered (bool)
- Meaning: boundary fired (true) vs informational snapshot (false).
- Expected: usually true when event_type=boundary.

boundary.details (object)
- Meaning: structured explanation / evidence for triggering.

---

## D) artifacts/robustness_report.json (mps.robustness.v1)

Top-level:
- pass (bool): overall robustness pass across included scenarios
- thresholds (object): thresholds used by runner (informational; expectations are in manifests)
- scenarios (object): per-scenario results

Per-scenario:
- pass (bool): scenario-level pass
- seed (int): seed used for deterministic generation
- baseline_len (int): series length
- perturbations (list): list of perturbation cases

Per-perturbation metrics:
- decision_flip_rate (float)
  - Meaning: fraction of steps where decision differs vs baseline.
  - Range: [0, 1].
  - Lower is better.
- status_flip_rate (float)
  - Meaning: fraction of steps where status differs vs baseline.
  - Range: [0, 1].
- risk_delta_p95 (float)
  - Meaning: p95 absolute delta between risk series (baseline vs perturbed).
  - Range: >= 0 (typically small).
- human_significance_delta_p95 (float)
  - Meaning: p95 absolute delta of human_significance series.
  - Range: >= 0.
- biometric_consensus_drop_p95 (float)
  - Meaning: p95 drop in sensor consensus (baseline - perturbed, clamped at 0).
  - Range: >= 0.
- axiom_lock_rate (float)
  - Meaning: fraction of steps that enter axiom lock in the perturbation run.
  - Range: [0, 1].
  - For benign perturbations: expected to be ~0 (bounded by expectations).

Validated by: scripts/validate_robustness.py (against robustness_expectations in scenario manifests).

---

## E) artifacts/robustness_golden.json (mps.robustness.golden.v1)

Purpose:
- reduced summary for regression drift detection (not a “desired output”)

Per-scenario keys:
- pass (bool)
- n_perturbations (int)
- decision_flip_rate.max, .median (float)
- status_flip_rate.max, .median (float)
- risk_delta_p95.max, .median (float)
- human_significance_delta_p95.max, .median (float)
- biometric_consensus_drop_p95.max, .median (float)
- axiom_lock_rate.max, .median (float)

Validated by: scripts/validate_robustness_goldens.py (drift within tolerances).

---

## Notes
- Expectations live in scenario manifests (robustness_expectations).
- Goldens are allowed to change only with explicit justification and review.

