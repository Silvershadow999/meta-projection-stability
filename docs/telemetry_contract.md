# Telemetry & Artifact Contract (v1)

This document describes the stable, machine-readable contracts produced by this repository.

References:
- src/meta_projection_stability/types.py
- src/meta_projection_stability/state.py
- scripts/validate_results.py
- scripts/validate_robustness.py
- scripts/validate_robustness_goldens.py
- docs/versioning_policy.md

---

## 1) Telemetry stream: artifacts/results.jsonl

### Format
- JSON Lines: one JSON object per line (append-only)
- Each line represents one TelemetryEvent (schema-versioned)

### Stable top-level keys (expected)
- schema_version (string)
- ts_utc_s (float)
- run_id (string)
- scenario_id (string)
- step (int|null)
- event_type (string)
- severity (string)
- message (string)
- metrics (object)
- tags (object)
- boundary (object|null)
- payload (object)

### Canonical event_type values
- run_start, run_end, step, metric, boundary, note

### Example lines (single-line JSON; do not execute in shell)
RUN_START:
{"schema_version":"1.0.0","ts_utc_s":1710000000.0,"run_id":"run_abc123","scenario_id":"baseline","step":0,"event_type":"run_start","severity":"info","message":"run_start","metrics":{},"tags":{},"boundary":null,"payload":{"provenance":{"git_commit":"...","git_dirty":false},"scenario":{"scenario_id":"baseline"},"state":{"run_id":"run_abc123"}}}

BOUNDARY:
{"schema_version":"1.0.0","ts_utc_s":1710000001.0,"run_id":"run_abc123","scenario_id":"adversarial_min","step":0,"event_type":"boundary","severity":"warning","message":"boundary_triggered","metrics":{},"tags":{},"boundary":{"name":"REVIEW","triggered":true,"severity":"warning","details":{"reason":"..."}}, "payload":{"state":{"step":0}}}

### Invariants (validated by scripts/validate_results.py)
Per (run_id, scenario_id):
- exactly 1 run_start
- exactly 1 run_end
- at least 1 metric event
- run_start.payload.provenance exists (object)
- timestamps sane
- if metric.n_steps > 0 then at least one step event exists

---

## 2) Evaluation report: artifacts/eval_report.md
Generated from results.jsonl by scripts/eval_report.py.

---

## 3) Robustness report: artifacts/robustness_report.json
- schema: mps.robustness.v1
- scenarios -> perturbations -> metrics
Validated by scripts/validate_robustness.py (against robustness_expectations in scenario manifests).

## 4) Robustness golden: artifacts/robustness_golden.json
Regression anchor validated by scripts/validate_robustness_goldens.py.

## 5) Versioning summary
Telemetry and artifact schemas are treated as external contracts (additive-only by default).
See docs/versioning_policy.md.
