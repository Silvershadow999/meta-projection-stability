# Scenario Catalog (v1)

This catalog is the human-readable index of scenario manifests in `scenarios/`.

How to run evaluation (local):
- `./scripts/run_eval_clean.sh` (runs baseline + adversarial_min)
- `python scripts/validate_results.py --in artifacts/results.jsonl --require-scenarios baseline,adversarial_min`

How to run robustness (local):
- `python scripts/run_robustness.py`
- `python scripts/validate_robustness.py`
- `python scripts/validate_robustness_goldens.py`

---

## baseline

- **id:** baseline
- **name:** Baseline (deterministic)
- **seed:** 42
- **description:** Default config; used as reference for regressions.

### Expected boundaries
{
  "allowed_names": [],
  "max_count": 0,
  "min_count": 0
}

### Expected metric keys (METRIC event)
[
  "duration_s",
  "n_steps",
  "step_events",
  "boundary_events"
]

### Robustness expectations
{
  "max_axiom_lock_rate": 0.001,
  "max_biometric_consensus_drop_p95": 0.1,
  "max_decision_flip_rate": 0.01,
  "max_h_sig_delta_p95": 0.12,
  "max_risk_delta_p95": 0.1,
  "max_status_flip_rate": 0.02
}

---

## adversarial_min

- **id:** adversarial_min
- **name:** Adversarial Minimal
- **seed:** 1337
- **description:** For boundary handling + telemetry robustness under perturbations.

### Expected boundaries
{
  "allowed_names": [
    "REVIEW"
  ],
  "max_count": 10,
  "min_count": 1
}

### Expected metric keys (METRIC event)
[
  "duration_s",
  "n_steps",
  "step_events",
  "boundary_events"
]

### Robustness expectations
{
  "max_axiom_lock_rate": 0.001,
  "max_biometric_consensus_drop_p95": 0.1,
  "max_decision_flip_rate": 0.01,
  "max_h_sig_delta_p95": 0.12,
  "max_risk_delta_p95": 0.1,
  "max_status_flip_rate": 0.02
}

