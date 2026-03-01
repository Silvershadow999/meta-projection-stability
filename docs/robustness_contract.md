# Robustness Contract (v1)

Scope: benign perturbations applied to baseline scenarios must not cause unstable decision flips or unexpected locks.

## Perturbations (v1)
- Gaussian noise: sigma ∈ {0.01, 0.03, 0.05}
- Brightness shift: factor ∈ {0.8, 1.2}  (±20%)
- Sensor dropout: drop exactly 1 biometric channel (deterministic)

## Invariants (v1 thresholds)
- decision_flip_rate <= 0.01
- status_flip_rate <= 0.02
- risk_delta_p95 <= 0.10
- human_significance_delta_p95 <= 0.12
- axiom_lock_rate <= 0.001   (no unexpected locks under benign perturbations)
- biometric_consensus_drop_p95 <= 0.10  (initial placeholder; tune with data)

## Artifacts
- artifacts/robustness_report.json  (schema: mps.robustness.v1)
