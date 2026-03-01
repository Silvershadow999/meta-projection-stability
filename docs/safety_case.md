# Safety Case (v1)

This document argues that the repository provides **credible, machine-checkable safety evaluation scaffolding**
for meta-stability / boundary monitoring experiments.

It does **not** claim deployment safety of any autonomous system.

---

## Top-level claim C0
**C0:** The repo enforces a reproducible, audit-friendly evaluation pipeline with explicit safety boundaries and regression protection.

### Strategy S0
Show that:
1) foundational safety invariants are protected from silent drift,
2) telemetry and artifacts are structured and validated,
3) robustness expectations are explicit and CI-gated,
4) regressions are detected via goldens with defined tolerances.

---

## Claim C1 — Integrity of foundational invariants (Level-0)
**C1:** Foundational invariants (Level-0 axiom / baseline safety text) cannot drift silently without CI failure.

**Evidence:**
- `axiom-guard` workflow (Level-0 fingerprint / integrity check).
- Repo policy: changes to axiom/fingerprint require explicit PR review.

**Residual risk:**
- Admin bypass can override checks (governance risk, not technical).

---

## Claim C2 — Structured telemetry with explicit safety boundary semantics
**C2:** Telemetry outputs are schema-stable and boundaries are represented explicitly (not implicit logs).

**Evidence:**
- Telemetry contract types: `src/meta_projection_stability/types.py` (events, provenance, boundary signals).
- Run state: `src/meta_projection_stability/state.py` (serializable boundary snapshots).
- `artifacts/results.jsonl` invariants validated by `scripts/validate_results.py`.

**Residual risk:**
- Schema evolution can break downstream consumers if not versioned carefully.
- Mitigation: additive-only policy + schema_version fields.

---

## Claim C3 — Reproducibility / provenance
**C3:** Runs can be traced to a specific code version and scenario configuration.

**Evidence:**
- Scenario manifests in `scenarios/` with deterministic seeds and config overrides.
- Run provenance captured in telemetry (`git_commit`, `git_dirty`, environment hints).
- CI runs from clean checkout (no dirty workspace).

**Residual risk:**
- Git metadata may be unavailable in some environments; mitigate by failing closed in CI.

---

## Claim C4 — Robustness under benign perturbations is CI-gated
**C4:** The repo provides a CI-gated robustness contract that detects unstable decision flips and unexpected locks under benign perturbations.

**Evidence:**
- Runner: `scripts/run_robustness.py` generates `artifacts/robustness_report.json` (schema `mps.robustness.v1`).
- Expectations in scenario manifests: `robustness_expectations` fields.
- Validator: `scripts/validate_robustness.py` enforces expectations.
- Regression gate: `scripts/validate_robustness_goldens.py` checks drift within tolerance.
- Workflow: `robustness-gate` runs runner + validators and uploads artifacts.

**Residual risk:**
- Benign perturbation suite is not a full adversarial robustness suite.
- Mitigation: expand perturbations and add adversarial threat-driven scenarios over time.

---

## Claim C5 — Regression protection exists for safety-critical metrics
**C5:** Safety-relevant metrics do not regress unnoticed.

**Evidence:**
- Golden snapshot: `artifacts/robustness_golden.json`
- Drift validator: `scripts/validate_robustness_goldens.py` with explicit tolerances
- Required check in branch rulesets ensures merge is blocked on regression.

**Residual risk:**
- Goldens can be changed in PRs; mitigation is explicit review and justification in PR template / policy.

---

## Summary of Evidence Artifacts (CI outputs)
- `artifacts/results.jsonl`
- `artifacts/eval_report.md`
- `artifacts/robustness_report.json`
- `artifacts/robustness_golden.json`
- CI logs for `axiom-guard`, `eval-gate`, `robustness-gate`

---

## Known limitations (explicitly not claimed here)
- This repo does not prove real-world safety of autonomous deployment.
- This repo does not provide formal verification.
- Threat model coverage is limited to the evaluation pipeline (not full infra/security).

