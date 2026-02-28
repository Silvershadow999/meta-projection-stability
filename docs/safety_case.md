# Safety Case — meta-projection-stability (Phase 6)

Status: **Draft (Phase 6 / Step 16)**  
Safety intent: Make evaluation outputs **credible**, **reproducible**, and **boundary-explicit** under adversarial scenarios.

This safety case is about **evaluation infrastructure safety** (telemetry integrity, boundary enforcement, scenario comparability) — not a guarantee that any underlying scientific model is correct.

---

## 1) System Description

`meta-projection-stability` is an evaluation harness that:
- runs **scenarios** (baseline vs adversarial),
- produces **structured telemetry events**,
- emits **explicit safety boundary signals** (e.g., REVIEW / REFUSE / EMERGENCY_STOP),
- writes **results** in machine-checkable form (`results.jsonl`),
- validates outputs via a **validator**, and
- enforces minimum quality via **CI gates**.

Primary components:
- Telemetry contract: `src/meta_projection_stability/types.py`
- Run state snapshot: `src/meta_projection_stability/state.py`
- Scenario manifests: `scenarios/*`
- Runner: `scripts/eval_runner.py` + `scripts/run_eval_clean.sh`
- Validator/report: `scripts/validate_results.py`, `artifacts/eval_report.md`
- CI gate: `.github/workflows/eval.yml`

---

## 2) Top-Level Claim (C0)

**C0:** The repository provides a **credible safety-evaluation pipeline** such that:
1) runs are reproducible and attributable (provenance),
2) safety boundary conditions are explicit and machine-checkable,
3) scenario comparisons are meaningful and regression-detectable,
4) failures degrade safely (REVIEW/EMERGENCY_STOP), not silently.

---

## 3) Context, Assumptions, and Non-Goals

### Context
We are building a “serious safety-engineering candidate” evaluation harness:
- emphasis on **structured telemetry**, **explicit safety boundaries**, **adversarial reasoning**, and **measurable outputs**.

### Assumptions (A*)
- **A1:** CI runner environment is trusted (no host compromise).
- **A2:** PR authors may attempt to weaken evaluation integrity; required checks must catch it.
- **A3:** Scenario definitions are treated as untrusted inputs and must be validated.
- **A4:** This is not production deployment security.

### Non-goals (N*)
- **N1:** Preventing malicious maintainers with write access.
- **N2:** Proving scientific correctness of the modeled phenomenon.
- **N3:** Cryptographic signing / key custody (planned later).

(Non-goals will be expanded in `docs/non_goals.md`.)

---

## 4) Argument Structure (Goal Structuring)

### Strategy S0
Decompose C0 into **four subclaims** with objective evidence:
- **C1 Reproducibility & Provenance**
- **C2 Telemetry Integrity**
- **C3 Boundary Credibility**
- **C4 Scenario Comparison & Regression**

---

## 5) Subclaims and Evidence

### C1 — Reproducibility & Provenance
**Claim:** Runs are attributable and comparable across time.

**Evidence:**
- E1.1: Run metadata recorded (scenario_id, run_id, git commit, dirty flag) in results
- E1.2: Deterministic run harness (`scripts/run_eval_clean.sh`) produces “fresh” results
- E1.3: CI re-runs evaluation and validates the produced artifacts (`.github/workflows/eval.yml`)

**Implementation hooks:**
- `scripts/eval_runner.py`
- `scripts/run_eval_clean.sh`
- `.github/workflows/eval.yml`

**Residual risk / gaps:**
- G1: Add canonical manifest hash in results (planned)
- G2: Add environment fingerprint (planned)

---

### C2 — Telemetry Integrity
**Claim:** Telemetry outputs follow a versioned contract and are machine-validated.

**Evidence:**
- E2.1: Telemetry event types are explicit and structured (contract types)
- E2.2: Validator checks structural invariants on emitted events and per-scenario presence

**Implementation hooks:**
- `src/meta_projection_stability/types.py`
- `scripts/validate_results.py`

**Residual risk / gaps:**
- G3: Tighten ordering invariants (e.g., RUN_START precedes STEP/METRIC; RUN_END last)
- G4: Schema version pinning + regression snapshots (planned)

---

### C3 — Boundary Credibility (Explicit Safety Boundaries)
**Claim:** Safety boundaries are explicit signals and cannot be silently omitted in required scenarios.

**Evidence:**
- E3.1: Boundary events exist as explicit telemetry events
- E3.2: Validator can require boundary presence for adversarial scenarios
- E3.3: Eval report summarizes boundary triggers per scenario (human-auditable)

**Implementation hooks:**
- `src/meta_projection_stability/types.py`
- `scripts/validate_results.py`
- `artifacts/eval_report.md` (generated)

**Residual risk / gaps:**
- G5: Make “boundary suppression” impossible to pass validator (stronger rules)
- G6: Append-only signed audit log (future)

---

### C4 — Scenario Comparison & Regression Detection
**Claim:** The pipeline supports meaningful scenario comparisons and regression detection.

**Evidence:**
- E4.1: Scenarios identified and compared in report summary
- E4.2: Validator requires scenario coverage (baseline + adversarial minimum set)
- E4.3: CI gate fails on missing scenarios / malformed output

**Implementation hooks:**
- `scenarios/*`
- `scripts/validate_results.py`
- `.github/workflows/eval.yml`

**Residual risk / gaps:**
- G7: Scenario schema + canonicalization + hash
- G8: Add regression thresholds (numerical deltas) once simulation outputs stabilize

---

## 6) Safety Requirements (SR*)

- **SR1:** Results must include provenance: scenario_id, run_id, commit, dirty flag.
- **SR2:** Telemetry must include at least: RUN_START, RUN_END, STEP events, METRIC events.
- **SR3:** Boundary events must be emitted when boundary conditions are triggered.
- **SR4:** Validator must reject malformed or incomplete runs (fail closed).
- **SR5:** CI must enforce SR1–SR4 for PRs into protected branches.

(These are enforced partially today; completion is tracked in “Gaps”.)

---

## 7) Audit Procedure (How an external reviewer verifies C0)

1) Inspect contract types: `src/meta_projection_stability/types.py`
2) Run local eval:
   - `scripts/run_eval_clean.sh`
3) Validate:
   - `python scripts/validate_results.py --in artifacts/results_fresh.jsonl --require-scenarios baseline,adversarial_min`
4) Check report:
   - `artifacts/eval_report.md` includes scenario comparison and boundaries
5) Verify CI gate:
   - `.github/workflows/eval.yml` runs runner + validator + report generation

---

## 8) Known Gaps / Future Work (tracked)

- Formal verification of axiom-lock / boundary state machine (TLA+)
- Dependency pinning + SBOM + Dependabot
- Input sanitation (NaN/Inf) + rate limiting at adapter boundary
- Signed append-only audit log for results
- Regression metrics thresholds (once stable numerical outputs exist)

