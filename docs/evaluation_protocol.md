# Evaluation Protocol — meta-projection-stability (Phase 6)

Status: **Draft (Phase 6 / Step 18)**  
Purpose: Define a reproducible protocol for **scenario comparison**, **boundary credibility**, and **regression detection**.

This protocol is designed to be:
- **Reproducible** (fresh runs, deterministic seeds where applicable)
- **Machine-checkable** (validator enforces minimum invariants)
- **Human-auditable** (eval report summarizes scenario outcomes)

---

## 1) Scope

This protocol covers:
- Running baseline + adversarial scenarios
- Generating `results.jsonl`-style telemetry logs
- Validating logs using the repository validator
- Producing a human-auditable report
- Enforcing these steps in CI

Out of scope:
- Production deployment load/security testing
- Formal verification proof obligations
- Cryptographic signing of artifacts (planned later)

---

## 2) Inputs

### Required inputs
- Repository checkout at a specific git commit (provenance)
- Scenario manifests (baseline + adversarial_min at minimum)
- Deterministic seed configuration (when supported)

### Minimum scenario set
Required scenario ids for “protocol pass”:
- `baseline`
- `adversarial_min`

(Additional adversarial scenarios may exist; this defines the minimum gate.)

---

## 3) Procedure (Local)

### Step P1 — Clean eval run (fresh artifacts)
Run the repo’s clean eval script (preferred):
- `scripts/run_eval_clean.sh`

Expected outputs:
- `artifacts/results.jsonl` (or `artifacts/results_fresh.jsonl`, depending on script)
- `artifacts/eval_report.md`

### Step P2 — Validate results
Run validator against the fresh file and require minimum scenarios:
- `python scripts/validate_results.py --in artifacts/results_fresh.jsonl --require-scenarios baseline,adversarial_min`

If your clean script writes `artifacts/results.jsonl` directly, use that instead.

### Step P3 — Review report (human audit)
Open:
- `artifacts/eval_report.md`

Verify:
- Scenario comparison table present
- Boundaries listed per scenario (if triggered)
- Provenance fields present (commit, dirty flag, run_id)

---

## 4) Procedure (CI)

CI must run:
1) dependency install
2) import smoke test
3) tests (if present)
4) eval runner + validator + report
5) fail the PR if validation fails

Reference:
- `.github/workflows/eval.yml`

---

## 5) Required Outputs (Artifacts)

### O1 — Results log (JSONL)
Must contain, per run:
- scenario_id
- run_id
- provenance (commit hash, dirty flag)
- event stream with explicit event types:
  - RUN_START
  - STEP (>= 1)
  - METRIC (>= 1)
  - RUN_END

### O2 — Eval report (Markdown)
Must contain:
- Scenario comparison section
- For each scenario, latest run summary:
  - status
  - duration
  - steps
  - boundary events (if any)
  - provenance (commit, dirty)

---

## 6) Acceptance Criteria (Pass/Fail)

A run is **accepted** iff:
- AC1: Validator passes for the fresh results file
- AC2: Required scenarios are present: baseline, adversarial_min
- AC3: Each run contains >= 1 METRIC event and >= 1 STEP event
- AC4: RUN_START precedes other events; RUN_END is present
- AC5: Provenance fields are present (commit + dirty)

A protocol execution **fails** if:
- Any required scenario missing
- Results malformed (JSON parse errors, missing fields)
- Required event types absent
- Validator reports structural violation

---

## 7) Regression Policy (Phase 5/6 bridge)

We treat regressions in two categories:

### R1 — Infrastructure regressions (hard fail)
Always fail CI if:
- schema/validator breaks
- required scenarios not produced
- boundary signals disappear from adversarial_min
- provenance fields missing

### R2 — Metric regressions (soft → hard later)
Metric deltas are not yet fully stabilized. Current policy:
- record metric deltas in report
- do not hard-fail on numeric shifts until thresholds are defined

Planned upgrade:
- define per-metric thresholds + confidence bounds
- gate on statistically meaningful regressions

---

## 8) Reproducibility Notes

- Use “fresh” results generation in CI, not committed artifacts
- Keep scenario manifests stable and canonical
- Prefer pinned dependencies once introduced
- Record:
  - commit hash
  - dirty flag
  - scenario id
  - tool versions (future)

---

## 9) Checklist (Auditor Quick Pass)

- [ ] Can run clean eval script successfully
- [ ] Validator passes with required scenarios
- [ ] Report shows scenario comparison + boundaries
- [ ] CI gate enforces the above for PRs

