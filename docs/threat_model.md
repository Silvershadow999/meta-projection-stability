# Threat Model — meta-projection-stability (Phase 6)

Status: **Draft (Phase 6 / Step 15)**  
Scope: **Evaluation harness + telemetry + boundary signalling**, not production deployment.

## 1) Purpose and Scope

This document models credible threats against **evaluation integrity** and **safety boundary enforcement** for the `meta-projection-stability` repository.

In-scope components (current repo structure):
- **Telemetry Contract** (`src/meta_projection_stability/types.py`)
- **Run State snapshotting** (`src/meta_projection_stability/state.py`)
- **Scenario manifests** (`scenarios/*.json` or equivalent)
- **Eval runner** (`scripts/eval_runner.py`, `scripts/run_eval_clean.sh`)
- **Results + validator + report** (`artifacts/results.jsonl`, `scripts/validate_results.py`, `artifacts/eval_report.md`)
- **CI eval gate** (`.github/workflows/eval.yml`)

Out-of-scope (for now):
- Full production deployment / external API service hardening
- Cryptographic signing / HSM key management (tracked as future work)
- Formal verification (TLA+/Coq) beyond lightweight invariants (tracked as future work)

Security goals:
- **G1: Reproducibility** — same scenario + code + seed ⇒ comparable outputs
- **G2: Provenance** — every run is attributable (commit, scenario id, tool version)
- **G3: Boundary credibility** — explicit boundary events cannot be silently suppressed
- **G4: Evaluation integrity** — results cannot be trivially forged without detection
- **G5: Safe failure** — malformed inputs degrade safely (REVIEW / EMERGENCY_STOP)

Non-goals (summary; detailed in `docs/non_goals.md` later):
- Preventing a determined attacker with full repo write access
- Defending against kernel/host compromise
- Guaranteeing correctness of the underlying scientific model (this is evaluation infra)

## 2) Assets

Primary assets we protect:
- **A1: Telemetry stream correctness** (event types, ordering, required fields)
- **A2: Boundary signals** (e.g., REFUSE/REVIEW/EMERGENCY_STOP) and their triggers
- **A3: Scenario definitions** (inputs/parameters that define comparisons)
- **A4: Results** (`results.jsonl`) + **validator acceptance criteria**
- **A5: Provenance metadata** (git commit, dirty flag, environment fingerprint)
- **A6: CI gate outcome** (pass/fail as a policy enforcement mechanism)

## 3) Trust Boundaries and Data Flows

### Trust boundary TB1 — Scenario input boundary
Untrusted:
- scenario manifest files (could be malformed / adversarial)
Trusted:
- schema validation + defaulting + explicit allowlist of scenario keys

### Trust boundary TB2 — Runtime signal boundary (adapter/runner)
Untrusted:
- raw signals passed into adapter/interpret layer (NaN/Inf/out-of-range, weird types)
Trusted:
- sanitization + clipping + explicit “invalid → boundary event” behavior

### Trust boundary TB3 — Results boundary (filesystem / artifacts)
Untrusted:
- artifacts directory contents
Trusted:
- validator rules + deterministic formatting + CI re-generation of “fresh” results

### Trust boundary TB4 — CI boundary
Untrusted:
- PR author environment
Trusted:
- GitHub Actions runner environment, pinned workflow, reproducible install

Data flow summary:
scenario manifest → eval_runner → adapter/state → telemetry events → results.jsonl → validate_results → eval_report → CI gate

## 4) Threats (STRIDE)

Legend:
- Impact: Low / Med / High
- Likelihood: Low / Med / High
- Detection: how we expect to catch it today

| ID | STRIDE | Threat | Asset(s) | Impact | Likelihood | Current controls | Gaps / Planned mitigations |
|---:|:------:|--------|----------|:------:|:----------:|------------------|----------------------------|
| T01 | S | Scenario spoofing (claim baseline but run different params) | A3,A4,A5 | High | Med | scenario_id in results; validator requires scenarios | Add scenario hash + manifest canonicalization |
| T02 | T | Results tampering (edit `results.jsonl`) | A4,A6 | High | Med | CI regenerates fresh results; validator checks structure | Add append-only signed log (future) |
| T03 | R | Repudiation of runs (“not my run”) | A5 | Med | Med | git commit + dirty flag | Add environment fingerprint + tool version pin |
| T04 | I | Leak of sensitive local paths / env | A5 | Low | Med | keep metadata minimal | Redact absolute paths; avoid dumping env vars |
| T05 | D | DoS via huge scenario / runaway loop | A1,A6 | Med | Med | n_steps bounded; timeouts in CI | Add explicit per-scenario max runtime + hard timeout |
| T06 | E | Privilege escalation via command injection in runner | A6 | High | Low | avoid shell eval; fixed scripts | Audit runner; never interpolate untrusted strings |
| T07 | T | Boundary suppression (emit no boundary event) | A2,A6 | High | Med | validator can require boundary events in adversarial scenario | Tighten validator invariants: required event ordering |
| T08 | T | Telemetry schema drift (silent breaking change) | A1,A6 | High | Med | typed contract; import smoke tests | Add versioned schema + regression snapshots |
| T09 | D | NaN/Inf poisoning in interpret() | A2,A4 | High | Med | partial clipping | Add input sanitation: finite checks → boundary event |
| T10 | S/T | Scenario manifest trick: path traversal / unintended file read | A3,A6 | High | Low | manifests should be data-only | Ensure manifests cannot reference filesystem paths |

## 5) Top Risks (Prioritized)

1) **Evaluation integrity bypass** (T01/T02/T07/T08)  
   Why: undermines the entire “safety engineering candidate” claim.
2) **NaN/Inf poisoning** (T09)  
   Why: causes undefined behavior and false safety confidence.
3) **DoS via adversarial scenario** (T05)  
   Why: blocks CI gate and slows iteration.

## 6) Mitigations Roadmap (mapped to Phases)

Immediate (Phase 5/6 adjacent):
- Enforce **scenario manifest schema** + canonical JSON and hash in results (T01)
- Strengthen validator: enforce **required event counts + ordering** (T07/T08)
- Add **finite checks** (NaN/Inf) in adapter interpret layer → boundary event (T09)

Next (Phase 6+):
- Add append-only **signed audit log** (T02/T03)
- Add dependency pinning + SBOM + Dependabot (supply-chain)
- Formalize invariants (TLA+) for “axiom lock” / boundary state machine

## 7) Evidence Hooks (What we can show auditors)

- Telemetry contract types: `src/meta_projection_stability/types.py`
- Run snapshot: `src/meta_projection_stability/state.py`
- Deterministic run metadata in results: commit, dirty, scenario_id
- Validator acceptance criteria: `scripts/validate_results.py`
- CI enforcement: `.github/workflows/eval.yml`

## 8) Assumptions

- Adversary can modify scenario manifests in PRs, but cannot bypass CI checks once required.
- CI runner environment is trusted (no host compromise).
- Threat model focuses on **integrity/credibility** of evaluation, not on production security.

