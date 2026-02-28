# Non-Goals — meta-projection-stability (Phase 6)

Status: **Draft (Phase 6 / Step 17)**  
This document defines what this repository **does not** claim to provide.

---

## 1) Not Production Security

This repository is an **evaluation harness**. It is **not** a hardened production service.

Non-goals:
- No guarantees against host/kernel compromise
- No secure multi-tenant isolation
- No hardened secret storage / HSM integration
- No production-grade authentication/authorization model

---

## 2) Not a Formal Proof of Correctness

The repo aims for **measurable evaluation credibility**, not mathematical proof.

Non-goals:
- No full formal verification (TLA+/Coq/Isabelle) of the system state machine (planned later)
- No invariant proof that the underlying model is correct
- No completeness proof of the threat model

---

## 3) Not a Claim of Scientific Validity

The evaluation infrastructure can be strong even if the modeled phenomenon is exploratory.

Non-goals:
- No claim that “meta projection stability” is a validated scientific theory
- No claim that φ/quasi-crystal heuristics are physically correct
- No claim of predictive power beyond documented evaluation protocols

---

## 4) Not Adversary-Resistant Under Maintainer Compromise

If an attacker has repository write access and can change CI rules, they can subvert evaluation.

Non-goals:
- Defending against malicious maintainers
- Defending against compromised CI configuration by an authorized actor

Mitigation direction (future):
- branch protection + required checks + signed commits (policy-level controls)

---

## 5) Not Cryptographically Signed Evidence (Yet)

Current outputs are validated structurally, but not cryptographically signed.

Non-goals (current state):
- No cryptographic signatures on `results.jsonl` / reports
- No append-only tamper-evident log with key custody

Planned (future):
- signed audit log + provenance strengthening

---

## 6) Not Exhaustive Test/Fuzz Coverage (Yet)

We aim to grow coverage, but we do not claim “complete” adversarial testing.

Non-goals (current state):
- No guarantee of 95%+ coverage at all times
- No guarantee that fuzzing finds all failure modes
- No claim of resilience against all malformed inputs

---

## 7) Not a Substitute for External Audit

This repo can support auditability, but does not replace independent review.

Non-goals:
- No claim that internal CI checks are equivalent to external audit
- No claim of compliance certifications (OpenSSF, ISO, etc.)

---

## 8) What we DO claim (minimal positive statement)

- Structured telemetry + explicit boundary signaling
- Reproducible evaluation runs with provenance metadata
- Validator + CI gate that fails closed on malformed outputs
- Clear documentation of threats, safety case, and non-goals

