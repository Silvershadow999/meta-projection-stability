# Non-Goals — meta-projection-stability

This document defines explicit **non-goals** to prevent over-claiming and scope creep.
Anything not listed as an intended capability should be treated as **out of scope** unless added explicitly.

## 1. Not a Production Safety System
- This repository is **not** a production-grade safety controller.
- It does **not** guarantee real-world safe operation of any physical system.

## 2. No Security Hardening Guarantee
- No guarantee of resistance against malicious local actors.
- No cryptographic signing/attestation of telemetry.
- No secure enclave, hardened runtime, or tamper-proof logging.

## 3. No Claims of Scientific or Physical Validity
- The framework does not claim to validate or prove physical theories.
- Outputs are evaluation artifacts, not scientific proof.

## 4. No Comprehensive Adversarial Coverage
- “Adversarial scenarios” are limited, curated inputs for evaluation.
- This does not cover all possible adversarial strategies or failure modes.

## 5. No General Alignment Guarantee
- The repository does not claim to solve alignment in general.
- It provides instrumentation and evaluation structure, not a universal safety solution.

## 6. No Cross-Environment Determinism Guarantee
- Reproducibility is best-effort via seed/scenario/provenance.
- Exact bit-for-bit determinism across different OS/CPU/Python builds is not guaranteed.

## 7. No Performance / Scalability Guarantees
- No guarantee for large-scale runs, distributed execution, or high-throughput logging.
- CI gating and scaling strategies are planned, not guaranteed.

## 8. No Hidden Behavior Claims
- Telemetry only represents what the current code emits.
- Absence of evidence in logs is not evidence of absence unless explicitly validated by invariants tests.

## 9. Deferred / Planned (not done yet)
The following items are planned but are **not** guaranteed to exist unless implemented:
- Strict telemetry schema validation per event line
- CI regression gates (pass/fail criteria enforced in CI)
- Real model execution integration via a bounded runner hook
- Comprehensive threat mitigation beyond local execution assumptions

## 10. Traceability
- Threat Model: `docs/threat_model.md`
- Safety Case: `docs/safety_case.md`
