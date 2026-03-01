# Non-Goals (v1)

This repository is an **evaluation and safety-engineering scaffold** for meta-stability monitoring, telemetry,
scenario comparison, and robustness regression checks.

It is intentionally **not** a production control system.

---

## Not a deployment safety guarantee
- Not a proof of safety for real-world autonomous systems.
- Not a certification artifact for ISO 26262 / IEC 61508 / DO-178C by itself.
- Not formal verification, and not a substitute for domain-specific safety engineering.

## Not an offensive or coercive control system
- No “takeover”, “override”, or “dominance” mechanisms are intended or claimed.
- “Axiom lock” / boundary signals are **evaluation outputs**, not enforcement.

## Not a full adversarial security product
- Not a complete threat detection or intrusion prevention system.
- Not hardened against real-world attackers beyond the CI-gated evaluation scope.
- No claims about resistance to sophisticated adaptive adversaries (yet).

## Not a privacy / compliance solution
- Not a GDPR/PII compliance framework.
- Artifacts may contain run metadata; do not include secrets in logs/artifacts.
- No guarantee of data minimization beyond documented conventions.

## Not a training / model-weights repository
- No claims about training data, model weights, or model governance.
- The repo focuses on evaluation scaffolding and control-logic experiments.

## Not a performance benchmark suite
- Not optimized for speed, cost, or scaling to large production workloads.
- CI timeouts and bounded runs prioritize determinism over throughput.

---

## Intended use (what it *is* for)
- Structured telemetry and explicit boundary semantics.
- Reproducible scenario comparison (baseline vs adversarial variants).
- Robustness checks under benign perturbations with CI gating.
- Regression detection via goldens and explicit tolerances.

See:
- `docs/threat_model.md`
- `docs/safety_case.md`

