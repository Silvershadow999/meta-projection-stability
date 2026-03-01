# Security Policy

## Overview
This repository focuses on **evaluation and safety-engineering scaffolding** (telemetry, scenario comparison,
robustness gates, regression checks). It is not a production security product and does not make deployment
safety guarantees.

See:
- `docs/non_goals.md`
- `docs/threat_model.md`
- `docs/safety_case.md`

## Reporting a vulnerability
If you believe you have found a security vulnerability or a safety-critical weakness in the evaluation gates,
please report it responsibly.

### Preferred reporting method
- Open a **private report** via GitHub Security Advisories (preferred), or
- If advisories are not available, open an issue with **minimal details** and request a private channel.

**Do not** include secrets, credentials, or exploit code in public issues.

## What counts as a security issue here
Examples:
- bypassing required CI gates (axiom/eval/robustness) or weakening them silently
- tampering with artifacts/goldens in a way that hides regressions
- validator logic flaws that allow unsafe regressions to pass
- supply-chain risks in build/workflow steps (malicious install, unpinned actions, etc.)
- injection of sensitive data into artifacts (secrets leakage)

## Out of scope
- production deployment security of autonomous systems (not provided by this repo)
- hardware/physical security
- network perimeter security beyond CI workflows
- vulnerabilities in third-party dependencies not used by this repo

## Coordinated disclosure
We aim to acknowledge reports and coordinate a fix before public disclosure when feasible.
Severity and timelines depend on impact and complexity.

## Hardening principles used
- required status checks in rulesets (axiom/eval/robustness/policy)
- CODEOWNERS-based review enforcement for safety-critical surfaces
- deterministic artifacts + validators + golden drift checks

Thank you for helping improve safety and integrity.
