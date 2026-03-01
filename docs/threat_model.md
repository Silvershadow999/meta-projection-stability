# Threat Model (v1)

## Scope
This threat model covers the **evaluation and safety-engineering pipeline** of `meta-projection-stability`:
- structured telemetry + artifacts (`results.jsonl`, `eval_report.md`, `robustness_report.json`)
- scenario manifests and reproducible runs (baseline + adversarial variants)
- CI gates (axiom integrity, eval validation, robustness validation)

**Out of scope (for now):**
- deployment as a real-time controller in production environments
- hardware security (TPM/secure enclaves), network perimeter security, physical threats
- model weights / training pipelines (not part of this repo)

## Assets (what we protect)
1. **Safety invariants / Level-0 axiom** (integrity, non-silent drift)
2. **Telemetry contract** (schema stability, unambiguous semantics)
3. **Reproducibility / provenance** (seed, manifest, git commit, deterministic artifacts)
4. **Evaluation artifacts** (append-only results, reports, golden references)
5. **Robustness contract** (expectations + validator + golden drift checks)

## Trust boundaries
- Developer workstation / Codespace vs. GitHub Actions runners
- Local artifacts (mutable) vs. CI artifacts (immutable per run)
- Scenario inputs (manifests) vs. produced outputs (JSONL/JSON reports)
- Human review boundary: when automation must escalate (REVIEW / REFUSE / EMERGENCY_STOP)

## Assumptions
- CI runners are ephemeral and start from clean checkouts.
- Branch protection rules enforce required checks before merge.
- Scenarios are treated as untrusted input (must be validated).
- Artifacts are not assumed to be confidential (public repo setting).

## Threat categories (STRIDE-inspired)

### 1) Spoofing
**Threat:** A malicious contributor spoofs scenario identity or provenance (e.g., claiming baseline while running altered parameters).  
**Mitigations:**
- run provenance captured in telemetry (git commit, dirty flag)
- scenario_id must match manifest id; loader enforces stable mapping
- CI gates run from clean checkout (no dirty workspace)
**Residual risk:** provenance can be incomplete if git metadata unavailable; mitigate by failing closed in CI.

### 2) Tampering
**Threat:** Modifying artifacts or manifests to hide regressions (e.g., editing robustness_report.json, changing goldens silently).  
**Mitigations:**
- CI regenerates artifacts and validates invariants
- golden drift check compares current report against committed golden with tolerances
- axiom / fingerprint guard prevents silent drift of foundational safety text
**Residual risk:** A PR can still change golden files; mitigation is explicit human review + PR description requiring justification.

### 3) Repudiation
**Threat:** Contributor denies responsibility for changed behavior ("it was always like that").  
**Mitigations:**
- append-only JSONL and deterministic reports
- explicit scenario manifests + seeds
- CI logs and artifacts attached to runs
**Residual risk:** none beyond normal git history ambiguity.

### 4) Information disclosure
**Threat:** leaking sensitive info into artifacts (paths, local usernames, private metadata).  
**Mitigations:**
- keep provenance minimal; avoid storing secrets in payloads
- do not log environment variables or file contents outside repo
**Residual risk:** developer may add sensitive fields; mitigate by code review and linting.

### 5) Denial of service (DoS) / resource exhaustion
**Threat:** scenarios or perturbations cause CI to time out (infinite loops, huge artifacts).  
**Mitigations:**
- workflow timeouts
- bounded steps, bounded perturbations count
- artifact size discipline (summary + golden, not raw dumps)
**Residual risk:** intentionally expensive PRs; mitigate by limiting CI minutes and adding per-run caps.

### 6) Elevation of privilege / policy bypass
**Threat:** bypassing safety gates (disabling validators, removing required checks).  
**Mitigations:**
- branch protection rulesets enforce required checks
- separate gates (axiom-guard, eval-gate, robustness-gate) reduce single-point failure
**Residual risk:** repository admin privileges can override; mitigated by governance, not code.

## Threat-driven tests (mapping to repo checks)
- **axiom-guard:** integrity boundary for Level-0 axiom / core invariants
- **eval-gate:** results.jsonl invariants + report generation
- **robustness-gate:** benign perturbations stability + expectations validator + golden drift check

## Open risks / next iterations
- Formalize "Non-goals" to prevent misuse interpretations.
- Add explicit "Safety Case" argument structure (claims → evidence → tests).
- Add scenario input validation (schema validation, bounds checking).
- Add adversarial "tamper" scenarios to test reporting integrity.

