# meta-projection-stability

Experimental control-logic prototype for **meta-projection stability** with **human-significance anchoring**, **trust-modulated risk regulation**, **structured telemetry**, **scenario comparison**, and **CI-gated robustness/regression checks**.

This repository is designed as a **simulation + evaluation environment** for studying how a bounded control system behaves under instability, contradiction, degraded trust, spoof-like signals, weak channel integrity, and hard safety boundary conditions — while preserving explicitly human-centered constraints.

## Philosophy in One Sentence
Adaptive under uncertainty, bounded under contradiction, defensive under instability, and locked under irreversible violation.

---

## What this repo is (and is not)

**This repo is:**
- A research/engineering sandbox to test **bounded decision logic** under stress
- A safety-engineering candidate focusing on:
  - structured telemetry
  - explicit safety boundaries
  - reproducibility + provenance
  - scenario comparison
  - robustness under benign perturbations
  - regression protection (goldens)

**This repo is not:**
- A production deployment safety guarantee
- A full adversarial security product
- A privacy/compliance framework
- A formal verification system

See: `docs/non_goals.md`

---

## Core idea

The framework combines several interacting layers:

- meta-stability adaptation
- trust dynamics
- human-significance anchoring
- risk smoothing / damping
- biometric proxy telemetry
- mutuality / support shaping
- adversarial scenario testing
- axiom-lock / irreversible boundary logic
- level-0 policy / fingerprint protection

The goal is to explore systems that remain:

- interpretable
- bounded
- trend-aware
- resistant to shallow spoofing
- resistant to domination-style collapse
- measurable through explicit telemetry
- inspectable through scenario-driven stress tests

---

## Repository structure (high level)

- `src/meta_projection_stability/` — library code (adapter, simulation, telemetry schema, etc.)
- `scenarios/` — scenario manifests (`baseline.json`, `adversarial_min.json`, …)
- `scripts/` — runners + validators (eval, robustness, reports)
- `artifacts/` — generated outputs (JSONL telemetry, reports, robustness artifacts)
- `tests/` — smoke + invariants tests (including robustness artifact checks)
- `docs/` — safety engineering docs (threat model, safety case, protocol, non-goals)
- `.github/workflows/` — CI gates

---

## Main components

### 1) MetaProjectionStabilityAdapter
Models:
- instability risk
- trust-level evolution
- momentum-sensitive warning logic
- hysteresis / cooldown handling
- human-significance recovery / decay
- biometric proxy penalties
- autonomy-sensitive blocking
- mutuality bonus signals
- axiom-lock conditions

Outputs are designed to be **inspectable** and **telemetry-friendly**:
- decision / status / reasons
- human_significance + EMA
- risk + damped risk
- trust_level
- momentum + coherence
- cooldown state
- biometric proxy signals (mean, consensus, critical-channel penalty)
- axiom lock indicators

### 2) Simulation layer
Synthetic environment for rapid testing without external systems:
- multi-layer magnitudes
- stress windows / drift patterns
- history tracking for risk / trust / significance
- optional plotting and summaries

### 3) Biometric proxy / signal integrity layer
Optional “soft” multi-signal robustness:
- consensus quality
- channel degradation
- critical-channel penalties
- autonomy-sensitive damping
- mutuality/support signals

Telemetry separates:
- biometric_proxy_mean
- biometric_proxy
- sensor_consensus
- critical_channel_min
- critical_channel_penalty

### 4) Scenario comparison + adversarial testing
The scenario suite supports:
- baseline reference
- adversarial minimal cases (boundary/telemetry robustness)
- structured comparison via reproducible artifacts

### 5) Level-0 axiom protection
Level-0 invariants are guarded against silent drift by a CI integrity gate (“axiom guard”).

---

## Installation

Editable install (recommended for development):

```bash
pip install -e .
