readme_content = """# meta-projection-stability

A research-oriented framework for **meta-stability simulation**, **trust-damped control**, **human-significance anchoring**, **sensor / biometric proxy evaluation**, **adversarial scenario testing**, and **axiom-bound safety logic**.

The repository is designed as a simulation and evaluation environment for studying how a bounded control system behaves under instability, contradiction, degraded trust, spoof-like signals, weak channel integrity, and hard safety boundary conditions — while preserving explicitly human-centered constraints.

---

## Why this repository exists

Most safety and control discussions stay either too abstract or too opaque.

This repository takes a different route:

- make internal state explicit
- make safety-relevant transitions measurable
- test behavior under stress instead of assuming stability
- keep decision logic inspectable rather than hidden behind a single score
- explore bounded control instead of unconstrained autonomy

The result is a compact sandbox for experimenting with **interpretable meta-stability dynamics** under realistic failure pressure.

---

## Core Idea

The framework combines several interacting layers:

- **meta-stability adaptation**
- **trust dynamics**
- **human-significance anchoring**
- **risk smoothing / damping**
- **sensor and biometric proxy telemetry**
- **mutuality / support shaping**
- **adversarial scenario simulation**
- **axiom-lock / irreversible boundary logic**
- **Level-0 policy / fingerprint protection**

The overall intention is to model systems that remain:

- interpretable
- bounded
- trend-aware
- resilient against shallow spoofing
- resistant to domination-style collapse
- auditable through explicit telemetry
- stress-testable under adverse conditions

---

## Project Status

This repository is an **experimental prototype**.

It is **not** presented as:

- a production-ready alignment solution
- a deployment-grade security product
- a proof of safe autonomous enforcement
- a complete real-world robotics control system

It is best understood as a **research and engineering sandbox** for studying:

- trust-aware gating
- significance decay / recovery
- bounded control behavior
- biometric / sensor-consensus penalties
- emergency lock conditions
- compatibility checks under contradiction
- policy integrity protection
- adversarial evaluation loops

---

## Main Components

### 1. MetaProjectionStabilityAdapter

The `MetaProjectionStabilityAdapter` is the core control component.

It models:

- instability risk
- trust-level evolution
- momentum-sensitive warning logic
- hysteresis / cooldown behavior
- human-significance recovery / decay
- biometric proxy penalties
- autonomy-sensitive blocking
- mutuality bonus signals
- axiom-lock conditions

Typical output fields include:

- `decision`
- `status`
- `decision_reason`
- `status_reason`
- `human_significance`
- `h_sig_ema`
- `instability_risk`
- `risk_raw_damped`
- `trust_level`
- `momentum`
- `coherence`
- `risk_input`
- `trust_damping`
- `cooldown_remaining`
- `biometric_proxy_mean`
- `biometric_proxy`
- `sensor_consensus`
- `critical_channel_penalty`
- `critical_channel_min`
- `bio_penalty`
- `autonomy_proxy`
- `autonomy_penalty`
- `consensus_penalty`
- `base_decay_effective`
- `mutual_bonus`
- `harm_commit_persistent`
- `axiom_locked_at_step`
- `near_axiom_lock`

---

### 2. Simulation Layer

`run_simulation(...)` provides a configurable synthetic environment for testing the control loop.

Depending on the active branch or module state, this may include:

- layer-based internal dynamics
- synthetic stress windows
- sensor-style proxy environments
- trend-sensitive updates
- history tracking for risk, trust, and significance
- optional plotting and summaries

This allows rapid iteration without depending on external hardware or live systems.

---

### 3. Sensor / Biometric Proxy Layer

The framework supports soft proxy logic for signal integrity and embodied-state style inputs, including:

- biometric proxy values
- sensor consensus quality
- weak-channel penalties
- autonomy-sensitive damping
- tamper suspicion
- support / mutuality signals

This helps move the system beyond a naive single-scalar trust gate.

The telemetry deliberately separates:

- `biometric_proxy_mean`
- `biometric_proxy`
- `sensor_consensus`
- `critical_channel_min`
- `critical_channel_penalty`

That distinction matters because a high average can still hide a weak critical channel.

---

### 4. Adversarial Scenario Runner

The adversarial tooling is used to stress the control loop under structured failure modes.

Scenarios may include patterns such as:

- spoof-like instability
- contradiction pressure
- trust degradation
- threshold-hover behavior
- weak consensus / weak channel structures
- axiom-trigger conditions
- restart / lock persistence edge cases

This is not meant as frontier-scale red-teaming.  
It is meant as **explicit stress evaluation of bounded control behavior**.

---

### 5. Level-0 Axiom Protection

The repository includes a **Level-0 Axiom** concept together with integrity / fingerprint checks.

Purpose:

- protect canonical policy text from silent drift
- ensure foundational safety material is not casually altered
- make baseline policy changes explicit and reviewable

This acts as a lightweight but important integrity boundary.

---

### 6. Experimental Axiom Handshake / Compatibility Logic

The repository also includes experimental **axiom-bound compatibility modules**.

These are intended to evaluate whether an external entity or process remains compatible with bounded, human-preserving control logic under contradiction.

Important:

- this is a **compatibility evaluation concept**
- it is **not** an overwrite mechanism
- it is **not** an offensive control interface
- it is a **bounded integrity / resonance / compatibility layer**

The question is intentionally narrow:

> Can an external decision process remain coherent, bounded, reversibility-aware, and non-dominating under contradiction?

If not, the system should degrade toward **rejection**, not cooperation.

---

## Design Philosophy

This repository emphasizes:

- **explicit state** over hidden magic
- **telemetry-rich outputs** over opaque scoring
- **boundedness** over unconstrained optimization
- **stress-testing** over aesthetic stability claims
- **human-centered constraints** over abstract capability maximization
- **inspectable failure modes** over vague safety language

In one sentence:

**Adaptive under uncertainty, bounded under contradiction, defensive under instability, and locked under irreversible violation.**

---

## Decision Philosophy

The adapter is built around layered behavioral modes.

### Normal Operation

- adaptive
- recoverable
- trust-damped
- significance-aware

### Transition Band

- cautious
- reflective
- warning-sensitive
- contradiction-aware

### Critical State

- reset-capable
- cooldown-aware
- safety-prioritized

### Terminal Axiom State

- hard lock
- non-recoverable through ordinary trust dynamics
- intended only for irreversible boundary-violation conditions

---

## Quick Start

### Install in editable mode

```bash
pip install -e .
