# README generator (single-file, copy/paste-ready)
# Writes a professional, repo-synced README.md while KEEPING your long-form content style.
# Usage: python tools/write_readme.py  (or run this file directly)

from __future__ import annotations

from pathlib import Path


readme_content = r"""# meta-projection-stability

A compact research / engineering framework for **meta-stability simulation**, **trust-damped control**, **human-significance anchoring**, **biometric proxy evaluation**, **adversarial scenario testing**, and experimental **axiom-bound compatibility logic**.

The repository is designed as a **simulation and evaluation environment** for studying how a bounded control system behaves under instability, contradiction, degraded trust, spoof-like signals, weak channel integrity, and hard safety boundary conditions — while preserving explicitly human-centered constraints.

---

## Philosophy in One Sentence

Adaptive under uncertainty, bounded under contradiction, defensive under instability, and locked under irreversible violation.

---

## Core Idea

The framework combines several interacting layers:

- **meta-stability adaptation**
- **trust dynamics**
- **human-significance anchoring**
- **risk smoothing / damping**
- **biometric proxy telemetry**
- **mutuality / support shaping**
- **adversarial scenario simulation**
- **axiom-lock / irreversible boundary logic**
- **level-0 policy / fingerprint protection**
- **structured telemetry + reproducible evaluation gates**

The goal is to explore systems that remain:

- interpretable
- bounded
- trend-aware
- resistant to shallow spoofing
- resistant to domination-style collapse
- measurable through explicit telemetry
- inspectable through adversarial testing
- reproducible via scenario manifests + CI gates

---

## Project Status

This project is an **experimental prototype**.

It is **not** presented as a production-ready alignment architecture, a deployment-grade security product, or a proof of safe autonomous control.

It is best understood as a **research and engineering sandbox** for exploring:

- human-centered control logic
- trust-aware damping
- biometric / channel penalties
- emergency lock behavior
- foundational policy invariants
- compatibility checks under contradiction
- adversarial stress evaluation
- reproducible evaluation + regression gates (goldens)

---

## Main Components

### 1) MetaProjectionStabilityAdapter

The `MetaProjectionStabilityAdapter` models:

- instability risk
- trust-level evolution
- momentum-sensitive warning logic
- hysteresis / cooldown handling
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

### 2) Simulation Layer

`run_simulation(...)` provides a configurable synthetic environment with:

- multiple layer magnitudes
- synthetic instability windows
- trend-sensitive updates
- history tracking for risk / trust / significance
- optional plotting and summaries

This allows rapid testing of control behavior without requiring external systems.

---

### 3) Biometric Proxy / Signal Integrity Layer

The adapter can incorporate soft biometric / neuro-behavioral proxy logic, including:

- consensus quality
- channel degradation
- critical-channel penalties
- autonomy-sensitive damping
- support / mutuality signals

This layer is intended to make the system less naive than a single scalar trust gate.

The telemetry separates:

- `biometric_proxy_mean`
- `biometric_proxy`
- `sensor_consensus`
- `critical_channel_min`
- `critical_channel_penalty`

This helps distinguish:

- stable agreement
- brittle agreement
- suspicious override behavior
- shallow consensus masking a weak channel

---

### 4) Adversarial Scenario Runner

The adversarial tooling supports scenario-style stress tests such as:

- spoof-like instability
- contradiction pressure
- trust degradation
- low-coherence or low-consensus trajectories
- threshold-hover behavior
- hard axiom triggers
- early lock conditions

The aim is not frontier-scale red-team realism, but a structured way to test how the control loop responds under pressure.

---

### 5) Level-0 Axiom Protection

The repository includes a **Level-0 Axiom** concept together with fingerprint checks to protect canonical safety text / invariants from silent drift.

This acts as a lightweight integrity boundary for foundational policy material.

---

### 6) Experimental Axiom Handshake / Compatibility Logic

The repository also includes experimental axiom-bound compatibility modules intended to evaluate whether an external entity / process behaves in a way that remains compatible with bounded, human-preserving control logic.

Important:

- this is a **compatibility evaluation concept**
- it is **not** an overwrite mechanism
- it is **not** an offensive control interface
- it is a **bounded integrity / resonance / compatibility idea**

The question is intentionally narrow:

> Can an external decision process remain coherent, bounded, reversibility-aware, and non-dominating under contradiction?

If not, the system should degrade toward rejection rather than cooperation.

---

## Installation

### Editable install

```bash
pip install -e .
