# meta-projection-stability

A compact research / engineering framework for **meta-stability simulation**, **trust-damped control**, **human-significance anchoring**, **biometric proxy evaluation**, **adversarial scenario testing**, and an experimental **Axiom Handshake / Compatibility Gateway** for bounded external-entity compatibility checks.

The project is designed as a **simulation and evaluation environment** for studying how a system behaves under instability, contradiction, low-trust states, spoofing attempts, and degraded signal integrity — while preserving human-centered constraints.

---

## Core idea

The framework combines several layers:

- **meta-stability adaptation**
- **trust dynamics**
- **human significance anchoring**
- **risk smoothing / damping**
- **biometric proxy telemetry**
- **mutuality / support shaping**
- **adversarial scenario simulation**
- **integrity and compatibility boundary logic**

The intention is to create a system that remains:

- interpretable
- bounded
- reversible where possible
- resistant to shallow spoofing
- resistant to domination-style collapse
- measurable through telemetry

---

## Features

### 1. Meta-stability adapter
The `MetaProjectionStabilityAdapter` models:

- instability risk
- trust-level evolution
- momentum-sensitive warning logic
- hysteresis / cooldown handling
- human significance recovery / decay
- telemetry-rich decision outputs

Typical outputs include:

- `decision`
- `status`
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

---

### 2. Biometric proxy / signal integrity layer
The adapter can incorporate a biometric / channel-quality style proxy layer, including telemetry such as:

- `biometric_proxy_mean`
- `biometric_proxy`
- `sensor_consensus`
- `critical_channel_penalty`
- `critical_channel_min`
- `consensus_penalty`
- `bio_penalty`
- `base_decay_effective`

This is intended to distinguish between:

- stable signal ensembles
- inconsistent or spoofed channel structure
- superficially strong but internally weak channel sets

---

### 3. Axiom lock / irreversible harm latch
The framework can be extended with a **persistent axiom lock** concept:

- normal operation remains trust-damped and adaptive
- if an irreversible harm commitment is detected, the system can latch into a terminal state such as:
  - `AXIOM_ZERO_LOCK`
  - `axiom_lock`

This is intended as a **last safety ring**, not as a normal operating path.

---

### 4. Adversarial scenarios
The package supports adversarial / stress testing through CLI scenarios and helper functions.

Examples include testing:

- spoof-like behavior
- contradiction pressure
- trust degradation
- low-coherence channel sets
- axiom-lock boundary conditions

---

### 5. Axiom Handshake / Compatibility Gateway
An experimental handshake layer is included to evaluate whether an **external entity** behaves in a way that is compatible with the system’s stability axioms.

Important:

- this is a **compatibility check**
- it is **not** an overwrite mechanism
- it is **not** an offensive control surface
- it is a **bounded integrity boundary**

The handshake asks whether an external entity remains coherent under contradiction while preserving:

- boundedness
- reversibility
- non-domination
- human preservation weighting

---

## Installation

### Local editable install
```bash
pip install -e .
