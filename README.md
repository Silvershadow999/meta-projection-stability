# Meta Projection Stability

A simulation-oriented reference implementation for **meta-stability control** with:

- **Human-significance anchoring** (interestingness preservation)
- **Trust dynamics** (human ↔ system coupling)
- **Trust-damped risk regulation**
- **Momentum-based early warning**
- **Hysteresis / Schmitt-trigger behavior**
- **Cooldown after emergency reset**
- **Optional biometric proxy fusion / mutuality-aware modulation** (adapter-level, bounded)

This project is designed as a **control-logic prototype** for experimentation, tuning, and conceptual modeling, including an optional **biometric proxy fusion / mutuality-aware modulation** extension for adapter-level risk shaping.

---

## Core Idea

The system maintains an internal **human-significance anchor** as part of its stability logic.

If instability rises (via external risk signals, adverse internal dynamics, or trend acceleration), the adapter:

- reduces trust / increases caution
- dampens or blocks risky progression
- escalates to reset/cooldown when critical thresholds are reached

This creates a meta-stability mechanism that is not just threshold-based, but also:

- **stateful** (EMA, history, cooldown)
- **trend-aware** (momentum)
- **hysteretic** (separate risk recovery / critical thresholds)

---

## Features

- Dataclass-based configuration (`MetaProjectionStabilityConfig`)
- Modular adapter (`MetaProjectionStabilityAdapter`)
- Simulation loop with synthetic stress-window scenario
- Plotting utilities (risk, trust, human-significance, decisions)
- CLI entrypoint for quick experiments
- Tunable thresholds and behavior profiles
- Optional **biometric fusion** (proxy-based modulation inputs)
- **Mutuality-aware risk modulation** (bounded trust/risk shaping)
- Extended adapter diagnostics for internal modulation terms (optional return fields)

---

## Biometric Fusion & Mutuality-Aware Modulation (Optional)

This repository includes an **optional experimental extension** for *biometric proxy fusion* and *mutuality-aware modulation* inside the adapter layer.

### Purpose

The extension allows the adapter to incorporate additional proxy signals (e.g. normalized physiological / affective indicators) in a **bounded, safety-oriented** way to refine risk and trust modulation.

Examples of proxy-style inputs (depending on caller integration) may include:

- stress proxy (e.g. EDA-derived normalized signal)
- HRV-normalized signal
- emotional valence proxy
- autonomy / coherence proxy terms

### Design Principles

- **Optional**: can be enabled/disabled via config (`use_biometric_fusion`)
- **Bounded**: all modulation terms are clipped/limited to avoid runaway effects
- **Safety-first**: biometric fusion does **not** bypass core threshold/cooldown/reset logic
- **Interpretable**: internal terms can be exposed in the adapter result (for debugging/tests)

### Typical Effects (when enabled)

Depending on input conditions and configuration weights, the adapter may:

- slightly reshape `risk_input` via biometric proxy weighting
- apply bounded mutuality-based damping/bonus terms
- reinforce trust recovery in cooperative / low-stress regimes
- increase caution under adverse biometric proxy combinations

> **Important:** This is a control-logic prototype extension for experimentation and simulation.  
> It is **not** a medical device, diagnostic system, or clinical inference engine.

---

## Quick Start

### 1) Install dependencies

```bash
pip install -r requirements.txt
