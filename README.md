# Meta Projection Stability

A simulation-oriented reference implementation for **meta-stability control** with:

- **Human-significance anchoring** (interestingness preservation)
- **Trust dynamics** (human ↔ system coupling)
- **Trust-damped risk regulation**
- **Momentum-based early warning**
- **Hysteresis / Schmitt-trigger behavior**
- **Cooldown after emergency reset**

This project is designed as a **control-logic prototype** for experimentation, tuning, and conceptual modeling.

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

---

## Repository Structure

```text
meta-projection-stability/
├─ README.md
├─ requirements.txt
├─ .gitignore
├─ LICENSE
├─ src/
│  └─ meta_projection_stability/
│     ├─ __init__.py
│     ├─ config.py
│     ├─ adapter.py
│     ├─ simulation.py
│     ├─ plotting.py
│     └─ cli.py
└─ examples/
   └─ run_balanced_humanistic.py
