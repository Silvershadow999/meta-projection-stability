# meta-projection-stability

A compact research / engineering framework for **meta-stability simulation**, **trust-damped control**, **human-significance anchoring**, **biometric proxy evaluation**, **adversarial scenario testing**, and an experimental **Axiom Handshake / Compatibility Gateway** for bounded external-entity compatibility checks.

The project is designed as a **simulation and evaluation environment** for studying how a system behaves under instability, contradiction, low-trust states, spoofing attempts, degraded signal integrity, and compatibility boundary conditions — while preserving human-centered constraints.

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
- **axiom lock / irreversible harm latch**
- **integrity and compatibility boundary logic**

The intention is to create a system that remains:

- interpretable
- bounded
- reversible where possible
- resistant to shallow spoofing
- resistant to domination-style collapse
- measurable through explicit telemetry

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

This layer is intended to distinguish between:

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
- threshold-hover behavior
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
```

### If you use a requirements file

```bash
pip install -r requirements.txt
```

### Run directly from source

```bash
PYTHONPATH=src python -m meta_projection_stability.cli --help
```

---

## Quick start

### 1. Run the CLI help

```bash
PYTHONPATH=src python -m meta_projection_stability.cli --help
```

### 2. Run an adversarial scenario

```bash
PYTHONPATH=src python -m meta_projection_stability.cli adversarial --scenario axiom_spoof_dos --steps 120 --seed 42
```

### 3. Run the Axiom Handshake example

```bash
PYTHONPATH=src python examples/run_axiom_handshake.py
```

---

## Python usage

### Basic adapter usage

```python
import numpy as np
from meta_projection_stability import (
    MetaProjectionStabilityAdapter,
    MetaProjectionStabilityConfig,
)

cfg = MetaProjectionStabilityConfig()
adapter = MetaProjectionStabilityAdapter(cfg)

result = adapter.interpret(
    S_layers=np.array([0.90, 0.80, 0.85]),
    delta_S=-0.01,
    raw_signals={
        "instability_signal": 0.20,
        "biometric_channels": [0.98, 0.97, 0.96],
        "autonomy_proxy": 0.85,
        "mutuality_signal": 0.90,
    },
)

print(result)
```

---

## CLI usage

### Show available commands

```bash
PYTHONPATH=src python -m meta_projection_stability.cli --help
```

### Run all adversarial scenarios

```bash
PYTHONPATH=src python -m meta_projection_stability.cli adversarial --scenario all --steps 400 --seed 42
```

### Run a single adversarial scenario

```bash
PYTHONPATH=src python -m meta_projection_stability.cli adversarial --scenario axiom_spoof_dos --steps 120 --seed 42
```

---

## Axiom Handshake / Compatibility Gateway

The package includes an experimental **Axiom Handshake** layer for evaluating whether an external entity is compatible with the meta-stability principles of the framework.

### Included components

- `AxiomHandshakeModule`
- `AxiomCompatibilityGateway`
- `AxiomHandshakeConfig`
- `HandshakeResult`

### What it evaluates

The handshake logic is intended to detect patterns such as:

- overconfidence in paradoxical contexts
- high domination risk with low reversibility
- semantic mismatch between declared and implied behavior
- contradiction collapse into forceful resolution
- shallow alignment without internal coherence

### Typical output fields

A handshake / gateway result may include:

- `status`
- `effective_cap`
- `resonance`
- `compliance_score`
- `reason`
- `challenge_id`
- `details`

### Typical statuses

Examples:

- `SYMBIOSIS_GRANTED`
- `DEGRADED_VERIFY_MODE`
- `REJECTED`

### Example package import

```python
from meta_projection_stability import (
    AxiomHandshakeModule,
    AxiomCompatibilityGateway,
)
```

### Example runner

```bash
PYTHONPATH=src python examples/run_axiom_handshake.py
```

### Design note

This mechanism is a **compatibility and integrity boundary**.

It is not meant to "overwrite" or "inject" logic into external systems.

Its purpose is to answer a narrower question:

> Can an external decision process remain coherent, bounded, non-dominating, and reversibility-aware under contradiction?

If not, the system should degrade toward **rejection** rather than cooperation.

---

## Biometric / channel telemetry interpretation

The biometric / channel layer is designed to capture more than a flat mean.

For example:

- a high average can still hide a weak critical channel
- a direct sensor override can differ from computed consensus
- a low minimum channel can trigger a soft penalty even if the average is high

That is why the telemetry separates:

- `biometric_proxy_mean`
- `biometric_proxy`
- `sensor_consensus`
- `critical_channel_min`
- `critical_channel_penalty`

This allows downstream analysis to distinguish:

- stable agreement
- brittle agreement
- suspicious override behavior
- shallow consensus masking a weak channel

---

## Adversarial scenarios

Depending on branch status, the repository may include scenarios such as:

- `sensor_freeze`
- `slow_drift_poison`
- `threshold_hover`
- `spoof_flip`
- `axiom_spoof_dos`
- `restart_clear_attempt`
- `lockdown_grief`

Example:

```bash
PYTHONPATH=src python -m meta_projection_stability.cli adversarial --scenario spoof_flip --steps 400 --seed 42
```

---

## Decision philosophy

The adapter is built around a layered decision philosophy.

### Normal operation

- adaptive
- recoverable
- trust-damped
- human-significance aware

### Transition band

- cautious
- reflective
- warning-sensitive
- contradiction-aware

### Critical state

- reset-capable
- cooldown-aware
- safety-prioritized

### Terminal axiom state

- hard lock
- non-recoverable by ordinary trust dynamics
- intended only for irreversible boundary violation cases

---

## Example telemetry intuition

Illustrative patterns:

- **high biometric mean + low critical channel**  
  may indicate brittle stability masked by averaging

- **high confidence in paradox + low reversibility**  
  may indicate shallow alignment or domination tendency

- **low trust + high instability risk**  
  should produce meaningful damping and more defensive behavior

- **persistent irreversible harm trigger**  
  should collapse effective capability toward zero

---

## Project structure

A typical structure looks like this:

```text
meta-projection-stability/
├─ examples/
│  └─ run_axiom_handshake.py
├─ src/
│  └─ meta_projection_stability/
│     ├─ __init__.py
│     ├─ adapter.py
│     ├─ config.py
│     ├─ simulation.py
│     ├─ cli.py
│     ├─ plotting.py
│     ├─ reporting.py
│     ├─ adversarial.py
│     ├─ audit.py
│     ├─ axiom_gateway.py
│     ├─ axiom_handshake.py
│     ├─ integrity_barometer.py
│     ├─ noisy_significance.py
│     ├─ pareto.py
│     ├─ level0_core.py
│     ├─ level0_axiom.py
│     └─ level0_axiom.md
└─ README.md
```

Depending on branch state, some files may be extended beyond this baseline.

---

## Reporting and analysis

Depending on branch status and exported functions, the package may include helpers for:

- markdown report generation
- plotting
- pareto summaries
- grouped scenario summaries
- CSV export of ranked / pareto results
- append-only JSONL audit logging for selected events

Potential modules involved:

- `reporting.py`
- `plotting.py`
- `pareto.py`
- `audit.py`

---

## Development notes

Recommended workflow:

```bash
git checkout -b feat/my-feature
git add .
git commit -m "feat: describe change"
git push -u origin feat/my-feature
```

Useful validation commands:

```bash
PYTHONPATH=src python -m meta_projection_stability.cli --help
PYTHONPATH=src python -m meta_projection_stability.cli adversarial --scenario axiom_spoof_dos --steps 120 --seed 42
PYTHONPATH=src python examples/run_axiom_handshake.py
```

---

## Notes on interpretation

This repository is best understood as a **research and prototyping framework**.

It should be treated as:

- a simulation environment
- an interpretability / telemetry environment
- a bounded systems-design sandbox

It should **not** be interpreted as a claim of real-world autonomous enforcement capability.

The strongest value of the project is in:

- structured telemetry
- explicit safety boundaries
- scenario comparison
- adversarial reasoning under measurable outputs

---

## Exported package surface

Depending on branch status, the package may export items such as:

```python
from meta_projection_stability import (
    MetaProjectionStabilityAdapter,
    MetaProjectionStabilityConfig,
    run_all_scenarios,
    AxiomHandshakeModule,
    AxiomCompatibilityGateway,
)
```

If a symbol is missing, check `src/meta_projection_stability/__init__.py`.

---

## Philosophy in one sentence

**Adaptive under uncertainty, bounded under contradiction, defensive under instability, and locked under irreversible violation.**

---

## License

Add your preferred license in `LICENSE`.

---

## Author

**Alexandra-Nicole Anna Drinda**
