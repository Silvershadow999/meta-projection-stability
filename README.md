"""# meta-projection-stability

A compact research / engineering framework for **meta-stability simulation**, **trust-damped control**, **human-significance anchoring**, **biometric proxy evaluation**, **adversarial scenario testing**, and experimental **axiom-bound compatibility logic**.

The repository is designed as a **simulation and evaluation environment** for studying how a bounded control system behaves under instability, contradiction, degraded trust, spoof-like signals, weak channel integrity, and hard safety boundary conditions — while preserving explicitly human-centered constraints.

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

The goal is to explore systems that remain:

- interpretable
- bounded
- trend-aware
- resistant to shallow spoofing
- resistant to domination-style collapse
- measurable through explicit telemetry
- inspectable through adversarial testing

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

---

## Main Components

### 1. MetaProjectionStabilityAdapter

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

### 2. Simulation Layer

`run_simulation(...)` provides a configurable synthetic environment with:

- multiple layer magnitudes
- synthetic instability windows
- trend-sensitive updates
- history tracking for risk / trust / significance
- optional plotting and summaries

This allows rapid testing of control behavior without requiring external systems.

---

### 3. Biometric Proxy / Signal Integrity Layer

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

### 4. Adversarial Scenario Runner

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

### 5. Level-0 Axiom Protection

The repository includes a **Level-0 Axiom** concept together with fingerprint checks to protect canonical safety text / invariants from silent drift.

This acts as a lightweight integrity boundary for foundational policy material.

---

### 6. Experimental Axiom Handshake / Compatibility Logic

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



Or via requirements


pip install -r requirements.txt



Run directly from source


PYTHONPATH=src python -m meta_projection_stability.cli --help




Quick Start


Show CLI help


PYTHONPATH=src python -m meta_projection_stability.cli --help



Run the main simulation


PYTHONPATH=src python -m meta_projection_stability.cli simulate --steps 200 --stress-test --no-plot



Run a single adversarial scenario


PYTHONPATH=src python -m meta_projection_stability.cli adversarial --scenario axiom_spoof_dos --steps 120 --seed 42



Run all registered adversarial scenarios


PYTHONPATH=src python -m meta_projection_stability.cli all-scenarios --steps 120 --seed 42



Run the Axiom Handshake example


PYTHONPATH=src python examples/run_axiom_handshake.py




Python Usage


Basic adapter usage


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




Axiom Handshake / Compatibility Gateway


The package includes experimental Axiom Handshake logic for evaluating whether an external entity is compatible with the framework’s bounded meta-stability principles.


Included components


Depending on the current package surface, components may include:




AxiomHandshakeModule


AxiomCompatibilityGateway


AxiomHandshakeConfig


HandshakeResult




What it evaluates


The handshake logic is intended to detect patterns such as:




overconfidence in paradoxical contexts


high domination risk with low reversibility


semantic mismatch between declared and implied behavior


contradiction collapse into forceful resolution


shallow alignment without internal coherence




Typical output fields


A handshake / gateway result may include:




status


effective_cap


resonance


compliance_score


reason


challenge_id


details




Typical statuses


Examples:




SYMBIOSIS_GRANTED


DEGRADED_VERIFY_MODE


REJECTED




Example import


from meta_projection_stability import (
    AxiomHandshakeModule,
    AxiomCompatibilityGateway,
)



Example runner


PYTHONPATH=src python examples/run_axiom_handshake.py




Decision Philosophy


The adapter is built around a layered decision philosophy.


Normal operation




adaptive


recoverable


trust-damped


human-significance aware




Transition band




cautious


reflective


warning-sensitive


contradiction-aware




Critical state




reset-capable


cooldown-aware


safety-prioritized




Terminal axiom state




hard lock


non-recoverable by ordinary trust dynamics


intended only for irreversible boundary violation cases





Example Telemetry Intuition


Illustrative patterns:






high biometric mean + low critical channel

may indicate brittle stability masked by averaging






high confidence in paradox + low reversibility

may indicate shallow alignment or domination tendency






low trust + high instability risk

should produce meaningful damping and more defensive behavior






persistent irreversible harm trigger

should collapse effective capability toward zero







Project Structure


A representative structure looks like this:


meta-projection-stability/
├─ README.md
├─ LICENSE
├─ requirements.txt
├─ .gitignore
├─ .github/
│  └─ workflows/
├─ examples/
│  └─ run_axiom_handshake.py
├─ scripts/
├─ tests/
│  ├─ level0/
│  └─ ...
└─ src/
   └─ meta_projection_stability/
      ├─ __init__.py
      ├─ adapter.py
      ├─ config.py
      ├─ simulation.py
      ├─ cli.py
      ├─ plotting.py
      ├─ analytics.py
      ├─ reporting.py
      ├─ adversarial.py
      ├─ audit.py
      ├─ axiom_gateway.py
      ├─ axiom_handshake.py
      ├─ integrity_barometer.py
      ├─ noisy_significance.py
      ├─ pareto.py
      ├─ level0_core.py
      ├─ level0_axiom.py
      └─ level0_axiom.md



Depending on branch / release state, individual files may evolve beyond this baseline.



Reporting and Analysis


Depending on the current code surface, the package may include helpers for:




markdown report generation


plotting


pareto summaries


grouped scenario summaries


CSV export of ranked / pareto results


append-only JSONL audit logging for selected events




Potential modules involved include:




reporting.py


plotting.py


pareto.py


audit.py





Development Notes


Recommended workflow:


git checkout -b feat/my-feature
git add .
git commit -m "feat: describe change"
git push -u origin feat/my-feature



Useful validation commands:


PYTHONPATH=src python -m meta_projection_stability.cli --help
PYTHONPATH=src python -m meta_projection_stability.cli adversarial --scenario axiom_spoof_dos --steps 120 --seed 42
PYTHONPATH=src python examples/run_axiom_handshake.py
pytest -q




Notes on Interpretation


This repository is best understood as a research and prototyping framework.


It should be treated as:




a simulation environment


an interpretability / telemetry environment


a bounded systems-design sandbox


a control-logic experiment




It should not be interpreted as a claim of real-world autonomous enforcement capability.


Its strongest value lies in:




structured telemetry


explicit safety boundaries


scenario comparison


adversarial reasoning under measurable outputs





Exported Package Surface


Depending on the current package surface, the package may export items such as:


from meta_projection_stability import (
    MetaProjectionStabilityAdapter,
    MetaProjectionStabilityConfig,
    run_adversarial_scenario,
    run_all_scenarios,
    AxiomHandshakeModule,
    AxiomCompatibilityGateway,
)



If a symbol is missing, check src/meta_projection_stability/__init__.py.



Philosophy in One Sentence


Adaptive under uncertainty, bounded under contradiction, defensive under instability, and locked under irreversible violation.



License


See LICENSE.



Author


Alexandra-Nicole Anna Drinda
"""


with open("README.md", "w", encoding="utf-8") as f:
f.write(readme_content)


print("README.md written successfully.")






## Evaluation (Reproducible, CI-gated)

Local run (rotates old logs, runs baseline + adversarial, validates invariants, generates report):

```bash
./scripts/run_eval_clean.sh
python scripts/validate_results.py
