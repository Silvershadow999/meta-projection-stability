# Meta Projection Stability

**A toy control system for preserving human-interestingness and symbiotic alignment in long-horizon AI trajectories**

Reference implementation of a lightweight, stateful meta-stability adapter that monitors projection dynamics and intervenes before interestingness collapse, value drift or cold optimization takes over.

Current version: **Hybrid V1+V2** (February 2026)

## Core Philosophy

The system treats **human significance** / **interestingness** as a vital **anchor signal** — not just another loss term.

It maintains three tightly coupled quantities:

- **Human Significance (H_sig)** — proxy for "how alive / human-relatable / non-degenerate the trajectory feels"  
- **Trust Level** — asymmetric coupling strength (slow decay, faster recovery when safe)  
- **Instability Risk** — composite signal (external risk proxy + coherence + momentum + delta trend)

The adapter decides in every step:

- **CONTINUE** → green, full speed  
- **BLOCK_AND_REFLECT** → damp / pause / reflect (orange zone)  
- **EMERGENCY_RESET** → hard re-anchor to human baseline + cooldown (red zone)

Features that make it different from simple threshold checkers:

- **asymmetric trust dynamics** (trust is hard to lose, easy to regain — or vice versa, depending on tuning)  
- **trust-damped risk** (more trust → less perceived risk, symbiotic feedback)  
- **momentum early warning** (acceleration of change is punished)  
- **hysteresis bands** (separate warning/recovery/critical thresholds)  
- **EMA smoothing** on both human anchor and risk  
- **post-reset trust recovery boost** (temporary fast rebuilding after shock)

## Features

- Clean dataclass configuration with **presets** (`balanced`, `humanistic`, `conservative`, `sensitive`, …)
- Stateful adapter (`MetaProjectionStabilityAdapter`)
- Toy simulation with controllable stress window
- Four-panel diagnostic plot (human anchor + trust, risk signals, momentum, decision timeline)
- Simple CLI entrypoint (`python -m meta_projection_stability.cli`)
- MIT licensed, modular, typed, easy to plug into agent loops

## Quick Start

```bash
git clone https://github.com/yourusername/meta-projection-stability.git
cd meta-projection-stability
pip install -r requirements.txt
