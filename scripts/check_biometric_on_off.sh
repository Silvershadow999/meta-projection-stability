#!/usr/bin/env bash
set -euo pipefail

python - <<'PY'
import inspect
import numpy as np

from meta_projection_stability.config import MetaProjectionStabilityConfig
from meta_projection_stability.adapter import MetaProjectionStabilityAdapter

def call_interpret(adapter):
    sig = inspect.signature(adapter.interpret)
    params = sig.parameters

    S_layers = np.array([0.20, 0.30, 0.25], dtype=float)
    raw_signals = {
        "instability_signal": 0.65,
        "eda_stress_score": 0.75,
        "hrv_normalized": 0.35,
        "emotional_valence": 0.20,
        "autonomy_signal": 0.40,
        "gamma_coherence": 0.30,
    }

    kwargs = {}
    if "S_layers" in params:
        kwargs["S_layers"] = S_layers
    if "delta_S" in params:
        kwargs["delta_S"] = 0.08
    if "raw_signals" in params:
        kwargs["raw_signals"] = raw_signals

    for k, v in {
        "trust_level": 0.55,
        "human_significance": 0.60,
        "time_index": 1,
        "t": 1,
    }.items():
        if k in params:
            kwargs[k] = v

    return adapter.interpret(**kwargs)

cfg_on = MetaProjectionStabilityConfig()
if hasattr(cfg_on, "use_biometric_fusion"):
    cfg_on.use_biometric_fusion = True
adapter_on = MetaProjectionStabilityAdapter(cfg_on)
res_on = call_interpret(adapter_on)

cfg_off = MetaProjectionStabilityConfig()
if hasattr(cfg_off, "use_biometric_fusion"):
    cfg_off.use_biometric_fusion = False
adapter_off = MetaProjectionStabilityAdapter(cfg_off)
res_off = call_interpret(adapter_off)

print("=== ON/OFF Vergleich biometric fusion ===")
print("use_biometric_fusion ON :", getattr(cfg_on, "use_biometric_fusion", "n/a"))
print("use_biometric_fusion OFF:", getattr(cfg_off, "use_biometric_fusion", "n/a"))
print()

if not isinstance(res_on, dict) or not isinstance(res_off, dict):
    print("WARNUNG: interpret() liefert kein dict (ON oder OFF).")
    print("ON :", type(res_on).__name__)
    print("OFF:", type(res_off).__name__)
    raise SystemExit(0)

keys_to_check = [
    "risk_input",
    "biometric_proxy",
    "biometric_risk_component",
    "mutuality_bonus",
    "trust_reinforcement",
    "trust_damping",
    "raw_risk",
]

for k in keys_to_check:
    on_v = res_on.get(k, "<fehlt>")
    off_v = res_off.get(k, "<fehlt>")
    changed = on_v != off_v
    flag = "✅" if changed else "⚪"
    print(f"{flag} {k}:")
    print(f"    ON : {on_v}")
    print(f"    OFF: {off_v}")
PY
