#!/usr/bin/env bash
set -euo pipefail

echo "== MetaProjectionStabilityAdapter / interpret() Check =="

python - <<'PY'
import inspect
import numpy as np

from meta_projection_stability.config import MetaProjectionStabilityConfig
from meta_projection_stability.adapter import MetaProjectionStabilityAdapter

cfg = MetaProjectionStabilityConfig()
adapter = MetaProjectionStabilityAdapter(cfg)

print("\n[1] interpret()-Signatur:")
print(inspect.signature(adapter.interpret))

# Beispielwerte (sicher / minimal)
S_layers = np.array([0.20, 0.30, 0.25], dtype=float)
delta_S = 0.01

# raw_signals als DICT (wichtig, damit .get(...) funktioniert)
raw_signals = {
    "instability_signal": 0.35,
    "eda_stress_score": 0.40,
    "hrv_normalized": 0.62,
    "emotional_valence": 0.55,
    "autonomy_signal": 0.70,
    "gamma_coherence": 0.58,
}

# Wir bauen den Call signatur-sicher dynamisch zusammen:
sig = inspect.signature(adapter.interpret)
params = sig.parameters

kwargs = {}

# Pflicht/übliche Parameter dynamisch setzen
if "S_layers" in params:
    kwargs["S_layers"] = S_layers
if "delta_S" in params:
    kwargs["delta_S"] = delta_S
if "raw_signals" in params:
    kwargs["raw_signals"] = raw_signals

# Optional bekannte Parameter nur setzen, wenn vorhanden
optional_defaults = {
    "trust_level": 0.72,
    "human_significance": 0.66,
    "time_index": 0,
    "t": 0,
}

for k, v in optional_defaults.items():
    if k in params:
        kwargs[k] = v

print("\n[2] Aufruf mit kwargs:")
for k, v in kwargs.items():
    print(f"  - {k}: {v}")

res = adapter.interpret(**kwargs)

print("\n[3] Rückgabetyp:")
print(type(res).__name__)

if isinstance(res, dict):
    print("\n[4] Rückgabe-Keys:")
    for k in sorted(res.keys()):
        print("  -", k)

    check_keys = [
        "biometric_proxy",
        "biometric_risk_component",
        "autonomy_proxy",
        "gamma_coherence_proxy",
        "mutuality_bonus",
        "trust_reinforcement",
        "cooldown_remaining",
    ]

    print("\n[5] Erwartete neue Felder:")
    missing = []
    for k in check_keys:
        if k in res:
            print(f"  ✅ {k}: {res.get(k)}")
        else:
            print(f"  ❌ {k}: FEHLT")
            missing.append(k)

    if missing:
        print("\nWARNUNG: Einige erwartete Felder fehlen:", missing)
    else:
        print("\nOK: Alle erwarteten Felder vorhanden.")
else:
    print("\nWARNUNG: interpret() liefert kein dict zurück; Tests müssen ggf. angepasst werden.")
PY
