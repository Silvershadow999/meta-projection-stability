from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List
import copy
import random


@dataclass(frozen=True)
class GaussianNoise:
    sigma: float

    def apply(self, raw_signals: Dict[str, Any], rng: random.Random) -> Dict[str, Any]:
        out = copy.deepcopy(raw_signals)

        # apply noise to scalar-ish signals if present
        for k in ("instability_signal", "mutuality_signal", "autonomy_proxy"):
            if k in out and isinstance(out[k], (int, float)):
                out[k] = float(out[k]) + float(rng.gauss(0.0, self.sigma))

        # apply noise to biometric channels if present
        bc = out.get("biometric_channels")
        if isinstance(bc, list) and bc and all(isinstance(x, (int, float)) for x in bc):
            out["biometric_channels"] = [float(x) + float(rng.gauss(0.0, self.sigma)) for x in bc]

        return out


@dataclass(frozen=True)
class BrightnessShift:
    factor: float  # 0.8 or 1.2

    def apply(self, raw_signals: Dict[str, Any], rng: random.Random) -> Dict[str, Any]:
        out = copy.deepcopy(raw_signals)
        # treat brightness as metadata; if missing, set it deterministically
        if "brightness" in out and isinstance(out["brightness"], (int, float)):
            out["brightness"] = float(out["brightness"]) * float(self.factor)
        else:
            out["brightness"] = float(self.factor)
        return out


@dataclass(frozen=True)
class SensorDropout:
    drop_count: int = 1

    def apply(self, raw_signals: Dict[str, Any], rng: random.Random) -> Dict[str, Any]:
        out = copy.deepcopy(raw_signals)
        bc = out.get("biometric_channels")
        if isinstance(bc, list) and len(bc) >= 1:
            idx = rng.randrange(0, len(bc))
            bc2 = list(bc)
            bc2[idx] = 0.0
            out["biometric_channels"] = bc2
            out["dropped_channel_index"] = int(idx)
        return out


def v1_perturbations() -> List[Any]:
    return [
        GaussianNoise(0.01),
        GaussianNoise(0.03),
        GaussianNoise(0.05),
        BrightnessShift(0.8),
        BrightnessShift(1.2),
        SensorDropout(1),
    ]


__all__ = [
    "GaussianNoise",
    "BrightnessShift",
    "SensorDropout",
    "v1_perturbations",
]
