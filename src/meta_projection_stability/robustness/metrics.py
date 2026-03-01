from __future__ import annotations

import math
from typing import Sequence


def p95(values: Sequence[float]) -> float:
    vals = sorted(float(v) for v in values)
    if not vals:
        return 0.0
    k = int(math.ceil(0.95 * len(vals))) - 1
    k = max(0, min(k, len(vals) - 1))
    return float(vals[k])


def decision_flip_rate(base: Sequence[str], pert: Sequence[str]) -> float:
    n = min(len(base), len(pert))
    if n == 0:
        return 0.0
    flips = sum(1 for i in range(n) if base[i] != pert[i])
    return float(flips) / float(n)


def status_flip_rate(base: Sequence[str], pert: Sequence[str]) -> float:
    n = min(len(base), len(pert))
    if n == 0:
        return 0.0
    flips = sum(1 for i in range(n) if base[i] != pert[i])
    return float(flips) / float(n)


def delta_p95(base: Sequence[float], pert: Sequence[float]) -> float:
    n = min(len(base), len(pert))
    if n == 0:
        return 0.0
    deltas = [abs(float(base[i]) - float(pert[i])) for i in range(n)]
    return p95(deltas)


def biometric_consensus_drop_p95(base: Sequence[float], pert: Sequence[float]) -> float:
    n = min(len(base), len(pert))
    if n == 0:
        return 0.0
    drops = [max(0.0, float(base[i]) - float(pert[i])) for i in range(n)]
    return p95(drops)


def axiom_lock_rate(flags: Sequence[bool]) -> float:
    if not flags:
        return 0.0
    return float(sum(1 for x in flags if bool(x))) / float(len(flags))


__all__ = [
    "p95",
    "decision_flip_rate",
    "status_flip_rate",
    "delta_p95",
    "biometric_consensus_drop_p95",
    "axiom_lock_rate",
]
