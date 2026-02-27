"""
state.py — Minimal, serializable run state

Design goals:
- JSON-serializable snapshot for reproducibility and regression.
- No side effects on import.
- Keep it independent from the simulation logic; runner can embed it.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any, Dict, Optional
import time
import uuid

from .types import BoundarySignal


def new_run_id(prefix: str = "run") -> str:
    # Stable-ish, human-friendly run id
    return f"{prefix}_{uuid.uuid4().hex[:12]}"


@dataclass
class RunState:
    """
    Canonical state for a single run.

    Notes:
    - Keep fields JSON-serializable.
    - Additive changes only (backward compatible).
    """
    run_id: str = field(default_factory=new_run_id)
    scenario_id: str = ""
    step: int = 0

    created_utc_s: float = field(default_factory=lambda: float(time.time()))
    last_update_utc_s: float = field(default_factory=lambda: float(time.time()))

    # Explicit safety boundary signals (latest snapshot)
    boundaries: Dict[str, BoundarySignal] = field(default_factory=dict)

    # Optional scratchpad for small scalar values (must stay JSON-serializable!)
    scalars: Dict[str, float] = field(default_factory=dict)

    # Optional free-form info (JSON-serializable only)
    meta: Dict[str, Any] = field(default_factory=dict)

    def touch(self) -> None:
        self.last_update_utc_s = float(time.time())

    def set_boundary(self, signal: BoundarySignal) -> None:
        self.boundaries[signal.name] = signal
        self.touch()

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        # dataclass -> dict turns BoundarySignal into dict already, but we ensure shape:
        d["boundaries"] = {k: v.to_dict() for k, v in self.boundaries.items()}
        return d

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "RunState":
        boundaries_raw = data.get("boundaries") or {}
        boundaries: Dict[str, BoundarySignal] = {}
        for name, b in boundaries_raw.items():
            if isinstance(b, BoundarySignal):
                boundaries[name] = b
            else:
                boundaries[name] = BoundarySignal(
                    name=b.get("name", name),
                    triggered=bool(b.get("triggered", False)),
                    severity=b.get("severity", "warning"),
                    details=b.get("details", {}) or {},
                )
        obj = cls(
            run_id=data.get("run_id", new_run_id()),
            scenario_id=data.get("scenario_id", ""),
            step=int(data.get("step", 0)),
            created_utc_s=float(data.get("created_utc_s", time.time())),
            last_update_utc_s=float(data.get("last_update_utc_s", time.time())),
            boundaries=boundaries,
            scalars=data.get("scalars", {}) or {},
            meta=data.get("meta", {}) or {},
        )
        return obj


__all__ = ["RunState", "new_run_id"]
