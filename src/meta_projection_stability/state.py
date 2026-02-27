"""
state.py — Minimal, serializable run state (Phase 5)

Goals:
- JSON-serializable snapshot for reproducibility + regression.
- Explicit boundary signals (telemetry contract).
- No side effects on import; stdlib only + local types.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any, Dict
import time
import uuid

from .types import BoundarySignal, Severity


def new_run_id(prefix: str = "run") -> str:
    return f"{prefix}_{uuid.uuid4().hex[:12]}"


def _parse_severity(value: Any) -> Severity:
    """
    Tolerant parsing:
    - accepts Severity enum
    - accepts strings like "warning"
    - falls back to WARNING
    """
    if isinstance(value, Severity):
        return value
    if isinstance(value, str):
        v = value.strip().lower()
        for s in Severity:
            if s.value == v:
                return s
    return Severity.WARNING


@dataclass
class RunState:
    """
    Canonical state for a single run.

    Keep fields JSON-serializable. Additive changes only.
    """
    run_id: str = field(default_factory=new_run_id)
    scenario_id: str = ""
    step: int = 0

    created_utc_s: float = field(default_factory=lambda: float(time.time()))
    last_update_utc_s: float = field(default_factory=lambda: float(time.time()))

    # Explicit safety boundaries (latest snapshot)
    boundaries: Dict[str, BoundarySignal] = field(default_factory=dict)

    # Small scalar signals (JSON-safe)
    scalars: Dict[str, float] = field(default_factory=dict)

    # Free-form JSON-safe metadata
    meta: Dict[str, Any] = field(default_factory=dict)

    def touch(self) -> None:
        self.last_update_utc_s = float(time.time())

    def set_boundary(self, signal: BoundarySignal) -> None:
        self.boundaries[signal.name] = signal
        self.touch()

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["boundaries"] = {k: v.to_dict() for k, v in self.boundaries.items()}
        return d

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "RunState":
        boundaries_raw = data.get("boundaries") or {}
        boundaries: Dict[str, BoundarySignal] = {}

        for name, b in boundaries_raw.items():
            if isinstance(b, BoundarySignal):
                boundaries[name] = b
                continue
            b = b or {}
            boundaries[name] = BoundarySignal(
                name=b.get("name", name),
                triggered=bool(b.get("triggered", False)),
                severity=_parse_severity(b.get("severity", "warning")),
                details=b.get("details", {}) or {},
            )

        return cls(
            run_id=data.get("run_id", new_run_id()),
            scenario_id=data.get("scenario_id", ""),
            step=int(data.get("step", 0)),
            created_utc_s=float(data.get("created_utc_s", time.time())),
            last_update_utc_s=float(data.get("last_update_utc_s", time.time())),
            boundaries=boundaries,
            scalars=data.get("scalars", {}) or {},
            meta=data.get("meta", {}) or {},
        )


__all__ = ["RunState", "new_run_id"]
