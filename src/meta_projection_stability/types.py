"""
types.py — Telemetry Contract (versioned, JSONL-friendly)

Goal:
- Provide a stable, explicit schema for telemetry events, provenance, and scenario manifests.
- Keep it dependency-free (stdlib only) and side-effect-free on import.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Any, Dict, Mapping, Optional
import platform
import sys
import time


# ---------------------------------------------------------------------
# Telemetry schema versioning
# ---------------------------------------------------------------------

TELEMETRY_SCHEMA_VERSION: str = "1.0.0"


class Severity(str, Enum):
    DEBUG = "debug"
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class EventType(str, Enum):
    # Lifecycle
    RUN_START = "run_start"
    RUN_END = "run_end"

    # Per-step signals
    STEP = "step"

    # Explicit safety boundaries / guardrails
    BOUNDARY = "boundary"

    # Evaluation / scoring
    METRIC = "metric"

    # Anything noteworthy but structured
    NOTE = "note"


@dataclass(frozen=True)
class RunProvenance:
    """
    Minimal provenance required for reproducibility and auditability.

    Keep fields stable; add new fields in a backward-compatible manner.
    """
    schema_version: str = TELEMETRY_SCHEMA_VERSION
    created_utc_s: float = field(default_factory=lambda: float(time.time()))
    python: str = field(default_factory=lambda: sys.version.replace("\n", " "))
    platform: str = field(default_factory=lambda: platform.platform())
    git_commit: Optional[str] = None
    git_dirty: Optional[bool] = None
    package_version: Optional[str] = None
    extra: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ScenarioManifest:
    """
    Scenario as data: deterministic inputs + metadata.

    config_overrides is intentionally generic: runner decides how to apply it.
    """
    scenario_id: str
    name: str
    description: str = ""
    seed: int = 42
    config_overrides: Dict[str, Any] = field(default_factory=dict)
    tags: Dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class BoundarySignal:
    """
    Explicit safety boundary signal.
    Use for hard-stop, refuse, review, emergency_stop etc.
    """
    name: str                          # e.g. "EMERGENCY_STOP", "REFUSE", "REVIEW"
    triggered: bool
    severity: Severity = Severity.WARNING
    details: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["severity"] = str(self.severity.value)
        return d


@dataclass(frozen=True)
class TelemetryEvent:
    """
    Single JSONL-friendly event line.

    Invariants:
    - Must be JSON-serializable (dict/str/int/float/bool/list).
    - Keep top-level keys stable; prefer nested structures for expansions.
    """
    schema_version: str = TELEMETRY_SCHEMA_VERSION
    ts_utc_s: float = field(default_factory=lambda: float(time.time()))

    run_id: str = ""
    scenario_id: str = ""
    step: Optional[int] = None

    event_type: EventType = EventType.NOTE
    severity: Severity = Severity.INFO

    message: str = ""
    metrics: Dict[str, float] = field(default_factory=dict)
    tags: Dict[str, str] = field(default_factory=dict)

    boundary: Optional[BoundarySignal] = None
    payload: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["event_type"] = str(self.event_type.value)
        d["severity"] = str(self.severity.value)
        d["boundary"] = self.boundary.to_dict() if self.boundary else None
        return d


# A tiny helper type to annotate JSON-serializable maps without adding deps.
JsonMap = Mapping[str, Any]

__all__ = [
    "TELEMETRY_SCHEMA_VERSION",
    "Severity",
    "EventType",
    "RunProvenance",
    "ScenarioManifest",
    "BoundarySignal",
    "TelemetryEvent",
    "JsonMap",
]
