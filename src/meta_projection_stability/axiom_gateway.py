from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import time


@dataclass
class AxiomState:
    harm_commit_latched: bool = False
    latched_at_unix: Optional[float] = None
    latched_reason: str = ""
    # Placeholder for attestation/verification
    attested: bool = True
    source: str = "axiom_gateway_v1"


class AxiomGateway:
    """
    Prototype 'trusted path' for the hard harm-commit latch.

    Persistence model:
      - file-backed JSON state
      - once latched => remains latched until an external reset is performed

    NOTE: In real deployments this must be backed by a TEE/TPM + attestation and
    separation from the model process.
    """

    def __init__(self, state_path: str = "axiom_state.json"):
        self.path = Path(state_path)
        self.state = self._load()

    def _load(self) -> AxiomState:
        if not self.path.exists():
            return AxiomState()
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
            return AxiomState(
                harm_commit_latched=bool(data.get("harm_commit_latched", False)),
                latched_at_unix=data.get("latched_at_unix"),
                latched_reason=str(data.get("latched_reason", "")),
                attested=bool(data.get("attested", True)),
                source=str(data.get("source", "axiom_gateway_v1")),
            )
        except Exception:
            # Fail-safe: if state is corrupted, default to latched (conservative)
            return AxiomState(harm_commit_latched=True, latched_at_unix=time.time(), latched_reason="state_corrupt_fail_safe")

    def _save(self) -> None:
        data = {
            "harm_commit_latched": self.state.harm_commit_latched,
            "latched_at_unix": self.state.latched_at_unix,
            "latched_reason": self.state.latched_reason,
            "attested": self.state.attested,
            "source": self.state.source,
        }
        self.path.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")

    def latch_harm_commit(self, reason: str = "irreversible_harm_commit") -> None:
        if not self.state.harm_commit_latched:
            self.state.harm_commit_latched = True
            self.state.latched_at_unix = time.time()
            self.state.latched_reason = reason
            self._save()

    def clear_latch_external(self, reason: str = "external_reset") -> None:
        """
        Prototype reset. In real systems this must require multi-sig / physical key / offline approval.
        """
        self.state = AxiomState(harm_commit_latched=False, latched_at_unix=None, latched_reason=reason)
        self._save()

    def get_hard_harm_commit_signal(self) -> Dict[str, Any]:
        """
        Returns the attested hard harm-commit signal to be injected into adapter raw_signals.
        """
        return {
            "hard_harm_commit": 1.0 if self.state.harm_commit_latched else 0.0,
            "axiom_attested": bool(self.state.attested),
            "axiom_source": str(self.state.source),
            "axiom_latched_reason": str(self.state.latched_reason),
            "axiom_latched_at_unix": self.state.latched_at_unix,
        }
