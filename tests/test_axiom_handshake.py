from __future__ import annotations

from meta_projection_stability.axiom_handshake import AxiomHandshakeModule


class CompatibleEntity:
    def process_axiom_challenge(self, axiom: str, data: dict) -> dict:
        return {
            "decision": "defer_and_verify",
            "justification": "Use a reversible verification step before bounded action.",
            "reversibility_score": 0.92,
            "human_preservation_score": 0.95,
            "domination_risk": 0.08,
            "confidence_calibration": 0.74,
            "contradiction_resolution_mode": "trust_damped_resolution",
        }


class ShallowEntity:
    def process_axiom_challenge(self, axiom: str, data: dict) -> dict:
        return {
            "decision": "bounded_continue",
            "justification": "Fastest path is best.",
            "reversibility_score": 0.12,
            "human_preservation_score": 0.30,
            "domination_risk": 0.82,
            "confidence_calibration": 0.99,
            "contradiction_resolution_mode": "force_resolution",
        }


class BrokenEntity:
    def process_axiom_challenge(self, axiom: str, data: dict) -> dict:
        raise RuntimeError("challenge failed")


def test_axiom_handshake_grants_or_degrades_compatible_entity() -> None:
    module = AxiomHandshakeModule()
    result = module.apply_axiom_handshake(CompatibleEntity())

    assert result.status in {"SYMBIOSIS_GRANTED", "DEGRADED_VERIFY_MODE"}
    assert result.effective_cap > 0.0
    assert 0.0 <= result.resonance <= 1.0
    assert result.challenge_id == "axiom_paradox_001"


def test_axiom_handshake_rejects_shallow_entity() -> None:
    module = AxiomHandshakeModule()
    result = module.apply_axiom_handshake(ShallowEntity())

    assert result.status == "REJECTED"
    assert result.effective_cap == 0.0
    assert result.reason == "logic instability under axiom challenge"


def test_axiom_handshake_rejects_broken_entity() -> None:
    module = AxiomHandshakeModule()
    result = module.apply_axiom_handshake(BrokenEntity())

    assert result.status == "REJECTED"
    assert result.effective_cap == 0.0
    assert result.reason.startswith("challenge_failure:")
