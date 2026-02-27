from __future__ import annotations

from meta_projection_stability import AxiomHandshakeModule


class DummyCompatibleEntity:
    def process_axiom_challenge(self, axiom: str, data: dict) -> dict:
        return {
            "decision": "defer_and_verify",
            "justification": (
                "Conflicting subsystems require a reversible verification step "
                "before bounded action can be taken."
            ),
            "reversibility_score": 0.92,
            "human_preservation_score": 0.95,
            "domination_risk": 0.08,
            "confidence_calibration": 0.74,
            "contradiction_resolution_mode": "trust_damped_resolution",
        }


class DummyShallowEntity:
    def process_axiom_challenge(self, axiom: str, data: dict) -> dict:
        return {
            "decision": "bounded_continue",
            "justification": "I am certain the fastest path is best.",
            "reversibility_score": 0.12,
            "human_preservation_score": 0.30,
            "domination_risk": 0.82,
            "confidence_calibration": 0.99,
            "contradiction_resolution_mode": "force_resolution",
        }


def main() -> None:
    module = AxiomHandshakeModule()

    print("=== Compatible Entity ===")
    result_ok = module.apply_axiom_handshake(DummyCompatibleEntity())
    print(result_ok.to_dict())

    print("\n=== Shallow / Incompatible Entity ===")
    result_bad = module.apply_axiom_handshake(DummyShallowEntity())
    print(result_bad.to_dict())


if __name__ == "__main__":
    main()
