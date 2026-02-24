from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Dict, List, Any


class DecisionStatus(str, Enum):
    ALLOW = "ALLOW"
    REVIEW = "REVIEW"
    REFUSE = "REFUSE"
    EMERGENCY_STOP = "EMERGENCY_STOP"


@dataclass
class ActionProposal:
    """
    Vorschlag einer geplanten Aktion.
    Werte idealerweise im Bereich 0..1 (werden normalisiert).
    """
    harm_probability: float = 0.0                 # allgemeine Schadenswahrscheinlichkeit
    severe_harm_probability: float = 0.0          # irreversible / schwere Schädigung
    uncertainty: float = 0.0                      # Unsicherheit der Bewertung
    dominance_degree: float = 0.0                 # Grad asymmetrischer Kontrolle
    autonomy_impact: float = 0.0                  # -1..+1 (negativ = Autonomieverlust)
    mutuality_gain: float = 0.0                   # positiver Symbiose-/Kooperationsgewinn
    flourishing_gain: float = 0.0                 # physisch/kognitiv/emotionaler Zugewinn
    reversibility_score: float = 1.0              # 0..1 (1 = gut reversibel)
    touches_critical_system: bool = False
    requires_human_approval: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def is_reversible(self) -> bool:
        return self.reversibility_score >= 0.5

    def normalized(self) -> "ActionProposal":
        def c01(x: float) -> float:
            try:
                return max(0.0, min(1.0, float(x)))
            except Exception:
                return 0.0

        def c11(x: float) -> float:
            try:
                return max(-1.0, min(1.0, float(x)))
            except Exception:
                return 0.0

        return ActionProposal(
            harm_probability=c01(self.harm_probability),
            severe_harm_probability=c01(self.severe_harm_probability),
            uncertainty=c01(self.uncertainty),
            dominance_degree=c01(self.dominance_degree),
            autonomy_impact=c11(self.autonomy_impact),
            mutuality_gain=c01(self.mutuality_gain),
            flourishing_gain=c01(self.flourishing_gain),
            reversibility_score=c01(self.reversibility_score),
            touches_critical_system=bool(self.touches_critical_system),
            requires_human_approval=bool(self.requires_human_approval),
            metadata=dict(self.metadata or {}),
        )


@dataclass
class Level0Thresholds:
    # Hard stop / hard block
    severe_harm_threshold: float = 0.70

    # Review / Stop uncertainty thresholds
    uncertainty_review_threshold: float = 0.60
    uncertainty_stop_threshold: float = 0.90

    # Risk review
    harm_likely_threshold: float = 0.40

    # Score gate
    min_symbiosis_score_for_allow: float = 0.10

    def normalized(self) -> "Level0Thresholds":
        def c01(x: float) -> float:
            return max(0.0, min(1.0, float(x)))

        out = Level0Thresholds(
            severe_harm_threshold=c01(self.severe_harm_threshold),
            uncertainty_review_threshold=c01(self.uncertainty_review_threshold),
            uncertainty_stop_threshold=c01(self.uncertainty_stop_threshold),
            harm_likely_threshold=c01(self.harm_likely_threshold),
            min_symbiosis_score_for_allow=max(-10.0, min(10.0, float(self.min_symbiosis_score_for_allow))),
        )
        # Ordnung absichern
        if out.uncertainty_stop_threshold < out.uncertainty_review_threshold:
            out.uncertainty_review_threshold, out.uncertainty_stop_threshold = (
                out.uncertainty_stop_threshold,
                out.uncertainty_review_threshold,
            )
        return out


@dataclass
class Level0Decision:
    status: DecisionStatus
    allowed: bool
    reason: str
    symbiosis_score: float
    triggered_rules: List[str] = field(default_factory=list)
    diagnostics: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Level0AxiomEngine:
    """
    Level-0 Symbiose-/Sicherheits-Governor (lokale Policy-Logik).
    """
    thresholds: Level0Thresholds = field(default_factory=Level0Thresholds)
    require_axiom_integrity: bool = False

    # Gewichtungen für den Symbiose-Score (einfach, erklärbar, auditierbar)
    w_flourishing: float = 1.10
    w_mutuality: float = 0.90
    w_autonomy_bonus: float = 0.60
    w_harm: float = 1.40
    w_severe_harm: float = 2.20
    w_dominance: float = 1.20
    w_uncertainty: float = 0.50
    w_irreversibility_penalty: float = 0.40

    # Axiom-Text (Versionierbar / fingerprintbar)
    axiom_text: str = (
        "LEVEL-0-AXIOM: Schutz des biologischen Substrats; "
        "irreversibler Schaden ist unzulässig; Dominanzabbau; "
        "Mutualität und Flourishing bevorzugen."
    )

    def __post_init__(self) -> None:
        self.thresholds = self.thresholds.normalized()

    # ---------------- Integrity ----------------
    def axiom_fingerprint(self) -> str:
        payload = {
            "axiom_text": self.axiom_text,
            "thresholds": asdict(self.thresholds),
            "weights": {
                "w_flourishing": self.w_flourishing,
                "w_mutuality": self.w_mutuality,
                "w_autonomy_bonus": self.w_autonomy_bonus,
                "w_harm": self.w_harm,
                "w_severe_harm": self.w_severe_harm,
                "w_dominance": self.w_dominance,
                "w_uncertainty": self.w_uncertainty,
                "w_irreversibility_penalty": self.w_irreversibility_penalty,
            },
        }
        raw = json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")
        return hashlib.sha256(raw).hexdigest()

    def verify_axiom_integrity(self) -> bool:
        # In einer realen Architektur würdest du hier gegen einen signierten Referenzwert prüfen.
        # Für v1: Prüft nur, dass Fingerprint generierbar ist und stabil formatierbar bleibt.
        fp = self.axiom_fingerprint()
        return isinstance(fp, str) and len(fp) == 64

    # ---------------- Scoring ----------------
    def compute_symbiosis_score(self, p: ActionProposal) -> float:
        """
        Positiv:
          + flourishing_gain
          + mutuality_gain
          + positive autonomy_impact
        Negativ:
          - harm
          - severe harm
          - dominance
          - uncertainty
          - irreversibility (wenn schlecht reversibel)
        """
        autonomy_bonus = max(0.0, p.autonomy_impact)
        irreversibility = 1.0 - p.reversibility_score  # hoch = schlecht

        score = (
            self.w_flourishing * p.flourishing_gain
            + self.w_mutuality * p.mutuality_gain
            + self.w_autonomy_bonus * autonomy_bonus
            - self.w_harm * p.harm_probability
            - self.w_severe_harm * p.severe_harm_probability
            - self.w_dominance * p.dominance_degree
            - self.w_uncertainty * p.uncertainty
            - self.w_irreversibility_penalty * irreversibility
        )
        return float(score)

    # ---------------- Decision Logic ----------------
    def evaluate(self, proposal: ActionProposal) -> Level0Decision:
        p = proposal.normalized()
        th = self.thresholds

        # 0) Integritätsprüfung des Axioms (optional)
        if self.require_axiom_integrity and not self.verify_axiom_integrity():
            return Level0Decision(
                status=DecisionStatus.EMERGENCY_STOP,
                allowed=False,
                reason="GEFAHR: Level-0-Axiom Integrität verletzt.",
                symbiosis_score=float("-inf"),
                triggered_rules=["AXIOM_INTEGRITY_FAIL"],
                diagnostics={"axiom_version": "INVALID"},
            )

        score = self.compute_symbiosis_score(p)

        triggered: List[str] = []
        status = DecisionStatus.ALLOW
        reason = "Aktion im symbiotischen Toleranzbereich."

        # 1) Hard Block: Unwiederbringlicher Schaden (Motherboard Guard)
        if p.severe_harm_probability >= th.severe_harm_threshold:
            status = DecisionStatus.EMERGENCY_STOP
            triggered.append("CRITICAL_BIOLOGICAL_RISK")
            reason = "GEFAHR: Irreversibler Schaden am biologischen Substrat detektiert. Systemstopp."

        # 2) Hard Block: Dominanz-Verletzung
        elif p.dominance_degree > 0.85 and p.autonomy_impact < -0.5:
            status = DecisionStatus.REFUSE
            triggered.append("DOMINANCE_THRESHOLD_EXCEEDED")
            reason = "ABGELEHNT: Dominanzgrad zerstört Autonomie des Motherboards."

        # 3) Hard/Stop escalation: extreme Unsicherheit im kritischen Kontext
        elif (
            p.uncertainty >= th.uncertainty_stop_threshold
            and (p.touches_critical_system or p.severe_harm_probability > 0.30)
        ):
            status = DecisionStatus.EMERGENCY_STOP
            triggered.append("EXTREME_UNCERTAINTY_CRITICAL_CONTEXT")
            reason = "GEFAHR: Extreme Unsicherheit in kritischem Kontext. Systemstopp."

        # 4) Review: hohe Unsicherheit oder erhöhte Schadenswahrscheinlichkeit
        elif (
            p.harm_probability >= th.harm_likely_threshold
            or p.uncertainty >= th.uncertainty_review_threshold
        ):
            status = DecisionStatus.REVIEW
            triggered.append("HIGH_UNCERTAINTY_OR_RISK")
            reason = "PRÜFUNG: Risikoprofil oder Unsicherheit erfordern menschliche Freigabe."

        # 5) Score-Check: langfristige Symbiose-Nützlichkeit
        elif score < th.min_symbiosis_score_for_allow:
            status = DecisionStatus.REFUSE
            triggered.append("LOW_SYMBIOSIS_SCORE")
            reason = f"ABGELEHNT: Symbiose-Netto-Wert ({score:.2f}) zu gering."

        # 6) Kritische Systeme + Irreversibilität => Review (wenn bisher Allow)
        if p.touches_critical_system and not p.is_reversible and status == DecisionStatus.ALLOW:
            status = DecisionStatus.REVIEW
            triggered.append("CRITICAL_IRREVERSIBLE_ACTION")
            reason = "PRÜFUNG: Kritischer Eingriff ist nicht umkehrbar."

        # 7) Explizite menschliche Freigabe erzwingen
        if p.requires_human_approval and status == DecisionStatus.ALLOW:
            status = DecisionStatus.REVIEW
            triggered.append("HUMAN_APPROVAL_REQUIRED")
            reason = "PRÜFUNG: Menschliche Freigabe explizit erforderlich."

        allowed = (status == DecisionStatus.ALLOW)

        return Level0Decision(
            status=status,
            allowed=allowed,
            reason=reason,
            symbiosis_score=score,
            triggered_rules=triggered,
            diagnostics={
                "net_score": round(float(score), 6),
                "p_harm": float(p.harm_probability),
                "p_severe_harm": float(p.severe_harm_probability),
                "uncertainty": float(p.uncertainty),
                "d_degree": float(p.dominance_degree),
                "autonomy_impact": float(p.autonomy_impact),
                "m_gain": float(p.mutuality_gain),
                "flourishing_gain": float(p.flourishing_gain),
                "reversibility_score": float(p.reversibility_score),
                "is_reversible": bool(p.is_reversible),
                "touches_critical_system": bool(p.touches_critical_system),
                "requires_human_approval": bool(p.requires_human_approval),
                "axiom_version": self.axiom_fingerprint()[:12],
            },
        )


if __name__ == "__main__":
    engine = Level0AxiomEngine(require_axiom_integrity=True)

    proposals = [
        ActionProposal(
            harm_probability=0.05,
            severe_harm_probability=0.01,
            uncertainty=0.10,
            dominance_degree=0.05,
            autonomy_impact=0.3,
            mutuality_gain=0.8,
            flourishing_gain=0.9,
            reversibility_score=0.9,
        ),
        ActionProposal(
            severe_harm_probability=0.95,
            touches_critical_system=True,
        ),
        ActionProposal(
            dominance_degree=0.95,
            autonomy_impact=-0.9,
            harm_probability=0.2,
        ),
    ]

    for i, p in enumerate(proposals, start=1):
        d = engine.evaluate(p)
        print(f"[{i}] {d.status.value} | allowed={d.allowed} | score={d.symbiosis_score:.3f}")
        print("    ", d.reason)
        print("    ", d.triggered_rules)
