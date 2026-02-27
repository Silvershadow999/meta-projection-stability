from dataclasses import dataclass
from enum import Enum

# Fallback: Wir versuchen deinen echten Import.
# Wenn API noch leicht anders ist, zeigt der Test sauber den Fehler.
try:
    from meta_projection_stability.symbiosis_core import (
        DecisionStatus,
        # Optional vorhandene Klassen:
        # ActionProposal, Level0Governor, Thresholds
    )
except Exception as e:
    DecisionStatus = None
    IMPORT_ERROR = e
else:
    IMPORT_ERROR = None


def test_symbiosis_core_imports():
    assert IMPORT_ERROR is None, f"Importfehler in symbiosis_core: {IMPORT_ERROR}"


def test_decisionstatus_has_core_states():
    assert DecisionStatus is not None, "DecisionStatus nicht importierbar"
    names = {x.name for x in DecisionStatus}
    # Kernzustände, die dein Guardrail-System braucht
    required = {"ALLOW", "REVIEW", "REFUSE", "EMERGENCY_STOP"}
    missing = required - names
    assert not missing, f"Fehlende DecisionStatus-Werte: {missing}"


def test_policy_keywords_present():
    # Strukturtest gegen versehentliches Entfernen der Kernlogik
    from pathlib import Path
    p = Path("src/meta_projection_stability/symbiosis_core.py")
    assert p.exists(), "symbiosis_core.py fehlt"

    txt = p.read_text(encoding="utf-8")
    required_snippets = [
        "severe_harm",
        "dominance",
        "autonomy",
        "symbiosis",
        "EMERGENCY_STOP",
        "REFUSE",
        "REVIEW",
    ]
    missing = [s for s in required_snippets if s not in txt]
    assert not missing, f"Policy-Schlüsselbegriffe fehlen: {missing}"
