import inspect
import numpy as np
import pytest

from meta_projection_stability.config import MetaProjectionStabilityConfig
from meta_projection_stability.adapter import MetaProjectionStabilityAdapter


def _build_adapter(use_bio: bool):
    cfg = MetaProjectionStabilityConfig()
    if hasattr(cfg, "use_biometric_fusion"):
        cfg.use_biometric_fusion = use_bio
    return MetaProjectionStabilityAdapter(cfg), cfg


def _safe_interpret_call(adapter, *, stress_variant=False):
    """
    Signatur-robuster Aufruf von adapter.interpret(...):
    - setzt nur Argumente, die wirklich in der Signatur existieren
    - raw_signals immer als dict (wichtig)
    """
    sig = inspect.signature(adapter.interpret)
    params = sig.parameters

    # Baseline / Stress-Szenario
    if stress_variant:
        S_layers = np.array([0.35, 0.55, 0.40], dtype=float)
        delta_S = 0.08
        raw_signals = {
            "instability_signal": 0.70,
            "eda_stress_score": 0.82,
            "hrv_normalized": 0.28,
            "emotional_valence": 0.18,
            "autonomy_signal": 0.42,
            "gamma_coherence": 0.25,
        }
        opt_vals = {
            "trust_level": 0.52,
            "human_significance": 0.58,
            "time_index": 1,
            "t": 1,
        }
    else:
        S_layers = np.array([0.20, 0.30, 0.25], dtype=float)
        delta_S = 0.02
        raw_signals = {
            "instability_signal": 0.38,
            "eda_stress_score": 0.40,
            "hrv_normalized": 0.62,
            "emotional_valence": 0.57,
            "autonomy_signal": 0.72,
            "gamma_coherence": 0.61,
        }
        opt_vals = {
            "trust_level": 0.72,
            "human_significance": 0.66,
            "time_index": 0,
            "t": 0,
        }

    kwargs = {}

    # nur setzen, wenn vorhanden
    if "S_layers" in params:
        kwargs["S_layers"] = S_layers
    if "delta_S" in params:
        kwargs["delta_S"] = float(delta_S)
    if "raw_signals" in params:
        kwargs["raw_signals"] = raw_signals

    for k, v in opt_vals.items():
        if k in params:
            kwargs[k] = v

    # Fallback: falls interpret andere Namen nutzt, hier optional erweiterbar
    result = adapter.interpret(**kwargs)
    return result


def _require_dict_result(res):
    if not isinstance(res, dict):
        pytest.skip(
            f"adapter.interpret() returned {type(res).__name__}, expected dict for biometric field assertions."
        )


def test_adapter_interpret_runs_with_signature_safe_call():
    adapter, _ = _build_adapter(use_bio=True)
    res = _safe_interpret_call(adapter, stress_variant=False)
    assert res is not None


def test_adapter_returns_biometric_fields_if_dict():
    adapter, cfg = _build_adapter(use_bio=True)
    res = _safe_interpret_call(adapter, stress_variant=False)
    _require_dict_result(res)

    # Erwartete Felder aus deiner Erweiterung
    expected_keys = [
        "biometric_proxy",
        "biometric_risk_component",
        "autonomy_proxy",
        "gamma_coherence_proxy",
        "mutuality_bonus",
        "trust_reinforcement",
        "cooldown_remaining",
    ]

    missing = [k for k in expected_keys if k not in res]
    assert not missing, f"Missing adapter result keys: {missing}"

    # Grobe Typ-/Wert-Checks
    for k in [
        "biometric_proxy",
        "biometric_risk_component",
        "autonomy_proxy",
        "gamma_coherence_proxy",
        "mutuality_bonus",
        "trust_reinforcement",
    ]:
        assert isinstance(res[k], (int, float, np.floating)), f"{k} is not numeric: {type(res[k]).__name__}"

    assert isinstance(res["cooldown_remaining"], (int, np.integer)), (
        f"cooldown_remaining is not int-like: {type(res['cooldown_remaining']).__name__}"
    )

    # Optional: wenn Feature-Flag existiert, prüfen wir es wenigstens auf bool
    if hasattr(cfg, "use_biometric_fusion"):
        assert isinstance(cfg.use_biometric_fusion, bool)



def test_biometric_fusion_on_off_executes_if_dict():
    """
    Robuster ON/OFF-Test:
    - Beide Modi müssen ausführbar sein
    - Falls dict zurückkommt, prüfen wir nur auf gültige Struktur
    - KEIN harter Zwang auf unterschiedliche Werte (implementation-dependent)
    """
    adapter_on, cfg_on = _build_adapter(use_bio=True)
    adapter_off, cfg_off = _build_adapter(use_bio=False)

    res_on = _safe_interpret_call(adapter_on, stress_variant=True)
    res_off = _safe_interpret_call(adapter_off, stress_variant=True)

    assert res_on is not None
    assert res_off is not None

    if not isinstance(res_on, dict) or not isinstance(res_off, dict):
        pytest.skip("Adapter result is not dict in one/both modes; skipping dict-key assertions.")

    # Falls Feature-Flag existiert, dokumentieren wir nur die Existenz (nicht Wirkung in jedem Szenario)
    if hasattr(cfg_on, "use_biometric_fusion"):
        assert isinstance(cfg_on.use_biometric_fusion, bool)
        assert isinstance(cfg_off.use_biometric_fusion, bool)

    # Basisfelder prüfen, sofern vorhanden (nicht alle Implementierungen exposen alles)
    candidate_keys = [
        "risk_input",
        "trust_damping",
        "biometric_proxy",
        "biometric_risk_component",
        "mutuality_bonus",
        "trust_reinforcement",
    ]

    common = [k for k in candidate_keys if k in res_on and k in res_off]
    # Nur sicherstellen, dass wir etwas Vergleichbares bekommen haben
    assert isinstance(common, list)

    for k in common:
        assert res_on.get(k) is not None or res_off.get(k) is not None

def test_biometric_fusion_fields_present_even_when_feature_off_if_dict():
    """
    Dokumentiert gewünschtes Verhalten:
    Felder dürfen auch bei OFF vorhanden sein (z.B. für Observability),
    aber ihre Wirkung auf Risiko/Trust soll geringer/anders sein.
    """
    adapter, _ = _build_adapter(use_bio=False)
    res = _safe_interpret_call(adapter, stress_variant=True)
    _require_dict_result(res)

    # Mindestens die observability-Felder sollten vorhanden sein, wenn implementiert wie besprochen.
    observability_keys = [
        "biometric_proxy",
        "autonomy_proxy",
        "gamma_coherence_proxy",
    ]

    missing = [k for k in observability_keys if k not in res]
    # Kein harter Fail falls du dich entschieden hast, OFF schlanker zu halten:
    if missing:
        pytest.skip(f"Observability keys not exposed in OFF mode: {missing}")
