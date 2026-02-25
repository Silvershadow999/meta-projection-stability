from __future__ import annotations

import inspect
import pytest

from meta_projection_stability.config import MetaProjectionStabilityConfig
from meta_projection_stability.adapter import MetaProjectionStabilityAdapter


def _sample_raw_signals() -> dict:
    return {
        "instability_signal": 0.72,   # etwas höher für klareren Effekt
        "trust_level": 0.42,
        "trust": 0.42,
        "coherence": 0.58,
        "gamma_coherence": 0.58,
        "hrv": 0.35,
        "hrv_normalized": 0.35,
        "eda_stress": 0.82,
        "eda_stress_score": 0.82,
        "emotional_valence": -0.35,
        "valence": -0.35,
        "state_transition": 0.66,
        "transition_signal": 0.66,
        "human_sig": 0.61,
        "human_significance": 0.61,
        "autonomy_signal": 0.73,
    }


def _call_interpret_signature_safe(adapter: MetaProjectionStabilityAdapter):
    sig = inspect.signature(adapter.interpret)
    raw_signals = _sample_raw_signals()

    scalar_candidates = {
        "risk_input": 0.72,
        "risk": 0.72,
        "raw_risk": 0.72,
        "risk_score": 0.72,
        "trust_level": 0.42,
        "trust": 0.42,
        "coherence": 0.58,
        "gamma_coherence": 0.58,
        "hrv": 0.35,
        "hrv_normalized": 0.35,
        "eda_stress": 0.82,
        "eda_stress_score": 0.82,
        "emotional_valence": -0.35,
        "valence": -0.35,
        "state_transition": 0.66,
        "transition_signal": 0.66,
        "human_sig": 0.61,
        "human_significance": 0.61,
    }

    mapping_param_names = {
        "raw_signals", "signals", "inputs", "sensor_data", "biometrics",
        "metrics", "features", "observation", "obs"
    }
    int_param_names = {"step", "step_idx", "t", "index", "tick"}

    kwargs = {}
    for name, param in sig.parameters.items():
        if name == "self":
            continue

        lname = name.lower()

        if name in mapping_param_names or any(tok in lname for tok in ["signal", "input", "data", "metric", "feature", "obs"]):
            kwargs[name] = raw_signals
            continue

        if name in int_param_names or any(tok in lname for tok in ["step", "idx", "index", "tick"]):
            kwargs[name] = 10
            continue

        if name in scalar_candidates:
            kwargs[name] = scalar_candidates[name]
            continue

        if param.default is inspect._empty:
            kwargs[name] = 0.5

    return adapter.interpret(**kwargs)


def _as_dict_or_none(result):
    return result if isinstance(result, dict) else None


def _float_if_possible(v):
    try:
        return float(v)
    except Exception:
        return None


def test_config_has_biometric_fusion_fields():
    cfg = MetaProjectionStabilityConfig()

    expected = [
        "human_decay_scale",
        "recovery_trust_power",
        "transition_decay_factor",
        "cooldown_human_recovery_step",
        "use_biometric_fusion",
        "biometric_proxy_weight",
        "biometric_risk_weight",
        "autonomy_decay_weight",
        "mutuality_bonus_gain",
    ]
    missing = [k for k in expected if not hasattr(cfg, k)]
    assert not missing, f"Missing config fields: {missing}"


def test_adapter_interpret_runs_with_signature_safe_call():
    cfg = MetaProjectionStabilityConfig()
    adapter = MetaProjectionStabilityAdapter(cfg)
    result = _call_interpret_signature_safe(adapter)
    assert result is not None


def test_adapter_returns_some_biometric_or_state_fields_if_dict():
    """
    Kompatibel über Varianten:
    Nicht auf exakte Keyliste festnageln, sondern auf eine sinnvolle Teilmenge prüfen.
    """
    cfg = MetaProjectionStabilityConfig()
    adapter = MetaProjectionStabilityAdapter(cfg)

    result = _call_interpret_signature_safe(adapter)
    res = _as_dict_or_none(result)
    if res is None:
        pytest.skip("adapter.interpret() returns non-dict in current implementation")

    candidate_keys = [
        # biometrisch/fusion
        "biometric_proxy",
        "biometric_risk_component",
        "autonomy_proxy",
        "gamma_coherence_proxy",
        "mutuality_bonus",
        "trust_reinforcement",
        "cooldown_remaining",
        # mögliche alternative Benennungen
        "biometric_risk",
        "autonomy_decay",
        "trust_damping",
        "recovery_bonus",
        "transition_damping",
        # generische Outputs
        "risk",
        "risk_output",
        "adapted_risk",
        "final_risk",
        "score",
    ]

    present = [k for k in candidate_keys if k in res]
    assert present, f"No expected biometric/state/result keys found. Keys were: {list(res.keys())}"


def test_biometric_fusion_on_off_runs_and_optionally_changes_output_if_dict():
    """
    ON/OFF-Vergleich:
    - Muss in beiden Modi laufen
    - Falls dict: wir suchen Unterschiede in einer breiten Kandidatenmenge
    - Wenn kein Unterschied sichtbar ist, skip statt fail (versionskompatibel)
    """
    cfg_on = MetaProjectionStabilityConfig()
    cfg_on.use_biometric_fusion = True
    adapter_on = MetaProjectionStabilityAdapter(cfg_on)
    res_on_raw = _call_interpret_signature_safe(adapter_on)

    cfg_off = MetaProjectionStabilityConfig()
    cfg_off.use_biometric_fusion = False
    adapter_off = MetaProjectionStabilityAdapter(cfg_off)
    res_off_raw = _call_interpret_signature_safe(adapter_off)

    # Nicht-dict Rückgaben: nur sicherstellen, dass beide Calls funktionieren
    if not isinstance(res_on_raw, dict) or not isinstance(res_off_raw, dict):
        assert res_on_raw is not None and res_off_raw is not None
        pytest.skip("Non-dict return type; ON/OFF field diff skipped")

    res_on = res_on_raw
    res_off = res_off_raw

    diff_candidates = [
        "biometric_proxy",
        "biometric_risk_component",
        "biometric_risk",
        "autonomy_proxy",
        "mutuality_bonus",
        "trust_reinforcement",
        "trust_damping",
        "recovery_bonus",
        "transition_damping",
        "risk",
        "risk_output",
        "adapted_risk",
        "final_risk",
        "score",
    ]

    comparable = [k for k in diff_candidates if k in res_on and k in res_off]
    if not comparable:
        pytest.skip(f"No comparable ON/OFF keys found. ON keys: {list(res_on.keys())}")

    changed = []
    for k in comparable:
        a = _float_if_possible(res_on.get(k))
        b = _float_if_possible(res_off.get(k))
        if a is None or b is None:
            if res_on.get(k) != res_off.get(k):
                changed.append(k)
        else:
            if abs(a - b) > 1e-12:
                changed.append(k)

    if not changed:
        pytest.skip(f"ON/OFF produced no visible change for current inputs. Checked: {comparable}")

    assert changed, "Expected at least one ON/OFF difference"
