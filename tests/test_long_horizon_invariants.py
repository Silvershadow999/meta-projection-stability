from __future__ import annotations

import math

import pytest

from meta_projection_stability.config import MetaProjectionStabilityConfig
from meta_projection_stability.simulation import run_long_horizon_simulation
from meta_projection_stability.analytics import compute_stability_metrics


ALLOWED_DECISIONS = {"CONTINUE", "BLOCK_AND_REFLECT", "EMERGENCY_RESET"}
ALLOWED_STATUSES = {"nominal", "transitioning", "cooldown", "critical_instability_reset"}


def _is_finite_number(x) -> bool:
    if x is None:
        return False
    try:
        xf = float(x)
    except (TypeError, ValueError):
        return False
    return math.isfinite(xf)


def _finite_list(values):
    out = []
    for v in values:
        if _is_finite_number(v):
            out.append(float(v))
    return out


def _run_smoke_long_horizon():
    cfg = MetaProjectionStabilityConfig(
        seed=42,
        enable_plot=False,
        debug=False,
        verbose=False,
    )
    return run_long_horizon_simulation(
        steps=1200,
        seed=42,
        cfg=cfg,
        use_noisy_significance=True,
        stress_events=[(150, 0.15), (151, 0.10), (600, 0.22), (900, 0.18)],
    )


def test_long_horizon_returns_expected_top_level_structure():
    res = _run_smoke_long_horizon()

    assert isinstance(res, dict)
    for key in ["history", "metrics", "stability", "config"]:
        assert key in res, f"Missing top-level key: {key}"

    assert isinstance(res["history"], dict)
    assert isinstance(res["metrics"], dict)


def test_history_lengths_are_consistent_enough():
    res = _run_smoke_long_horizon()
    h = res["history"]

    keys_expected_series = [
        "step",
        "h_sig",
        "h_ema",
        "risk",
        "risk_raw",
        "trust",
        "coherence",
        "decision",
        "status",
        "momentum",
        "risk_input",
        "trust_damping",
        "cooldown_remaining",
        "S2",
        "delta_S",
        "external_human_proxy",
        "stress_event_delta",
        "instability_signal_base",
        "instability_signal_final",
    ]

    lengths = {}
    for k in keys_expected_series:
        assert k in h, f"Missing history key: {k}"
        assert isinstance(h[k], list), f"History key {k} is not a list"
        lengths[k] = len(h[k])

    # all lengths should match exactly in current implementation
    unique_lengths = set(lengths.values())
    assert len(unique_lengths) == 1, f"Inconsistent history lengths: {lengths}"

    n = next(iter(unique_lengths))
    assert n > 0, "Empty history"
    assert n <= 1200, "History longer than requested steps"


def test_numeric_series_are_finite_and_bounded_where_expected():
    res = _run_smoke_long_horizon()
    h = res["history"]

    bounded_0_1_keys = [
        "risk",
        "risk_raw",
        "trust",
        "coherence",
        "risk_input",
        "trust_damping",
        "external_human_proxy",
        "instability_signal_base",
        "instability_signal_final",
    ]

    for k in bounded_0_1_keys:
        vals = _finite_list(h.get(k, []))
        assert vals, f"No finite numeric values for {k}"
        assert all(0.0 <= v <= 1.0 for v in vals), f"{k} out of [0,1] bounds"

    # human significance may be >1.0 depending on cfg.human_sig_max (e.g. 1.10)
    h_sig_vals = _finite_list(h.get("h_sig", []))
    h_ema_vals = _finite_list(h.get("h_ema", []))
    assert h_sig_vals and h_ema_vals
    assert all(v >= 0.0 for v in h_sig_vals), "h_sig has negative values"
    assert all(v >= 0.0 for v in h_ema_vals), "h_ema has negative values"

    # cooldown should never be negative
    cooldown_vals = _finite_list(h.get("cooldown_remaining", []))
    assert cooldown_vals, "No cooldown values"
    assert all(v >= 0.0 for v in cooldown_vals), "cooldown_remaining negative"

    # delta_S / momentum / S2 should be finite when present
    for k in ["delta_S", "momentum", "S2"]:
        vals = _finite_list(h.get(k, []))
        assert vals, f"No finite values in {k}"


def test_decision_and_status_domains_are_valid():
    res = _run_smoke_long_horizon()
    h = res["history"]

    decisions = h.get("decision", [])
    statuses = h.get("status", [])

    assert decisions, "No decisions recorded"
    assert statuses, "No statuses recorded"

    invalid_decisions = sorted(set(d for d in decisions if d not in ALLOWED_DECISIONS))
    invalid_statuses = sorted(set(s for s in statuses if s not in ALLOWED_STATUSES))

    assert not invalid_decisions, f"Invalid decisions found: {invalid_decisions}"
    assert not invalid_statuses, f"Invalid statuses found: {invalid_statuses}"


def test_metrics_payload_is_sane():
    res = _run_smoke_long_horizon()
    m = res["metrics"]

    assert "steps_executed" in m
    assert "decision_fractions" in m
    assert "risk_stats" in m
    assert "trust_stats" in m
    assert "h_sig_stats" in m

    assert isinstance(m["steps_executed"], int)
    assert m["steps_executed"] > 0

    fractions = m["decision_fractions"]
    for k in ["CONTINUE", "BLOCK_AND_REFLECT", "EMERGENCY_RESET"]:
        assert k in fractions, f"Missing fraction for {k}"
        v = float(fractions[k])
        assert 0.0 <= v <= 1.0, f"Fraction {k} out of bounds: {v}"

    cooldown_fraction = float(m.get("cooldown_fraction", 0.0))
    assert 0.0 <= cooldown_fraction <= 1.0

    for stats_key in ["risk_stats", "trust_stats", "h_sig_stats", "external_human_proxy_stats"]:
        stats = m.get(stats_key, {})
        if not isinstance(stats, dict):
            continue
        for sub in ["min", "max", "mean", "p95"]:
            if stats.get(sub) is not None:
                assert _is_finite_number(stats[sub]), f"{stats_key}.{sub} not finite"


def test_compute_stability_metrics_on_history_is_valid_and_bounded():
    res = _run_smoke_long_horizon()
    metrics = compute_stability_metrics(res["history"])

    assert isinstance(metrics, dict)
    assert metrics.get("valid") is True
    assert int(metrics.get("steps", 0)) > 0

    for k in [
        "continue_rate",
        "block_rate",
        "reset_rate",
        "nominal_fraction",
        "transitioning_fraction",
        "cooldown_fraction",
        "false_positive_block_rate",
        "stuck_transitioning_rate",
    ]:
        v = float(metrics[k])
        assert 0.0 <= v <= 1.0, f"{k} out of [0,1]: {v}"

    for k in ["max_block_streak", "max_reset_streak", "max_cooldown_streak"]:
        assert int(metrics[k]) >= 0

    for k in ["avg_block_streak", "avg_cooldown_streak"]:
        assert float(metrics[k]) >= 0.0


@pytest.mark.parametrize(
    "steps,events",
    [
        (300, []),
        (500, [(100, 0.20)]),
        (700, [(120, 0.15), (121, 0.15), (450, 0.25)]),
    ],
)
def test_invariants_hold_across_multiple_short_profiles(steps, events):
    cfg = MetaProjectionStabilityConfig(seed=7, enable_plot=False, debug=False, verbose=False)

    res = run_long_horizon_simulation(
        steps=steps,
        seed=7,
        cfg=cfg,
        use_noisy_significance=True,
        stress_events=events,
    )
    h = res["history"]

    # quick invariants
    assert len(h["decision"]) > 0
    assert all(d in ALLOWED_DECISIONS for d in h["decision"])
    assert all((float(c) >= 0.0) for c in h["cooldown_remaining"] if _is_finite_number(c))

    for key in ["risk", "trust", "coherence", "risk_input", "trust_damping"]:
        vals = _finite_list(h[key])
        assert vals
        assert all(0.0 <= v <= 1.0 for v in vals), f"{key} out of bounds in profile steps={steps}"
