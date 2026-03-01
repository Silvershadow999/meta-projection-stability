from meta_projection_stability.robustness.metrics import (
    p95,
    decision_flip_rate,
    status_flip_rate,
    delta_p95,
    biometric_consensus_drop_p95,
    axiom_lock_rate,
)

def test_p95_empty_and_basic():
    assert p95([]) == 0.0
    assert p95([1.0]) == 1.0
    assert p95([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]) >= 9.0

def test_flip_rates():
    base = ["A", "A", "B", "B"]
    pert = ["A", "B", "B", "C"]
    assert decision_flip_rate(base, pert) == 2 / 4
    assert status_flip_rate(base, pert) == 2 / 4
    assert decision_flip_rate([], []) == 0.0

def test_delta_p95_and_drop_p95():
    base = [0.9, 0.8, 0.7, 0.6]
    pert = [0.85, 0.7, 0.9, 0.6]
    d = delta_p95(base, pert)
    assert d >= 0.1  # at least one delta is 0.2

    drop = biometric_consensus_drop_p95([1.0, 0.5], [0.8, 0.6])
    # drops: [0.2, 0.0] -> p95 should be 0.2
    assert drop >= 0.2 - 1e-9

def test_axiom_lock_rate():
    assert axiom_lock_rate([]) == 0.0
    assert axiom_lock_rate([True, False, True, True]) == 3 / 4
