from __future__ import annotations

import importlib


def test_smoke_imports_phase5_heavy_modules():
    # These modules were 0% in the coverage report and are safe to import.
    mods = [
        "meta_projection_stability.baseline_sim",
        "meta_projection_stability.cli",
        "meta_projection_stability.cli_experiments",
        "meta_projection_stability.experiment_runner",
        "meta_projection_stability.globalsense",
    ]
    for m in mods:
        mod = importlib.import_module(m)
        assert mod is not None


def test_smoke_profiles_reporting_ranking_perturbations():
    mods = [
        "meta_projection_stability.profiles",
        "meta_projection_stability.reporting",
        "meta_projection_stability.ranking",
        "meta_projection_stability.robustness.perturbations",
    ]
    for m in mods:
        mod = importlib.import_module(m)
        assert mod is not None
