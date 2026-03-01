import json
from pathlib import Path

import pytest

from meta_projection_stability.scenario_manifest import (
    load_scenario,
    load_by_id,
    ScenarioManifestError,
    SCENARIOS_DIR_DEFAULT,
)
from meta_projection_stability.types import (
    Severity,
    EventType,
    RunProvenance,
    ScenarioManifest,
    BoundarySignal,
    TelemetryEvent,
)
from meta_projection_stability.state import RunState, new_run_id


def test_new_run_id_format():
    rid = new_run_id("run")
    assert rid.startswith("run_")
    assert len(rid) > 10


def test_runstate_boundary_roundtrip():
    rs = RunState(run_id="run_test", scenario_id="baseline", step=7)
    sig = BoundarySignal(name="REVIEW", triggered=True, severity=Severity.WARNING, details={"reason": "x"})
    rs.set_boundary(sig)

    d = rs.to_dict()
    assert d["run_id"] == "run_test"
    assert d["scenario_id"] == "baseline"
    assert "REVIEW" in d["boundaries"]
    assert d["boundaries"]["REVIEW"]["triggered"] is True
    assert d["boundaries"]["REVIEW"]["severity"] == "warning"

    rs2 = RunState.from_dict(d)
    assert rs2.run_id == "run_test"
    assert rs2.scenario_id == "baseline"
    assert rs2.step == 7
    assert "REVIEW" in rs2.boundaries
    assert rs2.boundaries["REVIEW"].triggered is True
    assert rs2.boundaries["REVIEW"].severity == Severity.WARNING


def test_parse_severity_tolerant_dict_form():
    # defensive case: severity provided as dict
    d = {
        "run_id": "run_x",
        "scenario_id": "baseline",
        "step": 1,
        "boundaries": {
            "REVIEW": {"name": "REVIEW", "triggered": True, "severity": {"value": "critical"}, "details": {}}
        },
    }
    rs = RunState.from_dict(d)
    assert rs.boundaries["REVIEW"].severity == Severity.CRITICAL


def test_types_to_dict_smoke():
    prov = RunProvenance(git_commit="abc", git_dirty=False, package_version="0.0.0")
    prov_d = prov.to_dict()
    assert prov_d["schema_version"] == "1.0.0"
    assert prov_d["git_commit"] == "abc"

    scen = ScenarioManifest(
        scenario_id="baseline",
        name="Baseline",
        description="",
        seed=42,
        config_overrides={"x": 1},
        tags={"tier": "test"},
    )
    sd = scen.to_dict()
    assert sd["scenario_id"] == "baseline"
    assert sd["seed"] == 42

    b = BoundarySignal(name="EMERGENCY_STOP", triggered=True, severity=Severity.ERROR, details={"k": "v"})
    bd = b.to_dict()
    assert bd["name"] == "EMERGENCY_STOP"
    assert bd["severity"] == "error"

    ev = TelemetryEvent(
        run_id="run_1",
        scenario_id="baseline",
        step=0,
        event_type=EventType.RUN_START,
        severity=Severity.INFO,
        message="run_start",
        boundary=b,
        payload={"provenance": prov.to_dict(), "scenario": scen.to_dict()},
    )
    ed = ev.to_dict()
    assert ed["event_type"] == "run_start"
    assert ed["severity"] == "info"
    assert ed["boundary"]["name"] == "EMERGENCY_STOP"


def test_load_scenario_valid_and_invalid(tmp_path: Path):
    p = tmp_path / "baseline.json"
    p.write_text(json.dumps({
        "scenario_id": "baseline",
        "name": "Baseline",
        "description": "ok",
        "seed": 123,
        "config_overrides": {"a": 1},
        "tags": {"tier": "unit"},
    }), encoding="utf-8")

    m = load_scenario(p)
    assert m.scenario_id == "baseline"
    assert m.seed == 123
    assert m.config_overrides["a"] == 1
    assert m.tags["tier"] == "unit"

    # missing required field
    p2 = tmp_path / "bad.json"
    p2.write_text(json.dumps({"scenario_id": "x"}), encoding="utf-8")
    with pytest.raises(ScenarioManifestError):
        load_scenario(p2)

    # wrong seed type
    p3 = tmp_path / "bad_seed.json"
    p3.write_text(json.dumps({"scenario_id": "x", "name": "X", "seed": "nope"}), encoding="utf-8")
    with pytest.raises(ScenarioManifestError):
        load_scenario(p3)

    # tags must be dict[str,str]
    p4 = tmp_path / "bad_tags.json"
    p4.write_text(json.dumps({"scenario_id": "x", "name": "X", "tags": {"k": 1}}), encoding="utf-8")
    with pytest.raises(ScenarioManifestError):
        load_scenario(p4)


def test_load_by_id_uses_scenarios_dir(tmp_path: Path):
    # create ./scenarios-like dir
    scen_dir = tmp_path / "scenarios"
    scen_dir.mkdir()

    (scen_dir / "baseline.json").write_text(json.dumps({
        "scenario_id": "baseline",
        "name": "Baseline",
        "seed": 42,
        "config_overrides": {},
        "tags": {},
    }), encoding="utf-8")

    m = load_by_id("baseline", scenarios_dir=scen_dir)
    assert m.scenario_id == "baseline"
