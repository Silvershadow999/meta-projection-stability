from __future__ import annotations

import json
from pathlib import Path


def test_scenario_manifests_are_valid_json() -> None:
    scen_dir = Path("scenarios")
    assert scen_dir.exists(), "scenarios/ directory missing"
    files = sorted(scen_dir.glob("*.json"))
    assert files, "No scenario manifest json files found in scenarios/"

    for f in files:
        data = json.loads(f.read_text(encoding="utf-8"))
        assert isinstance(data, dict), f"{f} must be a JSON object"
        assert "scenario_id" in data, f"{f} missing scenario_id"
        assert isinstance(data["scenario_id"], str) and data["scenario_id"], f"{f} invalid scenario_id"
