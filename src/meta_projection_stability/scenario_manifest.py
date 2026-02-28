"""
scenario_manifest.py — Scenario manifests as data (Phase 5)

Loads JSON scenario manifests from ./scenarios and returns ScenarioManifest.
- stdlib only
- light validation (required keys, types)
"""

from __future__ import annotations

from pathlib import Path
import json

from .types import ScenarioManifest


SCENARIOS_DIR_DEFAULT = Path("scenarios")

def _as_dict(x, default):
    return x if isinstance(x, dict) else default

def _as_list_str(x, default):
    if not isinstance(x, list):
        return default
    out = []
    for v in x:
        if isinstance(v, str) and v.strip():
            out.append(v.strip())
    return out


class ScenarioManifestError(ValueError):
    pass


def load_scenario(path: str | Path) -> ScenarioManifest:
    p = Path(path)
    if not p.exists():
        raise ScenarioManifestError(f"Scenario file not found: {p}")

    data = json.loads(p.read_text(encoding="utf-8"))

    for k in ("scenario_id", "name"):
        if k not in data or not isinstance(data[k], str) or not data[k].strip():
            raise ScenarioManifestError(f"Missing/invalid required field '{k}' in {p}")

    seed = data.get("seed", 42)
    if not isinstance(seed, int):
        raise ScenarioManifestError(f"'seed' must be int in {p}")

    config_overrides = data.get("config_overrides", {}) or {}
    if not isinstance(config_overrides, dict):
        raise ScenarioManifestError(f"'config_overrides' must be object/dict in {p}")

    tags = data.get("tags", {}) or {}
    if not isinstance(tags, dict) or not all(isinstance(k, str) and isinstance(v, str) for k, v in tags.items()):
        raise ScenarioManifestError(f"'tags' must be dict[str,str] in {p}")

    return ScenarioManifest(
        scenario_id=data["scenario_id"],
        name=data["name"],
        description=str(data.get("description", "")),
        seed=seed,
        config_overrides=config_overrides,
        tags=tags,
    )


def load_by_id(scenario_id: str, scenarios_dir: Path = SCENARIOS_DIR_DEFAULT) -> ScenarioManifest:
    return load_scenario(scenarios_dir / f"{scenario_id}.json")


__all__ = ["ScenarioManifestError", "load_scenario", "load_by_id", "SCENARIOS_DIR_DEFAULT"]
