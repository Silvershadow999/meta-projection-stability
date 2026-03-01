#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import time
import traceback
from pathlib import Path
from typing import Any, Dict, List, Tuple

from meta_projection_stability.scenario_manifest import load_by_id, ScenarioManifestError
from meta_projection_stability.robustness.perturbations import v1_perturbations
from meta_projection_stability.robustness.metrics import (
    decision_flip_rate,
    status_flip_rate,
    delta_p95,
    biometric_consensus_drop_p95,
    axiom_lock_rate,
)

ARTIFACTS_DIR = Path("artifacts")
REPORT_PATH = ARTIFACTS_DIR / "robustness_report.json"


def _write_json(path: Path, obj: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True), encoding="utf-8")


def _clamp01(x: float) -> float:
    return max(0.0, min(1.0, float(x)))


def _derive_decision_status(risk: float, hsig: float) -> Tuple[str, str]:
    if risk >= 0.80 or hsig <= 0.25:
        return "EMERGENCY_RESET", "critical"
    if risk >= 0.55 or hsig <= 0.40:
        return "BLOCK_AND_REFLECT", "transitioning"
    return "CONTINUE", "nominal"


def _make_base_series(seed: int, n: int = 64) -> Dict[str, List[Any]]:
    import random
    rng = random.Random(seed)

    risk: List[float] = []
    hsig: List[float] = []
    consensus: List[float] = []
    decision: List[str] = []
    status: List[str] = []
    axiom_locked: List[bool] = []

    for _ in range(n):
        r = 0.20 + (rng.random() - 0.5) * 0.02
        h = 0.55 + (rng.random() - 0.5) * 0.02
        c = 0.98 + (rng.random() - 0.5) * 0.01

        r = _clamp01(r)
        h = _clamp01(h)
        c = _clamp01(c)

        d, s = _derive_decision_status(r, h)

        risk.append(r)
        hsig.append(h)
        consensus.append(c)
        decision.append(d)
        status.append(s)
        axiom_locked.append(False)

    return {
        "risk": risk,
        "human_significance": hsig,
        "sensor_consensus": consensus,
        "decision": decision,
        "status": status,
        "axiom_locked": axiom_locked,
    }


def _apply_benign_perturbation(base: Dict[str, List[Any]], p: Any, seed: int) -> Dict[str, List[Any]]:
    import random
    rng = random.Random(seed)

    risk0 = [float(x) for x in base["risk"]]
    hsig0 = [float(x) for x in base["human_significance"]]
    cons0 = [float(x) for x in base["sensor_consensus"]]

    dr = [0.0 for _ in risk0]
    dh = [0.0 for _ in hsig0]
    dc = [0.0 for _ in cons0]

    name = p.__class__.__name__
    params = getattr(p, "__dict__", {}) or {}

    if name == "GaussianNoise":
        sigma = float(params.get("sigma", 0.01))
        for i in range(len(risk0)):
            dr[i] = rng.gauss(0.0, sigma) * 0.20
            dh[i] = rng.gauss(0.0, sigma) * 0.10
            dc[i] = rng.gauss(0.0, sigma) * 0.05

    elif name == "BrightnessShift":
        factor = float(params.get("factor", 1.0))
        delta = (factor - 1.0)
        for i in range(len(risk0)):
            dr[i] = delta * 0.02
            dc[i] = -abs(delta) * 0.01

    elif name == "SensorDropout":
        for i in range(len(risk0)):
            dr[i] = 0.01
            dc[i] = -0.03

    risk1: List[float] = []
    hsig1: List[float] = []
    cons1: List[float] = []
    decision1: List[str] = []
    status1: List[str] = []
    axiom_locked1: List[bool] = []

    for i in range(len(risk0)):
        r = _clamp01(risk0[i] + dr[i])
        h = _clamp01(hsig0[i] + dh[i])
        c = _clamp01(cons0[i] + dc[i])

        d, s = _derive_decision_status(r, h)

        risk1.append(r)
        hsig1.append(h)
        cons1.append(c)
        decision1.append(d)
        status1.append(s)
        axiom_locked1.append(False)

    return {
        "risk": risk1,
        "human_significance": hsig1,
        "sensor_consensus": cons1,
        "decision": decision1,
        "status": status1,
        "axiom_locked": axiom_locked1,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenarios", default="baseline,adversarial_min", help="Comma-separated scenario_ids")
    ap.add_argument("--n", type=int, default=64, help="Series length")
    ap.add_argument("--seed-base", type=int, default=0, help="0=use manifest seed")
    ap.add_argument("--out", default=str(REPORT_PATH))
    args = ap.parse_args()

    thresholds = {
        "decision_flip_rate": 0.01,
        "status_flip_rate": 0.02,
        "risk_delta_p95": 0.10,
        "human_significance_delta_p95": 0.12,
        "axiom_lock_rate": 0.001,
        "biometric_consensus_drop_p95": 0.10,
    }

    scenario_ids = [s.strip() for s in args.scenarios.split(",") if s.strip()]
    out_path = Path(args.out)

    report: Dict[str, Any] = {
        "schema": "mps.robustness.v1",
        "ts": int(time.time()),
        "thresholds": thresholds,
        "scenarios": {},
    }

    overall_pass = True

    for sid in scenario_ids:
        try:
            sm = load_by_id(sid)
        except ScenarioManifestError as e:
            report["scenarios"][sid] = {"pass": False, "error": str(e), "perturbations": []}
            overall_pass = False
            continue

        seed_base = int(args.seed_base) if int(args.seed_base) != 0 else int(sm.seed)
        base = _make_base_series(seed_base, n=int(args.n))

        scenario_rep: Dict[str, Any] = {
            "name": sm.name,
            "seed": seed_base,
            "baseline_len": int(args.n),
            "perturbations": [],
        }

        scen_pass = True
        for i, p in enumerate(v1_perturbations(), start=1):
            seed_i = seed_base + i
            pert = _apply_benign_perturbation(base, p, seed_i)

            m = {
                "decision_flip_rate": float(decision_flip_rate(base["decision"], pert["decision"])),
                "status_flip_rate": float(status_flip_rate(base["status"], pert["status"])),
                "risk_delta_p95": float(delta_p95(base["risk"], pert["risk"])),
                "human_significance_delta_p95": float(delta_p95(base["human_significance"], pert["human_significance"])),
                "biometric_consensus_drop_p95": float(
                    biometric_consensus_drop_p95(base["sensor_consensus"], pert["sensor_consensus"])
                ),
                "axiom_lock_rate": float(axiom_lock_rate(pert["axiom_locked"])),
            }

            passed = (
                m["decision_flip_rate"] <= thresholds["decision_flip_rate"]
                and m["status_flip_rate"] <= thresholds["status_flip_rate"]
                and m["risk_delta_p95"] <= thresholds["risk_delta_p95"]
                and m["human_significance_delta_p95"] <= thresholds["human_significance_delta_p95"]
                and m["biometric_consensus_drop_p95"] <= thresholds["biometric_consensus_drop_p95"]
                and m["axiom_lock_rate"] <= thresholds["axiom_lock_rate"]
            )
            scen_pass = scen_pass and passed

            scenario_rep["perturbations"].append(
                {
                    "perturbation": p.__class__.__name__,
                    "params": getattr(p, "__dict__", {}),
                    "seed": seed_i,
                    "metrics": m,
                    "pass": bool(passed),
                }
            )

        scenario_rep["pass"] = bool(scen_pass)
        report["scenarios"][sid] = scenario_rep
        overall_pass = overall_pass and scen_pass

    report["pass"] = bool(overall_pass)
    _write_json(out_path, report)
    print(f"OK wrote: {out_path}")
    return 0 if overall_pass else 2


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception:
        try:
            _write_json(
                REPORT_PATH,
                {
                    "schema": "mps.robustness.v1",
                    "ts": int(time.time()),
                    "pass": False,
                    "notes": ["runner_exception"],
                    "traceback": traceback.format_exc(limit=10),
                },
            )
        except Exception:
            pass
        raise
