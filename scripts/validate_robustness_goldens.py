#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict

def fail(msg: str) -> None:
    raise SystemExit(f"FAIL: {msg}")

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--report", default="artifacts/robustness_report.json")
    ap.add_argument("--golden", default="artifacts/robustness_golden.json")
    ap.add_argument("--scenarios", default="baseline,adversarial_min")
    args = ap.parse_args()

    report_p = Path(args.report)
    golden_p = Path(args.golden)
    if not report_p.exists():
        fail(f"Missing report: {report_p}")
    if not golden_p.exists():
        fail(f"Missing golden: {golden_p}")

    rep = json.loads(report_p.read_text(encoding="utf-8"))
    gol = json.loads(golden_p.read_text(encoding="utf-8"))

    rep_s = rep.get("scenarios") or {}
    gol_s = gol.get("scenarios") or {}

    scenario_ids = [s.strip() for s in args.scenarios.split(",") if s.strip()]

    # tolerances (v1)
    tol = {
        "decision_flip_rate": 0.005,
        "status_flip_rate": 0.005,
        "risk_delta_p95": 0.01,
        "human_significance_delta_p95": 0.01,
        "biometric_consensus_drop_p95": 0.01,
        "axiom_lock_rate": 0.001,
    }

    def max_metric(sdict: Dict[str, Any], key: str) -> float:
        # sdict is scenario object (from report or golden summary)
        # golden has {key: {max, median}}, report has perturbations list with metrics
        if "perturbations" in sdict:
            perts = sdict.get("perturbations") or []
            vals = []
            for p in perts:
                m = (p.get("metrics") or {})
                if key in m:
                    vals.append(float(m[key]))
            return max(vals) if vals else 0.0
        v = sdict.get(key)
        if isinstance(v, dict) and "max" in v:
            return float(v["max"])
        return 0.0

    for sid in scenario_ids:
        if sid not in rep_s:
            fail(f"Scenario missing in report: {sid}")
        if sid not in gol_s:
            fail(f"Scenario missing in golden: {sid}")

        rs = rep_s[sid]
        gs = gol_s[sid]

        # pass must not regress
        if bool(gs.get("pass", False)) and not bool(rs.get("pass", False)):
            fail(f"{sid}: pass regressed true -> false")

        for k, t in tol.items():
            rmax = max_metric(rs, k)
            gmax = max_metric(gs, k)
            if rmax > (gmax + t):
                fail(f"{sid}: {k} max drifted: {rmax} > {gmax}+{t}")

    print("PASS: robustness golden drift within tolerance")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
