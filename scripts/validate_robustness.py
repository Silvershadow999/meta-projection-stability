#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict

from meta_projection_stability.scenario_manifest import load_by_id


def fail(msg: str) -> None:
    raise SystemExit(f"FAIL: {msg}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--report", default="artifacts/robustness_report.json")
    ap.add_argument("--scenarios", default="baseline,adversarial_min")
    args = ap.parse_args()

    report_path = Path(args.report)
    if not report_path.exists():
        fail(f"Report not found: {report_path}")

    rep = json.loads(report_path.read_text(encoding="utf-8"))
    if rep.get("schema") != "mps.robustness.v1":
        fail(f"Unexpected schema: {rep.get('schema')}")

    scen_rep: Dict[str, Any] = rep.get("scenarios") or {}
    scenario_ids = [s.strip() for s in args.scenarios.split(",") if s.strip()]

    for sid in scenario_ids:
        sm = load_by_id(sid)
        exp = (sm.to_dict().get("robustness_expectations") or {})
        if not exp:
            continue

        s = scen_rep.get(sid)
        if not isinstance(s, dict):
            fail(f"Missing scenario in report: {sid}")

        perts = s.get("perturbations") or []
        if not perts:
            fail(f"No perturbations reported for scenario: {sid}")

        def chk(m: Dict[str, Any], metric_key: str, exp_key: str):
            if metric_key not in m:
                fail(f"{sid}: missing metric {metric_key}")
            if exp_key not in exp:
                return
            if float(m[metric_key]) > float(exp[exp_key]):
                fail(f"{sid}: {metric_key}={m[metric_key]} exceeds {exp_key}={exp[exp_key]}")

        for p in perts:
            m = (p.get("metrics") or {})
            if not isinstance(m, dict):
                fail(f"{sid}: perturbation missing metrics dict")

            chk(m, "decision_flip_rate", "max_decision_flip_rate")
            chk(m, "status_flip_rate", "max_status_flip_rate")
            chk(m, "risk_delta_p95", "max_risk_delta_p95")
            chk(m, "human_significance_delta_p95", "max_h_sig_delta_p95")
            chk(m, "axiom_lock_rate", "max_axiom_lock_rate")
            chk(m, "biometric_consensus_drop_p95", "max_biometric_consensus_drop_p95")

    print("PASS: robustness expectations satisfied")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
