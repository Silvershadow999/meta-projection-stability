# Evaluation Protocol (v1)

This protocol defines how to run, validate, and interpret evaluation outputs for `meta-projection-stability`.

## Inputs (sources of truth)
- Scenario manifests: `scenarios/*.json`
- Telemetry contract: `src/meta_projection_stability/types.py`
- Run state schema: `src/meta_projection_stability/state.py`
- Robustness expectations: `robustness_expectations` fields in manifests
- Golden references: `artifacts/robustness_golden.json`

## Outputs (artifacts)
- `artifacts/results.jsonl` — structured telemetry stream (events)
- `artifacts/eval_report.md` — human-readable evaluation summary
- `artifacts/robustness_report.json` — robustness metrics per scenario/perturbation
- `artifacts/robustness_golden.json` — reduced baseline summary for regression drift checks

## Local execution (recommended)
### A) Evaluation pipeline (telemetry + report)
```bash
./scripts/run_eval_clean.sh
python scripts/validate_results.py --in artifacts/results.jsonl --require-scenarios baseline,adversarial_min
python scripts/eval_report.py --in artifacts/results.jsonl --out artifacts/eval_report.md

