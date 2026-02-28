# Goldens (Regression Baselines)

This folder stores *expected summaries* for scenarios in a stable, versioned format.

Design goals:
- Small JSON files (commit-friendly)
- Derived from `artifacts/results.jsonl` (telemetry)
- Used by `scripts/validate_goldens.py` to detect regressions

File naming:
- `<scenario_id>.json` (e.g., `baseline.json`, `adversarial_min.json`)

Schema version:
- `schema_version`: "golden_summary_v1"
