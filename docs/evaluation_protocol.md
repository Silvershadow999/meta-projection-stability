# Evaluation Protocol — meta-projection-stability

## 1. Purpose
Define a repeatable evaluation protocol that produces **measurable outputs** and **pass/fail signals** suitable for regression gating.

## 2. Inputs
- Scenario manifests: `scenarios/*.json`
- Runner: `scripts/eval_runner.py`

## 3. Outputs (Artifacts)
- JSONL event log: `artifacts/results.jsonl`
- Human report: `artifacts/eval_report.md` (derived from JSONL)

## 4. Required Telemetry Invariants (Hard Requirements)
For each `(run_id, scenario_id)`:
1) Exactly one `run_start` event
2) Exactly one `run_end` event
3) If `n_steps > 0`, at least one `step` event must exist
4) A `metric` event must exist
5) Event ordering should be sane (start precedes end by timestamp)

## 5. Scenario Set (Minimum)
- `baseline` (deterministic reference)
- `adversarial_min` (boundary visibility & robustness)

## 6. Pass/Fail Criteria (Current)
### PASS if:
- All invariants hold for all runs parsed in the JSONL
- `baseline` produces **no** boundary events by default (unless explicitly expected)
- `adversarial_min` produces **at least one** boundary event (e.g., REVIEW) in the skeleton runner

### FAIL if:
- Any invariant is violated
- Missing provenance in run_start payload (git_commit may be null, but the field must exist)
- JSONL contains malformed lines (non-JSON)

## 7. Future Extensions (Not guaranteed yet)
- Allowlist of config overrides
- Per-scenario expected boundary sets and metric ranges
- CI regression baselines and delta thresholds
