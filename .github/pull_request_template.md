## Summary
Describe the change in 2–5 sentences.

## Why
What problem does this solve / what capability does it add?

## What changed (files / modules)
- [ ] Code
- [ ] Scenarios
- [ ] CI workflows
- [ ] Docs only

List key files:
- 

---

## Safety gates impact (required)
Which gates are affected by this PR?

- [ ] axiom-guard (Level-0 integrity / fingerprint)
- [ ] eval-gate (results.jsonl invariants + eval_report)
- [ ] robustness-gate (robustness_report + expectations + goldens)

If a gate was modified, explain why and what prevents weakening:
- 

---

## Artifacts / Evidence (required)
Attach or reference evidence from CI artifacts or local runs.

- `artifacts/results.jsonl` (if eval changes)
- `artifacts/eval_report.md` (if eval changes)
- `artifacts/robustness_report.json` (if robustness changes)
- `artifacts/robustness_golden.json` (if goldens changed)

Evidence notes:
- 

---

## Expectations / Goldens / Threshold changes (required when applicable)

### Scenario expectations
- [ ] `robustness_expectations` changed
- [ ] Expected boundaries / metrics keys changed

Justification:
- 

### Goldens (regression)
- [ ] `artifacts/robustness_golden.json` updated

Justification:
- 

---

## Risk / Limitations
What could break? What is intentionally not addressed?

- 

---

## Checklist (Definition of Done)
- [ ] Required checks are green (axiom/eval/robustness)
- [ ] Schema changes are additive/backwards compatible
- [ ] New/changed scenarios remain deterministic (seeded)
- [ ] If goldens/thresholds changed: justification included above
- [ ] Docs updated if behavior/contract changed
