# Threat Model — meta-projection-stability

## 1. Scope
This threat model covers the **evaluation pipeline** and **safety instrumentation** of the `meta-projection-stability` repository, with emphasis on:

- **Structured telemetry** (schema-versioned events)
- **Explicit safety boundaries** (e.g., REVIEW/REFUSE/EMERGENCY_STOP)
- **Scenario-based evaluation** (baseline vs adversarial)
- **Reproducibility & provenance** (git commit/dirty, deterministic scenarios)

**Out of scope (for now):**
- Production deployment hardening
- Networked / multi-tenant execution
- Cryptographic attestations beyond basic provenance fields

## 2. System Overview
### Components
- **Telemetry Contract:** `src/meta_projection_stability/types.py`
- **Run State:** `src/meta_projection_stability/state.py`
- **Scenario Manifests:** `scenarios/*.json` (+ loader)
- **Runner:** `scripts/eval_runner.py`
- **Artifacts:** `artifacts/results.jsonl` (JSONL event log), `artifacts/eval_report.md` (derived)
- **Report Generator:** `scripts/eval_report.py`

### Data Flow (high level)
1) `eval_runner` loads `ScenarioManifest`  
2) Initializes `RunProvenance` + `RunState`  
3) Emits JSONL telemetry events to `artifacts/results.jsonl`  
4) `eval_report` aggregates JSONL into `artifacts/eval_report.md`

## 3. Assets & Security Objectives
### Assets
- **A1: Telemetry integrity** — events reflect what actually happened
- **A2: Provenance quality** — runs are attributable to a commit + reproducible scenario
- **A3: Boundary correctness** — safety boundaries trigger reliably and are logged
- **A4: Evaluation comparability** — scenarios are comparable across commits

### Security / Safety Objectives
- **O1:** Prevent silent failure (missing boundary triggers, missing events)
- **O2:** Detect regressions in boundary behavior or metrics
- **O3:** Maintain deterministic evaluation under scenario + seed control
- **O4:** Avoid over-claiming: keep safety limitations explicit (see Non-Goals)

## 4. Assumptions
- Local developer execution (single user, trusted environment)
- `artifacts/` may be ignored in git; results can be regenerated
- Scenario manifests are treated as *inputs* and may be adversarial or malformed
- No secrets are stored in telemetry

## 5. Threat Actors
- **TA1:** Accidental developer error (misconfiguration, partial edits, wrong file paths)
- **TA2:** Malicious contributor (PR introduces misleading telemetry or disables boundaries)
- **TA3:** Adversarial scenario author (malformed manifests, extreme parameters)
- **TA4:** Tooling/environment drift (Python version differences, dependency changes)

## 6. Attack Surfaces
- **AS1:** Scenario manifest parsing (`scenarios/*.json`)
- **AS2:** Telemetry writer / schema drift (`results.jsonl`)
- **AS3:** Boundary signaling (logic that sets `BoundarySignal`)
- **AS4:** Runner CLI + defaults (`scripts/eval_runner.py`)
- **AS5:** Report aggregation (mis-parsing, silent truncation)

## 7. Threats, Failure Modes, and Mitigations

### T1 — Schema drift / incompatible telemetry
**Failure mode:** event fields change silently; report breaks or misreads.  
**Mitigations:**
- Version field: `TELEMETRY_SCHEMA_VERSION`
- Additive-only changes; keep top-level keys stable
- Report parser tolerant to missing fields
- Regression checks (later): validate schema per line

### T2 — Missing provenance / irreproducible runs
**Failure mode:** cannot attribute a run to a commit or environment.  
**Mitigations:**
- `RunProvenance` includes `git_commit`, `git_dirty`, python/platform
- Runner always emits RUN_START with provenance payload

### T3 — Boundary not triggered or not logged
**Failure mode:** safety boundary logic triggers but telemetry does not record it; or vice versa.  
**Mitigations:**
- Boundary recorded in `RunState.boundaries`
- Dedicated `BOUNDARY` events (explicit event type)
- Report surfaces triggered boundary names per run

### T4 — Scenario manipulation / malformed inputs
**Failure mode:** scenario files missing required fields, wrong types, or extreme overrides causing undefined behavior.  
**Mitigations:**
- Loader performs light validation (`ScenarioManifestError`)
- Keep overrides as data; apply in controlled layer (runner)
- Add later: allowlist of overridable config keys

### T5 — “Telemetry forgery” by malicious code changes
**Failure mode:** code emits “ok” metrics despite failures; hides boundaries.  
**Mitigations:**
- Make boundary logic explicit and centrally tested
- Add later: minimal invariants tests (e.g., RUN_START/END must exist per run_id)
- Add later: cross-check state transitions vs emitted events

### T6 — Report misinterpretation / silent truncation
**Failure mode:** report ignores some runs or merges runs incorrectly.  
**Mitigations:**
- Grouping by `(run_id, scenario_id)`
- Show table of all runs + latest-per-scenario comparison
- Add later: include count of parsed lines vs written lines

## 8. Abuse Cases (Adversarial Reasoning)
- **AC1:** Disable boundaries under stress inputs → must be detectable via comparison baseline vs adversarial
- **AC2:** Inflate metrics to hide regressions → require metrics + boundaries to be consistent with state log
- **AC3:** Create malformed manifest to crash evaluation → loader must fail fast with clear error

## 9. Residual Risk
- Local environment is not hardened; a malicious local actor can always tamper with artifacts.
- Without cryptographic signing, telemetry integrity is “best effort” (provenance only).
- Boundary logic correctness depends on model integration (future step).

## 10. Next Steps (links)
- Safety Case: `docs/safety_case.md` (claims/arguments/evidence)
- Non-Goals: `docs/non_goals.md`
- Evaluation Protocol: define pass/fail criteria and regression gates
