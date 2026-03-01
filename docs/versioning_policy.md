# Versioning Policy (v1)

## Scope
This policy defines how we evolve:
- package version (`pyproject.toml`)
- telemetry contract and artifact schemas
- scenario manifest fields
- CI gates and validators

## SemVer guidance (project-level)
- **MAJOR**: breaking changes to public APIs or artifact/telemetry schemas
- **MINOR**: backwards-compatible feature additions
- **PATCH**: bug fixes and internal improvements

## Telemetry schema policy (strict)
Telemetry and artifacts are treated as an external contract.

### Rules
1) **Additive-only by default**
   - New fields are allowed if optional or have safe defaults.
   - Existing field meanings must not change silently.

2) **Schema versioning**
   - `TELEMETRY_SCHEMA_VERSION` must be bumped when the JSON schema changes.
   - If change is additive and compatible: bump **MINOR** of schema version.
   - If breaking: bump **MAJOR** of schema version and document migration.

3) **Stable top-level keys**
   - Prefer nested structures for extension rather than adding many new top-level fields.

4) **Validators are source of truth**
   - `scripts/validate_results.py`, `scripts/validate_robustness.py`,
     and `scripts/validate_robustness_goldens.py` encode machine-checkable invariants.
   - Any validator change must be justified in PR template.

## Scenario manifests policy
- Treat scenarios as **untrusted input**.
- Add fields additively (defaults preserved).
- Expectations fields (e.g., `robustness_expectations`) must have explicit rationale.

## Goldens policy
- Goldens are **regression anchors**, not “desired outputs”.
- Updating goldens requires:
  - justification + evidence (artifacts)
  - reviewer attention (CODEOWNERS)

## CI workflow policy
- CI gates are safety-critical.
- Workflow changes must not reduce gate strength without explicit justification.

