# Release Checklist

## Preflight
- [ ] All required checks are green on `main` (axiom/eval/robustness/policy).
- [ ] README and safety docs reflect current behavior.
- [ ] Changelog updated (`CHANGELOG.md`).

## Version bump
- [ ] Decide SemVer bump: PATCH/MINOR/MAJOR.
- [ ] Update `pyproject.toml` version.
- [ ] Commit: `chore(release): bump version to X.Y.Z`.

## Tag
- [ ] Ensure you are on `main` and up to date: `git checkout main && git pull`.
- [ ] Create annotated tag: `git tag -a vX.Y.Z -m "Release vX.Y.Z"`.
- [ ] Push tag: `git push origin vX.Y.Z`.

## GitHub Release
- [ ] Draft a new release from tag `vX.Y.Z`.
- [ ] Title: `vX.Y.Z`.
- [ ] Paste relevant Changelog entries.
- [ ] Publish release.

## Post-release
- [ ] Start new “Unreleased” section in `CHANGELOG.md`.
- [ ] Optional: create a follow-up issue for next milestone items.
