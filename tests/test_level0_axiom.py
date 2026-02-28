def test_level0_axiom_file_exists_and_nonempty():
    """
    Minimal presence test required by policy/axiom guard checks.
    """
    import pathlib

    root = pathlib.Path(__file__).resolve().parents[1]
    candidates = [
        root / "tests" / "level0" / "level0_axiom.md",
        root / "level0_axiom.md",
        root / "axiom_state.json",
    ]

    existing = [p for p in candidates if p.exists()]
    assert existing, f"Expected one of these to exist: {candidates}"
    assert any(p.stat().st_size > 0 for p in existing), f"All candidates were empty: {existing}"
