import hashlib
from pathlib import Path

EXPECTED_SHA256 = "a43f4127d20b23d11973b289eb38497fbc2c0e591fbb47b4ad4107c2efeccd5f"


def test_level0_axiom_hash():
    p = Path("src/meta_projection_stability/level0_axiom.md")
    assert p.exists(), "level0_axiom.md fehlt"

    actual = hashlib.sha256(p.read_bytes()).hexdigest()
    assert actual == EXPECTED_SHA256, (
        "Level-0-Axiom wurde verändert! "
        f"expected={EXPECTED_SHA256}, actual={actual}"
    )
