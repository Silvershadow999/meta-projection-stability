from pathlib import Path
import hashlib

EXPECTED_SHA256 = "913fdde15cf7fa1326d16706ac96aa685bbf92d461d249376feb8ed706bd2ba7"

def test_level0_axiom_fingerprint():
    p = Path("src/meta_projection_stability/level0_axiom.md")
    assert p.exists(), "level0_axiom.md fehlt"

    actual = hashlib.sha256(p.read_bytes()).hexdigest()
    assert actual == EXPECTED_SHA256, (
        "Level-0-Axiom wurde verändert! "
        f"expected={EXPECTED_SHA256}, actual={actual}"
    )
