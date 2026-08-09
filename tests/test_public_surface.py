from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
README = ROOT / "README.md"


def test_readme_uses_real_paths_and_modeled_evidence_token() -> None:
    text = README.read_text(encoding="utf-8")
    assert "src/apple_ane_kv_quantizer.py" in text
    assert "src/quantizer.go" in text
    assert "src/MetalComputeEngine.swift" in text
    assert "MODELED_SCENARIO_NOT_HARDWARE_MEASUREMENT" in text


def test_readme_does_not_restore_nonexistent_or_unverified_front_door_claims() -> None:
    text = README.read_text(encoding="utf-8").casefold()
    forbidden = (
        "src/ane_quantizer.swift",
        "src/tensor_bridge.go",
        "src/quantizer_engine.py",
        "minimal accuracy loss",
        "mcp tool: `quantize_kv_cache()`",
    )
    assert all(marker not in text for marker in forbidden)


def test_readme_declares_non_affiliation_and_hardware_nonclaims() -> None:
    text = README.read_text(encoding="utf-8").casefold()
    assert "not affiliated with, endorsed by, or operated by apple" in text
    assert "does not claim proprietary apple access" in text
    assert "not proof of ane execution" in text
