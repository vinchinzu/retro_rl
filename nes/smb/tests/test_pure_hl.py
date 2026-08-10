"""Pure HappyLee track isolation (no emulator required for path gates)."""

from __future__ import annotations

import pytest

from smb.paths import MODELS_DIR
from smb.tas import pure_hl as ph


def test_track_status_blocks_8_4_without_gate() -> None:
    st = ph.track_status()
    assert st["track"] == "pure_hl"
    assert st["gate_8_3_open"] is False or ph.PURE_8_3_GATE.exists()
    # Hard block API always returns structured refusal when gate closed.
    r = ph.refuse_8_4_until_gate()
    if not ph.pure_8_3_gate_open():
        assert r["allowed"] is False
        assert "BLOCKED" in r["message"]


def test_assert_pure_write_path_accepts_track_dirs() -> None:
    ph.ensure_pure_dirs()
    ok = ph.assert_pure_write_path(ph.PURE_HL_EVIDENCE / "unit_test_ok.json")
    assert ok.name == "unit_test_ok.json"
    ok2 = ph.assert_pure_write_path(ph.PURE_HL_MODELS / "smb_8_3_pure_hl.json")
    assert ok2.parent == ph.PURE_HL_MODELS.resolve()


def test_assert_pure_write_path_rejects_outside_and_protected() -> None:
    with pytest.raises(RuntimeError, match="outside track"):
        ph.assert_pure_write_path(MODELS_DIR / "smb_1_1_to_ending_natural_82.json")
    with pytest.raises(RuntimeError, match="protected name"):
        # Even under pure_hl dir, forbidden basenames are refused.
        ph.assert_pure_write_path(
            ph.PURE_HL_MODELS / "smb_1_1_to_ending_natural_82.json"
        )
    with pytest.raises(RuntimeError, match="protected name"):
        ph.assert_pure_write_path(ph.PURE_HL_MODELS / "smb_happylee_hybrid_v2_fx84.json")
    with pytest.raises(RuntimeError, match="protected name"):
        ph.assert_pure_write_path(
            ph.PURE_HL_MODELS / "smb_8_3_stitchless_skills_leave.json"
        )


def test_write_json_only_under_pure_hl(tmp_path, monkeypatch) -> None:
    # Use real pure_hl dirs (assert is path-based); write then delete evidence scrap.
    path = ph.PURE_HL_EVIDENCE / "_unit_write_test.json"
    out = ph.write_json(path, {"ok": True, "track": "pure_hl"})
    assert out.exists()
    assert out.read_text(encoding="utf-8")
    out.unlink(missing_ok=True)


def test_select_leave_fan_prefers_fast_and_default() -> None:
    unique = [
        {"si82": 10850, "leave82": 2210, "timer": 300},
        {"si82": 10910, "leave82": 2209, "timer": 301},
        {"si82": 10920, "leave82": 2220, "timer": 290},
        {"si82": 10880, "leave82": 2250, "timer": 280},
        {"si82": 10900, "leave82": 2300, "timer": 270},
        {"si82": 10930, "leave82": 2400, "timer": 260},
    ]
    fan = ph.select_leave_fan(unique, top_leaves=3, default_si82=10910)
    assert fan[0]["leave82"] == 2209  # fastest first
    assert any(r["si82"] == 10910 for r in fan)
    assert len(fan) >= 3
    assert len(fan) <= 5


def test_select_leave_fan_empty() -> None:
    assert ph.select_leave_fan([]) == []

