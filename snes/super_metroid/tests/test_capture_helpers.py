"""Practice preset capture helpers (ROM-free)."""

from __future__ import annotations

import inspect

from super_metroid.paths import (
    INTEGRATION_DIR,
    PRACTICE_CONTRACTOR_STATE_DIR,
    PRACTICE_INTEGRATION_DIR,
    PRACTICE_REPERTOIRE_STATE_DIR,
)
from super_metroid.practice_repertoire.capture import (
    ADDR_CM_MENU_BANK,
    ADDR_LOAD_PRESET,
    XY_BAND,
    _fingerprint_ok,
    _session_rows,
    capture_sessions,
)
from super_metroid.practice_repertoire.catalog import PRODUCT_CATEGORY


class _State:
    def __init__(self, *, gs: int, room: int, x: int, y: int) -> None:
        self.game_state = gs
        self.room_id = room
        self.samus_x = x
        self.samus_y = y


def test_load_preset_address() -> None:
    assert ADDR_LOAD_PRESET == 0xFD5C
    assert ADDR_CM_MENU_BANK == 0xFE26
    assert XY_BAND == 8


def test_fingerprint_match_band() -> None:
    rec = {"room_id": 0x9E9F, "x": 0x0580, "y": 0x02A8}
    ok = _State(gs=8, room=0x9E9F, x=0x0580, y=0x02A8)
    assert _fingerprint_ok(ok, rec)
    near = _State(gs=8, room=0x9E9F, x=0x0583, y=0x02A4)
    assert _fingerprint_ok(near, rec)
    miss_room = _State(gs=8, room=0x91F8, x=0x0580, y=0x02A8)
    assert not _fingerprint_ok(miss_room, rec)
    miss_gs = _State(gs=9, room=0x9E9F, x=0x0580, y=0x02A8)
    assert not _fingerprint_ok(miss_gs, rec)
    miss_xy = _State(gs=8, room=0x9E9F, x=0x0580 + XY_BAND + 1, y=0x02A8)
    assert not _fingerprint_ok(miss_xy, rec)


def test_session_rows_kpdr25_and_ids() -> None:
    rows = _session_rows(category=PRODUCT_CATEGORY, limit=None)
    assert len(rows) >= 100
    assert all(rec["category"] == PRODUCT_CATEGORY for rec in rows)
    ids = ["kpdr25/crateria/morph", "kpdr25/wrecked_ship/phantoon"]
    picked = _session_rows(category=PRODUCT_CATEGORY, limit=None, ids=ids)
    assert {rec["id"] for rec in picked} == set(ids)


def test_contractor_state_dir_is_practice_integration() -> None:
    assert PRACTICE_CONTRACTOR_STATE_DIR == (
        PRACTICE_INTEGRATION_DIR / "practice_repertoire"
    )
    assert PRACTICE_REPERTOIRE_STATE_DIR == INTEGRATION_DIR / "practice_repertoire"
    assert PRACTICE_CONTRACTOR_STATE_DIR != PRACTICE_REPERTOIRE_STATE_DIR
    default = inspect.signature(capture_sessions).parameters["out_dir"].default
    assert default == PRACTICE_CONTRACTOR_STATE_DIR


def test_ensure_practice_integration_symlink_idempotent(
    tmp_path, monkeypatch
) -> None:
    from super_metroid.practice_repertoire import capture as cap

    rom = tmp_path / "practice.sfc"
    rom.write_bytes(b"rom")
    dest = tmp_path / "practice_int"
    vanilla = tmp_path / "vanilla"
    vanilla.mkdir()
    for name in ("data.json", "metadata.json", "scenario.json"):
        (vanilla / name).write_text("{}", encoding="utf-8")
    monkeypatch.setattr(cap, "PRACTICE_INTEGRATION_DIR", dest)
    monkeypatch.setattr(cap, "SHARED_PRACTICE_ROM", rom)
    monkeypatch.setattr(cap, "INTEGRATION_DIR", vanilla)

    cap.ensure_practice_integration()
    link = dest / "rom.sfc"
    assert link.is_symlink()
    assert link.resolve() == rom.resolve()
    first_ino = link.lstat().st_ino
    cap.ensure_practice_integration()
    assert link.is_symlink()
    assert link.lstat().st_ino == first_ino

    wrong = tmp_path / "other.sfc"
    wrong.write_bytes(b"nope")
    link.unlink()
    link.symlink_to(wrong)
    cap.ensure_practice_integration()
    assert link.resolve() == rom.resolve()
