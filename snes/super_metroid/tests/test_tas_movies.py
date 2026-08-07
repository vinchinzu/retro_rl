"""Unit tests for Super Metroid TAS movie import (no emulator)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from retro_harness.controls import (
    SNES_A,
    SNES_B,
    SNES_LEFT,
    SNES_RIGHT,
    SNES_START,
    SNES_UP,
)
from super_metroid.paths import GAME_DIR
from super_metroid.tas.bk2 import parse_bk2, parse_logkey_p1_to_env
from super_metroid.tas.lsmv import parse_lsmv
from super_metroid.tas.rle import (
    compress_snes12_rle,
    expand_snes12_rle,
    frames_to_snes12_rle_payload,
    load_snes12_rle_seed,
)
from super_metroid.tas.slice import (
    ANY_FRAMES,
    HUNDRED_FRAMES,
    REF_100,
    REF_ANY,
    REF_ANY_WIP,
    REF_SMTC4,
    SLICE_CATALOG,
    export_slice,
    finish_slice_ids,
    load_movie_frames,
    slice_frames,
)

REF = GAME_DIR / "tas" / "ref"


def test_pit_seed_first_jump_layout() -> None:
    """Product Pit seed: first jump at f150 hold 15; tail starts at f198."""
    path = GAME_DIR / "policies" / "morph" / "seg03_pit_room.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    btns = payload["raw_buttons"]
    assert len(btns) == 810
    # A rising edges
    jumps: list[tuple[int, int]] = []
    prev_a = 0
    i = 0
    while i < len(btns):
        a = int(btns[i][8])
        if a and not prev_a:
            j0 = i
            while i < len(btns) and int(btns[i][8]):
                i += 1
            jumps.append((j0, i - j0))
            prev_a = 0
            continue
        prev_a = a
        i += 1
    assert jumps[0] == (150, 15)
    assert jumps[1][0] == 198
    fj = payload.get("metadata", {}).get("first_jump", {})
    assert fj.get("hold_A") == 15
    assert fj.get("land_pin", {}).get("x") == 195
    assert fj.get("land_pin", {}).get("y") == 123


def test_parse_sniq_any_lsmv_summary() -> None:
    if not REF_ANY.exists():
        pytest.skip("missing sniq any% LSMV")
    movie = parse_lsmv(REF_ANY)
    assert movie.num_frames == ANY_FRAMES
    # First inputs: Start then A (menu mash), same as HappyLee style power-on
    assert movie.summary()["first_nonzero_frame"] == 0
    assert movie.frames[0][SNES_START] == 1
    assert movie.frames[1][SNES_A] == 1
    # Button field is BYsSudlrAXLR — raw length 12
    assert len(movie.raw_p1[0]) == 12
    assert movie.raw_p1[0][3] in "Ss"  # Start slot


def test_parse_sniq_100_bk2_summary() -> None:
    if not REF_100.exists():
        pytest.skip("missing sniq 100% BK2")
    movie = parse_bk2(REF_100)
    assert movie.num_frames == HUNDRED_FRAMES
    assert movie.header.get("GameName", "").lower().find("metroid") >= 0
    # Frame 0 idle, frame 1 Start (converter log)
    assert not any(movie.frames[0])
    assert movie.frames[1][SNES_START] == 1
    assert movie.frames[2][SNES_A] == 1


def test_lsmv_bk2_menu_alignment() -> None:
    """any% LSMV and 100% BK2 share the same early menu cadence."""
    if not (REF_ANY.exists() and REF_100.exists()):
        pytest.skip("missing refs")
    any_m = parse_lsmv(REF_ANY)
    hun = parse_bk2(REF_100)
    # Compare first 20 *pressed* patterns via START/A only
    any_sa = [(fr[SNES_START], fr[SNES_A]) for fr in any_m.frames[:40]]
    hun_sa = [(fr[SNES_START], fr[SNES_A]) for fr in hun.frames[1:41]]
    assert any_sa == hun_sa


def test_bk2_logkey_order_sniq_100() -> None:
    if not REF_100.exists():
        pytest.skip("missing ref")
    movie = parse_bk2(REF_100)
    assert movie.logkey is not None
    mapped = parse_logkey_p1_to_env(movie.logkey)
    assert mapped is not None
    # LogKey: Up Down Left Right Select Start Y B X A L R
    assert mapped[0] == SNES_UP
    assert mapped[5] == SNES_START
    assert mapped[7] == SNES_B
    assert mapped[9] == SNES_A


def test_parse_minimal_lsmv(tmp_path: Path) -> None:
    import zipfile

    buf = tmp_path / "tiny.lsmv"
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("gametype", "snes_ntsc\n")
        zf.writestr("systemid", "lsnes-rr1\n")
        zf.writestr("controlsversion", "0\n")
        zf.writestr("coreversion", "test\n")
        zf.writestr("projectid", "x\n")
        zf.writestr("rrdata", b"")
        zf.writestr(
            "input",
            # Positions: B Y s S u d l r A X L R (env order)
            "F.|............\n"
            "F.|...S........\n"
            "F.|B......r....\n",
        )
    movie = parse_lsmv(buf)
    assert movie.num_frames == 3
    assert movie.frames[1][SNES_START] == 1
    assert movie.frames[2][SNES_B] == 1
    assert movie.frames[2][SNES_RIGHT] == 1


def test_parse_minimal_bk2(tmp_path: Path) -> None:
    import zipfile

    buf = tmp_path / "tiny.bk2"
    log = (
        "[Input]\n"
        "LogKey:#Reset|Power|#P1 Up|P1 Down|P1 Left|P1 Right|"
        "P1 Select|P1 Start|P1 Y|P1 B|P1 X|P1 A|P1 L|P1 R|\n"
        "|..|............|\n"
        "|..|.....S......|\n"
        "|..|...R...B....|\n"
    )
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("Header.txt", "GameName Super Metroid\nPlatform SNES\n")
        zf.writestr("Input Log.txt", log)
    movie = parse_bk2(buf)
    assert movie.num_frames == 3
    assert movie.frames[1][SNES_START] == 1
    assert movie.frames[2][SNES_RIGHT] == 1
    assert movie.frames[2][SNES_B] == 1


def test_rle_roundtrip_preserves_lr() -> None:
    frames = [
        [0] * 12,
        [0] * 12,
    ]
    frames[1][SNES_LEFT] = 1
    frames[1][SNES_RIGHT] = 1
    frames[1][SNES_B] = 1
    segs = compress_snes12_rle(frames)
    back = expand_snes12_rle({"segments": segs})
    assert back == frames
    assert back[1][SNES_LEFT] and back[1][SNES_RIGHT]


def test_export_menu_slice(tmp_path: Path) -> None:
    if not REF_ANY.exists():
        pytest.skip("missing ref")
    out = tmp_path / "menu.json"
    payload = export_slice("sniq_any_menu", out_path=out)
    assert out.exists()
    assert payload["num_frames"] == 600
    assert payload["format"] == "snes12_rle"
    data = load_snes12_rle_seed(out)
    frames = expand_snes12_rle(data)
    assert len(frames) == 600
    assert frames[0][SNES_START] == 1


def test_export_finish_slices_exist_in_catalog() -> None:
    ids = finish_slice_ids()
    assert "sniq_any_full" in ids
    assert "sniq_100_full" in ids
    assert "sniq_any_tourian_escape" in ids
    assert "sniq_any_final_10k" in ids
    assert "sniq_100_final_15k" in ids
    for sid in ids:
        assert sid in SLICE_CATALOG


def test_final_10k_slice_length(tmp_path: Path) -> None:
    if not REF_ANY.exists():
        pytest.skip("missing ref")
    out = tmp_path / "final10k.json"
    payload = export_slice("sniq_any_final_10k", out_path=out)
    assert payload["num_frames"] == 10_000
    assert payload["movie_start_index"] == ANY_FRAMES - 10_000


def test_smtc4_short_bk2() -> None:
    if not REF_SMTC4.exists():
        pytest.skip("missing contest bk2")
    movie = parse_bk2(REF_SMTC4)
    assert movie.num_frames == 5_384
    assert movie.summary()["first_nonzero_frame"] is not None


def test_any_wip_frame_count() -> None:
    if not REF_ANY_WIP.exists():
        pytest.skip("missing wip")
    movie = parse_lsmv(REF_ANY_WIP)
    assert movie.num_frames == 55_037


def test_slice_frames_bounds() -> None:
    frames = load_movie_frames(REF_SMTC4, "bk2") if REF_SMTC4.exists() else [[0] * 12] * 100
    if not REF_SMTC4.exists():
        body = slice_frames(frames, 10, 20)
        assert len(body) == 10
        return
    body = slice_frames(frames, 0, 100)
    assert len(body) == 100


def test_payload_metadata_keys() -> None:
    frames = [[0] * 12, [0] * 12]
    frames[1][SNES_A] = 1
    payload = frames_to_snes12_rle_payload(
        frames, route_id="t", source="unit", extra={"movie_start_index": 3}
    )
    assert payload["movie_start_index"] == 3
    assert payload["num_frames"] == 2
    assert json.dumps(payload)  # serializable
