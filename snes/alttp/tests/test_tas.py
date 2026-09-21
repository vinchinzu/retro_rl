"""Unit tests for ALttP TAS movie import (no emulator, no ROM)."""

from __future__ import annotations

import zipfile
from pathlib import Path

import pytest

from retro_harness.controls import (
    SNES_A,
    SNES_B,
    SNES_LEFT,
    SNES_RIGHT,
    SNES_START,
    SNES_X,
)
from alttp.tas.bk2 import parse_bk2, parse_logkey_p1_to_env
from alttp.tas.catalog import MOVIES, SKIPPED, by_filename, fetchable
from alttp.tas.lsmv import parse_lsmv
from alttp.tas.rle import (
    GAME_NAME,
    compress_snes12_rle,
    expand_snes12_rle,
    frames_to_snes12_rle_payload,
    load_snes12_rle_seed,
)
from alttp.tas.slice import (
    GEG_FRAMES,
    REF_GEG,
    REF_WIP,
    SLICE_CATALOG,
    WIP_FRAMES,
    SliceSpec,
    export_slice,
    slice_frames,
    stub_slice_ids,
    verified_slice_ids,
)


def _write_lsmv(path: Path, input_text: str, *, extra: dict[str, str] | None = None) -> Path:
    with zipfile.ZipFile(path, "w") as zf:
        zf.writestr("gametype", "snes_ntsc\n")
        zf.writestr("systemid", "lsnes-rr1\n")
        zf.writestr("controlsversion", "0\n")
        zf.writestr("coreversion", "test\n")
        zf.writestr("projectid", "x\n")
        zf.writestr("rrdata", b"")
        zf.writestr("input", input_text)
        for key, value in (extra or {}).items():
            zf.writestr(key, value)
    return path


def _write_bk2(path: Path, log: str) -> Path:
    with zipfile.ZipFile(path, "w") as zf:
        zf.writestr("Header.txt", "GameName A Link to the Past\nPlatform SNES\n")
        zf.writestr("Input Log.txt", log)
    return path


def test_parse_minimal_lsmv(tmp_path: Path) -> None:
    buf = _write_lsmv(
        tmp_path / "tiny.lsmv",
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


def test_parse_ygamepad16_truncates_to_snes12(tmp_path: Path) -> None:
    """GEG #3898M uses 16-char P1 + extra ports; first 12 are BYsSudlrAXLR."""
    buf = _write_lsmv(
        tmp_path / "geg16.lsmv",
        "F.|................|................\n"
        "F.|.........X......|................\n"
        "F.|...S...r.XL..1..|........A.......\n",
        extra={"port1": "ygamepad16\n"},
    )
    movie = parse_lsmv(buf)
    assert movie.num_frames == 3
    assert not any(movie.frames[0])
    assert movie.frames[1][SNES_X] == 1
    assert movie.frames[2][SNES_START] == 1
    assert movie.frames[2][SNES_RIGHT] == 1
    assert movie.frames[2][SNES_X] == 1
    assert movie.raw_p1[1] == ".........X.."


def test_parse_minimal_bk2(tmp_path: Path) -> None:
    log = (
        "[Input]\n"
        "LogKey:#Reset|Power|#P1 Up|P1 Down|P1 Left|P1 Right|"
        "P1 Select|P1 Start|P1 Y|P1 B|P1 X|P1 A|P1 L|P1 R|\n"
        "|..|............|\n"
        "|..|.....S......|\n"
        "|..|...R...B....|\n"
    )
    movie = parse_bk2(_write_bk2(tmp_path / "tiny.bk2", log))
    assert movie.num_frames == 3
    assert movie.frames[1][SNES_START] == 1
    assert movie.frames[2][SNES_RIGHT] == 1
    assert movie.frames[2][SNES_B] == 1
    mapped = parse_logkey_p1_to_env(movie.logkey or "")
    assert mapped is not None
    assert mapped[5] == SNES_START
    assert mapped[7] == SNES_B
    assert mapped[9] == SNES_A


def test_rle_roundtrip_preserves_lr() -> None:
    frames = [[0] * 12, [0] * 12]
    frames[1][SNES_LEFT] = 1
    frames[1][SNES_RIGHT] = 1
    frames[1][SNES_B] = 1
    segs = compress_snes12_rle(frames)
    back = expand_snes12_rle({"segments": segs})
    assert back == frames
    assert back[1][SNES_LEFT] and back[1][SNES_RIGHT]


def test_rle_payload_is_alttp_game() -> None:
    payload = frames_to_snes12_rle_payload(
        [[0] * 12],
        route_id="unit",
        source="test",
    )
    assert payload["format"] == "snes12_rle"
    assert payload["game_name"] == GAME_NAME
    assert payload["num_frames"] == 1


def test_export_slice_from_synthetic_lsmv(tmp_path: Path) -> None:
    movie = _write_lsmv(
        tmp_path / "open.lsmv",
        "F.|............\n" * 10 + "F.|...S........\n" * 5,
    )
    spec = SliceSpec(
        id="synthetic_menu",
        movie=movie,
        kind="lsmv",
        start=8,
        end=15,
        source="unit",
        notes="synthetic",
        tags=("test",),
    )
    out = tmp_path / "menu.json"
    payload = export_slice(spec, out_path=out)
    assert out.exists()
    assert payload["num_frames"] == 7
    assert payload["game_name"] == GAME_NAME
    assert payload["verified"] is True
    frames = expand_snes12_rle(load_snes12_rle_seed(out))
    assert len(frames) == 7
    assert frames[2][SNES_START] == 1


def test_export_refuses_unverified_stubs() -> None:
    with pytest.raises(ValueError, match="unverified stub"):
        export_slice("castle")
    with pytest.raises(ValueError, match="unverified stub"):
        export_slice("sewers")
    with pytest.raises(ValueError, match="unverified stub"):
        export_slice("sanctuary")


def test_opening_stubs_are_catalogued_not_claimed() -> None:
    for sid in ("castle", "sewers", "sanctuary"):
        sp = SLICE_CATALOG[sid]
        assert sp.verified is False
        assert sp.start is None
        assert "stub" in sp.tags
        assert "Do not claim this plays" in sp.notes
    assert stub_slice_ids() == ["castle", "sewers", "sanctuary"]
    for sid in verified_slice_ids():
        assert SLICE_CATALOG[sid].verified is True
        assert SLICE_CATALOG[sid].start is not None


def test_catalog_urls_and_shas() -> None:
    names = [m.filename for m in MOVIES]
    assert len(names) == len(set(names))
    assert all(m.url.startswith("https://tasvideos.org/") for m in fetchable())
    assert all(len(m.sha256) == 64 for m in MOVIES)
    geg = by_filename("fmp_geg_3898M.lsmv")
    assert geg.expected_frames == GEG_FRAMES
    assert geg.kind == "lsmv"
    wip = by_filename("m_riss_100_wip_10k.bk2")
    assert wip.expected_frames == WIP_FRAMES
    assert wip.region == "US"
    reasons = " ".join(s.skip_reason or "" for s in SKIPPED)
    assert "RAM watch" in reasons
    assert "Lua" in reasons


def test_slice_frames_bounds() -> None:
    frames = [[0] * 12] * 100
    body = slice_frames(frames, 10, 20)
    assert len(body) == 10
    with pytest.raises(ValueError):
        slice_frames(frames, -1, 10)


def test_parse_vendored_geg_if_present() -> None:
    if not REF_GEG.exists():
        pytest.skip("missing fmp GEG LSMV (run python -m alttp.tas.fetch_refs)")
    movie = parse_lsmv(REF_GEG)
    assert movie.num_frames == GEG_FRAMES
    assert movie.summary()["first_nonzero_frame"] == 395
    assert movie.frames[395][SNES_X] == 1
    assert movie.meta.get("gametype") == "snes_ntsc"


def test_parse_vendored_wip_if_present() -> None:
    if not REF_WIP.exists():
        pytest.skip("missing m_riss WIP BK2 (run python -m alttp.tas.fetch_refs)")
    movie = parse_bk2(REF_WIP)
    assert movie.num_frames == WIP_FRAMES
    assert movie.summary()["first_nonzero_frame"] == 958
    assert movie.frames[958][SNES_START] == 1
    assert "Link to the Past" in (movie.header.get("GameName") or "")
