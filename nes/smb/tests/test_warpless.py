"""Warpless / 32-exit TAS import (HappyLee & Mars608 #3728M). No emulator."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from retro_harness.controls import NES_LEFT, NES_RIGHT, NES_START
from smb.tas.annotate import AnnotateState, dash_key, is_live_control, stage_label
from smb.tas.fm2 import parse_fm2, parse_movie
from smb.paths import MODELS_DIR
from smb.policy import expand_nes9_rle, load_nes9_rle_seed
from smb.scripts.annotate_fm2 import export_1_3_slice
from smb.tas.warpless_extract import (
    STALL_FRAMES,
    _trial_score,
    export_warpless_slice,
)
from smb.tas.warpless import (
    CHAIN_TARGETS,
    WARPLESS_BK2,
    WARPLESS_FIRST_LR,
    WARPLESS_FM2,
    WARPLESS_FRAMES,
    WARPLESS_LEGS,
    WARPLESS_PUBLICATION_ID,
    WARPLESS_START_FRAME,
    WL_1_1_FM2_START,
    WL_1_1_LEAVE_FRAMES,
    WL_1_1_SETTLE,
    WL_1_2_CTRL_WAIT,
    WL_1_2_FM2_START,
    WL_1_2_LEAVE_FRAMES,
    WL_1_3_CTRL_WAIT,
    WL_1_3_FM2_HINT,
    WL_1_3_FM2_START,
    WL_1_3_LEAVE_FRAMES,
    WL_1_4_CTRL_WAIT,
    WL_1_4_FM2_HINT,
    WL_1_4_FM2_START,
    WL_1_4_LEAVE_FRAMES,
    WL_1_4_SEED,
    WL_2_1_CTRL_WAIT,
    WL_2_1_FM2_HINT,
    WL_2_1_FM2_START,
    WL_2_1_LEAVE_FRAMES,
    WL_2_1_SEED,
    WL_2_2_FM2_HINT,
    fm2_hint,
    get_leg,
    load_warpless_slice,
    require_warpless_slice,
    slice_path,
    slices_present,
    summary_dict,
    warpless_present,
)


def test_warpless_summary_offline() -> None:
    info = summary_dict()
    assert info["publication"] == "https://tasvideos.org/3728M"
    assert info["num_frames"] == WARPLESS_FRAMES
    assert info["route_id"] == "smb_all_exits"
    assert info["fm2_present"] is warpless_present()


def test_dash_clock_ignores_area_number() -> None:
    assert stage_label(0, 2) == "1-3"
    ug = SimpleNamespace(world=0, level=2, dash_level=1, oper_mode=1)
    assert dash_key(ug) == (0, 1)
    assert stage_label(*dash_key(ug)) == "1-2"


def test_annotate_state_records_1_2_then_1_3() -> None:
    state = AnnotateState()

    def snap(**kw: object) -> SimpleNamespace:
        base: dict[str, object] = {
            "world": 0,
            "level": 0,
            "dash_level": 0,
            "oper_mode": 1,
            "player_state": 8,
            "dying": False,
            "timer": 400,
            "player_x": 40,
            "player_y": 176,
            "lives": 2,
        }
        base.update(kw)
        return SimpleNamespace(**base)

    state.observe(snap(), 100)
    assert state.order == ["1-1"]
    assert state.marks["1-1"].first_control == 100
    # 1-2 underground: AreaNumber=2, LevelNumber stays 1
    state.observe(snap(level=2, dash_level=1, player_x=400), 500)
    assert state.order == ["1-1", "1-2"]
    assert state.marks["1-1"].leave_to == "1-2"
    assert is_live_control(snap(dash_level=2, player_x=40))
    state.observe(snap(dash_level=2, player_x=40), 900)
    assert state.order == ["1-1", "1-2", "1-3"]
    assert state.marks["1-2"].leave_to == "1-3"
    assert state.marks["1-3"].first_control == 900


def test_parse_warpless_fm2_if_present() -> None:
    if not warpless_present():
        return
    movie = parse_fm2(WARPLESS_FM2)
    assert movie.num_frames == WARPLESS_FRAMES
    assert movie.author and "HappyLee" in movie.author
    assert "Mars608" in (movie.author or "")
    assert movie.summary()["first_nonzero_frame"] == WARPLESS_START_FRAME
    assert movie.frames[WARPLESS_START_FRAME][NES_START] == 1
    assert movie.frames[WARPLESS_FIRST_LR][NES_LEFT] == 1
    assert movie.frames[WARPLESS_FIRST_LR][NES_RIGHT] == 1
    # 32-exit movie is much longer than warps (17,868)
    assert movie.num_frames > 60_000


def test_warpless_1_1_slice_metadata_if_present() -> None:
    path = MODELS_DIR / "smb_1_1_warpless_slice.json"
    if not path.exists():
        return
    data = load_nes9_rle_seed(path)
    assert data["num_frames"] == WL_1_1_LEAVE_FRAMES
    assert data.get("fm2_start_index") == WL_1_1_FM2_START
    assert data.get("route_id") == "smb_all_exits"
    assert data.get("stage_id") == "1-1"
    assert len(expand_nes9_rle(data)) == WL_1_1_LEAVE_FRAMES


def test_warpless_1_2_flag_slice_metadata_if_present() -> None:
    path = MODELS_DIR / "smb_1_2_warpless_flag_slice.json"
    if not path.exists():
        return
    data = load_nes9_rle_seed(path)
    assert data["num_frames"] == WL_1_2_LEAVE_FRAMES
    assert data.get("fm2_start_index") == WL_1_2_FM2_START
    assert data.get("target") == "1_3_control"
    assert data.get("route_id") == "smb_all_exits"
    assert "flag" in str(data.get("note", "")).lower()
    assert len(expand_nes9_rle(data)) == WL_1_2_LEAVE_FRAMES


def test_warpless_1_3_hint_follows_1_2_flag_leave() -> None:
    assert WL_1_3_FM2_HINT == WL_1_2_FM2_START + WL_1_2_LEAVE_FRAMES == 4653
    assert WL_1_3_FM2_START == WL_1_3_FM2_HINT
    assert WL_1_3_LEAVE_FRAMES == 1740
    assert WL_1_4_FM2_HINT == WL_1_3_FM2_START + WL_1_3_LEAVE_FRAMES == 6393
    assert fm2_hint("1-4") == WL_1_4_FM2_HINT == 6393
    assert WL_1_4_FM2_START == WL_1_4_FM2_HINT
    assert WL_1_4_LEAVE_FRAMES == 1702
    assert WL_1_4_CTRL_WAIT == 0
    assert WL_2_1_FM2_HINT == WL_1_4_FM2_START + WL_1_4_LEAVE_FRAMES == 8095
    assert fm2_hint("2-1") == WL_2_1_FM2_HINT
    assert WL_2_1_FM2_START == 7999
    assert WL_2_1_LEAVE_FRAMES == 2440
    assert WL_2_1_CTRL_WAIT == 0
    assert WL_2_2_FM2_HINT == WL_2_1_FM2_START + WL_2_1_LEAVE_FRAMES == 10439
    assert fm2_hint("2-2") == WL_2_2_FM2_HINT


def test_warpless_legs_cover_32_exits() -> None:
    assert len(WARPLESS_LEGS) == 32
    assert WARPLESS_LEGS[0].id == "1-1"
    assert WARPLESS_LEGS[-1].id == "8-4"
    assert WARPLESS_LEGS[-1].leave_ending
    castle = get_leg("1-4")
    assert castle.world == 0 and castle.dash == 3
    assert castle.leave_world == 1 and castle.leave_dash == 0
    assert castle.leave_outcome == "2_1_control"
    assert castle.seed_name == WL_1_4_SEED
    assert castle.max_play == 2500
    flag22 = get_leg("2-2")
    assert flag22.world == 1 and flag22.dash == 1
    assert flag22.leave_id == "2-3"
    assert "flag" in flag22.note.lower()
    assert get_leg("8-4").leave_outcome == "ending_axe"
    snap_14 = SimpleNamespace(
        world=0, level=4, dash_level=3, oper_mode=1, player_state=7,
        timer=301, player_x=40, player_y=80, dying=False, lives=2,
    )
    snap_21 = SimpleNamespace(
        world=1, level=0, dash_level=0, oper_mode=1, player_state=7,
        timer=301, player_x=40, player_y=176, dying=False, lives=2,
    )
    assert get_leg("1-3").leave(snap_14)
    assert get_leg("1-4").control(snap_14)
    assert get_leg("1-4").leave(snap_21)
    assert not get_leg("1-4").leave(snap_14)
    drop = SimpleNamespace(
        world=1, level=2, dash_level=1, oper_mode=1, player_state=7,
        timer=401, player_x=40, player_y=0, dying=False, lives=2,
    )
    land = SimpleNamespace(
        world=1, level=2, dash_level=1, oper_mode=1, player_state=8,
        timer=399, player_x=40, player_y=176, dying=False, lives=2,
    )
    assert get_leg("2-1").leave(drop)
    assert get_leg("2-1").leave(land)
    assert get_leg("2-2").control(drop)
    assert get_leg("2-2").control(land)
    falling = SimpleNamespace(**{**drop.__dict__, "player_y": 80})
    assert not get_leg("2-2").control(falling)


def test_trial_score_prefers_clear_then_short_then_few_leads() -> None:
    center = 10451
    miss = {"start_idx": 10451, "max_x": 1315, "leave_frame": None, "lead_idle": 0}
    hit = {
        "start_idx": 10440,
        "max_x": 3100,
        "leave_frame": 2400,
        "lead_idle": 3,
        "warped": False,
    }
    closer = {**hit, "start_idx": 10451, "lead_idle": 3}
    fewer = {**hit, "start_idx": 10451, "lead_idle": 0}
    assert _trial_score(hit, center) > _trial_score(miss, center)
    assert _trial_score(fewer, center) > _trial_score(closer, center)
    near_miss = {"start_idx": 10439, "max_x": 563, "leave_frame": None, "lead_idle": 0}
    far_clip = {"start_idx": 10451, "max_x": 2225, "leave_frame": None, "lead_idle": 0}
    assert _trial_score(far_clip, center) > _trial_score(near_miss, center)
    assert STALL_FRAMES >= 120


def test_export_2_2_slice_stores_lead_idle(tmp_path: Path) -> None:
    frames = [[0] * 9 for _ in range(40)]
    frames[15][7] = 1
    dest = tmp_path / "smb_2_2_warpless_slice.json"
    payload = export_warpless_slice(
        frames,
        stage_id="2-2",
        start_idx=15,
        body_frames=12,
        fm2_path=Path("happylee_mars608_warpless_3728M.fm2"),
        out_path=dest,
        lead_idle=4,
    )
    assert dest.is_file()
    assert payload["num_frames"] == 12
    assert payload["fm2_start_index"] == 15
    assert payload["lead_idle"] == 4
    assert payload["target"] == "2_3_control"
    assert payload["stage_id"] == "2-2"
    assert "flag" in str(payload.get("note", "")).lower()
    assert require_warpless_slice(payload, stage_id="2-2") is payload
    assert len(expand_nes9_rle(payload)) == 12


def test_export_1_3_slice_metadata(tmp_path: Path) -> None:
    frames = [[0] * 9 for _ in range(40)]
    frames[10][7] = 1
    dest = tmp_path / "smb_1_3_warpless_slice.json"
    payload = export_1_3_slice(
        frames,
        start_idx=10,
        body_frames=20,
        fm2_path=Path("warpless.fm2"),
        out_path=dest,
    )
    assert dest.is_file()
    assert payload["num_frames"] == 20
    assert payload["fm2_start_index"] == 10
    assert payload["target"] == "1_4_control"
    assert payload["stage_id"] == "1-3"
    assert payload["route_id"] == "smb_all_exits"
    assert "1-3" in str(payload.get("note", ""))
    assert len(expand_nes9_rle(payload)) == 20


def test_export_1_4_slice_metadata(tmp_path: Path) -> None:
    frames = [[0] * 9 for _ in range(40)]
    frames[12][7] = 1
    dest = tmp_path / "smb_1_4_warpless_slice.json"
    payload = export_warpless_slice(
        frames,
        stage_id="1-4",
        start_idx=12,
        body_frames=18,
        fm2_path=Path("happylee_mars608_warpless_3728M.fm2"),
        out_path=dest,
    )
    assert dest.is_file()
    assert payload["num_frames"] == 18
    assert payload["fm2_start_index"] == 12
    assert payload["target"] == "2_1_control"
    assert payload["stage_id"] == "1-4"
    assert payload["route_id"] == "smb_all_exits"
    assert "3728M" in str(payload.get("source", ""))
    assert "warpless" in str(payload.get("source", "")).lower()
    assert require_warpless_slice(payload, stage_id="1-4") is payload
    assert len(expand_nes9_rle(payload)) == 18


def test_warpless_2_1_slice_metadata_if_present() -> None:
    path = MODELS_DIR / WL_2_1_SEED
    if not path.exists() or not WL_2_1_LEAVE_FRAMES:
        return
    data = load_nes9_rle_seed(path)
    assert data["num_frames"] == WL_2_1_LEAVE_FRAMES
    assert data.get("fm2_start_index") == WL_2_1_FM2_START
    assert data.get("target") == "2_2_control"
    assert data.get("route_id") == "smb_all_exits"
    assert data.get("stage_id") == "2-1"
    assert "3728M" in str(data.get("source", ""))
    assert "flag" not in str(data.get("note", "")).lower()
    assert len(expand_nes9_rle(data)) == WL_2_1_LEAVE_FRAMES


def test_warpless_1_4_slice_metadata_if_present() -> None:
    path = MODELS_DIR / WL_1_4_SEED
    if not path.exists() or not WL_1_4_LEAVE_FRAMES:
        return
    data = load_nes9_rle_seed(path)
    assert data["num_frames"] == WL_1_4_LEAVE_FRAMES
    assert data.get("fm2_start_index") == WL_1_4_FM2_START
    assert data.get("target") == "2_1_control"
    assert data.get("route_id") == "smb_all_exits"
    assert data.get("stage_id") == "1-4"
    assert "3728M" in str(data.get("source", ""))
    assert len(expand_nes9_rle(data)) == WL_1_4_LEAVE_FRAMES


def test_warpless_1_3_slice_metadata_if_present() -> None:
    path = MODELS_DIR / "smb_1_3_warpless_slice.json"
    if not path.exists() or not WL_1_3_LEAVE_FRAMES:
        return
    data = load_nes9_rle_seed(path)
    assert data["num_frames"] == WL_1_3_LEAVE_FRAMES
    assert data.get("fm2_start_index") == WL_1_3_FM2_START
    assert data.get("target") == "1_4_control"
    assert data.get("route_id") == "smb_all_exits"
    assert data.get("stage_id") == "1-3"
    assert len(expand_nes9_rle(data)) == WL_1_3_LEAVE_FRAMES


def test_require_warpless_slice_rejects_warp_any_percent() -> None:
    warp = {
        "source": "HappyLee warps #1715M FM2 @190",
        "route_id": "smb_1_1_happylee",
        "stage_id": "1-1",
        "note": "isolated Level1_1 warp slice",
    }
    with pytest.raises(ValueError, match="1715M"):
        require_warpless_slice(warp, stage_id="1-1")
    hand = {
        "source": "HappyLee 1-2 UG prefix + lift A19",
        "route_id": "smb_1_2_flag",
        "stage_id": "1-2",
        "note": "hand-built flag tail",
    }
    with pytest.raises(ValueError, match="warpless"):
        require_warpless_slice(hand, stage_id="1-2")
    ok = {
        "source": "HappyLee & Mars608 warpless #3728M FM2 @4653",
        "route_id": "smb_all_exits",
        "stage_id": "1-3",
        "note": "32-exit 1-3 athletic",
    }
    assert require_warpless_slice(ok, stage_id="1-3") is ok
    ok14 = {
        "source": "HappyLee & Mars608 warpless #3728M FM2 @6393",
        "route_id": "smb_all_exits",
        "stage_id": "1-4",
        "note": "32-exit 1-4 castle",
    }
    assert require_warpless_slice(ok14, stage_id="1-4") is ok14
    with pytest.raises(ValueError, match="1715M"):
        require_warpless_slice(
            {
                "source": "HappyLee warps #1715M FM2 @190",
                "route_id": "smb_all_exits",
                "stage_id": "1-4",
            },
            stage_id="1-4",
        )


def test_warpless_same_file_cuts_chain() -> None:
    """1-1 / 1-2 flag / 1-3 are consecutive movie windows, not warp leftovers."""
    assert WL_1_1_SETTLE == 2
    assert WL_1_1_FM2_START + WL_1_1_LEAVE_FRAMES + WL_1_2_CTRL_WAIT == WL_1_2_FM2_START
    assert WL_1_2_FM2_START + WL_1_2_LEAVE_FRAMES + WL_1_3_CTRL_WAIT == WL_1_3_FM2_START
    assert WL_1_3_CTRL_WAIT == 0
    assert WL_1_3_FM2_START + WL_1_3_LEAVE_FRAMES + WL_1_4_CTRL_WAIT == WL_1_4_FM2_START
    assert WL_1_4_FM2_START + WL_1_4_LEAVE_FRAMES + WL_2_1_CTRL_WAIT == WL_2_1_FM2_HINT
    assert WL_2_1_FM2_START + WL_2_1_LEAVE_FRAMES == WL_2_2_FM2_HINT
    assert CHAIN_TARGETS == ("1-1", "1-2", "1-3", "1-4", "2-1")
    for sid in CHAIN_TARGETS:
        path = slice_path(sid)
        if not path.is_file():
            continue
        data, frames = load_warpless_slice(sid)
        assert WARPLESS_PUBLICATION_ID in str(data.get("source", ""))
        assert "1715M" not in str(data.get("source", ""))
        assert data.get("route_id") == "smb_all_exits"
        assert Path(str(data.get("fm2", "x"))).name.startswith(
            "happylee_mars608_warpless_3728M"
        )
        assert len(frames) == int(data["num_frames"])
    assert slices_present("2-1") is all(slice_path(s).is_file() for s in CHAIN_TARGETS)


def test_on_disk_warp_seeds_are_not_warpless_cuts() -> None:
    """#1715M 1-1 and the hand-built flag body must not pass provenance."""
    warp_11 = MODELS_DIR / "smb_1_1_happylee_slice.json"
    if warp_11.is_file():
        with pytest.raises(ValueError):
            require_warpless_slice(load_nes9_rle_seed(warp_11), stage_id="1-1")
    hand_12 = MODELS_DIR / "smb_1_2_flag.json"
    if hand_12.is_file():
        with pytest.raises(ValueError):
            require_warpless_slice(load_nes9_rle_seed(hand_12), stage_id="1-2")
    warp_12 = MODELS_DIR / "smb_1_2_happylee_slice.json"
    if warp_12.is_file():
        with pytest.raises(ValueError):
            require_warpless_slice(load_nes9_rle_seed(warp_12), stage_id="1-2")


def test_parse_warpless_bk2_if_present() -> None:
    if not WARPLESS_BK2.exists():
        return
    fm2 = parse_fm2(WARPLESS_FM2)
    bk2 = parse_movie(WARPLESS_BK2)
    assert bk2.num_frames == fm2.num_frames
    assert bk2.frames == fm2.frames
