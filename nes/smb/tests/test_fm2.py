"""Unit tests for FCEUX FM2 import + StageSpec surface (no emulator required)."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from retro_harness.controls import NES_A, NES_B, NES_LEFT, NES_RIGHT, NES_START
from smb.paths import GAME_DIR, MODELS_DIR
from smb.policy import expand_nes9_rle, load_nes9_rle_seed
from smb.tas.fm2 import parse_fm2
from smb.tas.replay import to_action9
from smb.tas.slice import (
    HL_1_2_FM2_START,
    HL_1_2_W4_FRAMES,
    HL_4_1_FM2_START,
    HL_4_1_LEAVE_FRAMES,
    HL_4_2_FM2_START,
    HL_4_2_W8_FRAMES,
    SliceProbe,
    export_1_2_slice,
    export_4_1_slice,
    export_4_2_slice,
    export_stage_slice,
    is_4_1_control,
    is_4_2_control,
)
from smb.tas.stages import (
    STAGE_1_2,
    STAGES,
    get_stage,
)

REF = GAME_DIR / "tas" / "ref" / "happylee_warps_1715M.fm2"


def test_parse_happylee_summary() -> None:
    if not REF.exists():
        return  # optional vendored artifact
    movie = parse_fm2(REF)
    assert movie.num_frames == 17_868
    assert movie.lr_frames == 85
    # First non-idle is Start (FM2 letter T = sTart), not Select
    assert movie.summary()["first_nonzero_frame"] == 41
    assert movie.frames[41][NES_START] == 1
    assert movie.frames[41][2] == 0  # SELECT off


def test_fm2_button_map_lr_and_a() -> None:
    if not REF.exists():
        return
    movie = parse_fm2(REF)
    # frame 196: RL......  → Left+Right
    assert movie.raw_p1[196] == "RL......"
    assert movie.frames[196][NES_LEFT] == 1
    assert movie.frames[196][NES_RIGHT] == 1
    # frame 197: R......A
    assert movie.raw_p1[197].endswith("A")
    assert movie.frames[197][NES_RIGHT] == 1
    assert movie.frames[197][NES_A] == 1


def test_parse_minimal_inline(tmp_path: Path) -> None:
    path = tmp_path / "tiny.fm2"
    path.write_text(
        "version 3\n"
        "romFilename test.nes\n"
        "|0|........||\n"
        "|0|R.....B.||\n"
        "|0|RL.....A||\n",
        encoding="utf-8",
    )
    movie = parse_fm2(path)
    assert movie.num_frames == 3
    assert movie.frames[1][NES_RIGHT] == 1
    assert movie.frames[1][NES_B] == 1
    assert movie.frames[2][NES_LEFT] == 1
    assert movie.frames[2][NES_RIGHT] == 1
    assert movie.frames[2][NES_A] == 1
    assert movie.lr_frames == 1


def test_happylee_1_1_slice_metadata() -> None:
    path = MODELS_DIR / "smb_1_1_happylee_slice.json"
    if not path.exists():
        return
    data = load_nes9_rle_seed(path)
    assert data["num_frames"] == 1733
    assert data.get("fm2_start_index") == 190
    frames = expand_nes9_rle(data)
    assert len(frames) == 1733
    # First non-idle-ish run uses L+R accel (do not sanitize)
    lr = sum(1 for f in frames if f[NES_LEFT] and f[NES_RIGHT])
    assert lr >= 1


def test_stage_1_2_table_matches_constants() -> None:
    """STAGE_1_2 fm2 start/body match module constants; get_stage works."""
    assert STAGE_1_2.fm2_start == HL_1_2_FM2_START == 2109
    assert STAGE_1_2.body_frames == HL_1_2_W4_FRAMES == 1657
    assert get_stage("1-2") is STAGE_1_2
    assert get_stage("1_2") is STAGE_1_2
    assert "1-2" in STAGES
    assert STAGES["1-2"].seed_name == "smb_1_2_happylee_slice.json"


def test_slice_probe_leave_frame_w4_ok() -> None:
    """SliceProbe.leave_frame is canonical; .w4 is a read/write alias; .ok."""
    p = SliceProbe(start_idx=2109, leave_frame=1657)
    assert p.leave_frame == 1657
    assert p.w4 == 1657
    assert p.ok is True

    p.w4 = 100
    assert p.leave_frame == 100
    assert p.w4 == 100

    dead = SliceProbe(start_idx=0, leave_frame=50, death=40)
    assert dead.ok is False
    no_leave = SliceProbe(start_idx=0, max_x=900)
    assert no_leave.ok is False
    assert no_leave.w4 is None

    d = p.to_dict()
    assert d["leave_frame"] == 100
    assert d["w4"] == 100  # legacy JSON key


def test_to_action9_preserves_left_right() -> None:
    """to_action9 must not sanitize simultaneous Left+Right."""
    frame = [0] * 9
    frame[NES_LEFT] = 1
    frame[NES_RIGHT] = 1
    frame[NES_B] = 1
    action = to_action9(frame)
    assert isinstance(action, np.ndarray)
    assert action.dtype == np.int8
    assert int(action[NES_LEFT]) == 1
    assert int(action[NES_RIGHT]) == 1
    assert int(action[NES_B]) == 1
    # short frames still pad to 9
    short = to_action9([1, 0, 0])
    assert short.shape == (9,)
    assert int(short[0]) == 1


def test_export_1_2_slice_length(tmp_path: Path) -> None:
    if not REF.exists():
        return
    out = tmp_path / "smb_1_2_hl.json"
    payload = export_1_2_slice(
        fm2_path=REF,
        start_idx=HL_1_2_FM2_START,
        w4_frames=HL_1_2_W4_FRAMES,
        out_path=out,
    )
    assert payload["num_frames"] == HL_1_2_W4_FRAMES
    assert out.exists()
    data = load_nes9_rle_seed(out)
    assert data["fm2_start_index"] == HL_1_2_FM2_START
    assert len(expand_nes9_rle(data)) == HL_1_2_W4_FRAMES


def test_export_stage_slice_via_get_stage(tmp_path: Path) -> None:
    """export_stage_slice(str|StageSpec) matches export_1_2_slice for 1-2."""
    if not REF.exists():
        return
    out = tmp_path / "smb_1_2_via_stage.json"
    payload = export_stage_slice(
        "1-2",
        fm2_path=REF,
        start_idx=HL_1_2_FM2_START,
        body_frames=HL_1_2_W4_FRAMES,
        out_path=out,
    )
    assert payload["num_frames"] == HL_1_2_W4_FRAMES
    assert payload.get("stage_id") == "1-2"
    data = load_nes9_rle_seed(out)
    assert data["fm2_start_index"] == HL_1_2_FM2_START
    assert len(expand_nes9_rle(data)) == HL_1_2_W4_FRAMES


def test_export_4_1_and_4_2_slice_lengths(tmp_path: Path) -> None:
    if not REF.exists():
        return
    out41 = tmp_path / "smb_4_1_hl.json"
    p41 = export_4_1_slice(
        fm2_path=REF,
        start_idx=HL_4_1_FM2_START,
        leave_frames=HL_4_1_LEAVE_FRAMES,
        out_path=out41,
    )
    assert p41["num_frames"] == HL_4_1_LEAVE_FRAMES
    data41 = load_nes9_rle_seed(out41)
    assert data41["fm2_start_index"] == HL_4_1_FM2_START
    assert len(expand_nes9_rle(data41)) == HL_4_1_LEAVE_FRAMES

    out42 = tmp_path / "smb_4_2_hl.json"
    p42 = export_4_2_slice(
        fm2_path=REF,
        start_idx=HL_4_2_FM2_START,
        w8_frames=HL_4_2_W8_FRAMES,
        out_path=out42,
    )
    assert p42["num_frames"] == HL_4_2_W8_FRAMES
    data42 = load_nes9_rle_seed(out42)
    assert data42["fm2_start_index"] == HL_4_2_FM2_START
    assert data42.get("target") == "world_8_entry"
    assert len(expand_nes9_rle(data42)) == HL_4_2_W8_FRAMES
    # L+R preserved somewhere in the warps movie body
    frames = expand_nes9_rle(data42)
    lr = sum(1 for f in frames if f[NES_LEFT] and f[NES_RIGHT])
    assert lr >= 1


def test_4_1_4_2_control_gates() -> None:
    """Unit predicates match the verified control fingerprints."""

    class _S:
        def __init__(self, **kw: int) -> None:
            self.world = kw.get("world", 3)
            self.level = kw.get("level", 0)
            self.oper_mode = kw.get("oper_mode", 1)
            self.player_state = kw.get("player_state", 7)
            self.dying = bool(kw.get("dying", 0))
            self.timer = kw.get("timer", 401)
            self.player_x = kw.get("player_x", 40)

    assert is_4_1_control(_S())
    assert not is_4_1_control(_S(timer=0))
    assert not is_4_1_control(_S(level=1))
    # 4-2: timer may be 0 at first control
    assert is_4_2_control(_S(level=1, timer=0, player_x=40))
    assert not is_4_2_control(_S(level=1, player_x=200))
    assert not is_4_2_control(_S(level=0, timer=0))


def test_happylee_4_1_4_2_slice_metadata() -> None:
    p41 = MODELS_DIR / "smb_4_1_happylee_slice.json"
    p42 = MODELS_DIR / "smb_4_2_happylee_slice.json"
    if not p41.exists() or not p42.exists():
        return
    d41 = load_nes9_rle_seed(p41)
    d42 = load_nes9_rle_seed(p42)
    assert d41["num_frames"] == HL_4_1_LEAVE_FRAMES
    assert d41.get("fm2_start_index") == HL_4_1_FM2_START
    assert d42["num_frames"] == HL_4_2_W8_FRAMES
    assert d42.get("fm2_start_index") == HL_4_2_FM2_START


def test_record_happylee_cli_targets() -> None:
    """record_happylee is importable and exposes verified chain targets."""
    from smb.scripts import record_happylee as rh

    assert rh.TARGETS == ("1-1", "w4", "w8", "ending")
    assert rh.SEED_1_1.name == "smb_1_1_happylee_slice.json"
    assert rh.SEED_4_2.name == "smb_4_2_happylee_slice.json"
    assert rh.SEED_HYBRID_ENDING.name == "smb_happylee_hybrid_v2_fx84.json"


def test_happylee_8_1_8_2_and_hybrid_metadata() -> None:
    p81 = MODELS_DIR / "smb_8_1_happylee_slice.json"
    p82 = MODELS_DIR / "smb_8_2_happylee_slice.json"
    ph = MODELS_DIR / "smb_happylee_hybrid_ending.json"
    if not p81.exists() or not p82.exists():
        return
    from smb.tas.slice import HL_8_1_FM2_START, HL_8_1_LEAVE_FRAMES, HL_8_2_FM2_START, HL_8_2_LEAVE_FRAMES

    d81 = load_nes9_rle_seed(p81)
    d82 = load_nes9_rle_seed(p82)
    assert d81["num_frames"] == HL_8_1_LEAVE_FRAMES
    assert d81.get("fm2_start_index") == HL_8_1_FM2_START
    assert d82["num_frames"] == HL_8_2_LEAVE_FRAMES
    assert d82.get("fm2_start_index") == HL_8_2_FM2_START
    if ph.exists():
        dh = load_nes9_rle_seed(ph)
        assert dh["num_frames"] == 18_769
        assert dh.get("target") == "8_4_ending"
        assert "hybrid" in dh
