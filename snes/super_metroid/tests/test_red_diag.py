"""Unit tests for pure RED auto-capture (no emulator fight required)."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import numpy as np

from super_metroid.ram import (
    ADDR_DOOR_DEF_PTR,
    ADDR_DOOR_TRANSITION,
    ADDR_GAME_STATE,
    parse_state,
)
from super_metroid.scripts.probe.red_diag import (
    FrameRing,
    attach_red_diag,
    build_door_plm_snapshot,
    capture_red_artifacts,
    default_red_diag_dir,
    display_path,
    write_frame_dump,
)


def _nav_state(**kwargs):
    base = parse_state(np.zeros(0x2000, dtype=np.uint8))
    return replace(base, **kwargs)


class _FakeEnv:
    """Minimal env for peek_wram (low WRAM via get_ram)."""

    def __init__(self) -> None:
        self._ram = np.zeros(0x2000, dtype=np.uint8)
        # door_def_ptr, door_transition, game_state as little-endian u16
        self._ram[ADDR_DOOR_DEF_PTR] = 0x52
        self._ram[ADDR_DOOR_DEF_PTR + 1] = 0x8F
        self._ram[ADDR_DOOR_TRANSITION] = 0
        self._ram[ADDR_GAME_STATE] = 8

    def get_ram(self):
        return self._ram


def test_display_path_repo_relative(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    target = root / "super_metroid" / "debug" / "pin.json"
    target.parent.mkdir(parents=True)
    target.write_text("{}", encoding="utf-8")
    assert display_path(target, root=root) == "super_metroid/debug/pin.json"


def test_frame_ring_maxlen_and_copy() -> None:
    ring = FrameRing(maxlen=3)
    a = np.zeros((4, 4, 3), dtype=np.uint8)
    ring.push(a)
    a[0, 0, 0] = 99
    assert ring.frames()[0][0, 0, 0] == 0
    for i in range(5):
        ring.push(np.full((2, 2, 3), i, dtype=np.uint8))
    assert len(ring) == 3
    assert int(ring.frames()[0][0, 0, 0]) == 2


def test_build_door_plm_snapshot_marks_plm_blocked() -> None:
    st = _nav_state(
        room_id=0xB167,
        pose=137,
        samus_x=107,
        samus_y=171,
        door_transition=0,
        game_state=8,
    )
    env = _FakeEnv()
    snap = build_door_plm_snapshot(
        env,
        st,
        error="door missed",
        segment="frog-save-to-speedway",
        source="super_metroid/custom_integrations/x.state",
        frames=400,
    )
    assert snap["kind"] == "pure_red_door_plm_snapshot"
    assert snap["plmRecords"]["status"] == "blocked"
    assert snap["doorNav"]["door_transition"] == 0
    assert snap["doorNav"]["room"] == "0xB167"
    assert snap["wramPeeks"]["door_definition_ptr"] == 0x8F52
    assert snap["wramPeeks"]["game_state"] == 8
    assert snap["probePin"]["x"] == 107


def test_build_snapshot_state_only_without_env() -> None:
    st = _nav_state(room_id=0xA59F, pose=1, samus_x=10, samus_y=20)
    snap = build_door_plm_snapshot(None, st, error="no env")
    assert snap["wramPeeks"] == {}
    assert snap["probePin"]["room"] == "0xA59F"


def test_write_frame_dump_pngs(tmp_path: Path) -> None:
    frames = [
        np.zeros((8, 8, 3), dtype=np.uint8),
        np.full((8, 8, 3), 128, dtype=np.uint8),
    ]
    written = write_frame_dump(frames, tmp_path / "frames")
    assert len(written) == 2
    assert all(p.suffix == ".png" and p.is_file() for p in written)


def test_capture_red_artifacts_surfaces_paths(tmp_path: Path) -> None:
    st = _nav_state(
        room_id=0xB167,
        pose=137,
        samus_x=107,
        samus_y=171,
        door_transition=0,
    )
    frames = [np.zeros((16, 16, 3), dtype=np.uint8) for _ in range(4)]
    out = tmp_path / "red_diag" / "run1"
    report = {
        "success": False,
        "error": "frog_save_to_speedway: right door missed",
        "frames": 400,
        "probePin": {"room": "0xB167", "pose": 137, "x": 107, "y": 171},
        "residualPinLine": "room=0xB167 pose=137 x=107 y=171 door_transition=0 frames=400",
    }
    artifacts = capture_red_artifacts(
        env=_FakeEnv(),
        state=st,
        frames=frames,
        segment="frog-save-to-speedway",
        error=str(report["error"]),
        source="super_metroid/custom_integrations/scratch/x.state",
        probe_frames=400,
        out_dir=out,
        pin_json=out / "pin.json",
        report=report,
        write_pin=True,
        root=tmp_path,
    )
    assert artifacts.snapshot_path.is_file()
    assert len(artifacts.frame_paths) == 4
    assert artifacts.pin_path is not None and artifacts.pin_path.is_file()

    pin = json.loads(artifacts.pin_path.read_text(encoding="utf-8"))
    assert "redDiag" in pin
    red = pin["redDiag"]
    assert red["frameCount"] == 4
    assert red["medium"] == "frame_dump"
    assert "door_plm_snapshot.json" in red["snapshotPath"]
    assert red["snapshotPath"].startswith("red_diag/") or "red_diag" in red["snapshotPath"]
    assert pin["residualArtifactLine"].startswith("snapshot=")

    attach_red_diag(report, artifacts, root=tmp_path)
    assert report["redDiag"]["frameCount"] == 4
    assert "snapshot=" in report["residualArtifactLine"]


def test_default_red_diag_dir_sanitizes_segment(tmp_path: Path) -> None:
    d = default_red_diag_dir(
        segment="frog-save-to-speedway",
        stamp="20260801T120000Z",
        base=tmp_path,
    )
    assert d.name == "20260801T120000Z_frog-save-to-speedway"
    assert d.parent == tmp_path


def test_green_report_schema_has_no_red_diag_requirement() -> None:
    """Document that GREEN pure reports need not include redDiag."""
    green = {
        "success": True,
        "roomIdHex": "0xB106",
        "probePin": {"room": "0xB106"},
    }
    assert "redDiag" not in green
