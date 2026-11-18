"""Play the USA Tompa SMV on stable-retro and stop at the first desync.

The movie is Snes9x 1.43, reset-anchored, SRAM filled with ``0x60``. This
does not play all 274k frames. It stops at Sanctuary or at the first pose
that is no longer a room: a menu after play, a walk while the name
menu is still up, movement while the intro or file select is still up,
an indoor wall walk, or a one-frame teleport.

    cd snes && SDL_VIDEODRIVER=dummy uv run python alttp/scripts/tas_sync.py
"""

from __future__ import annotations

import json
import os

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

import argparse
import sys
from pathlib import Path
from typing import Any

import numpy as np

from alttp.dungeon.tilemap import read_room_tiles, tile_kind
from alttp.paths import GAME, GAME_DIR, RECORDINGS_DIR
from alttp.ram import (
    HYRULE_CASTLE_B1_EAST_ROOM,
    HYRULE_CASTLE_B1_NORTH_ZELDA_WEST_ROOM,
    HYRULE_CASTLE_B1_PIT_ROOM,
    HYRULE_CASTLE_B1_WEST_ROOM,
    HYRULE_CASTLE_MAIN_EAST_ROOM,
    HYRULE_CASTLE_MAIN_HALL_ROOM,
    HYRULE_CASTLE_MAIN_WEST_ROOM,
    HYRULE_CASTLE_NE_ROOM,
    HYRULE_CASTLE_NORTH_CONNECTOR_ROOM,
    HYRULE_CASTLE_NW_ROOM,
    HYRULE_CASTLE_SCREEN,
    HYRULE_CASTLE_SECRET_ENTRANCE_ROOM,
    HYRULE_CASTLE_THRONE_APPROACH_ROOM,
    SANCTUARY_ROOM,
    ZELDA_CELL_ROOM,
    read_snapshot,
    read_u8,
)
from alttp.startup import parse_s9x_blocks
from alttp.tas.catalog import by_filename
from alttp.tas.fetch_refs import fetch_one
from alttp.tas.smv import parse_smv_env
from alttp.walk_physics import LINK_BODY
from retro_harness.controls import (
    SNES_DOWN,
    SNES_LEFT,
    SNES_RIGHT,
    SNES_START,
    SNES_UP,
)
from retro_harness.env import make_env

# Census (ram_map.md): 0x00 intro, 0x01 file select, 0x04 name file.
# Play is 0x07 / 0x09 with submodule 0. ram.py's title/file names disagree.
_INTRO = 0x00
_FILE_SELECT = 0x01
_NAME = 0x04
_PLAY = frozenset({0x07, 0x09})
_MENU_AFTER_PLAY = frozenset({_INTRO, _FILE_SELECT, _NAME})
# pin_probe: 0x27 blocks the same as a wall. Unknown bytes are not walls.
_BLOCK = 0x27
# Indoor rooms on the vanilla castle path. Not coordinates.
_CASTLE_ROOMS = frozenset(
    {
        HYRULE_CASTLE_SECRET_ENTRANCE_ROOM,
        HYRULE_CASTLE_MAIN_HALL_ROOM,
        HYRULE_CASTLE_MAIN_WEST_ROOM,
        HYRULE_CASTLE_MAIN_EAST_ROOM,
        HYRULE_CASTLE_NW_ROOM,
        HYRULE_CASTLE_NORTH_CONNECTOR_ROOM,
        HYRULE_CASTLE_THRONE_APPROACH_ROOM,
        HYRULE_CASTLE_NE_ROOM,
        HYRULE_CASTLE_B1_NORTH_ZELDA_WEST_ROOM,
        HYRULE_CASTLE_B1_PIT_ROOM,
        HYRULE_CASTLE_B1_WEST_ROOM,
        HYRULE_CASTLE_B1_EAST_ROOM,
        ZELDA_CELL_ROOM,
        SANCTUARY_ROOM,
    }
)
_EARLY_FRAMES = 7000
_WALL_FRAMES = 8
_MENU_HOLD = 90
# Down on the bottom name row (or Up on the top) does not move the cursor.
# A few frames of slack is a late release. Longer than this is a walk.
_PIN_FRAMES = 8
_NAME_COL = 0xB10
_NAME_ROW = 0xB15
_NAME_ANIM_Y = 0xB14
_NAME_SUB = 0x03
_START_WINDOW = 120
_STEP_LIMIT = 32
_SANCTUARY_HOLD = 30
_TRACE_CAP = 120
_DEFAULT_CAP = 90000
REPORT_PATH = RECORDINGS_DIR / "ap006_tas_sync.json"
_DPAD = (SNES_UP, SNES_DOWN, SNES_LEFT, SNES_RIGHT)


class StartStateError(RuntimeError):
    """Movie header parsed, but the reset SRAM could not be installed."""


def _install_sram(env: object, sram: bytes) -> dict[str, int]:
    """Copy movie SRAM into the power-on SRA block. No checksum rewrite."""
    if len(sram) < 0x2000:
        raise StartStateError(f"movie SRAM is {len(sram)} bytes")
    state = bytes(env.em.get_state())  # type: ignore[attr-defined]
    blocks = parse_s9x_blocks(state)
    sra = blocks.get("SRA")
    if sra is None:
        raise StartStateError("power-on state has no SRA block")
    size = int(sra["size"])
    if size not in (0x2000, 0x8000, 0x20000):
        raise StartStateError(f"SRA block is {size} bytes")
    copied = min(size, len(sram))
    modified = bytearray(state)
    off = int(sra["data_offset"])
    modified[off : off + copied] = sram[:copied]
    env.em.set_state(bytes(modified))  # type: ignore[attr-defined]
    check = parse_s9x_blocks(bytes(env.em.get_state()))  # type: ignore[attr-defined]
    back = check.get("SRA")
    if back is None or bytes(env.em.get_state())[back["data_offset"]] != sram[0]:  # type: ignore[attr-defined]
        raise StartStateError("SRA write did not stick")
    return {"sra_size": size, "copied": copied, "fill": int(sram[0])}


def _center_tile(grid: np.ndarray, x: int, y: int) -> int:
    """Attribute under the middle of ``LINK_BODY``. Room tiles wrap at 64."""
    x0, y0, x1, y1 = LINK_BODY
    col = ((x + (x0 + x1) // 2) >> 3) & 63
    row = ((y + (y0 + y1) // 2) >> 3) & 63
    return int(grid[row, col])


def _is_wall(value: int) -> bool:
    return tile_kind(value) == "wall" or value == _BLOCK


def _pose(snap: Any, frame: int) -> dict[str, int]:
    return {
        "frame": frame,
        "module": int(snap.game_mode),
        "sub": int(snap.submodule),
        "room": int(snap.room_id),
        "x": int(snap.link_x),
        "y": int(snap.link_y),
        "screen": int(snap.screen_id),
        "indoors": int(bool(snap.indoors)),
        "follower": int(snap.follower),
        "sword": int(snap.sword_level),
    }


def _play(
    frames: np.ndarray,
    sram: bytes | None,
    *,
    lag: int,
    inject: bool,
    max_frames: int,
) -> dict[str, Any]:
    """One power-on pass. ``lag`` skips that many leading movie samples."""
    env = make_env(GAME, "NONE", GAME_DIR, render_mode=None)
    buf = np.zeros(12, dtype=np.uint8)
    result: dict[str, Any] = {
        "lag": lag,
        "injected_sram": inject,
        "seen_control": False,
        "seen_zelda": False,
        "seen_castle": False,
        "first_desync_frame": None,
        "reason": "",
        "sanctuary_frame": None,
        "frames_played": 0,
        "at_stop": None,
        "trace": [],
        "sram": None,
    }
    try:
        env.reset()
        if inject:
            if sram is None:
                raise StartStateError("reset movie has no SRAM")
            result["sram"] = _install_sram(env, sram)
        wall_run = 0
        wall_from = 0
        menu_run = 0
        menu_from = 0
        sanctuary_run = 0
        start_age = _START_WINDOW
        prev: dict[str, int] | None = None
        prev_wall = False
        last_key: tuple[int, int, int] | None = None
        limit = min(max_frames, len(frames) - lag)
        for step in range(limit):
            movie_frame = step + lag
            buf[:] = frames[movie_frame]
            start_age = 0 if buf[SNES_START] else start_age + 1
            env.em.set_button_mask(buf, 0)  # type: ignore[attr-defined]
            env.em.step()  # type: ignore[attr-defined]
            ram = np.asarray(env.get_ram(), dtype=np.uint8)
            snap = read_snapshot(ram)
            result["frames_played"] = step + 1
            control = bool(snap.has_control)
            if control:
                result["seen_control"] = True
            if int(snap.follower) == 1:
                result["seen_zelda"] = True
            if control and (
                (snap.indoors and int(snap.room_base_id) in _CASTLE_ROOMS)
                or ((not snap.indoors) and int(snap.screen_id) == HYRULE_CASTLE_SCREEN)
            ):
                result["seen_castle"] = True
            # Submodule chatter during naming would fill the trace.
            place = int(snap.room_id) if snap.indoors else int(snap.screen_id)
            key = (int(snap.game_mode), int(bool(snap.indoors)), place)
            if key != last_key and len(result["trace"]) < _TRACE_CAP:
                result["trace"].append(_pose(snap, movie_frame))
                last_key = key
            if step and step % 2000 == 0:
                print(
                    f"f={movie_frame} mode={snap.game_mode:#04x}/{snap.submodule:#04x} "
                    f"room={snap.room_id:#06x} xy=({snap.link_x},{snap.link_y}) "
                    f"scr={snap.screen_id:#04x} in={int(snap.indoors)}",
                    file=sys.stderr,
                    flush=True,
                )
            reason = _nonsense(
                snap,
                ram,
                buf,
                movie_frame=movie_frame,
                control=control,
                seen_control=bool(result["seen_control"]),
                seen_zelda=bool(result["seen_zelda"]),
                seen_castle=bool(result["seen_castle"]),
                start_age=start_age,
                prev=prev,
                prev_wall=prev_wall,
                wall_run=wall_run,
                menu_run=menu_run,
            )
            if reason is not None:
                text, wall_run, wall_from, menu_run, menu_from, prev_wall, hit = reason
                if hit is not None:
                    result["first_desync_frame"] = hit
                    result["reason"] = text
                    # Pose is unchanged across a short pin; stamp the hit frame.
                    result["at_stop"] = _pose(snap, hit)
                    return result
            else:
                wall_run, prev_wall = 0, False
                menu_run = 0
            if snap.in_sanctuary and control and result["seen_zelda"]:
                sanctuary_run += 1
                if sanctuary_run >= _SANCTUARY_HOLD:
                    result["sanctuary_frame"] = movie_frame - _SANCTUARY_HOLD + 1
                    result["reason"] = "sync held through sanctuary"
                    result["at_stop"] = _pose(snap, movie_frame)
                    return result
            else:
                sanctuary_run = 0
            if step + 1 == _EARLY_FRAMES and not result["seen_control"]:
                result["first_desync_frame"] = movie_frame
                result["reason"] = "no playable control by this frame"
                result["at_stop"] = _pose(snap, movie_frame)
                return result
            prev = _pose(snap, movie_frame)
            prev["control"] = int(control)
        result["reason"] = "cap reached before sanctuary and before a desync"
        if prev is not None:
            result["at_stop"] = {k: v for k, v in prev.items() if k != "control"}
        return result
    finally:
        env.close()


def _nonsense(
    snap: Any,
    ram: np.ndarray,
    buttons: np.ndarray,
    *,
    movie_frame: int,
    control: bool,
    seen_control: bool,
    seen_zelda: bool,
    seen_castle: bool,
    start_age: int,
    prev: dict[str, int] | None,
    prev_wall: bool,
    wall_run: int,
    menu_run: int,
) -> tuple[str, int, int, int, int, bool, int | None] | None:
    """Return updated streaks and a hit frame, or None when the pose is fine.

    The hit frame is the first frame of the streak, not the frame we noticed.
    """
    del seen_zelda, seen_castle
    module = int(snap.game_mode)
    if (not seen_control) and module == _NAME and int(snap.submodule) == _NAME_SUB:
        pinned = _name_pin(ram, buttons)
        if pinned is not None:
            menu_run += 1
            menu_from = movie_frame if menu_run == 1 else movie_frame - menu_run + 1
            if menu_run >= _PIN_FRAMES or pinned == "wobble":
                return (
                    f"name menu still up during a walk ({pinned})",
                    wall_run,
                    movie_frame,
                    menu_run,
                    menu_from,
                    False,
                    movie_frame if pinned == "wobble" else menu_from,
                )
            return ("", wall_run, movie_frame, menu_run, menu_from, False, None)
        menu_run = 0
        menu_from = movie_frame
        return ("", wall_run, movie_frame, menu_run, menu_from, False, None)
    if seen_control and module in _MENU_AFTER_PLAY and start_age > _START_WINDOW:
        return (
            f"menu ${module:#04x} after play without Start",
            wall_run,
            movie_frame,
            menu_run,
            movie_frame,
            False,
            movie_frame,
        )
    dpad = any(int(buttons[i]) for i in _DPAD)
    if (not seen_control) and module in (_INTRO, _FILE_SELECT) and dpad:
        menu_run += 1
        menu_from = movie_frame if menu_run == 1 else movie_frame - menu_run + 1
        if menu_run >= _MENU_HOLD:
            return (
                f"direction held in module ${module:#04x}",
                wall_run,
                movie_frame,
                menu_run,
                menu_from,
                False,
                menu_from,
            )
    else:
        menu_run = 0
        menu_from = movie_frame
    if (
        control
        and prev is not None
        and prev.get("control")
        and int(prev["indoors"]) == int(bool(snap.indoors))
        and (
            (snap.indoors and int(prev["room"]) == int(snap.room_id))
            or ((not snap.indoors) and int(prev["screen"]) == int(snap.screen_id))
        )
    ):
        jump = max(abs(int(snap.link_x) - int(prev["x"])), abs(int(snap.link_y) - int(prev["y"])))
        if jump > _STEP_LIMIT:
            return (
                f"step of {jump}px in one frame",
                wall_run,
                movie_frame,
                menu_run,
                menu_from,
                False,
                movie_frame,
            )
    center_wall = False
    if control and snap.indoors and int(snap.layer) in (0, 1):
        grid = read_room_tiles(ram, int(snap.layer))
        center_wall = _is_wall(_center_tile(grid, int(snap.link_x), int(snap.link_y)))
        moved = prev is not None and (
            int(prev["x"]) != int(snap.link_x) or int(prev["y"]) != int(snap.link_y)
        )
        same_room = prev is not None and int(prev["room"]) == int(snap.room_id)
        if center_wall and prev_wall and moved and same_room:
            wall_run += 1
            wall_from = movie_frame if wall_run == 1 else movie_frame - wall_run + 1
            if wall_run >= _WALL_FRAMES:
                return (
                    "indoor wall walk",
                    wall_run,
                    wall_from,
                    menu_run,
                    menu_from,
                    True,
                    wall_from,
                )
        else:
            wall_run = 1 if center_wall else 0
    else:
        wall_run = 0
    return ("", wall_run, movie_frame, menu_run, menu_from, center_wall, None)


def _name_pin(ram: np.ndarray, buttons: np.ndarray) -> str | None:
    """A direction the name grid is not consuming, or None while it still is.

    Left+right and up+down are the walk wobble. A pure vertical hold on the
    blocked edge is the same walk after the cursor has already arrived.
    """
    up = int(buttons[SNES_UP])
    down = int(buttons[SNES_DOWN])
    left = int(buttons[SNES_LEFT])
    right = int(buttons[SNES_RIGHT])
    if (up and down) or (left and right):
        return "wobble"
    row = read_u8(ram, _NAME_ROW)
    anim_y = read_u8(ram, _NAME_ANIM_Y)
    if down and not (up or left or right) and row == 3 and anim_y == 0:
        return "Down on row 3"
    if up and not (down or left or right) and row == 0 and anim_y == 0:
        return "Up on row 0"
    return None


def _choose(tries: list[dict[str, Any]]) -> dict[str, Any]:
    """Prefer a pass that reached control, else the aligned lag-0 desync."""
    alive = [row for row in tries if not row.get("error")]
    if not alive:
        raise StartStateError(tries[-1].get("error") or "start state did not load")
    played = [row for row in alive if row.get("seen_control") or row.get("sanctuary_frame")]
    if played:
        return max(played, key=lambda row: int(row["frames_played"]))
    desynced = [row for row in alive if row.get("first_desync_frame") is not None]
    if desynced:
        return min(
            desynced,
            key=lambda row: (int(row.get("lag") or 0), int(row["first_desync_frame"])),
        )
    return max(alive, key=lambda row: int(row["frames_played"]))


def _public(row: dict[str, Any]) -> dict[str, Any]:
    keep = (
        "lag",
        "injected_sram",
        "seen_control",
        "seen_zelda",
        "seen_castle",
        "first_desync_frame",
        "reason",
        "sanctuary_frame",
        "frames_played",
        "sram",
        "error",
    )
    return {key: row.get(key) for key in keep}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-frames", type=int, default=_DEFAULT_CAP)
    parser.add_argument("--lag", type=int, default=None, help="Force one alignment")
    args = parser.parse_args(argv)
    ref = by_filename("tompa_1269M.smv")
    path = fetch_one(ref)
    movie = parse_smv_env(path)
    raw = movie.raw
    if raw.version not in (1, 4, 5):
        raise StartStateError(f"unsupported SMV version {raw.version}")
    frames = np.asarray(movie.frames, dtype=np.uint8)
    if frames.ndim != 2 or frames.shape[1] != 12:
        raise StartStateError(f"movie frame shape {frames.shape}")
    sram = raw.save_ram
    if raw.starts_from_reset and (sram is None or len(set(sram)) != 1):
        # A reset movie whose SRAM is not one fill byte is still installed.
        pass
    if not raw.starts_from_reset:
        raise StartStateError(
            "movie is snapshot-anchored; the embedded Snes9x savestate "
            "is not a stable-retro state"
        )
    plan = [(0, True)] if args.lag is None else [(int(args.lag), True)]
    if args.lag is None:
        plan.append((1, True))
    tries: list[dict[str, Any]] = []
    for lag, inject in plan:
        print(f"try lag={lag} inject={inject}", file=sys.stderr, flush=True)
        try:
            row = _play(
                frames,
                sram,
                lag=lag,
                inject=inject,
                max_frames=int(args.max_frames),
            )
        except StartStateError as exc:
            row = {
                "lag": lag,
                "injected_sram": inject,
                "error": str(exc),
                "frames_played": 0,
                "seen_control": False,
                "first_desync_frame": None,
                "reason": str(exc),
                "sanctuary_frame": None,
                "trace": [],
                "at_stop": None,
            }
        tries.append(row)
        print(
            f"stop lag={lag} frames={row.get('frames_played')} "
            f"desync={row.get('first_desync_frame')} {row.get('reason')}",
            file=sys.stderr,
            flush=True,
        )
        if row.get("seen_control") or row.get("sanctuary_frame") is not None:
            break
        if row.get("error") and inject:
            # Second honest try: same lag, core SRAM, only after a bad install.
            continue
    best = _choose(tries)
    sanctuary = best.get("sanctuary_frame")
    report = {
        "movie": ref.filename,
        "path": str(path),
        "sha256": ref.sha256,
        "smv_version": raw.version,
        "emulator_version": raw.emulator_version,
        "starts_from_reset": raw.starts_from_reset,
        "rom_crc32": raw.rom_crc32,
        "rom_name": raw.rom_name,
        "header_frame_count": raw.header_frame_count,
        "num_input_samples": raw.num_frames,
        "sram_bytes": len(sram) if sram is not None else 0,
        "lag": best.get("lag"),
        "injected_sram": best.get("injected_sram"),
        "frames_played": best.get("frames_played"),
        "first_desync_frame": best.get("first_desync_frame"),
        "reason": best.get("reason"),
        "sanctuary_frame": sanctuary,
        "oracle_usable_to": sanctuary,
        "at_stop": best.get("at_stop"),
        "trace": best.get("trace"),
        "tries": [_public(row) for row in tries],
    }
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(_public(best), indent=2))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except StartStateError as exc:
        print(f"start state: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc
