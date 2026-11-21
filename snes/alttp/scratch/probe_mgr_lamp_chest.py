"""Open the Link's House lamp chest from Scan_0x0104. Two tries. No RAM writes.

Try 1 is the fresh-profile wake plus the lamp-chest button script.
Try 2 walks to a floor tile beside chest (56, 54) and presses A facing it.
A rise is saved as Play_Scan0x0104_lamp with provenance kind resumed.

    cd snes && SDL_VIDEODRIVER=dummy uv run python alttp/scratch/probe_mgr_lamp_chest.py
"""

from __future__ import annotations

import json

import numpy as np

from alttp.dungeon.nav import WalkCommit, find_path, room_origin, room_step
from alttp.dungeon.tilemap import read_room_tiles, tile_kind
from alttp.opening_route.castle_to_sword import dismiss_hold_up_item
from alttp.opening_route.room_engine import DEST_SETTLE_MAX_FRAMES
from alttp.paths import GAME, GAME_DIR, RECORDINGS_DIR
from alttp.primitives import settle_control
from alttp.ram import LINK_ITEM_LAMP, read_inventory, read_u16, wram_index
from alttp.scripts.pin_probe import chest_lines, describe
from alttp.spine.ledger import RunLedger
from alttp.spine.tape import read_pin_provenance, write_pin_provenance
from alttp.startup import (
    FRESH_PROFILE_LAMP_CHEST_SCRIPT,
    FRESH_PROFILE_WAKE_SCRIPT,
    action_for,
    no_action,
    step_frames,
)
from alttp.walk_physics import LINK_BODY
from retro_harness.env import make_env, read_state_bytes, save_state
from retro_harness.segment_runner import configure_headless

SCAN = RECORDINGS_DIR / "scan" / "Scan_0x0104.state"
OUT = RECORDINGS_DIR / "probe_mgr_lamp_chest.json"
PIN = "Play_Scan0x0104_lamp"
ROOM = 0x104
CHEST = (56, 54)
# West, east, north, south of the chest, and the button that faces it.
_STANDS = (
    (55, 54, "RIGHT"),
    (57, 54, "LEFT"),
    (56, 53, "DOWN"),
    (56, 55, "UP"),
)
LINES: list[str] = []


def _log(text: str) -> None:
    print(text, flush=True)
    LINES.append(text)


def _ram(env: object) -> np.ndarray:
    return np.asarray(env.get_ram(), dtype=np.uint8)  # type: ignore[attr-defined]


def _lamp(env: object) -> int:
    return int(_ram(env)[wram_index(LINK_ITEM_LAMP)])


def _snap_line(env: object) -> str:
    from alttp.ram import read_snapshot

    snap = read_snapshot(_ram(env))
    return (
        f"room=0x{snap.room_id:04X} xy=({snap.link_x},{snap.link_y}) "
        f"state={snap.link_state} sub=0x{snap.submodule:02X} "
        f"mode=0x{snap.game_mode:02X} lamp={_lamp(env)} "
        f"magic={int(read_inventory(_ram(env)).magic)}"
    )


def _in_room(env: object) -> bool:
    from alttp.ram import read_snapshot

    snap = read_snapshot(_ram(env))
    return snap.room_base_id == (ROOM & 0xFF) and snap.game_mode != 0x12 and snap.indoors


def _settle(env: object) -> None:
    from alttp.ram import read_snapshot

    snap = read_snapshot(_ram(env))
    if snap.submodule != 0 or snap.game_mode not in (0x07, 0x09):
        settle_control(env, max_frames=DEST_SETTLE_MAX_FRAMES)


def _script(env: object, script: tuple[tuple[str, int], ...], label: str) -> str:
    """Play one button script. Stop if the lamp rises or the room changes."""
    for button, frames in script:
        action = no_action() if button == "NONE" else action_for(button)
        for _ in range(frames):
            step_frames(env, action, 1)
            if _lamp(env) != 0:
                _settle(env)
                return "lamp"
            from alttp.ram import read_snapshot

            snap = read_snapshot(_ram(env))
            if snap.game_mode == 0x0E:
                step_frames(env, action_for("A"), 2)
            if not _in_room(env):
                _log(f"  {label} left {_snap_line(env)}")
                return "left"
        _log(f"  {label} {button}:{frames} {_snap_line(env)}")
    return "done"


def _floor(env: object) -> tuple[np.ndarray, int]:
    ram = _ram(env)
    from alttp.ram import read_snapshot

    snap = read_snapshot(ram)
    grid = read_room_tiles(ram, snap.layer).copy()
    for row in range(grid.shape[0]):
        for col in range(grid.shape[1]):
            if tile_kind(int(grid[row, col])) != "floor":
                grid[row, col] = 0x04
    mask = read_u16(ram, 0x00EC)
    return grid, mask


def _walk_to(env: object, col: int, row: int) -> bool:
    """Floor path to one tile. Door and chest bytes stay walls."""
    commit = WalkCommit()
    goal_local = (col * 8, row * 8)
    for _ in range(500):
        if not _in_room(env):
            return False
        from alttp.ram import read_snapshot

        snap = read_snapshot(_ram(env))
        if snap.submodule != 0 or snap.game_mode == 0x0E:
            _settle(env)
            continue
        grid, mask = _floor(env)
        start = ((int(snap.link_x) & mask) & ~7, (int(snap.link_y) & mask) & ~7)
        if abs(start[0] - goal_local[0]) <= 8 and abs(start[1] - goal_local[1]) <= 8:
            return True
        path = find_path(grid, start, goal_local, LINK_BODY)
        if not path or len(path) < 2:
            return False
        ox = int(snap.link_x) - (int(snap.link_x) & mask)
        oy = int(snap.link_y) - (int(snap.link_y) & mask)
        wx, wy = ox + path[1][0], oy + path[1][1]
        dx, dy = wx - snap.link_x, wy - snap.link_y
        if abs(dx) >= abs(dy) and dx != 0:
            desired: str | None = "RIGHT" if dx > 0 else "LEFT"
        elif dy != 0:
            desired = "DOWN" if dy > 0 else "UP"
        else:
            desired = None
        button, commit = room_step(commit, desired)
        step_frames(env, no_action() if button is None else action_for(button), 1)
    return False


def _press_facing(env: object, facing: str) -> None:
    step_frames(env, action_for(facing), 10)
    step_frames(env, action_for("A"), 4)
    step_frames(env, no_action(), 20)
    if _lamp(env) or not _in_room(env):
        return
    from alttp.ram import read_snapshot

    snap = read_snapshot(_ram(env))
    if snap.is_hold_up_item or snap.link_state == 21:
        dismiss_hold_up_item(env)
    _settle(env)


def _try_script(env: object) -> None:
    _log(f"try1 start {_snap_line(env)}")
    _log("\n".join(describe(env)))
    _log("\n".join(chest_lines(env)))
    reason = _script(env, FRESH_PROFILE_WAKE_SCRIPT, "wake")
    _log(f"try1 wake {reason} {_snap_line(env)}")
    if reason == "lamp" or not _in_room(env):
        return
    reason = _script(env, FRESH_PROFILE_LAMP_CHEST_SCRIPT, "chest")
    _settle(env)
    if _lamp(env):
        dismiss_hold_up_item(env)
    _log(f"try1 chest {reason} {_snap_line(env)}")
    _log("\n".join(chest_lines(env)))


def _try_walk(env: object) -> None:
    _log(f"try2 start {_snap_line(env)}")
    # Short wake only. The long DOWN in the profile script is try 1.
    for button, frames in (("NONE", 40), ("B", 8), ("NONE", 16), ("A", 8), ("NONE", 40)):
        action = no_action() if button == "NONE" else action_for(button)
        step_frames(env, action, frames)
        if not _in_room(env) or _lamp(env):
            _log(f"try2 wake left {_snap_line(env)}")
            return
    _log(f"try2 awake {_snap_line(env)}")
    ox, oy = room_origin(*_xy(env))
    _log(f"try2 origin=({ox},{oy}) chest_px=({ox + CHEST[0] * 8},{oy + CHEST[1] * 8})")
    for col, row, facing in _STANDS:
        if not _in_room(env) or _lamp(env):
            break
        ok = _walk_to(env, col, row)
        _log(f"  stand ({col},{row}) face {facing} ok={ok} {_snap_line(env)}")
        if not ok or not _in_room(env):
            continue
        _press_facing(env, facing)
        _log(f"  pressed {_snap_line(env)}")
        _log("\n".join(chest_lines(env)))
        if _lamp(env):
            break
    _log(f"try2 end {_snap_line(env)}")


def _xy(env: object) -> tuple[int, int]:
    from alttp.ram import read_snapshot

    snap = read_snapshot(_ram(env))
    return int(snap.link_x), int(snap.link_y)


def _save(env: object, ledger: RunLedger) -> None:
    path = save_state(env, GAME_DIR, GAME, PIN)
    write_pin_provenance(
        path,
        source_run="play:Scan_0x0104",
        frame=int(ledger.frame),
        kind="resumed",
    )
    kind = (read_pin_provenance(path) or {}).get("kind")
    _log(f"saved {PIN} lamp={_lamp(env)} kind={kind} frame={ledger.frame}")


def _changes(ledger: RunLedger) -> list[dict[str, object]]:
    return [change.row() for change in ledger.changes]


def main() -> int:
    if not SCAN.is_file():
        raise SystemExit(f"missing {SCAN}")
    configure_headless()
    env = make_env(GAME, "NONE", GAME_DIR, render_mode=None)
    blob = read_state_bytes(SCAN)
    ledger = RunLedger(print_on_close=False)
    saved = False
    try:
        env.reset()
        env.em.set_state(blob)
        ledger.attach(env)
        ledger.begin_stage("try1")
        _try_script(env)
        if _lamp(env) != 0:
            _save(env, ledger)
            saved = True
        else:
            env.em.set_state(blob)
            ledger._held = None
            ledger._link = None
            ledger.begin_stage("try2")
            _try_walk(env)
            if _lamp(env) != 0:
                _save(env, ledger)
                saved = True
        rows = _changes(ledger)
        lamp_rows = [row for row in rows if row["field"] == "lamp"]
        _log(f"lamp changes {lamp_rows}")
        _log(f"saved={saved}")
        OUT.write_text(
            json.dumps(
                {
                    "pin": PIN if saved else None,
                    "lamp": _lamp(env),
                    "saved": saved,
                    "lamp_changes": lamp_rows,
                    "changes": rows,
                    "log": LINES,
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
    finally:
        env.close()
    return 0 if saved else 2


if __name__ == "__main__":
    raise SystemExit(main())
