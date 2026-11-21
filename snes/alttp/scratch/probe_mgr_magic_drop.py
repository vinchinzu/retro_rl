"""Take a magic drop in play from Scan_desert_0x84. Two pots. No chest opens.

The room 0x74 north-lip pot already raised magic 0 to 16. This stays in
0x84 and lifts two different 0x70-0x73 pots. A rise saves
Play_Scan_desert_0x84_magic with provenance kind resumed.

    cd snes && SDL_VIDEODRIVER=dummy uv run python alttp/scratch/probe_mgr_magic_drop.py
"""

from __future__ import annotations

import json

import numpy as np

from alttp.dungeon.nav import WalkCommit, room_origin
from alttp.dungeon.tilemap import read_room_tiles
from alttp.opening_route.room_engine import DEST_SETTLE_MAX_FRAMES, _walk_waypoint
from alttp.paths import GAME, GAME_DIR, RECORDINGS_DIR
from alttp.primitives import Waypoint, active_sprites, lift, settle_control, throw
from alttp.ram import read_inventory, read_snapshot, wram_index
from alttp.spine.ledger import RunLedger
from alttp.spine.tape import read_pin_provenance, write_pin_provenance
from alttp.sprite_ids import sprite_name
from alttp.startup import action_for, no_action, step_frames
from retro_harness.env import make_env, read_state_bytes, save_state
from retro_harness.segment_runner import configure_headless

SCAN = RECORDINGS_DIR / "scan" / "Scan_desert_0x84.state"
OUT = RECORDINGS_DIR / "probe_mgr_magic_drop.json"
PIN = "Play_Scan_desert_0x84_magic"
ROOM = 0x84
POTS = range(0x70, 0x74)
MAGIC = 0xF36E
FILLER = 0xF373
HP = 0xF36D
SMALL_MAGIC = 0xDF
FULL_MAGIC = 0xE0
# Map path to the north hall. The next step, north_door_approach, enters 0x74.
_HALL = (
    (2296, 4496, "alcove_south"),
    (2296, 4415, "alcove_north"),
    (2174, 4415, "west_around"),
    (2174, 4304, "west_jog"),
    (2283, 4196, "hall_center"),
)
LINES: list[str] = []
REFILLS: list[str] = []


def _log(text: str) -> None:
    print(text, flush=True)
    LINES.append(text)


def _ram(env: object) -> np.ndarray:
    return np.asarray(env.get_ram(), dtype=np.uint8)  # type: ignore[attr-defined]


def _snap(env: object):
    return read_snapshot(_ram(env))


def _magic(env: object) -> tuple[int, int]:
    ram = _ram(env)
    return int(ram[wram_index(MAGIC)]), int(ram[wram_index(FILLER)])


def _line(env: object) -> str:
    snap = _snap(env)
    magic, filler = _magic(env)
    return (
        f"room=0x{snap.room_base_id:02X} xy=({snap.link_x},{snap.link_y}) "
        f"state={snap.link_state} hp=0x{snap.hp:02X} "
        f"magic={magic} filler={filler}"
    )


def _alive(env: object) -> bool:
    snap = _snap(env)
    return snap.room_base_id == ROOM and snap.game_mode != 0x12 and snap.hp > 0


def _settle(env: object) -> None:
    snap = _snap(env)
    if snap.submodule != 0 or snap.game_mode not in (0x07, 0x09):
        settle_control(env, max_frames=DEST_SETTLE_MAX_FRAMES)


def _refill_if_last_heart(env: object) -> None:
    """HP only, and only at the last heart. Inventory bytes are not written."""
    snap = _snap(env)
    if snap.hp <= 0 or snap.hp > 8:
        return
    cap = int(read_inventory(_ram(env)).max_hp)
    bus = 0x7E0000 + HP
    unwrapped = getattr(env, "unwrapped", env)
    unwrapped.data.memory.assign(bus, "|u1", cap & 0xFF)
    text = f"assist Survival hp 0x{snap.hp:02X}->0x{cap:02X} at {_line(env)}"
    REFILLS.append(text)
    _log(text)


def _pot_tiles(env: object) -> list[tuple[int, int, int]]:
    snap = _snap(env)
    grid = read_room_tiles(_ram(env), snap.layer)
    found: list[tuple[int, int, int]] = []
    for row in range(grid.shape[0]):
        for col in range(grid.shape[1]):
            byte = int(grid[row, col]) & 0xFF
            if byte in POTS:
                found.append((col, row, byte))
    return found


def _clusters(tiles: list[tuple[int, int, int]]) -> list[list[tuple[int, int, int]]]:
    pending = list(tiles)
    groups: list[list[tuple[int, int, int]]] = []
    while pending:
        seed = pending.pop(0)
        group = [seed]
        grew = True
        while grew:
            grew = False
            rest: list[tuple[int, int, int]] = []
            for tile in pending:
                if any(max(abs(tile[0] - got[0]), abs(tile[1] - got[1])) <= 2 for got in group):
                    group.append(tile)
                    grew = True
                else:
                    rest.append(tile)
            pending = rest
        groups.append(group)
    groups.sort(key=lambda group: (min(tile[0] for tile in group), min(tile[1] for tile in group)))
    return groups


def _walk(env: object, x: int, y: int, label: str) -> bool:
    commit = WalkCommit()
    point = Waypoint(x, y, tolerance=12, room=ROOM, label=label)
    result, _commit = _walk_waypoint(env, point, room=ROOM, commit=commit, max_frames=700)
    _refill_if_last_heart(env)
    _log(f"  walk {label} ok={result.ok} {result.reason} {_line(env)}")
    return bool(result.ok) and _alive(env)


def _to_hall(env: object) -> bool:
    for x, y, label in _HALL:
        if not _alive(env):
            return False
        if not _walk(env, x, y, label):
            return False
    return _alive(env)


def _sprites(env: object) -> str:
    parts = [
        f"{sprite_name(row.sprite_type)}:{row.sprite_type:02X}@({row.x},{row.y})"
        for row in active_sprites(env)
    ]
    return " ".join(parts)


def _break_pot(env: object, col: int, row: int) -> bool:
    """Stand on a side of one pot tile and lift. Stay in 0x84."""
    snap = _snap(env)
    ox, oy = room_origin(snap.link_x, snap.link_y)
    stands = (
        (col, row + 1, "UP"),
        (col, row - 1, "DOWN"),
        (col - 1, row, "RIGHT"),
        (col + 1, row, "LEFT"),
    )
    for stand_col, stand_row, facing in stands:
        if not _alive(env):
            return False
        if not _walk(env, ox + stand_col * 8 + 4, oy + stand_row * 8 + 4, f"pot_{col}_{row}"):
            continue
        step_frames(env, action_for(facing), 8)
        _refill_if_last_heart(env)
        if not _alive(env):
            return False
        lifted = lift(env)
        _log(f"  lift ({col},{row}) {facing} ok={lifted.ok} {lifted.reason} {_line(env)}")
        if not lifted.ok:
            continue
        thrown = throw(env)
        _log(f"  throw ok={thrown.ok} {thrown.reason} {_sprites(env)}")
        return True
    return False


def _collect(env: object, before: int) -> int:
    """Idle, then step onto a magic sprite. Returns the magic byte."""
    for _ in range(90):
        step_frames(env, no_action(), 1)
        magic, filler = _magic(env)
        if magic > before or filler:
            break
        if not _alive(env):
            return _magic(env)[0]
    for _ in range(160):
        magic, filler = _magic(env)
        if magic > before and filler == 0:
            return magic
        if not _alive(env):
            return magic
        target = None
        for row in active_sprites(env):
            if row.sprite_type in (SMALL_MAGIC, FULL_MAGIC):
                target = (row.x, row.y)
                break
        if target is None:
            step_frames(env, no_action(), 1)
            continue
        snap = _snap(env)
        dx, dy = target[0] - snap.link_x, target[1] - snap.link_y
        if abs(dx) <= 4 and abs(dy) <= 4:
            step_frames(env, no_action(), 1)
            continue
        if abs(dx) >= abs(dy):
            button = "RIGHT" if dx > 0 else "LEFT"
        else:
            button = "DOWN" if dy > 0 else "UP"
        step_frames(env, action_for(button), 1)
        _refill_if_last_heart(env)
    # Filler $F373 drains onto $F36E one point per frame.
    for _ in range(140):
        magic, filler = _magic(env)
        if filler == 0:
            break
        step_frames(env, no_action(), 1)
    magic, filler = _magic(env)
    _log(f"  collect magic={magic} filler={filler} {_sprites(env)}")
    return magic


def _try_cluster(env: object, group: list[tuple[int, int, int]], label: str) -> bool:
    _log(f"{label} tiles={group} {_line(env)}")
    if not _to_hall(env):
        _log(f"{label} no hall {_line(env)}")
        return False
    before = _magic(env)[0]
    broke = False
    for col, row, _byte in group:
        if _break_pot(env, col, row):
            broke = True
            break
    if not broke:
        _log(f"{label} no lift {_line(env)}")
        return False
    after = _collect(env, before)
    _log(f"{label} magic {before}->{after} {_line(env)}")
    return after > before and _snap(env).room_base_id == ROOM


def _save(env: object, ledger: RunLedger) -> None:
    path = save_state(env, GAME_DIR, GAME, PIN)
    write_pin_provenance(
        path,
        source_run="play:Scan_desert_0x84",
        frame=int(ledger.frame),
        kind="resumed",
    )
    kind = (read_pin_provenance(path) or {}).get("kind")
    magic, filler = _magic(env)
    _log(f"saved {PIN} magic={magic} filler={filler} kind={kind} frame={ledger.frame}")


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
        _settle(env)
        tiles = _pot_tiles(env)
        groups = _clusters(tiles)
        _log(f"start {_line(env)} pots={len(tiles)} clusters={len(groups)}")
        _log(f"inventory magic={int(read_inventory(_ram(env)).magic)}")
        if len(groups) < 1:
            _log("no 0x70-0x73 pots")
        else:
            ledger.attach(env)
            picks = groups[:2]
            for index, group in enumerate(picks, start=1):
                if index > 1:
                    env.em.set_state(blob)
                    ledger._held = None
                    ledger._link = None
                    _settle(env)
                ledger.begin_stage(f"try{index}")
                if _try_cluster(env, group, f"try{index}"):
                    _save(env, ledger)
                    saved = True
                    break
        rows = [change.row() for change in ledger.changes]
        magic_rows = [row for row in rows if row["field"] == "magic"]
        _log(f"magic changes {magic_rows}")
        OUT.write_text(
            json.dumps(
                {
                    "pin": PIN if saved else None,
                    "saved": saved,
                    "magic": _magic(env)[0],
                    "magic_changes": magic_rows,
                    "changes": rows,
                    "refills": REFILLS,
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
