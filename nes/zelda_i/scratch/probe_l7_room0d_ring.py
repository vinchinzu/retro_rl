"""Live: L7 play 0x0D ring walk-on of the hidden staircase (no position pokes).

Measured with `zelda_i.dungeon.tilemap` (cart-WRAM `$6530` room tile map),
not `$049E` sweeps. Room 0x0D interior, 16x16 cells:

        x=  32  48  64  80  96 112 128 144 160 176 192 208
 y= 96       .   .   .   .   .   .   .   .   .   .   .   .
 y=112       .   #   #   #   #   #   #   #   #   #   #   .
 y=128       .   .   .   .   .   .   .   .   .   .   #   .
 y=144       .   .   .   .   .   .   .   .   .   .   B   .   <- west door (16,144)
 y=160       .   .   .   .   .   .   .   .   .   .   #   .
 y=176       .   #   #   #   #   #   #   #   #   #   #   .
 y=192       .   .   .   .   .   .   .   .   .   .   .   .

Only x=32 and x=208 cross the y=112 / y=176 solid bands. Every prior sitting
clamped to `INLAND_X = (64, 192)` (a stale wallmaster guard from the
*uncleared* recon) and so excluded both corridors.

RAM claim (written before the first live trial):
  R1  UP to y=125, LEFT to x=32 along the y=128 row. MISS if LEFT pins x>34.
  R2  UP the x=32 column to y=93. MISS if it pins at y>=101.
  R3  pre-push `stair_cells()` is empty. MISS if stairs already exist.
  R4  the verified RIGHT push puts the block quad at (208,144) and the
      stairs at (208,96); the "(208,96) block" note is a RAM-read artifact.
      MISS if the tile map shows the block quad at (208,96).
  R5  RIGHT along the y=96 row onto (208,93) enters cellar 0x7B mode 9.
      MISS if Link stands at (208,93) in play 0x0D with no mode change.

    QT_QPA_PLATFORM=offscreen PYTHONPATH=.:nes:snes uv run python \\
        nes/zelda_i/scratch/probe_l7_room0d_ring.py --tag 0d_ring_v1
"""

from __future__ import annotations

import argparse
import json

from retro_harness.env import make_env, reset_obs, save_state, state_path
from retro_harness.nes import nes_action, nes_idle_action
from retro_harness.segment_runner import configure_headless, save_rgb_png
from zelda_i.dungeon.tilemap import (
    ascii_room,
    door_cells,
    find_cells,
    link_cell,
    stair_cells,
    BLOCK_TILES,
)
from zelda_i.dungeon.trace import compact_snapshot, write_state_provenance
from zelda_i.paths import GAME, GAME_DIR, RECORDINGS_DIR
from zelda_i.ram import (
    ADDR_BOMBS,
    ADDR_CANDLE,
    ADDR_FOOD,
    ADDR_KEYS,
    ADDR_LADDER,
    ADDR_WHISTLE,
    PLAY_MODE,
    read_snapshot,
    read_u8,
)
from zelda_i.runner import make_assist

ROOM = 0x0D
CELLAR_MODES = {9, 10, 11, 16}
BLOCK_TYPE = 0x68
# Cell rows Link stands on, as his stored y.
ROW_Y = {96: 93, 112: 109, 128: 125, 144: 141, 160: 157, 176: 173, 192: 189}
WEST_COLUMN_X = 32
EAST_COLUMN_X = 208
DOOR_ROW_CELL_Y = 144


def _s(env):
    return read_snapshot(env.get_ram())


def _step(env, a, btn, f):
    if btn is None:
        env.step(nes_idle_action())
    elif isinstance(btn, tuple):
        env.step(nes_action(*btn))
    else:
        env.step(nes_action(btn))
    if a:
        a.apply_env(env, frame=f)
    return f + 1


def _blocks(s):
    return [
        {"slot": int(o.slot), "x": int(o.x), "y": int(o.y)}
        for o in s.objects
        if 1 <= int(o.slot) <= 12 and int(o.type_id) == BLOCK_TYPE
    ]


def _glance(env) -> dict:
    ram = env.get_ram()
    s = read_snapshot(ram)
    return {
        "screen": f"0x{int(s.screen):02x}",
        "mode": int(s.mode),
        "xy": [int(s.link_x), int(s.link_y)],
        "cell": list(link_cell(int(s.link_x), int(s.link_y))),
        "colliding_tile": int(s.colliding_tile),
        "level": int(s.level),
        "cur_opened_doors": int(s.cur_opened_doors),
        "room_all_dead": int(s.room_all_dead),
        "blocks": _blocks(s),
        "stair_cells": [list(c) for c in stair_cells(ram)],
        "keys": int(read_u8(ram, ADDR_KEYS)),
        "bombs": int(read_u8(ram, ADDR_BOMBS)),
        "candle": int(read_u8(ram, ADDR_CANDLE)),
        "food": int(read_u8(ram, ADDR_FOOD)),
        "whistle": int(read_u8(ram, ADDR_WHISTLE)),
        "ladder": int(read_u8(ram, ADDR_LADDER)),
        "triforce": int(s.triforce),
    }


def _left_would_exit(s) -> bool:
    """Never press LEFT on the west door row: it leaves 0x0D for 0x79."""
    cx, cy = link_cell(int(s.link_x), int(s.link_y))
    return cy == DOOR_ROW_CELL_Y and cx <= WEST_COLUMN_X


def _leg(env, a, f, btn, done, *, budget, trace, tag, stall=48):
    """Hold one cardinal until ``done(snap)``. Returns (f, ok, note)."""
    last = None
    same = 0
    for _ in range(budget):
        s = _s(env)
        if int(s.mode) in CELLAR_MODES:
            return f, True, "cellar"
        if int(s.screen) != ROOM:
            return f, False, f"left_room_0x{int(s.screen):02x}"
        if done(s):
            return f, True, "reached"
        if btn == "LEFT" and _left_would_exit(s):
            return f, False, "door_row_guard"
        xy = (int(s.link_x), int(s.link_y))
        if xy == last:
            same += 1
            if same >= stall:
                trace.append(
                    {
                        "leg": tag,
                        "stalled": list(xy),
                        "tile": int(s.colliding_tile),
                        "cell": list(link_cell(*xy)),
                    }
                )
                return f, False, f"stall_{xy[0]}_{xy[1]}"
        else:
            same = 0
            last = xy
        f = _step(env, a, btn, f)
    return f, False, "budget"


def _push(env, a, f, face, trace):
    """Verified push: stand on the block's ``face`` side and hold."""
    s = _s(env)
    bl = _blocks(s)
    if not bl:
        return f, False, "no_block"
    bx, by = bl[0]["x"], bl[0]["y"]
    stand = {
        "RIGHT": (bx - 16, by),
        "LEFT": (bx + 16, by),
        "UP": (bx, by + 16),
        "DOWN": (bx, by - 16),
    }[face]
    # Reach the face along the y=144 row from the west (never the door row
    # LEFT-most cell; we only travel east here).
    f, ok, note = _leg(
        env, a, f, "UP",
        lambda s: int(s.link_y) <= ROW_Y[144],
        budget=200, trace=trace, tag="push_row",
    )
    f, ok, note = _leg(
        env, a, f, "RIGHT",
        lambda s: int(s.link_x) >= stand[0],
        budget=400, trace=trace, tag="push_east",
    )
    if not ok:
        return f, False, f"push_east_{note}"
    for _ in range(40):
        s = _s(env)
        if abs(int(s.link_y) - by) <= 1:
            break
        f = _step(env, a, "DOWN" if int(s.link_y) < by else "UP", f)
    started = False
    for _ in range(160):
        s = _s(env)
        cb = _blocks(s)
        cx = cb[0]["x"] if cb else bx
        cy = cb[0]["y"] if cb else by
        if not started and (cx != bx or cy != by):
            started = True
        if started and (abs(cx - bx) >= 16 or abs(cy - by) >= 16):
            break
        f = _step(env, a, None if started else face, f)
    for _ in range(30):
        f = _step(env, a, None, f)
    return f, True, "pushed"


def _tilemap_block_cells(ram):
    return [list(c) for c in find_cells(ram, BLOCK_TILES)]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="0d_ring_v1")
    ap.add_argument("--from-state", default="Level7Interior0DClearedReconFixture")
    ap.add_argument("--push-dir", default="RIGHT",
                    choices=["RIGHT", "LEFT", "UP", "DOWN", "NONE"])
    ap.add_argument("--ring", default="north", choices=["north", "south"])
    ap.add_argument("--save-fixture", default="")
    args = ap.parse_args()
    configure_headless()
    a = make_assist(True)
    env = make_env(GAME, args.from_state, GAME_DIR, render_mode="rgb_array")
    RECORDINGS_DIR.mkdir(parents=True, exist_ok=True)
    out: dict = {
        "route_eligible": False,
        "push_dir": args.push_dir,
        "ring": args.ring,
        "position_writes": 0,
    }
    trace: list[dict] = []
    try:
        reset_obs(env)
        for _ in range(2):
            env.step(nes_idle_action())
        ram = env.get_ram()
        out["start"] = _glance(env)
        out["map_before"] = ascii_room(ram)
        out["stairs_before"] = [list(c) for c in stair_cells(ram)]
        out["doors"] = [list(c) for c in door_cells(ram)]
        out["block_cells_before"] = _tilemap_block_cells(ram)
        print("START", out["start"])
        print(out["map_before"])
        print("stairs_before", out["stairs_before"], "doors", out["doors"])
        save_rgb_png(env.render(), RECORDINGS_DIR / f"{args.tag}_start.png")
        f = 0

        if args.push_dir != "NONE":
            f, ok, note = _push(env, a, f, args.push_dir, trace)
            ram = env.get_ram()
            out["push"] = {"ok": ok, "note": note, "glance": _glance(env)}
            out["map_after_push"] = ascii_room(ram)
            out["stairs_after_push"] = [list(c) for c in stair_cells(ram)]
            out["block_cells_after_push"] = _tilemap_block_cells(ram)
            print("PUSH", ok, note, out["push"]["glance"])
            print(out["map_after_push"])
            print("stairs_after_push", out["stairs_after_push"])
            print("block_cells_after_push", out["block_cells_after_push"])
            save_rgb_png(env.render(), RECORDINGS_DIR / f"{args.tag}_pushed.png")

        if args.ring == "north":
            legs = (
                ("west_off_door_row", "LEFT",
                 lambda s: int(s.link_x) <= 160, 400),
                ("up_to_row128", "UP",
                 lambda s: int(s.link_y) <= ROW_Y[128], 200),
                ("west_column", "LEFT",
                 lambda s: int(s.link_x) <= WEST_COLUMN_X, 600),
                ("north_column", "UP",
                 lambda s: int(s.link_y) <= ROW_Y[96], 300),
                ("east_top_row", "RIGHT",
                 lambda s: int(s.link_x) >= EAST_COLUMN_X, 700),
            )
        else:
            legs = (
                ("west_off_door_row", "LEFT",
                 lambda s: int(s.link_x) <= 160, 400),
                ("down_to_row160", "DOWN",
                 lambda s: int(s.link_y) >= ROW_Y[160], 200),
                ("west_column", "LEFT",
                 lambda s: int(s.link_x) <= WEST_COLUMN_X, 600),
                ("south_column", "DOWN",
                 lambda s: int(s.link_y) >= ROW_Y[192], 300),
                ("east_bottom_row", "RIGHT",
                 lambda s: int(s.link_x) >= EAST_COLUMN_X, 700),
                ("east_column_up", "UP",
                 lambda s: int(s.link_y) <= ROW_Y[96], 400),
            )
        hops = []
        for tag, btn, done, budget in legs:
            f, ok, note = _leg(
                env, a, f, btn, done, budget=budget, trace=trace, tag=tag
            )
            g = _glance(env)
            rec = {"leg": tag, "btn": btn, "ok": ok, "note": note,
                   "xy": g["xy"], "cell": g["cell"], "mode": g["mode"],
                   "screen": g["screen"], "tile": g["colliding_tile"],
                   "frames": f}
            hops.append(rec)
            print("LEG", rec)
            save_rgb_png(
                env.render(), RECORDINGS_DIR / f"{args.tag}_{tag}.png"
            )
            if int(_s(env).mode) in CELLAR_MODES:
                break
            if not ok:
                break
        out["legs"] = hops
        out["trace"] = trace

        if int(_s(env).mode) not in CELLAR_MODES and int(_s(env).screen) == ROOM:
            # Stand on the target cell and let CheckWarps settle.
            for _ in range(120):
                if int(_s(env).mode) in CELLAR_MODES:
                    break
                f = _step(env, a, None, f)
            for btn in ("UP", None, "RIGHT", None, "DOWN", None, "LEFT", None):
                if int(_s(env).mode) in CELLAR_MODES:
                    break
                f = _step(env, a, btn, f)
            out["after_nudge"] = _glance(env)
            print("AFTER NUDGE", out["after_nudge"])

        if int(_s(env).mode) in CELLAR_MODES:
            out["cellar"] = _glance(env)
            print("CELLAR", out["cellar"])
            save_rgb_png(env.render(), RECORDINGS_DIR / f"{args.tag}_cellar.png")
            for _ in range(240):
                s = _s(env)
                if int(s.mode) == PLAY_MODE and int(s.screen) != ROOM:
                    break
                f = _step(env, a, None, f)
            out["cellar_settled"] = _glance(env)
            print("CELLAR SETTLED", out["cellar_settled"])

        end = _glance(env)
        end["deaths"] = int(a.telemetry.deaths)
        end["progression_writes"] = int(a.telemetry.progression_writes)
        end["capacity_writes"] = int(a.telemetry.capacity_writes)
        end["frames"] = f
        out["end"] = end
        save_rgb_png(env.render(), RECORDINGS_DIR / f"{args.tag}_final.png")
        if args.save_fixture and int(_s(env).screen) not in {0x79}:
            path = save_state(env, GAME_DIR, GAME, args.save_fixture)
            src = state_path(GAME_DIR, GAME, args.from_state)
            write_state_provenance(
                path,
                source_state_path=src if src.exists() else None,
                request={
                    "bead": "rr-8t4.3",
                    "phase": "level7_nose_cellar_walkon",
                    "track": "recon_fixture",
                    "route_eligible": False,
                    "fixture_only": True,
                    "natural_entry": False,
                    "development_only": True,
                    "fixture_writes": [],
                    "notes": [
                        f"Derived from {args.from_state}: 0x0D {args.push_dir} "
                        "push then ring walk onto the (208,96) staircase. "
                        "No position/door/TF writes; position_writes=0.",
                        "UnlimitedHealthAssist traversal aid only.",
                    ],
                },
                selected_trial={
                    "ok": True,
                    "state": compact_snapshot(_s(env)),
                    "glance": end,
                },
                natural_entry=False,
            )
            out["saved_fixture"] = str(path)
            print("saved", path)
        (RECORDINGS_DIR / f"{args.tag}.json").write_text(json.dumps(out, indent=1))
        print("END", end)
    finally:
        env.close()


if __name__ == "__main__":
    main()
