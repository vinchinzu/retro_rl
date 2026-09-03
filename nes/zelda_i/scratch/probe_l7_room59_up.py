"""Recon: from Level7Interior59ReconFixture, clear the 0x59 goriya 0x05/0x06,
then take the perimeter waypoint route around the central obstacle to the UP
door (x~120, plane y~93) -> live $EB=0x49 (GORIYA_BUBBLE: goriya 0x05 + keese
0x1b + bubble residual 0x2b, entry (120,205) S mouth).

Waypoints (0x59 is lit; central mass fills ~x100..190 / y118..165):
  clear -> y~100 (open west band) -> x~44 -> y~64 (open top band)
  -> x~120 -> push UP with x-align.

    QT_QPA_PLATFORM=offscreen uv run python \\
        nes/zelda_i/scratch/probe_l7_room59_up.py --tag 59_up_v2 [--save-fixture]
"""

from __future__ import annotations

import argparse
import json

from retro_harness.env import make_env, reset_obs, save_state, state_path
from retro_harness.nes import nes_action, nes_idle_action
from retro_harness.segment_runner import configure_headless, save_rgb_png
from zelda_i.combat import nearest_enemy
from zelda_i.dungeon.behaviors import EnemyKind, engagement_hint
from zelda_i.dungeon.ids import object_name, room_item_name
from zelda_i.dungeon.trace import compact_snapshot, write_state_provenance
from zelda_i.level7.path import live_goriyas
from zelda_i.paths import GAME, GAME_DIR, RECORDINGS_DIR
from zelda_i.ram import (
    ADDR_BOMBS, ADDR_CANDLE, ADDR_FOOD, ADDR_KEYS, PLAY_MODE,
    read_snapshot, read_u8,
)
from zelda_i.runner import make_assist

ROOM_59 = 0x59
DEST_FIXTURE = "Level7Interior49ReconFixture"


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


def _glance(env) -> dict:
    ram = env.get_ram()
    s = read_snapshot(ram)
    types = sorted({int(o.type_id) for o in s.objects
                    if 1 <= int(o.slot) <= 12 and int(o.type_id) not in (0, 0xFF)})
    return {
        "screen": f"0x{int(s.screen):02x}", "mode": int(s.mode),
        "xy": [int(s.link_x), int(s.link_y)],
        "room_all_dead": int(s.room_all_dead),
        "room_item_id": int(s.room_item_id),
        "room_item_name": room_item_name(int(s.room_item_id)),
        "cur_opened_doors": int(s.cur_opened_doors),
        "open_doorway_mask": int(s.open_doorway_mask),
        "obj_types": [f"0x{t:02x}:{object_name(t)}" for t in types],
        "keys": int(read_u8(ram, ADDR_KEYS)), "bombs": int(read_u8(ram, ADDR_BOMBS)),
        "candle": int(read_u8(ram, ADDR_CANDLE)), "food": int(read_u8(ram, ADDR_FOOD)),
    }


def _clear(env, a, f, budget=2600):
    saw = False
    for i in range(budget):
        s = _s(env)
        if int(s.screen) != ROOM_59:
            return f, "left"
        live = live_goriyas(s)
        if live:
            saw = True
        elif saw and i > 60:
            return f, "clear"
        else:
            _step(env, a, "DOWN", f); f += 1; continue
        tgt = nearest_enemy(s.link_x, s.link_y, live)
        hint = engagement_hint(EnemyKind.GORIYA, s, tgt)
        _step(env, a, (hint.face, "A") if (i % 8) < 4 else hint.face, f); f += 1
    return f, "timeout"


def _reach_y(env, a, ty, f, budget=240):
    for _ in range(budget):
        s = _s(env)
        if int(s.screen) != ROOM_59:
            return f, False
        y = int(s.link_y)
        if abs(y - ty) <= 3:
            return f, True
        _step(env, a, "UP" if y > ty else "DOWN", f); f += 1
    return f, False


def _go_x(env, a, tx, f, budget=140):
    for _ in range(budget):
        s = _s(env)
        if int(s.screen) != ROOM_59:
            return f, False
        x = int(s.link_x)
        if abs(x - tx) <= 3:
            return f, True
        _step(env, a, "LEFT" if x > tx else "RIGHT", f); f += 1
    return f, True


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="59_up_v2")
    ap.add_argument("--from-state", default="Level7Interior59ReconFixture")
    ap.add_argument("--save-fixture", action="store_true")
    args = ap.parse_args()
    configure_headless()
    a = make_assist(True)
    env = make_env(GAME, args.from_state, GAME_DIR, render_mode="rgb_array")
    RECORDINGS_DIR.mkdir(parents=True, exist_ok=True)
    out: dict = {"route_eligible": False}
    try:
        reset_obs(env)
        for _ in range(2):
            env.step(nes_idle_action())
        out["start"] = _glance(env)
        print("START", out["start"])
        f = 0
        f, status = _clear(env, a, f)
        out["clear_status"] = status
        out["after_clear"] = _glance(env)
        print("AFTER CLEAR", status, out["after_clear"])
        # perimeter waypoints
        for (wx, wy) in [(None, 100), (44, None), (None, 64), (120, None)]:
            if wy is not None:
                f, _ = _reach_y(env, a, wy, f)
            if wx is not None:
                f, _ = _go_x(env, a, wx, f)
        s = _s(env)
        out["pre_push"] = [int(s.link_x), int(s.link_y)]
        print("PRE PUSH", out["pre_push"])
        hit = None
        for _ in range(220):
            s = _s(env)
            if int(s.screen) != ROOM_59 and int(s.mode) == PLAY_MODE and not s.transitioning:
                hit = (int(s.screen), f)
                break
            x = int(s.link_x)
            b = "UP" if abs(x - 120) <= 4 else ("LEFT" if x > 120 else "RIGHT")
            _step(env, a, b, f); f += 1
        if hit is None:
            e = _s(env)
            out["result"] = "blocked"
            out["end_xy"] = [int(e.link_x), int(e.link_y)]
            out["tile"] = int(e.colliding_tile)
        else:
            out["result"] = f"0x{hit[0]:02x}"
            out["arrived_frame"] = hit[1]
            for _ in range(160):
                _step(env, a, None, f); f += 1
            out["dest"] = _glance(env)
            print("DEST", out["dest"])
            if args.save_fixture and out["result"] == "0x49":
                path = save_state(env, GAME_DIR, GAME, DEST_FIXTURE)
                src = state_path(GAME_DIR, GAME, args.from_state)
                write_state_provenance(
                    path, source_state_path=src if src.exists() else None,
                    request={"bead": "rr-8t4.2", "phase": "level7_interior_49_recon",
                             "track": "recon_fixture", "route_eligible": False,
                             "fixture_only": True, "natural_entry": False,
                             "development_only": True, "fixture_writes": [],
                             "notes": [f"Derived from {args.from_state}: 0x59 goriya "
                                       "0x05/0x06 cleared + perimeter waypoint route to "
                                       "the UP door -> 0x49 (GORIYA_BUBBLE). No set_state, "
                                       "no Candle/Food/TF/door/key/bomb writes.",
                                       "UnlimitedHealthAssist traversal aid only."]},
                    selected_trial={"ok": True, "state": compact_snapshot(_s(env)),
                                    "glance": out["dest"]},
                    natural_entry=False)
                out["saved_fixture"] = str(path)
                print("saved", path)
        end = _glance(env)
        end["deaths"] = int(a.telemetry.deaths)
        end["progression_writes"] = int(a.telemetry.progression_writes)
        end["capacity_writes"] = int(a.telemetry.capacity_writes)
        out["end"] = end
        save_rgb_png(env.render(), RECORDINGS_DIR / f"{args.tag}_final.png")
        (RECORDINGS_DIR / f"{args.tag}.json").write_text(json.dumps(out, indent=1))
        print("END", end)
        print("wrote", RECORDINGS_DIR / f"{args.tag}.json")
    finally:
        env.close()


if __name__ == "__main__":
    main()
