"""Recon: 0x1C FORCED_DIGDOGGER whistle shrink 0x38→0x18, then sword.

Pin: Level7Interior1CReconFixture — L7 play 0x1C (16,141) W mouth, type
0x38, keys 2, bombs 7, candle 2, whistle 1. North door is KILL_CLEAR.

Recipe (L5, no ADDR_SELECTED_ITEM poke): pause-select recorder=5, 12×B.
Stand is L7-remeasured; L5 (120,141) is the first hypothesis.

    QT_QPA_PLATFORM=offscreen PYTHONPATH=.:nes:snes uv run python \\
        nes/zelda_i/scratch/probe_l7_forced_digdogger.py --tag 1c_wh_v1
"""

from __future__ import annotations

import argparse
import json

from retro_harness.env import make_env, reset_obs, save_state, state_path
from retro_harness.nes import nes_action, nes_idle_action
from retro_harness.segment_runner import configure_headless, save_rgb_png
from zelda_i.combat import nearest_enemy, should_swing_at
from zelda_i.dungeon.behaviors import (
    DIGDOGGER_SHRUNK_TYPE,
    DIGDOGGER_TYPE,
    EnemyKind,
    engagement_hint,
)
from zelda_i.dungeon.ids import object_name
from zelda_i.dungeon.ops import idle
from zelda_i.dungeon.trace import compact_snapshot, write_state_provenance
from zelda_i.level5.boss_path import WHISTLE_B_SLOT
from zelda_i.level5.whistle_path import select_b_item_menu
from zelda_i.paths import GAME, GAME_DIR, RECORDINGS_DIR
from zelda_i.ram import (
    ADDR_BOMBS,
    ADDR_CANDLE,
    ADDR_KEYS,
    ADDR_SELECTED_ITEM,
    ADDR_WHISTLE,
    PLAY_MODE,
    read_snapshot,
    read_u8,
)
from zelda_i.runner import make_assist

ROOM = 0x1C


def _s(env):
    return read_snapshot(env.get_ram())


def _step(env, a, btn, f, total):
    if btn is None:
        env.step(nes_idle_action())
    elif isinstance(btn, tuple):
        env.step(nes_action(*btn))
    else:
        env.step(nes_action(btn))
    if a:
        a.apply_env(env, frame=f)
    total[0] += 1
    return total[0]


def _glance(env) -> dict:
    ram = env.get_ram()
    s = read_snapshot(ram)
    types = sorted(
        {
            int(o.type_id)
            for o in s.objects
            if 1 <= int(o.slot) <= 12 and int(o.type_id) not in (0, 0xFF)
        }
    )
    return {
        "screen": f"0x{int(s.screen):02x}",
        "mode": int(s.mode),
        "xy": [int(s.link_x), int(s.link_y)],
        "selected": int(read_u8(ram, ADDR_SELECTED_ITEM)),
        "obj_types": [f"0x{t:02x}:{object_name(t)}" for t in types],
        "objects": [
            {
                "slot": int(o.slot),
                "type": f"0x{int(o.type_id):02x}",
                "hp": int(o.hp),
                "xy": [int(o.x), int(o.y)],
            }
            for o in s.objects
            if 1 <= int(o.slot) <= 12 and int(o.type_id) not in (0, 0xFF)
        ],
        "keys": int(read_u8(ram, ADDR_KEYS)),
        "bombs": int(read_u8(ram, ADDR_BOMBS)),
        "candle": int(read_u8(ram, ADDR_CANDLE)),
        "whistle": int(read_u8(ram, ADDR_WHISTLE)),
        "triforce": int(s.triforce),
        "room_all_dead": int(s.room_all_dead),
        "cur_opened_doors": int(s.cur_opened_doors),
    }


def _reach(env, a, tx, ty, f, total, budget=300):
    for _ in range(budget):
        s = _s(env)
        x, y = int(s.link_x), int(s.link_y)
        if abs(x - tx) <= 3 and abs(y - ty) <= 3:
            return f, True, [x, y]
        if abs(y - ty) > 3:
            btn = "UP" if y > ty else "DOWN"
        else:
            btn = "LEFT" if x > tx else "RIGHT"
        f = _step(env, a, btn, f, total)
    e = _s(env)
    return f, False, [int(e.link_x), int(e.link_y)]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="1c_wh_v1")
    ap.add_argument("--from-state", default="Level7Interior1CReconFixture")
    ap.add_argument("--stand-x", type=int, default=120)
    ap.add_argument("--stand-y", type=int, default=141)
    ap.add_argument("--save-fixture", default="")
    args = ap.parse_args()
    configure_headless()
    a = make_assist(True)
    env = make_env(GAME, args.from_state, GAME_DIR, render_mode="rgb_array")
    RECORDINGS_DIR.mkdir(parents=True, exist_ok=True)
    total = [0]
    out: dict = {"route_eligible": False, "stand": [args.stand_x, args.stand_y]}
    try:
        reset_obs(env)
        for _ in range(2):
            env.step(nes_idle_action())
        out["start"] = _glance(env)
        print("START", out["start"])
        save_rgb_png(env.render(), RECORDINGS_DIR / f"{args.tag}_start.png")
        f = 0
        f, ok, xy = _reach(env, a, args.stand_x, args.stand_y, f, total)
        out["at_stand"] = {"ok": ok, "xy": xy, "glance": _glance(env)}
        print("STAND", out["at_stand"])
        save_rgb_png(env.render(), RECORDINGS_DIR / f"{args.tag}_stand.png")
        idle(env, a, total, 8)
        menu = select_b_item_menu(env, a, total, WHISTLE_B_SLOT)
        out["menu"] = menu
        print("MENU", menu)
        shrunk = False
        for attempt in range(4):
            for _ in range(12):
                f = _step(env, a, "B", f, total)
            for _ in range(20):
                idle(env, a, total, 12)
                s = _s(env)
                types = [
                    int(o.type_id)
                    for o in s.objects
                    if 1 <= int(o.slot) <= 12 and int(o.type_id) not in (0, 0xFF)
                ]
                rec = {
                    "attempt": attempt,
                    "types": [f"0x{t:02x}" for t in types],
                    "xy": [int(s.link_x), int(s.link_y)],
                    "mode": int(s.mode),
                }
                print("BLOW", rec)
                if DIGDOGGER_SHRUNK_TYPE in types or (
                    types and DIGDOGGER_TYPE not in types
                ):
                    shrunk = True
                    out["shrunk"] = rec
                    break
            if shrunk:
                break
            save_rgb_png(
                env.render(),
                RECORDINGS_DIR / f"{args.tag}_blow_{attempt}.png",
            )
        out["shrunk_ok"] = shrunk
        save_rgb_png(env.render(), RECORDINGS_DIR / f"{args.tag}_after_blow.png")
        out["after_blow"] = _glance(env)
        print("AFTER BLOW", out["after_blow"])
        if shrunk:
            killed = False
            for i in range(4000):
                s = _s(env)
                live = [
                    o
                    for o in s.objects
                    if 1 <= int(o.slot) <= 12
                    and int(o.type_id) == DIGDOGGER_SHRUNK_TYPE
                    and int(o.hp) > 0
                ]
                if not live:
                    killed = True
                    out["killed_frame"] = total[0]
                    print("KILLED", _glance(env))
                    save_rgb_png(
                        env.render(),
                        RECORDINGS_DIR / f"{args.tag}_killed.png",
                    )
                    break
                tgt = nearest_enemy(s.link_x, s.link_y, live)
                if tgt is None:
                    f = _step(env, a, None, f, total)
                    continue
                hint = engagement_hint(EnemyKind.DIGDOGGER, s, tgt)
                if should_swing_at(
                    s.link_x, s.link_y, hint.face, (tgt,), hint=hint
                ) and (i % 8) < 4:
                    btn = (hint.face, "A")
                elif hint.retreat:
                    btn = {"UP": "DOWN", "DOWN": "UP", "LEFT": "RIGHT",
                           "RIGHT": "LEFT"}.get(hint.face, "DOWN")
                else:
                    btn = hint.face
                f = _step(env, a, btn, f, total)
                if i > 0 and i % 400 == 0:
                    save_rgb_png(
                        env.render(),
                        RECORDINGS_DIR / f"{args.tag}_fight_{i}.png",
                    )
                    print("FIGHT", i, _glance(env)["xy"], "n", len(live))
            out["killed"] = killed
            if killed:
                f, ok, xy = _reach(env, a, 120, 93, f, total, budget=400)
                for _ in range(240):
                    s = _s(env)
                    if int(s.screen) != ROOM and int(s.mode) == PLAY_MODE:
                        break
                    f = _step(env, a, "UP", f, total)
                for _ in range(180):
                    s = _s(env)
                    if int(s.mode) == PLAY_MODE and not s.transitioning:
                        break
                    f = _step(env, a, None, f, total)
                out["after_north"] = _glance(env)
                print("NORTH", out["after_north"])
                save_rgb_png(
                    env.render(), RECORDINGS_DIR / f"{args.tag}_north.png"
                )
                if args.save_fixture:
                    path = save_state(env, GAME_DIR, GAME, args.save_fixture)
                    src = state_path(GAME_DIR, GAME, args.from_state)
                    write_state_provenance(
                        path,
                        source_state_path=src if src.exists() else None,
                        request={
                            "bead": "rr-8t4.3",
                            "phase": "level7_interior_0c_recon",
                            "track": "recon_fixture",
                            "route_eligible": False,
                            "fixture_only": True,
                            "natural_entry": False,
                            "development_only": True,
                            "fixture_writes": [],
                            "notes": [
                                f"Derived from {args.from_state}: 0x1C "
                                "Whistle shrink 0x38->0x18, sword-kill, "
                                f"KILL-CLEAR north -> {out['after_north']['screen']}. "
                                "No Candle/TF/door writes. Pause-select recorder=5.",
                                "UnlimitedHealthAssist traversal aid only.",
                            ],
                        },
                        selected_trial={
                            "ok": True,
                            "state": compact_snapshot(_s(env)),
                            "glance": out["after_north"],
                        },
                        natural_entry=False,
                    )
                    out["saved_fixture"] = str(path)
                    print("saved", path)
        end = _glance(env)
        end["deaths"] = int(a.telemetry.deaths)
        end["progression_writes"] = int(a.telemetry.progression_writes)
        end["capacity_writes"] = int(a.telemetry.capacity_writes)
        end["selected_writes"] = "none (pause-select)"
        out["end"] = end
        out["frames"] = total[0]
        save_rgb_png(env.render(), RECORDINGS_DIR / f"{args.tag}_final.png")
        (RECORDINGS_DIR / f"{args.tag}.json").write_text(json.dumps(out, indent=1))
        print("END", end)
        print("wrote", RECORDINGS_DIR / f"{args.tag}.json")
    finally:
        env.close()


if __name__ == "__main__":
    main()
