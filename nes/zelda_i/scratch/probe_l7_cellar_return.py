"""Recon: Red Candle cellar 0x4A stairs return UP to live CANDLE_PUSH.

Pin: Level7Interior4AReconFixture — L7 cellar $EB=0x4A mode 9 (136,141),
Candle 2 NATURAL, keys 3, bombs 8, food 0, whistle 1, ladder 1, TF 0.

Dead: walk off the candle pad at y=141 (tile 243). Inbound climbed the east
ladder then LEFT at y=125 onto the pad; reverse starts UP, not LEFT/RIGHT
at y=141.

    QT_QPA_PLATFORM=offscreen uv run python \\
        nes/zelda_i/scratch/probe_l7_cellar_return.py --tag 4a_ret_v1
"""

from __future__ import annotations

import argparse
import json

from retro_harness.env import make_env, reset_obs, save_state, state_path
from retro_harness.nes import nes_action, nes_idle_action
from retro_harness.segment_runner import configure_headless, save_rgb_png
from zelda_i.dungeon.ids import object_name, room_item_name
from zelda_i.dungeon.trace import compact_snapshot, write_state_provenance
from zelda_i.paths import GAME, GAME_DIR, RECORDINGS_DIR
from zelda_i.ram import (
    ADDR_BOMBS,
    ADDR_CANDLE,
    ADDR_FOOD,
    ADDR_HEALTH,
    ADDR_HEART_PARTIAL,
    ADDR_KEYS,
    ADDR_LADDER,
    ADDR_SELECTED_ITEM,
    ADDR_WHISTLE,
    PLAY_MODE,
    read_snapshot,
    read_u8,
)
from zelda_i.runner import make_assist

CELLAR = 0x4A
CELLAR_MODES = {9, 10, 11}
PLAY = 0x1A
STAIRS_TILES = range(0x70, 0x74)


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
    types = sorted(
        {
            int(o.type_id)
            for o in s.objects
            if 1 <= int(o.slot) <= 12 and int(o.type_id) not in (0, 0xFF)
        }
    )
    return {
        "screen": f"0x{int(s.screen):02x}",
        "screen_int": int(s.screen),
        "mode": int(s.mode),
        "xy": [int(s.link_x), int(s.link_y)],
        "colliding_tile": int(s.colliding_tile),
        "cur_opened_doors": int(s.cur_opened_doors),
        "open_doorway_mask": int(s.open_doorway_mask),
        "room_item_id": int(s.room_item_id),
        "room_item_name": room_item_name(int(s.room_item_id)),
        "obj_types": [f"0x{t:02x}:{object_name(t)}" for t in types],
        "keys": int(read_u8(ram, ADDR_KEYS)),
        "bombs": int(read_u8(ram, ADDR_BOMBS)),
        "candle": int(read_u8(ram, ADDR_CANDLE)),
        "food": int(read_u8(ram, ADDR_FOOD)),
        "whistle": int(read_u8(ram, ADDR_WHISTLE)),
        "ladder": int(read_u8(ram, ADDR_LADDER)),
        "selected": int(read_u8(ram, ADDR_SELECTED_ITEM)),
        "triforce": int(s.triforce),
        "health": int(read_u8(ram, ADDR_HEALTH)),
        "heart_partial": int(read_u8(ram, ADDR_HEART_PARTIAL)),
        "level": int(s.level),
        "transitioning": bool(s.transitioning),
    }


def _play_dest(s) -> bool:
    return (
        int(s.mode) == PLAY_MODE
        and not s.transitioning
        and int(s.screen) != CELLAR
    )


def _parse_seq(txt: str) -> list[tuple[str | tuple[str, ...] | None, int]]:
    """'UP:80,RIGHT:40,LEFT+DOWN:30,IDLE:20' -> [(btn, frames), ...]."""
    out = []
    for chunk in txt.replace(";", ",").split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        if ":" in chunk:
            name, n = chunk.rsplit(":", 1)
            frames = int(n)
        else:
            name, frames = chunk, 80
        name = name.strip().upper()
        if name in ("IDLE", "NONE", ""):
            btn: str | tuple[str, ...] | None = None
        elif "+" in name:
            btn = tuple(p.strip() for p in name.split("+") if p.strip())
        else:
            btn = name
        out.append((btn, frames))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="4a_ret_v1")
    ap.add_argument("--from-state", default="Level7Interior4AReconFixture")
    ap.add_argument("--seq", default="UP:120")
    ap.add_argument("--save-fixture", default="")
    args = ap.parse_args()
    configure_headless()
    a = make_assist(True)
    env = make_env(GAME, args.from_state, GAME_DIR, render_mode="rgb_array")
    RECORDINGS_DIR.mkdir(parents=True, exist_ok=True)
    out: dict = {"route_eligible": False, "seq": args.seq, "samples": []}
    try:
        reset_obs(env)
        for _ in range(2):
            env.step(nes_idle_action())
        out["start"] = _glance(env)
        print("START", out["start"])
        save_rgb_png(env.render(), RECORDINGS_DIR / f"{args.tag}_start.png")
        f = 0
        dest = None
        for btn, n in _parse_seq(args.seq):
            label = "IDLE" if btn is None else (
                "+".join(btn) if isinstance(btn, tuple) else btn
            )
            stuck = 0
            last = None
            for i in range(n):
                s = _s(env)
                if _play_dest(s):
                    dest = _glance(env)
                    dest["arrived_frame"] = f
                    print("DEST", dest)
                    save_rgb_png(
                        env.render(), RECORDINGS_DIR / f"{args.tag}_dest.png"
                    )
                    break
                xy = (int(s.link_x), int(s.link_y), int(s.mode), int(s.screen))
                if xy == last:
                    stuck += 1
                else:
                    stuck = 0
                    last = xy
                if i == 0 or i % 20 == 0 or stuck == 12:
                    rec = {
                        "f": f,
                        "btn": label,
                        "xy": [int(s.link_x), int(s.link_y)],
                        "mode": int(s.mode),
                        "screen": f"0x{int(s.screen):02x}",
                        "tile": int(s.colliding_tile),
                        "stuck": stuck,
                    }
                    out["samples"].append(rec)
                    print("SAMP", rec)
                if stuck >= 40:
                    rec = {
                        "f": f,
                        "btn": label,
                        "xy": [int(s.link_x), int(s.link_y)],
                        "tile": int(s.colliding_tile),
                        "stand": True,
                    }
                    out["samples"].append(rec)
                    print("STAND", rec)
                    save_rgb_png(
                        env.render(),
                        RECORDINGS_DIR
                        / f"{args.tag}_stand_{int(s.link_x)}_{int(s.link_y)}.png",
                    )
                    break
                _step(env, a, btn, f)
                f += 1
                if f > 0 and f % 80 == 0:
                    save_rgb_png(
                        env.render(),
                        RECORDINGS_DIR / f"{args.tag}_f{f}.png",
                    )
            if dest is not None:
                break
        if dest is None:
            for _ in range(80):
                s = _s(env)
                if _play_dest(s):
                    dest = _glance(env)
                    dest["arrived_frame"] = f
                    break
                _step(env, a, None, f)
                f += 1
        if dest is not None:
            for _ in range(170):
                s = _s(env)
                if int(s.mode) == PLAY_MODE and not s.transitioning:
                    break
                _step(env, a, None, f)
                f += 1
            out["dest"] = _glance(env)
            out["result"] = f"0x{int(_s(env).screen):02x}"
            print("SETTLED", out["dest"])
            save_rgb_png(env.render(), RECORDINGS_DIR / f"{args.tag}_dest.png")
            if args.save_fixture:
                path = save_state(env, GAME_DIR, GAME, args.save_fixture)
                src = state_path(GAME_DIR, GAME, args.from_state)
                write_state_provenance(
                    path,
                    source_state_path=src if src.exists() else None,
                    request={
                        "bead": "rr-8t4.3",
                        "phase": "level7_cellar_return_recon",
                        "track": "recon_fixture",
                        "route_eligible": False,
                        "fixture_only": True,
                        "natural_entry": False,
                        "development_only": True,
                        "fixture_writes": [],
                        "notes": [
                            f"Derived from {args.from_state}: cellar 0x4A "
                            f"stairs return seq={args.seq} -> "
                            f"{out['result']}. No Candle/TF/ladder/door writes.",
                            "UnlimitedHealthAssist traversal aid only.",
                        ],
                    },
                    selected_trial={
                        "ok": True,
                        "state": compact_snapshot(_s(env)),
                        "glance": out["dest"],
                    },
                    natural_entry=False,
                )
                out["saved_fixture"] = str(path)
                print("saved", path)
        else:
            out["result"] = "blocked"
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
