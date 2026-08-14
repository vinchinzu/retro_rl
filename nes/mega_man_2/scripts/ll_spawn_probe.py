"""Decode LL spawn live: watch $0400 types 0x3D/0x3E and mapset entry."""
from __future__ import annotations

import json
from pathlib import Path

from mega_man_2.paths import GAME, GAME_DIR, RECORDINGS_DIR
from mega_man_2.policy import AirManPolicy
from mega_man_2.ram import (
    ADDR_CAMERA_X,
    ADDR_CAMERA_X_SCREEN,
    ADDR_HEALTH,
    ADDR_TILE_FEET,
    camera_progress_x,
    is_fallen,
    player_screen_x,
    player_screen_y,
)
from retro_harness.env import make_env
from retro_harness.nes import nes_action, nes_idle_action
from retro_harness.segment_runner import configure_headless, save_rgb_png, write_json_report

# Disasm (lsmmega/mm2)
ADDR_OBJ_PTR = 0x0400       # aobject_pointer = type ID
ADDR_OBJ_FLAG = 0x0420      # bit7 exist
ADDR_OBJ_SCREEN = 0x0440
ADDR_OBJ_X = 0x0460
ADDR_OBJ_Y = 0x04A0
ADDR_ENEMIES_FLAG = 0x0100  # aenemies_flag base (page)
ADDR_SCREEN_ID = 0x0020     # zscreen_id (same as camera screen?)
ADDR_START_MAP = 0x0014
ADDR_END_MAP = 0x0015
ADDR_LEFT_EN_IDX = 0x0048
ADDR_RIGHT_EN_IDX = 0x0049

LL_TYPES = {0x3D, 0x3E, 0x3F}  # goro_move, goro, bolt
GOBLIN_TYPES = {0x40, 0x41, 0x42, 0x43, 0x44, 0x45}
PIPI_TYPES = {0x37, 0x38, 0x39, 0x3A, 0x3B, 0x3C}

TYPE_NAMES = {
    0x36: "matasaburo",
    0x37: "pipi",
    0x38: "pipi_move",
    0x39: "pipi_remove",
    0x3A: "pipi_egg",
    0x3B: "pipi_egg_crack",
    0x3C: "copipi",
    0x3D: "kaminari_goro_move",
    0x3E: "kaminari_goro",
    0x3F: "lightning_bolt",
    0x40: "goblin_1",
    0x41: "goblin_2",
}

def snap_objects(ram):
    objs = []
    types = set()
    for i in range(32):
        flag = int(ram[ADDR_OBJ_FLAG + i])
        if not (flag & 0x80):
            continue
        t = int(ram[ADDR_OBJ_PTR + i])
        types.add(t)
        objs.append({
            "i": i,
            "t": t,
            "name": TYPE_NAMES.get(t, f"unk_{t:#x}"),
            "x": int(ram[ADDR_OBJ_X + i]),
            "y": int(ram[ADDR_OBJ_Y + i]),
            "scr": int(ram[ADDR_OBJ_SCREEN + i]),
            "flag": flag,
        })
    return objs, types

def enemies_flag_dump(ram, n=40):
    return [int(ram[ADDR_ENEMIES_FLAG + i]) for i in range(n)]

def meta(ram):
    return {
        "cam_scr": int(ram[ADDR_CAMERA_X_SCREEN]),
        "cam_x": int(ram[ADDR_CAMERA_X]),
        "prog": camera_progress_x(ram),
        "sx": player_screen_x(ram),
        "sy": player_screen_y(ram),
        "hp": int(ram[ADDR_HEALTH]),
        "feet": int(ram[ADDR_TILE_FEET]),
        "screen_id": int(ram[ADDR_SCREEN_ID]),
        "start_map": int(ram[ADDR_START_MAP]),
        "end_map": int(ram[ADDR_END_MAP]),
        "len_idx": int(ram[ADDR_LEFT_EN_IDX]),
        "ren_idx": int(ram[ADDR_RIGHT_EN_IDX]),
        "fallen": is_fallen(ram),
    }

def run_probe(state: str, max_frames: int, mode: str, out: Path):
    env = make_env(GAME, state, GAME_DIR)
    obs, _ = env.reset()
    ram = env.get_ram()
    policy = AirManPolicy(target_camera_screen=6, start="screen2")

    timeline = []
    type_events = []
    prev_types = set()
    ll_seen = []
    max_prog = 0
    min_sy_at_scr4 = 255
    first_scr4 = None

    # initial snap
    objs, types = snap_objects(ram)
    m0 = meta(ram)
    m0["objs"] = objs
    m0["types"] = sorted(types)
    m0["eflags"] = enemies_flag_dump(ram)
    timeline.append({"f": 0, **m0})
    type_events.append({"f": 0, "types": sorted(types), "prog": m0["prog"]})
    prev_types = set(types)

    for f in range(1, max_frames + 1):
        if mode == "policy":
            fa = policy.tick(
                frame=f,
                health=int(env.get_ram()[ADDR_HEALTH]),
                camera_x_screen=int(env.get_ram()[ADDR_CAMERA_X_SCREEN]),
                fallen=is_fallen(env.get_ram()),
            )
            action = fa.action
        elif mode == "edge_jump":
            # From AirFan: walk right to edge then long jump, hold RIGHT+A
            ram = env.get_ram()
            sx = player_screen_x(ram)
            feet = int(ram[ADDR_TILE_FEET])
            prog = camera_progress_x(ram)
            if prog < 970 and feet == 1:
                # walk right
                action = nes_action("RIGHT")
            elif prog < 984 and feet == 1:
                # approach edge, prep rising A
                if f % 40 < 2:
                    action = nes_action("RIGHT")  # release A
                else:
                    action = nes_action("RIGHT", "A")
            else:
                # air: RIGHT + A hold for max height, shoot occasionally
                if f % 20 < 2:
                    action = nes_action("RIGHT", "A", "B")
                else:
                    action = nes_action("RIGHT", "A")
        elif mode == "high_camp":
            # walk to right edge, jump repeatedly trying to stay high as cam advances
            ram = env.get_ram()
            feet = int(ram[ADDR_TILE_FEET])
            prog = camera_progress_x(ram)
            sy = player_screen_y(ram)
            if feet == 1 and prog < 978:
                action = nes_action("RIGHT")
            elif feet == 1:
                # rising edge jump
                if f % 2 == 0:
                    action = nes_action("RIGHT")
                else:
                    action = nes_action("RIGHT", "A")
            else:
                # in air prefer A held for height; if sy high enough release to float?
                action = nes_action("RIGHT", "A")
        else:
            action = nes_idle_action()

        obs, _, term, trunc, _ = env.step(action)
        ram = env.get_ram()
        m = meta(ram)
        objs, types = snap_objects(ram)
        max_prog = max(max_prog, m["prog"])
        if m["cam_scr"] >= 4:
            if first_scr4 is None:
                first_scr4 = f
            min_sy_at_scr4 = min(min_sy_at_scr4, m["sy"])

        new_types = types - prev_types
        if types != prev_types:
            type_events.append({
                "f": f,
                "types": sorted(types),
                "new": sorted(new_types),
                "prog": m["prog"],
                "sy": m["sy"],
                "cam_scr": m["cam_scr"],
            })
            prev_types = set(types)

        ll_objs = [o for o in objs if o["t"] in LL_TYPES]
        if ll_objs:
            ll_seen.append({"f": f, **m, "ll": ll_objs})

        if f % 20 == 0 or ll_objs or m["fallen"] or m["hp"] <= 0:
            timeline.append({
                "f": f,
                **m,
                "objs": objs,
                "types": sorted(types),
                "eflags": enemies_flag_dump(ram) if (f % 40 == 0 or ll_objs) else None,
            })

        if m["fallen"] or m["hp"] <= 0 or term or trunc:
            # final
            timeline.append({"f": f, **m, "objs": objs, "types": sorted(types),
                             "eflags": enemies_flag_dump(ram)})
            try:
                save_rgb_png(obs, out / f"death_{mode}_p{m['prog']}.png")
            except Exception:
                pass
            break

    env.close()
    report = {
        "state": state,
        "mode": mode,
        "max_prog": max_prog,
        "first_scr4_f": first_scr4,
        "min_sy_at_scr4": min_sy_at_scr4 if first_scr4 else None,
        "ll_events": len(ll_seen),
        "ll_seen_sample": ll_seen[:5],
        "type_events": type_events[:40],
        "all_types_seen": sorted({t for e in type_events for t in e["types"]}),
        "timeline_tail": timeline[-8:],
        "timeline_head": timeline[:3],
        "decode_expected": {
            "LL_type": "0x3E objects_kaminari_goro",
            "first_LL": "mapset=4 x=0xC0 y=0x20",
            "goblin": "0x40/0x41",
            "pipi": "0x37",
        },
    }
    write_json_report(out / f"probe_{mode}.json", report)
    return report

def main():
    configure_headless()
    out = RECORDINGS_DIR / "air_post4_fpd6"
    out.mkdir(parents=True, exist_ok=True)

    reports = {}
    for state, mode, frames in [
        ("AirFanPlatform", "edge_jump", 200),
        ("AirFanPlatform", "high_camp", 250),
        ("AirScreen2", "policy", 700),
        ("AirFanPlatform", "policy", 400),
    ]:
        print(f"=== {state} / {mode} ===", flush=True)
        r = run_probe(state, frames, mode, out)
        reports[f"{state}_{mode}"] = {
            "max_prog": r["max_prog"],
            "ll_events": r["ll_events"],
            "all_types": r["all_types_seen"],
            "min_sy_scr4": r["min_sy_at_scr4"],
            "first_scr4": r["first_scr4_f"],
            "type_events_n": len(r["type_events"]),
            "head_types": r["timeline_head"][0]["types"] if r["timeline_head"] else None,
            "tail": r["timeline_tail"][-1] if r["timeline_tail"] else None,
        }
        print(json.dumps(reports[f"{state}_{mode}"], indent=2), flush=True)

    write_json_report(out / "summary.json", reports)
    print("DONE", out)

if __name__ == "__main__":
    main()
