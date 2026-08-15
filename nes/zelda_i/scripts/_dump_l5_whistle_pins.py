"""Dump Level5Whistle* / Level5TF pins. No nav, no pokes."""
from __future__ import annotations

from retro_harness.env import make_env, reset_obs
from retro_harness.nes import nes_idle_action
from retro_harness.segment_runner import configure_headless, write_json_report
from zelda_i.dungeon_ids import object_name
from zelda_i.dungeon_ops import idle
from zelda_i.paths import GAME, GAME_DIR, RECORDINGS_DIR
from zelda_i.ram import ADDR_TRIFORCE, ADDR_WHISTLE, read_snapshot, read_u8

ROOM_NAMES = {
    0x04: "Recorder / Whistle cellar",
    0x05: "six-Darknut",
    0x06: "empty passage east of 0x05",
    0x07: "cellar stairs (to Digdogger side)",
    0x14: "L5 triforce",
    0x24: "Digdogger",
    0x25: "west Pols Voice",
    0x26: "west Gibdos",
    0x27: "mixed Pols/Gibdo/Keese",
    0x37: "Darknuts + compass",
    0x47: "north Gibdos",
    0x55: "west Zols",
    0x56: "north Dodongos",
    0x57: "east Zols",
    0x64: "Blue Darknut stairs",
    0x65: "west Gibdo pocket",
    0x66: "3x Gibdo first key",
    0x76: "L5 entrance",
}

PINS = [
    "Level5Whistle",
    "Level5Whistle05",
    "Level5Whistle06",
    "Level5Whistle07",
    "Level5WhistleFloor",
    "Level5Whistle64",
    "Level5Whistle65",
    "Level5Whistle66",
    "Level5Whistle66Cleared",
    "Level5Whistle56",
    "Level5Whistle57",
    "Level5Whistle47",
    "Level5Whistle37",
    "Level5Whistle27",
    "Level5Whistle26",
    "Level5Whistle25",
    "Level5Whistle24",
    "Level5TF",
]


def dump_one(name: str) -> dict:
    env = make_env(GAME, name, GAME_DIR, render_mode="rgb_array")
    try:
        reset_obs(env)
        env.step(nes_idle_action())
        n = [1]
        idle(env, None, n, 8)
        ram = env.get_ram()
        s = read_snapshot(ram)
        tf = int(read_u8(ram, ADDR_TRIFORCE))
        objs = []
        for o in s.objects:
            if 1 <= o.slot <= 12 and o.type_id not in (0, 0xFF):
                objs.append({
                    "type": o.type_id,
                    "type_hex": f"0x{o.type_id:02x}",
                    "name": object_name(o.type_id),
                    "hp": o.hp,
                    "x": o.x,
                    "y": o.y,
                })
        rec = {
            "state": name,
            "room_name": ROOM_NAMES.get(s.screen, f"room 0x{s.screen:02x}"),
            "screen": f"0x{s.screen:02x}",
            "mode": s.mode,
            "level": s.level,
            "xy": [s.link_x, s.link_y],
            "keys": int(s.keys),
            "bombs": int(s.bombs),
            "whistle_0x065C": int(read_u8(ram, ADDR_WHISTLE)),
            "triforce_0x0671": tf,
            "tf_hex": hex(tf),
            "tf_l5_bit": bool(tf & 0x10),
            "doors": int(s.cur_opened_doors),
            "mask": int(s.open_doorway_mask),
            "item": int(s.room_item_id),
            "all_dead": int(s.room_all_dead),
            "objects": objs,
        }
        print(
            f"PIN {name:24} {rec['screen']} {rec['room_name']:28} "
            f"mode={rec['mode']} xy={rec['xy']} keys={rec['keys']} bombs={rec['bombs']} "
            f"whistle={rec['whistle_0x065C']} tf={rec['tf_hex']} bit10={rec['tf_l5_bit']} "
            f"doors={rec['doors']} mask={rec['mask']}",
            flush=True,
        )
        return rec
    finally:
        env.close()


def main() -> dict:
    configure_headless()
    integ = GAME_DIR / "custom_integrations" / GAME
    listed = sorted(p.stem for p in integ.glob("Level5*.state"))
    pins = []
    for name in PINS:
        state = integ / f"{name}.state"
        if not state.exists():
            pins.append({"state": name, "missing": True})
            print(f"PIN {name:24} MISSING", flush=True)
            continue
        try:
            pins.append(dump_one(name))
        except Exception as e:
            pins.append({"state": name, "error": str(e)})
            print(f"PIN {name:24} ERROR {e}", flush=True)
    body = {
        "listed_Level5_states": listed,
        "pins": pins,
        "pokes": False,
        "status_claim": False,
    }
    path = RECORDINGS_DIR / "l5_whistle_pins_dump.json"
    write_json_report(path, body)
    print("WROTE", path, "n", len(pins), "listed", len(listed), flush=True)
    return body


if __name__ == "__main__":
    main()
