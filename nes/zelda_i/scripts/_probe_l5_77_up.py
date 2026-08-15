"""One-off live recon: dump L5 0x77 from Level5EastKey, then natural UP only.

No pokes. No bomb/candle. Not a route runner. Not Clean STATUS.
"""
from __future__ import annotations

from zelda_i.door_graph.core import DoorDir, dirs_from_mask
from zelda_i.dungeon import DoorRoute
from zelda_i.dungeon_lab import _drive_exit
from zelda_i.dungeon_trace import compact_snapshot
from zelda_i.paths import GAME, GAME_DIR, RECORDINGS_DIR
from zelda_i.ram import PLAY_MODE, read_snapshot
from retro_harness.env import make_env, reset_obs
from retro_harness.nes import nes_idle_action
from retro_harness.segment_runner import (
    configure_headless,
    save_rgb_png,
    write_json_report,
)

STATE = "Level5EastKey"
ROOM = 0x77


def decode_doors(mask: int) -> dict:
    value = int(mask) & 0x0F
    return {
        "raw": value,
        "raw_hex": f"0x{value:02x}",
        "east": bool(value & DoorDir.RIGHT),
        "west": bool(value & DoorDir.LEFT),
        "south": bool(value & DoorDir.DOWN),
        "north": bool(value & DoorDir.UP),
        "open": sorted(d.name for d in dirs_from_mask(value)),
    }


def dump_live(snap) -> dict:
    compact = compact_snapshot(snap)
    compact["doors"] = decode_doors(snap.cur_opened_doors)
    compact["doorway_mask"] = decode_doors(snap.open_doorway_mask)
    compact["room_hex"] = f"0x{snap.screen:02x}"
    compact["next_room_hex"] = f"0x{snap.next_screen:02x}"
    return compact


def main() -> dict:
    configure_headless()
    RECORDINGS_DIR.mkdir(parents=True, exist_ok=True)
    env = make_env(GAME, STATE, GAME_DIR, render_mode="rgb_array")
    try:
        obs, _ = reset_obs(env)
        for _ in range(20):
            obs, *_ = env.step(nes_idle_action())
        snap = read_snapshot(env.get_ram())
        start = dump_live(snap)
        start_png = RECORDINGS_DIR / "l5_77_recon.png"
        save_rgb_png(obs, start_png)
        state_data = env.em.get_state()
    finally:
        env.close()

    up_png = RECORDINGS_DIR / "l5_77_up.png"
    # Align to north-door band then hold UP. Controller only — no pokes.
    up = _drive_exit(
        state_data,
        spec_room=ROOM,
        route=DoorRoute("UP", ((120, 141), (120, 93))),
        screenshot_path=up_png,
        max_frames=900,
    )
    dest_room = up.get("room")
    opened = bool(up.get("success")) and dest_room is not None and dest_room != ROOM
    report = {
        "ok": True,
        "status_claim": None,
        "from_state": STATE,
        "expected_room": f"0x{ROOM:02x}",
        "pokes": False,
        "bomb_or_candle": False,
        "dump": start,
        "screenshot": str(start_png.resolve()),
        "up_attempt": {
            "direction": "UP",
            "method": "dungeon_lab._drive_exit DoorRoute UP ((120,141),(120,93))",
            "opened": opened,
            "sealed": not opened,
            "dest_room": f"0x{dest_room:02x}" if dest_room is not None else None,
            "dest_pose": {"x": up.get("x"), "y": up.get("y")} if not opened else None,
            "lab": up,
            "screenshot": str(up_png.resolve()),
        },
    }
    if opened:
        report["up_attempt"]["dest_pose"] = None
    write_json_report(RECORDINGS_DIR / "l5_77_recon.json", report)
    return report


if __name__ == "__main__":
    report = main()
    dump = report["dump"]
    up = report["up_attempt"]
    print("ROOM", dump.get("room_hex"), "xy", dump.get("x"), dump.get("y"),
          "keys", dump.get("keys"), "doors", dump.get("doors"))
    print("OBJECTS", dump.get("objects"))
    print("UP opened", up["opened"], "sealed", up["sealed"],
          "dest", up["dest_room"], "lab", up["lab"])
    print("PNG", report["screenshot"])
    print("JSON", RECORDINGS_DIR / "l5_77_recon.json")
