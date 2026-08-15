"""Materialize untouched Level 9 room ``0x62`` and dump probe artifacts."""

from __future__ import annotations

import argparse
from typing import Any

from retro_harness.env import make_env, reset_obs
from retro_harness.nes import nes_action
from retro_harness.segment_runner import (
    configure_headless,
    save_rgb_png,
    write_json_report,
)
from zelda_i.dungeon_trace import compact_snapshot
from zelda_i.level9_ganon import LEVEL9
from zelda_i.paths import GAME, GAME_DIR, RECORDINGS_DIR
from zelda_i.ram import (
    ADDR_CUR_OPENED_DOORS,
    ADDR_LEVEL,
    ADDR_LINK_X,
    ADDR_LINK_Y,
    ADDR_MODE,
    ADDR_NEXT_SCREEN,
    ADDR_OPEN_DOORWAY_MASK,
    ADDR_SCREEN,
    PLAY_MODE,
    read_snapshot,
)
from zelda_i.scripts.run_level9_ganon import FULL_LOADOUT, _assign, _idle, _step

SOURCE_STATE = "Level9EntranceReconFixture"
SOURCE_ROOM = 0x52
TARGET_ROOM = 0x62
SETTLE_FRAMES = 20
MAX_LOAD_FRAMES = 500
DOOR_BITS = (("N", 0x08), ("S", 0x04), ("W", 0x02), ("E", 0x01))


def _in_target_room(snap: Any) -> bool:
    return (
        snap.level == LEVEL9
        and snap.screen == TARGET_ROOM
        and snap.mode == PLAY_MODE
    )


def _live_objects(snap: Any) -> list[dict[str, int]]:
    return [
        {
            "slot": int(obj.slot),
            "type": int(obj.type_id),
            "hp": int(obj.hp),
            "x": int(obj.x),
            "y": int(obj.y),
        }
        for obj in snap.objects
        if 1 <= obj.slot <= 12 and (obj.type_id or obj.hp)
    ]


def _format_door_bits(mask: int) -> str:
    return " ".join(
        f"{name}(0x{bit:02X})={int(bool(mask & bit))}" for name, bit in DOOR_BITS
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tag", default="l9_room62_probe")
    args = parser.parse_args()

    configure_headless()
    RECORDINGS_DIR.mkdir(parents=True, exist_ok=True)
    env = make_env(GAME, SOURCE_STATE, GAME_DIR, render_mode="rgb_array")
    total = [0]
    try:
        obs, _ = reset_obs(env)

        for _, address, value in FULL_LOADOUT:
            _assign(env, address, value)

        # Invert the 0x62 -> 0x52 fixture loader to enter 0x62 from the north.
        for address, value in (
            (ADDR_LEVEL, LEVEL9),
            (ADDR_MODE, PLAY_MODE),
            (ADDR_SCREEN, SOURCE_ROOM),
            (ADDR_NEXT_SCREEN, TARGET_ROOM),
            (ADDR_LINK_X, 0x78),
            (ADDR_LINK_Y, 0xCD),
            (ADDR_CUR_OPENED_DOORS, 0x0F),
            (ADDR_OPEN_DOORWAY_MASK, 0x0F),
        ):
            _assign(env, address, value)

        loaded = False
        for _ in range(MAX_LOAD_FRAMES):
            obs = _step(env, nes_action("DOWN"), assist=None, total=total)
            if _in_target_room(read_snapshot(env.get_ram())):
                loaded = True
                break

        if loaded:
            obs = _idle(env, SETTLE_FRAMES, assist=None, total=total)

        # Read-only boundary: do not modify objects, doors, or room state below.
        ram = env.get_ram()
        snap = read_snapshot(ram)
        snapshot = compact_snapshot(snap)
        live_objects = _live_objects(snap)

        png_path = RECORDINGS_DIR / f"{args.tag}.png"
        ram_path = RECORDINGS_DIR / f"{args.tag}.ram.bin"
        json_path = RECORDINGS_DIR / f"{args.tag}.json"
        save_rgb_png(obs, png_path)
        ram_path.write_bytes(bytes(ram))
        write_json_report(json_path, snapshot)

        print(
            f"ROOM level={snap.level} room=0x{snap.screen:02X} mode={snap.mode} "
            f"frames={total[0]}"
        )
        print("LIVE_OBJECTS", live_objects)
        print(
            f"DOORS cur=0x{snap.cur_opened_doors:02X}",
            _format_door_bits(snap.cur_opened_doors),
        )
        print(
            f"DOORWAY_MASK raw=0x{snap.open_doorway_mask:02X}",
            _format_door_bits(snap.open_doorway_mask),
        )
        print("ARTIFACTS", png_path, ram_path, json_path)
        return 0 if _in_target_room(snap) else 1
    finally:
        env.close()


if __name__ == "__main__":
    raise SystemExit(main())
