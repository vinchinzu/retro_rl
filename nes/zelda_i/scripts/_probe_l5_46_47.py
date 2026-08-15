"""Safer 0x46 push (no south walk-out) + east wall shot; dump 0x47 from 0x57 UP."""
from __future__ import annotations

from retro_harness.env import make_env, reset_obs
from retro_harness.nes import nes_action, nes_idle_action
from retro_harness.segment_runner import configure_headless, save_rgb_png, write_json_report
from zelda_i.assist import UnlimitedHealthAssist
from zelda_i.door_graph.core import DoorDir, dirs_from_mask
from zelda_i.dungeon_ops import exit_door, goto, idle
from zelda_i.dungeon_trace import compact_snapshot
from zelda_i.level9_stairs import BLOCK_STAIRS_X, BLOCK_STAIRS_Y, stair_transition_modes
from zelda_i.paths import GAME, GAME_DIR, RECORDINGS_DIR
from zelda_i.ram import ADDR_CANDLE, ADDR_COLLIDING_TILE, ADDR_MAP, ADDR_WHISTLE, PLAY_MODE, read_snapshot, read_u8

SAFE_46 = (
    ((136, 144), "LEFT"),
    ((128, 144), "LEFT"),
    ((104, 144), "RIGHT"),
    ((96, 144), "RIGHT"),
    ((80, 141), "RIGHT"),
    ((160, 141), "LEFT"),
    ((120, 125), "UP"),
    ((120, 157), "UP"),
    ((80, 125), "RIGHT"),
    ((160, 125), "LEFT"),
    ((80, 157), "RIGHT"),
    ((160, 157), "LEFT"),
)


def decode(mask: int) -> dict:
    v = int(mask) & 0x0F
    return {
        "raw": v,
        "raw_hex": f"0x{v:02x}",
        "east": bool(v & DoorDir.RIGHT),
        "west": bool(v & DoorDir.LEFT),
        "south": bool(v & DoorDir.DOWN),
        "north": bool(v & DoorDir.UP),
        "open": sorted(d.name for d in dirs_from_mask(v)),
    }


def dump(env) -> dict:
    ram = env.get_ram()
    snap = read_snapshot(ram)
    c = compact_snapshot(snap)
    c["doors"] = decode(snap.cur_opened_doors)
    c["doorway_mask"] = decode(snap.open_doorway_mask)
    c["room_hex"] = f"0x{snap.screen:02x}"
    c["whistle_0x065C"] = int(read_u8(ram, ADDR_WHISTLE))
    c["candle_0x065B"] = int(read_u8(ram, ADDR_CANDLE))
    c["map_0x0668"] = int(read_u8(ram, ADDR_MAP))
    c["colliding_tile"] = int(read_u8(ram, ADDR_COLLIDING_TILE))
    c["objects"] = [
        {
            "slot": obj.slot,
            "type_id": obj.type_id,
            "type_hex": f"0x{obj.type_id:02x}",
            "x": obj.x,
            "y": obj.y,
            "hp": obj.hp,
        }
        for obj in snap.objects
        if 1 <= obj.slot <= 12 and obj.type_id not in (0, 0xFF)
    ]
    return c


def open_env(state: str):
    env = make_env(GAME, state, GAME_DIR, render_mode="rgb_array")
    assist = UnlimitedHealthAssist(enabled=True)
    obs, _ = reset_obs(env)
    obs, *_ = env.step(nes_idle_action())
    assist.apply_env(env, frame=0)
    return env, assist, obs


def step(env, assist, total, action):
    obs, *_ = env.step(action)
    total[0] += 1
    assist.apply_env(env, frame=total[0])
    return obs


def session_46() -> dict:
    configure_headless()
    env, assist, obs = open_env("Level5Entered46")
    total = [1]
    try:
        idle(env, assist, total, 24)
        start = dump(env)
        goto(env, assist, total, 120, 141, tol=3, max_f=400)
        idle(env, assist, total, 60)
        after_map = dump(env)
        obs, *_ = env.step(nes_idle_action())
        save_rgb_png(obs, RECORDINGS_DIR / "l5_46_after_map.png")

        opened = None
        tried = []
        room = 0x46
        for stand, direction in SAFE_46:
            if read_snapshot(env.get_ram()).screen != room:
                break
            sx, sy = stand
            goto(env, assist, total, sx, sy, tol=3, max_f=350)
            if read_snapshot(env.get_ram()).screen != room:
                break
            doors0 = read_snapshot(env.get_ram()).cur_opened_doors
            mask0 = read_snapshot(env.get_ram()).open_doorway_mask
            x0 = read_snapshot(env.get_ram()).link_x
            for i in range(70):
                step(env, assist, total, nes_action(direction))
                snap = read_snapshot(env.get_ram())
                if snap.screen != room or stair_transition_modes(snap.mode):
                    opened = {"kind": "left_room", "stand": [sx, sy], "direction": direction, "dump": dump(env)}
                    break
                if snap.cur_opened_doors != doors0 or snap.open_doorway_mask != mask0:
                    opened = {"kind": "door_bits", "stand": [sx, sy], "direction": direction, "dump": dump(env)}
                    break
                # abort if we slide into the south mouth
                if snap.link_y >= 200:
                    break
                if direction == "LEFT" and snap.link_x <= 24:
                    break
                if direction == "RIGHT" and snap.link_x >= 216:
                    break
            tried.append({"stand": [sx, sy], "direction": direction, "xy": [read_snapshot(env.get_ram()).link_x, read_snapshot(env.get_ram()).link_y]})
            if opened:
                break
        after = dump(env)
        obs, *_ = env.step(nes_idle_action())
        save_rgb_png(obs, RECORDINGS_DIR / "l5_46_after_push.png")

        # stair spawn tile
        if read_snapshot(env.get_ram()).screen == room:
            goto(env, assist, total, BLOCK_STAIRS_X, BLOCK_STAIRS_Y, tol=3, max_f=300)
            idle(env, assist, total, 16)
        stair = dump(env)
        obs, *_ = env.step(nes_idle_action())
        save_rgb_png(obs, RECORDINGS_DIR / "l5_46_stair_check.png")

        # east wall close-up (ROM E=open; bomb only if cracked)
        east = None
        if read_snapshot(env.get_ram()).screen == room:
            goto(env, assist, total, 208, 141, tol=3, max_f=400)
            for _ in range(8):
                step(env, assist, total, nes_action("RIGHT"))
            idle(env, assist, total, 6)
            east = dump(env)
            obs, *_ = env.step(nes_idle_action())
            save_rgb_png(obs, RECORDINGS_DIR / "l5_46_wall_east.png")

        report = {
            "from_state": "Level5Entered46",
            "pokes": False,
            "status_claim": None,
            "start": start,
            "after_map": after_map,
            "push_opened": opened,
            "tried_n": len(tried),
            "after": after,
            "stair": stair,
            "east": east,
            "bomb": {
                "tried": False,
                "reason": "ROM E=open not bomb; east wall shot for crack check. No candle (dark room) — do not invent a bomb.",
            },
            "whistle_0x065C": after.get("whistle_0x065C"),
        }
        write_json_report(RECORDINGS_DIR / "l5_46_blocks.json", report)
        print("46 opened", bool(opened), "map", after_map.get("map_0x0668"), "room", after.get("room_hex"), flush=True)
        return report
    finally:
        env.close()


def session_47() -> dict:
    configure_headless()
    env, assist, obs = open_env("Level5Cleared57")
    total = [1]
    try:
        idle(env, assist, total, 24)
        start = dump(env)
        hop = exit_door(env, assist, total, "UP")
        idle(env, assist, total, 50)
        dest = dump(env)
        obs, *_ = env.step(nes_idle_action())
        save_rgb_png(obs, RECORDINGS_DIR / "l5_47_from57.png")
        report = {
            "from_state": "Level5Cleared57",
            "pokes": False,
            "status_claim": None,
            "hop": {"changed": hop.get("changed_room"), "result": hop.get("result"), "after": hop.get("after")},
            "start57": start,
            "dest": dest,
            "whistle_0x065C": dest.get("whistle_0x065C"),
        }
        write_json_report(RECORDINGS_DIR / "l5_47_from57.json", report)
        print("47", dest.get("room_hex"), dest.get("doors"), dest.get("doorway_mask"), dest.get("room_item_id"), dest.get("objects"), flush=True)
        return report
    finally:
        env.close()


if __name__ == "__main__":
    a = session_46()
    b = session_47()
    print("WHISTLE", b.get("whistle_0x065C"), flush=True)
