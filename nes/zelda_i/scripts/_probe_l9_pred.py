
"""Retarget probe: 0x62 y-first sides + 0x51/0x53 into 0x52."""
from __future__ import annotations

from retro_harness.env import make_env, reset_obs
from retro_harness.nes import nes_action, nes_idle_action
from retro_harness.segment_runner import configure_headless, save_rgb_png, write_json_report
from zelda_i.combat import in_sword_hitbox
from zelda_i.dungeon_ids import KEESE_OBJECT_TYPE, object_name
from zelda_i.dungeon_trace import compact_snapshot
from zelda_i.level9_ganon import LEVEL9
from zelda_i.level9_patra import PATRA_EYE_COUNT, final_patra_live, patra_eyes
from zelda_i.level9_room62 import (
    LOADER_CANDIDATES,
    ROOM_LEVEL9_62,
    door_bits,
    in_room_62,
    room62_object_summary,
)
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
from zelda_i.scripts.run_level9_ganon import (
    FIXTURE_SOURCE,
    FULL_LOADOUT,
    _assign,
    _idle,
    _step,
)
from zelda_i.scripts.run_level9_room62 import SETTLE_IDLE_FRAMES, _apply_loader, _hold_until_room62


def live_keese(snap):
    return [
        obj
        for obj in snap.objects
        if 1 <= obj.slot <= 12 and obj.type_id == KEESE_OBJECT_TYPE
    ]


def chase_sword(snap, cooldown):
    enemies = live_keese(snap)
    if not enemies:
        return nes_idle_action(), max(0, cooldown - 1)
    if cooldown > 0:
        return nes_idle_action(), cooldown - 1
    target = min(
        enemies,
        key=lambda o: abs(int(o.x) - snap.link_x) + abs(int(o.y) - snap.link_y),
    )
    for direction in ("UP", "DOWN", "LEFT", "RIGHT"):
        if in_sword_hitbox(
            snap.link_x, snap.link_y, direction, target.x, target.y,
            reach=24, half_width=16,
        ):
            return nes_action(direction, "A"), 10
    dx = int(target.x) - int(snap.link_x)
    dy = int(target.y) - int(snap.link_y)
    if abs(dx) >= abs(dy):
        return nes_action("RIGHT" if dx > 0 else "LEFT"), 0
    return nes_action("DOWN" if dy > 0 else "UP"), 0


def walk_waypoints(env, total, points, frames_each=400):
    obs = None
    for x, y in points:
        for _ in range(frames_each):
            snap = read_snapshot(env.get_ram())
            dx = int(x) - int(snap.link_x)
            dy = int(y) - int(snap.link_y)
            if abs(dx) <= 3 and abs(dy) <= 3:
                break
            # y-first so we hit dungeon side-door bands
            if abs(dy) > 3:
                action = nes_action("DOWN" if dy > 0 else "UP")
            else:
                action = nes_action("RIGHT" if dx > 0 else "LEFT")
            obs = _step(env, action, assist=None, total=total)
    return obs, read_snapshot(env.get_ram())


def hold(env, total, direction, frames=350):
    obs = None
    start = read_snapshot(env.get_ram()).screen
    for _ in range(frames):
        snap = read_snapshot(env.get_ram())
        if snap.screen != start and not snap.transitioning:
            return obs, snap
        obs = _step(env, nes_action(direction), assist=None, total=total)
    return obs, read_snapshot(env.get_ram())


def info(snap):
    return {
        "room": int(snap.screen),
        "mode": int(snap.mode),
        "link": {"x": snap.link_x, "y": snap.link_y},
        "doors": door_bits(snap.cur_opened_doors),
        "mask": door_bits(snap.open_doorway_mask),
        "objects": room62_object_summary(snap),
        "final_patra_live": bool(final_patra_live(snap)),
        "patra_eyes": len(patra_eyes(snap)),
        "room_item_id": int(snap.room_item_id),
        "room_all_dead": int(snap.room_all_dead),
    }


def materialize(env, total, from_room, next_room, direction, link_x=0x78, link_y=0x58):
    reset_obs(env)
    for _, address, value in FULL_LOADOUT:
        _assign(env, address, value)
    for address, value in (
        (ADDR_LEVEL, LEVEL9),
        (ADDR_MODE, PLAY_MODE),
        (ADDR_SCREEN, from_room),
        (ADDR_NEXT_SCREEN, next_room),
        (ADDR_LINK_X, link_x),
        (ADDR_LINK_Y, link_y),
        (ADDR_CUR_OPENED_DOORS, 0x0F),
        (ADDR_OPEN_DOORWAY_MASK, 0x0F),
    ):
        _assign(env, address, value)
    obs = None
    for _ in range(500):
        obs = _step(env, nes_action(direction), assist=None, total=total)
        snap = read_snapshot(env.get_ram())
        if snap.mode == PLAY_MODE and snap.level == LEVEL9 and snap.screen == next_room:
            obs = _idle(env, SETTLE_IDLE_FRAMES, assist=None, total=total)
            return obs, True
    return obs, False


def main():
    configure_headless()
    RECORDINGS_DIR.mkdir(parents=True, exist_ok=True)
    report = {"ok": True, "room62": [], "candidates": []}

    env = make_env(GAME, FIXTURE_SOURCE, GAME_DIR, render_mode="rgb_array")
    try:
        # 0x62: y-first to south band, then sides; also center collect + north
        for label, waypoints, direction in (
            ("left_via_south", ((120, 189), (32, 189)), "LEFT"),
            ("right_via_south", ((120, 189), (208, 189)), "RIGHT"),
            ("center_then_north", ((120, 141),), "UP"),
            ("south_then_north", ((120, 189), (120, 93)), "UP"),
        ):
            env.close()
            env = make_env(GAME, FIXTURE_SOURCE, GAME_DIR, render_mode="rgb_array")
            total = [0]
            reset_obs(env)
            _apply_loader(env, LOADER_CANDIDATES[0])
            _, loaded = _hold_until_room62(env, LOADER_CANDIDATES[0], total=total)
            if not loaded:
                report["room62"].append({"label": label, "error": "settle failed"})
                continue
            _idle(env, SETTLE_IDLE_FRAMES, assist=None, total=total)
            cooldown = 0
            for _ in range(4000):
                snap = read_snapshot(env.get_ram())
                if in_room_62(snap) and not live_keese(snap):
                    break
                action, cooldown = chase_sword(snap, cooldown)
                _step(env, action, assist=None, total=total)
            _idle(env, 25, assist=None, total=total)
            obs, at = walk_waypoints(env, total, waypoints)
            obs, snap = hold(env, total, direction, 300)
            row = {"label": label, "after_walk": info(at), "after_hold": info(snap)}
            report["room62"].append(row)
            save_rgb_png(obs, RECORDINGS_DIR / f"l9_room62_{label}.png")
            print(
                "62", label, "->", hex(snap.screen),
                "xy", snap.link_x, snap.link_y,
                "patra", final_patra_live(snap),
            )

        # Candidates 0x51 and 0x53
        for name, from_room, next_room, load_dir, walk_dir, walk_points, lx, ly in (
            ("room51_from52_left", 0x52, 0x51, "LEFT", "RIGHT", ((208, 189),), 0x20, 0xBD),
            ("room53_from52_right", 0x52, 0x53, "RIGHT", "LEFT", ((32, 189),), 0xD0, 0xBD),
            ("room51_from61_up", 0x61, 0x51, "UP", "RIGHT", ((208, 189),), 0x78, 0x58),
            ("room53_from63_up", 0x63, 0x53, "UP", "LEFT", ((32, 189),), 0x78, 0x58),
        ):
            env.close()
            env = make_env(GAME, FIXTURE_SOURCE, GAME_DIR, render_mode="rgb_array")
            total = [0]
            obs, loaded = materialize(env, total, from_room, next_room, load_dir, lx, ly)
            row = {"name": name, "loaded": loaded}
            if loaded:
                snap = read_snapshot(env.get_ram())
                row["settled"] = info(snap)
                save_rgb_png(obs, RECORDINGS_DIR / f"l9_{name}_settle.png")
                walk_waypoints(env, total, walk_points)
                obs, snap = hold(env, total, walk_dir, 350)
                row["after_walk"] = info(snap)
                save_rgb_png(obs, RECORDINGS_DIR / f"l9_{name}_walk.png")
                print(
                    name, "settled", hex(row["settled"]["room"]),
                    "objs", [o["type_name"] for o in row["settled"]["objects"]],
                    "walk->", hex(snap.screen),
                    "patra", final_patra_live(snap),
                    "eyes", len(patra_eyes(snap)),
                )
            else:
                print(name, "LOAD FAIL")
            report["candidates"].append(row)
    finally:
        env.close()

    write_json_report(RECORDINGS_DIR / "l9_pred_retarget_probe.json", report)
    return report


if __name__ == "__main__":
    main()
