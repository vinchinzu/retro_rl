
"""Find live neighbors of 0x52 and retry 0x62 side doors at y=189."""
from __future__ import annotations

from retro_harness.env import make_env, reset_obs
from retro_harness.nes import nes_action, nes_idle_action
from retro_harness.segment_runner import configure_headless, save_rgb_png, write_json_report
from zelda_i.combat import in_sword_hitbox
from zelda_i.dungeon_ids import KEESE_OBJECT_TYPE
from zelda_i.dungeon_trace import compact_snapshot
from zelda_i.level9_patra import final_patra_live, patra_eyes
from zelda_i.level9_room62 import (
    LOADER_CANDIDATES,
    ROOM_LEVEL9_62,
    door_bits,
    in_room_62,
    room62_object_summary,
)
from zelda_i.paths import GAME, GAME_DIR, RECORDINGS_DIR
from zelda_i.ram import read_snapshot
from zelda_i.scripts.run_level9_ganon import FIXTURE_SOURCE, _idle, _step
from zelda_i.scripts.run_level9_room62 import (
    SETTLE_IDLE_FRAMES,
    _apply_loader,
    _hold_until_room62,
)


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


def walk_to(env, total, x, y, frames=500):
    obs = None
    for _ in range(frames):
        snap = read_snapshot(env.get_ram())
        dx = int(x) - int(snap.link_x)
        dy = int(y) - int(snap.link_y)
        if abs(dx) <= 3 and abs(dy) <= 3:
            return obs, True, snap
        if abs(dx) >= abs(dy) and abs(dx) > 3:
            action = nes_action("RIGHT" if dx > 0 else "LEFT")
        else:
            action = nes_action("DOWN" if dy > 0 else "UP")
        obs = _step(env, action, assist=None, total=total)
    return obs, False, read_snapshot(env.get_ram())


def hold(env, total, direction, frames=300):
    obs = None
    start_room = read_snapshot(env.get_ram()).screen
    for _ in range(frames):
        snap = read_snapshot(env.get_ram())
        if snap.screen != start_room and not snap.transitioning:
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
        "compact": compact_snapshot(snap),
    }


def main():
    configure_headless()
    RECORDINGS_DIR.mkdir(parents=True, exist_ok=True)
    report = {"ok": True}

    # --- A: live Patra 0x52 neighbors ---
    env = make_env(GAME, "Level9FinalPatraReconFixture", GAME_DIR, render_mode="rgb_array")
    total = [0]
    try:
        obs, _ = reset_obs(env)
        start = read_snapshot(env.get_ram())
        report["patra_start"] = info(start)
        save_rgb_png(obs, RECORDINGS_DIR / "l9_patra_neighbor_start.png")
        neighbors = []
        for direction, target in (
            ("DOWN", (120, 205)),
            ("LEFT", (32, 189)),
            ("RIGHT", (208, 189)),
        ):
            env.close()
            env = make_env(GAME, "Level9FinalPatraReconFixture", GAME_DIR, render_mode="rgb_array")
            total = [0]
            obs, _ = reset_obs(env)
            walk_to(env, total, *target)
            obs, snap = hold(env, total, direction, 350)
            row = {"direction": direction, **info(snap)}
            neighbors.append(row)
            save_rgb_png(obs, RECORDINGS_DIR / f"l9_patra_neighbor_{direction.lower()}.png")
            print(
                "PATRA", direction, "->", hex(snap.screen),
                "patra_live", final_patra_live(snap),
                "xy", snap.link_x, snap.link_y,
            )
        report["patra_neighbors"] = neighbors
    finally:
        env.close()

    # --- B: 0x62 side doors at y=189 ---
    env = make_env(GAME, FIXTURE_SOURCE, GAME_DIR, render_mode="rgb_array")
    total = [0]
    side = []
    try:
        for direction, target in (
            ("LEFT", (32, 189)),
            ("RIGHT", (208, 189)),
            ("LEFT", (32, 141)),
            ("RIGHT", (208, 141)),
        ):
            env.close()
            env = make_env(GAME, FIXTURE_SOURCE, GAME_DIR, render_mode="rgb_array")
            total = [0]
            reset_obs(env)
            _apply_loader(env, LOADER_CANDIDATES[0])
            _, loaded = _hold_until_room62(env, LOADER_CANDIDATES[0], total=total)
            if not loaded:
                side.append({"direction": direction, "error": "settle failed"})
                continue
            _idle(env, SETTLE_IDLE_FRAMES, assist=None, total=total)
            cooldown = 0
            for _ in range(4000):
                snap = read_snapshot(env.get_ram())
                if in_room_62(snap) and not live_keese(snap):
                    break
                action, cooldown = chase_sword(snap, cooldown)
                _step(env, action, assist=None, total=total)
            _idle(env, 20, assist=None, total=total)
            _, reached, at = walk_to(env, total, *target)
            obs, snap = hold(env, total, direction, 350)
            row = {
                "direction": direction,
                "target": target,
                "reached_target": reached,
                "at_target": {"x": at.link_x, "y": at.link_y},
                **info(snap),
            }
            side.append(row)
            tag = f"{direction.lower()}_{target[1]}"
            save_rgb_png(obs, RECORDINGS_DIR / f"l9_room62_side_{tag}.png")
            print(
                "62", direction, "y", target[1], "->", hex(snap.screen),
                "xy", snap.link_x, snap.link_y, "reached", reached,
            )
        report["room62_sides"] = side
    finally:
        env.close()

    write_json_report(RECORDINGS_DIR / "l9_patra_neighbor_probe.json", report)
    return report


if __name__ == "__main__":
    main()
