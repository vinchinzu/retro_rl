
"""Probe every live exit from settled L9 0x62. Not a route runner."""
from __future__ import annotations

from retro_harness.env import make_env, reset_obs
from retro_harness.nes import nes_action, nes_idle_action
from retro_harness.segment_runner import configure_headless, save_rgb_png, write_json_report
from zelda_i.combat import in_sword_hitbox
from zelda_i.dungeon_ids import KEESE_OBJECT_TYPE, object_name
from zelda_i.dungeon_trace import compact_snapshot
from zelda_i.level9_ganon import B_ITEM_BOMBS
from zelda_i.level9_patra import final_patra_live, patra_eyes
from zelda_i.level9_room62 import (
    LOADER_CANDIDATES,
    ROOM_LEVEL9_62,
    door_bits,
    in_room_62,
    room62_object_summary,
)
from zelda_i.paths import GAME, GAME_DIR, RECORDINGS_DIR
from zelda_i.ram import ADDR_SELECTED_ITEM, read_snapshot
from zelda_i.scripts.run_level9_ganon import FIXTURE_SOURCE, _assign, _idle, _step
from zelda_i.scripts.run_level9_room62 import (
    SETTLE_IDLE_FRAMES,
    _apply_loader,
    _hold_until_room62,
)

DOOR_TARGETS = {
    "RIGHT": (208, 141),
    "LEFT": (32, 141),
    "UP": (120, 93),
    "DOWN": (120, 205),
}


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


def settle_room62(env, total):
    candidate = LOADER_CANDIDATES[0]
    reset_obs(env)
    _apply_loader(env, candidate)
    obs, loaded = _hold_until_room62(env, candidate, total=total)
    if not loaded:
        return None, False
    obs = _idle(env, SETTLE_IDLE_FRAMES, assist=None, total=total)
    return obs, True


def clear_keese(env, total):
    cooldown = 0
    for _ in range(4000):
        snap = read_snapshot(env.get_ram())
        if in_room_62(snap) and not live_keese(snap):
            _idle(env, 30, assist=None, total=total)
            return True
        action, cooldown = chase_sword(snap, cooldown)
        _step(env, action, assist=None, total=total)
    return False


def walk_to(env, total, x, y, frames=400):
    obs = None
    for _ in range(frames):
        snap = read_snapshot(env.get_ram())
        dx = int(x) - int(snap.link_x)
        dy = int(y) - int(snap.link_y)
        if abs(dx) <= 3 and abs(dy) <= 3:
            return obs, True
        if abs(dx) >= abs(dy) and abs(dx) > 3:
            action = nes_action("RIGHT" if dx > 0 else "LEFT")
        else:
            action = nes_action("DOWN" if dy > 0 else "UP")
        obs = _step(env, action, assist=None, total=total)
    return obs, False


def push_dir(env, total, direction, frames):
    obs = None
    for _ in range(frames):
        snap = read_snapshot(env.get_ram())
        if snap.screen != ROOM_LEVEL9_62 and not snap.transitioning:
            return obs, snap
        if snap.transitioning:
            obs = _step(env, nes_action(direction), assist=None, total=total)
            continue
        obs = _step(env, nes_action(direction), assist=None, total=total)
    return obs, read_snapshot(env.get_ram())


def transition_info(snap):
    return {
        "room": int(snap.screen),
        "mode": int(snap.mode),
        "level": int(snap.level),
        "link": {"x": snap.link_x, "y": snap.link_y},
        "doors": door_bits(snap.cur_opened_doors),
        "mask": door_bits(snap.open_doorway_mask),
        "objects": room62_object_summary(snap),
        "final_patra_live": bool(final_patra_live(snap)),
        "patra_eyes": len(patra_eyes(snap)),
        "room_item_id": int(snap.room_item_id),
        "still_62": in_room_62(snap),
    }


def try_exit(env, total, direction, *, clear=True, select_bombs=False, bomb_stand=None):
    obs, loaded = settle_room62(env, total)
    if not loaded:
        return {"ok": False, "error": "settle failed"}
    if clear:
        clear_keese(env, total)
    if select_bombs:
        _assign(env, ADDR_SELECTED_ITEM, B_ITEM_BOMBS)
    if bomb_stand is not None:
        walk_to(env, total, bomb_stand[0], bomb_stand[1])
        for _ in range(6):
            _step(env, nes_action(direction), assist=None, total=total)
        _step(env, nes_action(direction, "B"), assist=None, total=total)
        back = {"UP": "DOWN", "DOWN": "UP", "LEFT": "RIGHT", "RIGHT": "LEFT"}[direction]
        for _ in range(8):
            _step(env, nes_action(back), assist=None, total=total)
        _idle(env, 110, assist=None, total=total)
    tx, ty = DOOR_TARGETS[direction]
    walk_to(env, total, tx, ty)
    obs, snap = push_dir(env, total, direction, 220)
    info = transition_info(snap)
    info["direction"] = direction
    info["cleared"] = clear
    info["bombed"] = bomb_stand
    return info, obs


def main():
    configure_headless()
    RECORDINGS_DIR.mkdir(parents=True, exist_ok=True)
    env = make_env(GAME, FIXTURE_SOURCE, GAME_DIR, render_mode="rgb_array")
    total = [0]
    results = []
    try:
        # 1) walk all 4 dirs after kill-clear
        for direction in ("UP", "DOWN", "LEFT", "RIGHT"):
            info, obs = try_exit(env, total, direction, clear=True)
            save_rgb_png(obs, RECORDINGS_DIR / f"l9_room62_exit_{direction.lower()}.png")
            results.append(info)
            print("EXIT", direction, "room", info.get("room"), "still62", info.get("still_62"),
                  "patra", info.get("final_patra_live"))

        # 2) walk to center item after clear (possible door trigger)
        obs, loaded = settle_room62(env, total)
        clear_keese(env, total)
        walk_to(env, total, 120, 141)
        _idle(env, 40, assist=None, total=total)
        snap = read_snapshot(env.get_ram())
        center = transition_info(snap)
        center["tag"] = "center_after_clear"
        save_rgb_png(obs, RECORDINGS_DIR / "l9_room62_exit_center.png")
        results.append(center)
        print("CENTER", center["room"], "item", center["room_item_id"], "objects", center["objects"])

        # 3) bomb more north stands
        for stand in ((120, 93), (120, 101), (112, 101), (128, 101), (120, 109)):
            info, obs = try_exit(
                env, total, "UP", clear=True, select_bombs=True, bomb_stand=stand
            )
            info["tag"] = f"bomb_north_{stand[0]}_{stand[1]}"
            save_rgb_png(obs, RECORDINGS_DIR / f"l9_room62_bomb_{stand[0]}_{stand[1]}.png")
            results.append(info)
            print("BOMB", stand, "room", info.get("room"), "still62", info.get("still_62"))

        # 4) bomb west (visually open? or bomb recess)
        info, obs = try_exit(
            env, total, "LEFT", clear=True, select_bombs=True, bomb_stand=(48, 141)
        )
        info["tag"] = "bomb_west"
        save_rgb_png(obs, RECORDINGS_DIR / "l9_room62_bomb_west.png")
        results.append(info)
        print("BOMB WEST", info.get("room"), "still62", info.get("still_62"))

        report = {"ok": True, "results": results, "total_frames": total[0]}
        write_json_report(RECORDINGS_DIR / "l9_room62_exit_probe.json", report)
        return report
    finally:
        env.close()


if __name__ == "__main__":
    main()
