
"""One-off door-contract experiment for L9 0x62. Not a route runner."""
from __future__ import annotations

from retro_harness.env import make_env, reset_obs
from retro_harness.nes import nes_action, nes_idle_action
from retro_harness.segment_runner import configure_headless, save_rgb_png, write_json_report
from zelda_i.combat import in_sword_hitbox
from zelda_i.dungeon_ids import KEESE_OBJECT_TYPE
from zelda_i.level9_ganon import B_ITEM_BOMBS
from zelda_i.level9_patra import final_patra_live, patra_eyes
from zelda_i.level9_room62 import (
    LOADER_CANDIDATES,
    ROOM_LEVEL9_62,
    door_bits,
    in_room_62,
    room62_object_summary,
    room62_to_patra_step,
)
from zelda_i.paths import GAME, GAME_DIR, RECORDINGS_DIR
from zelda_i.ram import ADDR_SELECTED_ITEM, read_snapshot
from zelda_i.scripts.run_level9_ganon import FIXTURE_SOURCE, _assign, _idle, _step
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
        return nes_idle_action(), "cleared", max(0, cooldown - 1)
    if cooldown > 0:
        return nes_idle_action(), "cooldown", cooldown - 1
    target = min(
        enemies,
        key=lambda o: abs(int(o.x) - snap.link_x) + abs(int(o.y) - snap.link_y),
    )
    for direction in ("UP", "DOWN", "LEFT", "RIGHT"):
        if in_sword_hitbox(
            snap.link_x,
            snap.link_y,
            direction,
            target.x,
            target.y,
            reach=24,
            half_width=16,
        ):
            return nes_action(direction, "A"), "sword", 10
    dx = int(target.x) - int(snap.link_x)
    dy = int(target.y) - int(snap.link_y)
    if abs(dx) >= abs(dy):
        return nes_action("RIGHT" if dx > 0 else "LEFT"), "chase_x", 0
    return nes_action("DOWN" if dy > 0 else "UP"), "chase_y", 0


def main():
    configure_headless()
    RECORDINGS_DIR.mkdir(parents=True, exist_ok=True)
    env = make_env(GAME, FIXTURE_SOURCE, GAME_DIR, render_mode="rgb_array")
    total = [0]
    report = {"ok": False}
    try:
        obs, _ = reset_obs(env)
        candidate = LOADER_CANDIDATES[0]
        _apply_loader(env, candidate)
        obs, loaded = _hold_until_room62(env, candidate, total=total)
        if not loaded:
            report["error"] = "did not settle 0x62"
            return report
        obs = _idle(env, SETTLE_IDLE_FRAMES, assist=None, total=total)
        start = read_snapshot(env.get_ram())
        report["start"] = {
            "objects": room62_object_summary(start),
            "doors": door_bits(start.cur_opened_doors),
            "mask": door_bits(start.open_doorway_mask),
            "room_all_dead": start.room_all_dead,
        }
        save_rgb_png(obs, RECORDINGS_DIR / "l9_room62_exp_start.png")

        cooldown = 0
        cleared = False
        for _ in range(4000):
            snap = read_snapshot(env.get_ram())
            if in_room_62(snap) and not live_keese(snap):
                cleared = True
                break
            action, reason, cooldown = chase_sword(snap, cooldown)
            obs = _step(env, action, assist=None, total=total)
        obs = _idle(env, 40, assist=None, total=total)
        after_kill = read_snapshot(env.get_ram())
        report["after_kill"] = {
            "cleared": cleared,
            "keese": len(live_keese(after_kill)),
            "doors": door_bits(after_kill.cur_opened_doors),
            "mask": door_bits(after_kill.open_doorway_mask),
            "room_all_dead": after_kill.room_all_dead,
            "room_obj_count": after_kill.room_obj_count,
            "objects": room62_object_summary(after_kill),
            "link": {"x": after_kill.link_x, "y": after_kill.link_y},
            "frames": total[0],
        }
        save_rgb_png(obs, RECORDINGS_DIR / "l9_room62_exp_after_kill.png")

        entered = None
        for _ in range(240):
            snap = read_snapshot(env.get_ram())
            if snap.screen != ROOM_LEVEL9_62 and not snap.transitioning:
                entered = {
                    "to_room": snap.screen,
                    "final_patra_live": bool(final_patra_live(snap)),
                    "eyes": len(patra_eyes(snap)),
                    "objects": room62_object_summary(snap),
                    "doors": door_bits(snap.cur_opened_doors),
                }
                break
            frame = room62_to_patra_step(snap)
            obs = _step(env, frame.action, assist=None, total=total)
        end = read_snapshot(env.get_ram())
        report["after_kill_north"] = {
            "entered": entered,
            "end_room": end.screen,
            "link": {"x": end.link_x, "y": end.link_y},
        }
        save_rgb_png(obs, RECORDINGS_DIR / "l9_room62_exp_after_kill_north.png")

        if entered is not None:
            report["ok"] = True
            report["method"] = "kill_clear_north"
            return report

        _assign(env, ADDR_SELECTED_ITEM, B_ITEM_BOMBS)
        bombs_before = int(read_snapshot(env.get_ram()).bombs)
        for _ in range(400):
            snap = read_snapshot(env.get_ram())
            dx = 120 - int(snap.link_x)
            dy = 101 - int(snap.link_y)
            if abs(dx) <= 3 and abs(dy) <= 3:
                break
            if abs(dx) >= abs(dy) and abs(dx) > 3:
                action = nes_action("RIGHT" if dx > 0 else "LEFT")
            else:
                action = nes_action("DOWN" if dy > 0 else "UP")
            obs = _step(env, action, assist=None, total=total)
        for _ in range(6):
            obs = _step(env, nes_action("UP"), assist=None, total=total)
        obs = _step(env, nes_action("UP", "B"), assist=None, total=total)
        for _ in range(8):
            obs = _step(env, nes_action("DOWN"), assist=None, total=total)
        for _ in range(110):
            obs = _step(env, nes_idle_action(), assist=None, total=total)
        after_bomb = read_snapshot(env.get_ram())
        report["after_bomb"] = {
            "bombs_before": bombs_before,
            "bombs_after": after_bomb.bombs,
            "selected_item": int(env.get_ram()[ADDR_SELECTED_ITEM]),
            "doors": door_bits(after_bomb.cur_opened_doors),
            "mask": door_bits(after_bomb.open_doorway_mask),
            "link": {"x": after_bomb.link_x, "y": after_bomb.link_y},
        }
        save_rgb_png(obs, RECORDINGS_DIR / "l9_room62_exp_after_bomb.png")

        entered = None
        for _ in range(300):
            snap = read_snapshot(env.get_ram())
            if snap.screen != ROOM_LEVEL9_62 and not snap.transitioning:
                entered = {
                    "to_room": snap.screen,
                    "final_patra_live": bool(final_patra_live(snap)),
                    "eyes": len(patra_eyes(snap)),
                    "objects": room62_object_summary(snap),
                    "doors": door_bits(snap.cur_opened_doors),
                }
                break
            frame = room62_to_patra_step(snap)
            obs = _step(env, frame.action, assist=None, total=total)
        end = read_snapshot(env.get_ram())
        report["after_bomb_north"] = {
            "entered": entered,
            "end_room": end.screen,
            "link": {"x": end.link_x, "y": end.link_y},
        }
        save_rgb_png(obs, RECORDINGS_DIR / "l9_room62_exp_after_bomb_north.png")
        if entered is not None:
            report["ok"] = True
            report["method"] = "bomb_north"
        else:
            report["error"] = "neither kill-clear nor bomb-north entered another room"
        return report
    finally:
        env.close()


if __name__ == "__main__":
    report = main()
    write_json_report(RECORDINGS_DIR / "l9_room62_door_experiment.json", report)
    print("METHOD", report.get("method"), "OK", report.get("ok"), "ERR", report.get("error"))
    print("AFTER_KILL", report.get("after_kill"))
    print("AFTER_KILL_NORTH", report.get("after_kill_north"))
    print("AFTER_BOMB", report.get("after_bomb"))
    print("AFTER_BOMB_NORTH", report.get("after_bomb_north"))
