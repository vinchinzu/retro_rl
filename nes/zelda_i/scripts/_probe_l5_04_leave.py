"""Confirm Level5Whistle 0x065C=1, raw-walk 0x04 cellar exit to 0x05. No pokes."""
from __future__ import annotations

from retro_harness.env import make_env, reset_obs, state_path, write_state_bytes
from retro_harness.nes import nes_action, nes_idle_action
from retro_harness.segment_runner import configure_headless, save_rgb_png, write_json_report
from zelda_i.assist import UnlimitedHealthAssist
from zelda_i.dungeon_ops import idle
from zelda_i.dungeon_trace import write_state_provenance
from zelda_i.paths import GAME, GAME_DIR, RECORDINGS_DIR
from zelda_i.ram import (
    ADDR_CANDLE,
    ADDR_SELECTED_ITEM,
    ADDR_TRIFORCE,
    ADDR_WHISTLE,
    PLAY_MODE,
    read_snapshot,
    read_u8,
)

STATE = "Level5Whistle"


def step(env, assist, total, a):
    env.step(a)
    total[0] += 1
    assist.apply_env(env, frame=total[0])


def dump(env):
    s = read_snapshot(env.get_ram())
    ram = env.get_ram()
    return {
        "L": s.level,
        "sc": f"0x{s.screen:02x}",
        "next": f"0x{s.next_screen:02x}",
        "mode": s.mode,
        "sub": int(s.submode),
        "xy": [s.link_x, s.link_y],
        "tile": int(s.colliding_tile),
        "whistle": int(read_u8(ram, ADDR_WHISTLE)),
        "selected": int(read_u8(ram, ADDR_SELECTED_ITEM)),
        "candle": int(read_u8(ram, ADDR_CANDLE)),
        "tf": int(read_u8(ram, ADDR_TRIFORCE)),
        "tf_l5": bool(int(read_u8(ram, ADDR_TRIFORCE)) & 0x10),
        "keys": int(s.keys),
        "bombs": int(s.bombs),
        "health": int(s.health),
        "item": int(s.room_item_id),
        "doors": int(s.cur_opened_doors),
        "mask": int(s.open_doorway_mask),
    }


def raw_axis(env, assist, total, axis, tgt, max_f=500):
    last = None
    stall = 0
    for _ in range(max_f):
        s = read_snapshot(env.get_ram())
        if s.mode == PLAY_MODE and s.screen != 0x04:
            return True, [s.link_x, s.link_y]
        if axis == "x":
            if abs(s.link_x - tgt) <= 1:
                return True, [s.link_x, s.link_y]
            a = nes_action("RIGHT" if s.link_x < tgt else "LEFT")
        else:
            if abs(s.link_y - tgt) <= 1:
                return True, [s.link_x, s.link_y]
            a = nes_action("DOWN" if s.link_y < tgt else "UP")
        step(env, assist, total, a)
        s2 = read_snapshot(env.get_ram())
        pos = (s2.link_x, s2.link_y)
        if pos == last:
            stall += 1
            if stall >= 35:
                return False, [s2.link_x, s2.link_y]
        else:
            stall = 0
        last = pos
    s = read_snapshot(env.get_ram())
    return False, [s.link_x, s.link_y]


def climb(env, assist, total, direction, frames=280):
    room0 = read_snapshot(env.get_ram()).screen
    mode0 = read_snapshot(env.get_ram()).mode
    for _ in range(frames):
        s = read_snapshot(env.get_ram())
        if s.mode == PLAY_MODE and (s.screen != room0 or mode0 != PLAY_MODE):
            return True
        if s.mode == PLAY_MODE and s.screen != 0x04:
            return True
        step(env, assist, total, nes_action(direction))
    idle(env, assist, total, 16)
    s = read_snapshot(env.get_ram())
    return s.mode == PLAY_MODE and s.screen != 0x04


def main():
    configure_headless()
    RECORDINGS_DIR.mkdir(parents=True, exist_ok=True)
    env = None
    total = [1]
    log = []
    try:
        env = make_env(GAME, STATE, GAME_DIR, render_mode="rgb_array")
        assist = UnlimitedHealthAssist(enabled=True)
        reset_obs(env)
        env.step(nes_idle_action())
        assist.apply_env(env, frame=0)
        idle(env, assist, total, 20)
        start = dump(env)
        log.append({"tag": "start", **start})
        print("START", start, flush=True)
        save_rgb_png(env.step(nes_idle_action())[0], RECORDINGS_DIR / "l5_whistle_confirm.png")
        if start["whistle"] != 1:
            write_json_report(
                RECORDINGS_DIR / "l5_04_leave.json",
                {"ok": False, "reason": "whistle_not_1", "start": start, "pokes": False, "status_claim": None},
            )
            print("WHISTLE_FAIL", start, flush=True)
            return

        # Fanfare settle: keep tapping LEFT until x changes or 8s of idle.
        moved = False
        for n in range(8):
            idle(env, assist, total, 40)
            x0 = read_snapshot(env.get_ram()).link_x
            for _ in range(16):
                step(env, assist, total, nes_action("LEFT"))
            x1 = read_snapshot(env.get_ram()).link_x
            rec = {"tag": f"fanfare_{n}", "x0": x0, "x1": x1, **dump(env)}
            log.append(rec)
            print("FANFARE", rec, flush=True)
            if x1 != x0:
                moved = True
                break
        print("MOVED", moved, dump(env), flush=True)

        # Platform y=141, short ladder x=176, floor y=189, mouths x=48 / x=192.
        plans = [
            (("x", 176), ("y", 189), ("x", 48), "UP"),
            (("x", 176), ("y", 189), ("x", 192), "UP"),
            (("x", 176), ("y", 189), ("x", 120), "UP"),
            (("y", 189), ("x", 48), "UP"),
            (("y", 189), ("x", 192), "UP"),
            (("x", 176), ("y", 93), "UP"),
            (("y", 93), ("x", 48), "UP"),
            (("y", 93), ("x", 192), "UP"),
        ]
        left = False
        for i, plan in enumerate(plans):
            s = read_snapshot(env.get_ram())
            if s.mode == PLAY_MODE and s.screen != 0x04:
                left = True
                break
            print("PLAN", i, plan, dump(env), flush=True)
            for item in plan:
                if isinstance(item, tuple):
                    ok, xy = raw_axis(env, assist, total, item[0], item[1])
                    rec = {"tag": f"plan{i}_{item[0]}{item[1]}", "ok": ok, "xy": xy, **dump(env)}
                    log.append(rec)
                    print("WALK", rec, flush=True)
                    s = read_snapshot(env.get_ram())
                    if s.mode == PLAY_MODE and s.screen != 0x04:
                        left = True
                        break
                else:
                    climbed = climb(env, assist, total, item, frames=320)
                    rec = {"tag": f"plan{i}_climb_{item}", "climbed": climbed, **dump(env)}
                    log.append(rec)
                    print("CLIMB", rec, flush=True)
                    if climbed:
                        left = True
                        break
            if left:
                break

        idle(env, assist, total, 24)
        for _ in range(200):
            s = read_snapshot(env.get_ram())
            if s.mode == PLAY_MODE and not s.transitioning:
                break
            step(env, assist, total, nes_idle_action())
        idle(env, assist, total, 12)
        final = dump(env)
        log.append({"tag": "final", **final})
        print("FINAL", final, flush=True)
        save_rgb_png(env.step(nes_idle_action())[0], RECORDINGS_DIR / "l5_04_leave.png")

        ok = final["whistle"] == 1 and (
            (final["mode"] == PLAY_MODE and final["sc"] != "0x04")
            or (final["sc"] == "0x05")
        )
        if ok:
            write_state_bytes(state_path(GAME_DIR, GAME, "Level5Whistle05"), env.em.get_state())
            write_state_provenance(
                state_path(GAME_DIR, GAME, "Level5Whistle05"),
                source_state_path=GAME_DIR / "custom_integrations" / GAME / "Level5Whistle.state",
                request={
                    "segment": "Level5Whistle05",
                    "predecessor_entry": True,
                    "start_state": STATE,
                    "via": "0x04 cellar raw-walk exit",
                    "key_poke": False,
                    "door_poke": False,
                    "bomb_count_poke": False,
                    "selected_item_poke": False,
                },
                selected_trial={
                    "success": True,
                    "room": int(read_snapshot(env.get_ram()).screen),
                    "whistle_0x065C": 1,
                    "mode": final["mode"],
                },
                natural_entry=False,
            )
            print("SAVED Level5Whistle05", final, flush=True)

        body = {
            "ok": ok,
            "pokes": False,
            "status_claim": None,
            "whistle_0x065C": final["whistle"],
            "start": start,
            "final": final,
            "log": log,
        }
        write_json_report(RECORDINGS_DIR / "l5_04_leave.json", body)
        print("OK", ok, "WHISTLE", final["whistle"], "ROOM", final["sc"], "MODE", final["mode"])
    finally:
        if env is not None:
            env.close()


if __name__ == "__main__":
    main()
