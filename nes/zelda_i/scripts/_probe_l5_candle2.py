"""L5 OW: hop 0x0B→shop 0x5E one screen at a time, farm, buy candle.

Reuse OverworldPathController + OverworldToCandleShopController buy.
No pokes. No Digdogger. No Clean STATUS.
"""
from __future__ import annotations

from retro_harness.env import make_env, reset_obs, state_path, write_state_bytes
from retro_harness.nes import nes_action, nes_idle_action
from retro_harness.segment_runner import configure_headless, save_rgb_png, write_json_report
from zelda_i.assist import UnlimitedHealthAssist
from zelda_i.dungeon_ops import idle, push_dir
from zelda_i.dungeon_trace import write_state_provenance
from zelda_i.heart_farm import FARM_SWING_HOLD, FARM_SWING_PERIOD
from zelda_i.level8_overworld import (
    CANDLE_SHOP_PRICE,
    LEVEL8_5C_MAZE_WAYPOINTS,
    OverworldToCandleShopController,
    is_5c_maze_hop,
)
from zelda_i.nav_common import swing_action
from zelda_i.overworld import ScreenHop
from zelda_i.ow_path import OverworldPathController
from zelda_i.paths import GAME, GAME_DIR, RECORDINGS_DIR
from zelda_i.ram import ADDR_CANDLE, ADDR_WHISTLE, PLAY_MODE, read_snapshot, read_u8

STATE = "Level5EntranceFromL4"
RUPEE_TYPES = frozenset({0x5D, 0x60, 0x61, 0x62})
# Tektite 0x10, Octorok 0x03/0x04, Moblin 0x0D/0x0E, Lynel 0x11/0x12, Armos 0x1E
FARM_ENEMIES = frozenset({0x03, 0x04, 0x0D, 0x0E, 0x10, 0x11, 0x12, 0x1E, 0x27})

HOPS_A: tuple[ScreenHop, ...] = (
    ScreenHop(0x1B, "DOWN", align_x=112),
    ScreenHop(0x1C, "RIGHT", align_y=140),
)
HOPS_A_ALT: tuple[ScreenHop, ...] = (
    ScreenHop(0x2B, "DOWN", align_x=112),
)
HOPS_B: tuple[ScreenHop, ...] = (
    ScreenHop(0x2C, "DOWN", align_x=48),
    ScreenHop(0x2B, "LEFT", align_y=85),
    ScreenHop(0x3B, "DOWN", align_x=48),
    ScreenHop(0x3A, "LEFT", align_y=140),
    ScreenHop(0x4A, "DOWN", align_x=112),
)
HOPS_C: tuple[ScreenHop, ...] = (
    ScreenHop(0x5A, "DOWN", align_x=112),
    ScreenHop(0x5B, "RIGHT", y_band_lo=130, y_band_hi=150),
    ScreenHop(0x5C, "RIGHT", y_band_lo=80, y_band_hi=95),
    ScreenHop(0x5D, "RIGHT", y_band_lo=120, y_band_hi=140),
    ScreenHop(0x5E, "RIGHT", y_band_lo=130, y_band_hi=150),
)


def inv(ram):
    snap = read_snapshot(ram)
    return {
        "R": int(snap.rupees),
        "candle": int(read_u8(ram, ADDR_CANDLE)),
        "whistle": int(read_u8(ram, ADDR_WHISTLE)),
        "L": int(snap.level),
        "sc": f"0x{snap.screen:02x}",
        "mode": int(snap.mode),
        "xy": [int(snap.link_x), int(snap.link_y)],
        "bombs": int(snap.bombs),
        "keys": int(snap.keys),
    }


def open_env(state=STATE):
    env = make_env(GAME, state, GAME_DIR, render_mode="rgb_array")
    assist = UnlimitedHealthAssist(enabled=True)
    reset_obs(env)
    env.step(nes_idle_action())
    assist.apply_env(env, frame=0)
    return env, assist


def step(env, assist, total, action):
    env.step(action)
    total[0] += 1
    assist.apply_env(env, frame=total[0])


def shot(env, name):
    obs, *_ = env.step(nes_idle_action())
    p = RECORDINGS_DIR / f"{name}.png"
    save_rgb_png(obs, p)
    return str(p)


def exit_l5(env, assist, total):
    idle(env, assist, total, 16)
    # 0x76 south mouth
    for _ in range(80):
        snap = read_snapshot(env.get_ram())
        if snap.level == 0:
            break
        if abs(snap.link_x - 120) > 3:
            step(env, assist, total, nes_action("LEFT" if snap.link_x > 120 else "RIGHT"))
        else:
            step(env, assist, total, nes_action("DOWN"))
    push_dir(env, assist, total, "DOWN", frames=300)
    for _ in range(240):
        snap = read_snapshot(env.get_ram())
        if snap.level == 0 and snap.mode == PLAY_MODE and not snap.transitioning:
            idle(env, assist, total, 20)
            break
        step(env, assist, total, nes_idle_action())
    return inv(env.get_ram())


def run_hops(env, assist, total, hops, tag, max_f=8000) -> dict:
    if not hops:
        return {"ok": True, "tag": tag, "skipped": True, "final": inv(env.get_ram())}
    nav = OverworldPathController(
        hops=hops,
        maze_waypoints=LEVEL8_5C_MAZE_WAYPOINTS if any(h.target == 0x5D for h in hops) else (),
        maze_hop_pred=is_5c_maze_hop if any(h.target == 0x5D for h in hops) else None,
        require_sword=False,
        max_frames=max_f,
    )
    trail = []
    last = None
    for _ in range(max_f):
        snap = read_snapshot(env.get_ram())
        key = (snap.level, snap.screen, snap.mode)
        if key != last:
            trail.append({"f": total[0], **inv(env.get_ram())})
            print(tag, trail[-1], flush=True)
            last = key
        if nav.success or (hasattr(nav.phase, "name") and nav.phase.name == "FAILED"):
            break
        act = nav.step(snap)
        step(env, assist, total, act.action)
    snap = read_snapshot(env.get_ram())
    ok = snap.level == 0 and snap.screen == hops[-1].target and snap.mode == PLAY_MODE
    rec = {
        "ok": ok or bool(nav.success),
        "tag": tag,
        "trail": trail,
        "nav": nav.report(),
        "final": inv(env.get_ram()),
    }
    print(tag, "DONE", rec["ok"], rec["final"], "notes", rec["nav"].get("notes", [])[-8:], flush=True)
    return rec


def farm(env, assist, total, need=60, budget=14000) -> dict:
    start = int(read_snapshot(env.get_ram()).rupees)
    if start >= need:
        return {"ok": True, "start": start, "end": start}
    farm_sc = read_snapshot(env.get_ram()).screen
    empty = 0
    bounce = "RIGHT"
    for i in range(budget):
        snap = read_snapshot(env.get_ram())
        if snap.rupees >= need:
            break
        if snap.level != 0 or snap.mode == 17:
            step(env, assist, total, nes_action("DOWN"))
            continue
        if snap.transitioning or snap.mode != PLAY_MODE:
            step(env, assist, total, nes_idle_action())
            continue
        if snap.screen != farm_sc:
            empty += 1
            step(env, assist, total, nes_action("LEFT" if bounce == "RIGHT" else "RIGHT"))
            if empty > 120:
                farm_sc = snap.screen
                empty = 0
            continue
        drops = [o for o in snap.objects if 1 <= o.slot <= 12 and o.type_id in RUPEE_TYPES]
        enemies = [
            o
            for o in snap.objects
            if 1 <= o.slot <= 12 and o.type_id in FARM_ENEMIES and 30 < o.y < 220
        ]
        tgt = None
        if drops:
            tgt = min(drops, key=lambda o: abs(o.x - snap.link_x) + abs(o.y - snap.link_y))
            empty = 0
        elif enemies:
            tgt = min(enemies, key=lambda o: abs(o.x - snap.link_x) + abs(o.y - snap.link_y))
            empty = 0
        else:
            empty += 1
        if tgt is not None:
            dx, dy = tgt.x - snap.link_x, tgt.y - snap.link_y
            if abs(dx) >= abs(dy) and abs(dx) > 3:
                d = "RIGHT" if dx > 0 else "LEFT"
            elif abs(dy) > 3:
                d = "DOWN" if dy > 0 else "UP"
            else:
                d = "UP"
            act = swing_action(total[0], d, "farm", period=FARM_SWING_PERIOD, hold=FARM_SWING_HOLD)
            step(env, assist, total, act.action)
        elif empty > 100:
            step(env, assist, total, nes_action(bounce))
            if empty > 180:
                bounce = "LEFT" if bounce == "RIGHT" else "RIGHT"
                empty = 0
                print("FARM_BOUNCE", snap.rupees, hex(snap.screen), flush=True)
        else:
            d = "RIGHT" if (total[0] // 50) % 2 == 0 else "LEFT"
            if abs(snap.link_y - 141) > 10:
                d = "DOWN" if snap.link_y < 141 else "UP"
            act = swing_action(total[0], d, "pat", period=FARM_SWING_PERIOD, hold=FARM_SWING_HOLD)
            step(env, assist, total, act.action)
        if i % 500 == 0:
            print("FARM", snap.rupees, hex(snap.screen), flush=True)
    end = int(read_snapshot(env.get_ram()).rupees)
    print("FARM_DONE", start, "->", end, flush=True)
    return {"ok": end >= need, "start": start, "end": end, "sc": f"0x{read_snapshot(env.get_ram()).screen:02x}"}


def buy_5e(env, assist, total) -> dict:
    nav = OverworldToCandleShopController(
        hops=(ScreenHop(0x5E, "UP", align_x=112),),
        enter_cave=True,
        buy_candle=True,
        max_frames=8000,
    )
    from zelda_i.level8_overworld import CandleShopNavPhase

    nav.phase = CandleShopNavPhase.DOOR
    nav.hop_index = 1
    trail = []
    last = None
    for _ in range(8000):
        ram = env.get_ram()
        snap = read_snapshot(ram)
        nav.candle_value = int(read_u8(ram, ADDR_CANDLE))
        key = (snap.level, snap.screen, snap.mode)
        if key != last:
            trail.append({"f": total[0], **inv(ram)})
            print("BUY", trail[-1], flush=True)
            last = key
        if nav.success or (hasattr(nav.phase, "name") and nav.phase.name == "FAILED"):
            break
        if nav.candle_value:
            nav.success = True
            break
        act = nav.step(snap)
        step(env, assist, total, act.action)
    # fallback geometry
    ram = env.get_ram()
    if int(read_u8(ram, ADDR_CANDLE)) == 0 and read_snapshot(ram).mode == 11:
        for i in range(700):
            snap = read_snapshot(env.get_ram())
            if int(read_u8(env.get_ram(), ADDR_CANDLE)):
                break
            if snap.mode != 11:
                break
            if i < 90 and snap.link_y > 200:
                step(env, assist, total, nes_idle_action())
            elif snap.link_y > 150:
                step(env, assist, total, nes_action("UP"))
            elif snap.link_x < 152:
                step(env, assist, total, nes_action("RIGHT"))
            else:
                step(env, assist, total, nes_action("UP"))
    candle = int(read_u8(env.get_ram(), ADDR_CANDLE))
    return {"ok": candle == 1, "candle": candle, "trail": trail, "final": inv(env.get_ram()), "nav": nav.report()}


def main():
    configure_headless()
    RECORDINGS_DIR.mkdir(parents=True, exist_ok=True)
    env = None
    total = [1]
    report = {
        "ok": False,
        "pokes": False,
        "status_claim": None,
        "commands": [
            "PYTHONPATH=nes:. uv run python nes/zelda_i/scripts/_probe_l5_candle2.py"
        ],
        "from_state": STATE,
    }
    try:
        env, assist = open_env()
        report["exit"] = exit_l5(env, assist, total)
        print("EXIT", report["exit"], flush=True)
        shot(env, "l5_c2_ow")

        a = run_hops(env, assist, total, HOPS_A, "A", max_f=10000)
        report["hops_a"] = a
        shot(env, "l5_c2_a")
        snap = read_snapshot(env.get_ram())
        if snap.screen == 0x1B:
            alt = run_hops(env, assist, total, HOPS_A_ALT, "A_ALT", max_f=8000)
            report["hops_a_alt"] = alt
            shot(env, "l5_c2_a_alt")

        snap = read_snapshot(env.get_ram())
        # continue from wherever we are
        hops_b = HOPS_B
        targets = [h.target for h in hops_b]
        if snap.screen in targets:
            hops_b = hops_b[targets.index(snap.screen) + 1 :]
        elif snap.screen == 0x1C:
            hops_b = HOPS_B
        elif snap.screen == 0x2B:
            hops_b = hops_b[targets.index(0x2B) + 1 :] if 0x2B in targets else hops_b[2:]
        b = run_hops(env, assist, total, hops_b, "B", max_f=16000)
        report["hops_b"] = b
        shot(env, "l5_c2_b")

        snap = read_snapshot(env.get_ram())
        report["farm"] = farm(env, assist, total, need=CANDLE_SHOP_PRICE, budget=16000)
        shot(env, "l5_c2_farm")

        snap = read_snapshot(env.get_ram())
        hops_c = HOPS_C
        targets = [h.target for h in hops_c]
        if snap.screen in targets:
            hops_c = hops_c[targets.index(snap.screen) + 1 :]
        elif snap.screen == 0x4A:
            hops_c = HOPS_C
        c = run_hops(env, assist, total, hops_c, "C", max_f=20000)
        report["hops_c"] = c
        shot(env, "l5_c2_c")

        snap = read_snapshot(env.get_ram())
        if snap.level == 0 and int(snap.rupees) < CANDLE_SHOP_PRICE:
            report["farm2"] = farm(env, assist, total, need=CANDLE_SHOP_PRICE, budget=12000)

        snap = read_snapshot(env.get_ram())
        if snap.level == 0 and snap.screen == 0x5E:
            report["buy"] = buy_5e(env, assist, total)
            shot(env, "l5_c2_buy")

        ram = env.get_ram()
        report["final"] = inv(ram)
        report["ok"] = int(read_u8(ram, ADDR_CANDLE)) == 1
        report["got_candle"] = report["ok"]
        report["whistle_0x065C"] = int(read_u8(ram, ADDR_WHISTLE))
        report["frames_total"] = total[0]
        shot(env, "l5_c2_final")
        if report["ok"]:
            path = write_state_bytes(
                state_path(GAME_DIR, GAME, "Level5HasCandle"),
                env.em.get_state(),
            )
            write_state_provenance(
                path,
                source_state_path=GAME_DIR / "custom_integrations" / GAME / f"{STATE}.state",
                request={
                    "segment": "Level5HasCandle",
                    "start_state": STATE,
                    "key_poke": False,
                    "via": "OW hops + farm + 0x5E buy",
                },
                selected_trial={
                    "success": True,
                    "frames": total[0],
                    "candle_0x065B": 1,
                    "rupees": int(read_snapshot(ram).rupees),
                },
                natural_entry=False,
            )
            report["checkpoint"] = "Level5HasCandle"
        write_json_report(RECORDINGS_DIR / "l5_candle2.json", report)
        return report
    finally:
        if env is not None:
            env.close()


if __name__ == "__main__":
    r = main()
    print("OK", r.get("ok"), "FINAL", r.get("final"))
    print("FARM", r.get("farm"), r.get("farm2"))
    print("BUY", r.get("buy"))
    print("CP", r.get("checkpoint"))
    print("STATUS_CLAIM", r.get("status_claim"), "POKES", r.get("pokes"))
