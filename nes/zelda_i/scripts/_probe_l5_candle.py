"""Exit L5 to OW, farm >=60 rupees, buy blue candle (0x065B=1).

Start: Level5EntranceFromL4 (proven 0x76 SOUTH → OW 0x0B).
Helpers reused:
- OverworldToCandleShopController + CANDLE_SHOP_HOPS / buy (0x5E)
- CANDLE_SHOP_MOUNTAIN 0x0C hop from item_gate_hops (0x0B RIGHT)
- HeartFarm-style chase for rupee drops (no poke)
- OverworldToLevel5Controller to return after candle

No pokes, no Clean STATUS, no Digdogger, no east67, no 0x65 bombs.
"""
from __future__ import annotations

from zelda_i.assist import UnlimitedHealthAssist
from zelda_i.dungeon_ops import idle, push_dir
from zelda_i.dungeon_trace import compact_snapshot, write_state_provenance
from zelda_i.heart_farm import (
    DEFAULT_4A_WAYPOINTS,
    FARM_SWING_HOLD,
    FARM_SWING_PERIOD,
    HeartFarmController,
)
from zelda_i.item_gate_hops import SCREEN_CANDLE_SHOP_MOUNTAIN
from zelda_i.level5_overworld import (
    LEVEL5_PATH_HOPS,
    OverworldToLevel5Controller,
)
from zelda_i.level8_overworld import (
    CANDLE_SHOP_HOPS,
    CANDLE_SHOP_PRICE,
    LEVEL8_5C_MAZE_WAYPOINTS,
    OverworldToCandleShopController,
    is_5c_maze_hop,
)
from zelda_i.nav_common import swing_action
from zelda_i.overworld import ScreenHop
from zelda_i.ow_path import OverworldPathController, PathNavPhase
from zelda_i.paths import GAME, GAME_DIR, RECORDINGS_DIR
from zelda_i.ram import (
    ADDR_CANDLE,
    ADDR_WHISTLE,
    PLAY_MODE,
    ZeldaSnapshot,
    read_snapshot,
    read_u8,
)
from retro_harness.env import make_env, reset_obs, state_path, write_state_bytes
from retro_harness.nes import nes_action, nes_idle_action
from retro_harness.segment_runner import (
    configure_headless,
    save_rgb_png,
    write_json_report,
)

STATE_EXIT = "Level5EntranceFromL4"
ROOM_76 = 0x76
OW_L5 = 0x0B
OW_HILLS = 0x1B
OW_SHOP_5E = 0x5E
OW_SHOP_0C = 0x0C
RUPEE_TYPES = frozenset({0x60, 0x61, 0x62})
ENEMY_SKIP = frozenset({0, 0xFF, 0x5A, 0x4F, 0x4E, 0x40})

# Reverse Lost Hills → mid-east → L8 candle corridor.
FROM_0B_TO_5E: tuple[ScreenHop, ...] = (
    ScreenHop(0x1B, "DOWN", align_x=112),
    ScreenHop(0x1C, "RIGHT", align_y=165),
    ScreenHop(0x2C, "DOWN", align_x=48),
    ScreenHop(0x2B, "LEFT", align_y=85),
    ScreenHop(0x3B, "DOWN", align_x=48),
    ScreenHop(0x3A, "LEFT", align_y=140),
    ScreenHop(0x4A, "DOWN", align_x=112),
    ScreenHop(0x5A, "DOWN", align_x=112),
    ScreenHop(0x5B, "RIGHT", y_band_lo=130, y_band_hi=150),
    ScreenHop(0x5C, "RIGHT", y_band_lo=80, y_band_hi=95),
    ScreenHop(0x5D, "RIGHT", y_band_lo=120, y_band_hi=140),
    ScreenHop(0x5E, "RIGHT", y_band_lo=130, y_band_hi=150),
)

FROM_5E_TO_L5: tuple[ScreenHop, ...] = (
    ScreenHop(0x5D, "LEFT", y_band_lo=130, y_band_hi=150),
    ScreenHop(0x5C, "LEFT", y_band_lo=120, y_band_hi=140),
    ScreenHop(0x5B, "LEFT", y_band_lo=80, y_band_hi=95),
    ScreenHop(0x5A, "LEFT", y_band_lo=130, y_band_hi=150),
    ScreenHop(0x4A, "UP", align_x=112),
    *LEVEL5_PATH_HOPS,
)


def inv(ram) -> dict:
    snap = read_snapshot(ram)
    return {
        "rupees": int(snap.rupees),
        "keys": int(snap.keys),
        "bombs": int(snap.bombs),
        "candle_0x065B": int(read_u8(ram, ADDR_CANDLE)),
        "whistle_0x065C": int(read_u8(ram, ADDR_WHISTLE)),
        "level": int(snap.level),
        "screen": f"0x{snap.screen:02x}",
        "mode": int(snap.mode),
        "xy": [int(snap.link_x), int(snap.link_y)],
    }


def dump(snap: ZeldaSnapshot, ram) -> dict:
    body = compact_snapshot(snap)
    body["inventory"] = inv(ram)
    body["room_hex"] = f"0x{snap.screen:02x}"
    return body


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


def shot(env, assist, total, name: str) -> str:
    png = RECORDINGS_DIR / f"{name}.png"
    obs, *_ = env.step(nes_idle_action())
    total[0] += 1
    assist.apply_env(env, frame=total[0])
    save_rgb_png(obs, png)
    return str(png.resolve())


def walk_axis(env, assist, total, axis: str, target: int, max_f: int = 400) -> bool:
    last = None
    stall = 0
    for _ in range(max_f):
        snap = read_snapshot(env.get_ram())
        if axis == "x":
            if abs(snap.link_x - target) <= 2:
                return True
            step(env, assist, total, nes_action("RIGHT" if snap.link_x < target else "LEFT"))
        else:
            if abs(snap.link_y - target) <= 2:
                return True
            step(env, assist, total, nes_action("DOWN" if snap.link_y < target else "UP"))
        pos = (snap.link_x, snap.link_y)
        if pos == last:
            stall += 1
            if stall >= 50:
                return False
        else:
            stall = 0
        last = pos
    return False


def exit_l5(env, assist, total) -> dict:
    idle(env, assist, total, 20)
    start = dump(read_snapshot(env.get_ram()), env.get_ram())
    walk_axis(env, assist, total, "x", 120, max_f=300)
    walk_axis(env, assist, total, "y", 205, max_f=400)
    push_dir(env, assist, total, "DOWN", frames=280)
    extra = 0
    while extra < 300:
        snap = read_snapshot(env.get_ram())
        if snap.level == 0 and snap.mode == PLAY_MODE and not snap.transitioning:
            idle(env, assist, total, 20)
            break
        step(env, assist, total, nes_idle_action())
        extra += 1
    ram = env.get_ram()
    snap = read_snapshot(ram)
    rec = {
        "start": start,
        "after": dump(snap, ram),
        "on_overworld": snap.level == 0,
        "ow_screen": f"0x{snap.screen:02x}" if snap.level == 0 else None,
        "rupees": int(snap.rupees),
        "candle_0x065B": int(read_u8(ram, ADDR_CANDLE)),
    }
    print("EXIT", rec["on_overworld"], rec["ow_screen"], "R", rec["rupees"], flush=True)
    return rec


def probe_dir(env, assist, total, direction: str, frames: int = 220) -> dict:
    snap0 = read_snapshot(env.get_ram())
    sc0, lv0 = snap0.screen, snap0.level
    push_dir(env, assist, total, direction, frames=frames)
    idle(env, assist, total, 12)
    extra = 0
    while extra < 80:
        snap = read_snapshot(env.get_ram())
        if not snap.transitioning and snap.mode in (PLAY_MODE, 11):
            break
        step(env, assist, total, nes_idle_action())
        extra += 1
    snap = read_snapshot(env.get_ram())
    rec = {
        "dir": direction,
        "from": f"L{lv0}:0x{sc0:02x}",
        "to_level": snap.level,
        "to_screen": f"0x{snap.screen:02x}",
        "mode": snap.mode,
        "xy": [snap.link_x, snap.link_y],
        "changed": snap.screen != sc0 or snap.level != lv0,
    }
    print("PROBE", rec, flush=True)
    return rec


def drive_nav(env, assist, total, nav, max_f: int, label: str) -> dict:
    trail = []
    last = None
    for _ in range(max_f):
        ram = env.get_ram()
        snap = read_snapshot(ram)
        if hasattr(nav, "candle_value"):
            nav.candle_value = int(read_u8(ram, ADDR_CANDLE))
        key = (snap.level, snap.screen, snap.mode)
        if key != last:
            trail.append(
                {
                    "f": total[0],
                    "L": snap.level,
                    "sc": f"0x{snap.screen:02x}",
                    "mode": snap.mode,
                    "xy": [snap.link_x, snap.link_y],
                    "R": int(snap.rupees),
                    "candle": int(read_u8(ram, ADDR_CANDLE)),
                }
            )
            print(label, trail[-1], flush=True)
            last = key
        if getattr(nav, "success", False):
            break
        phase = getattr(nav, "phase", None)
        if phase is not None and getattr(phase, "name", "") == "FAILED":
            break
        act = nav.step(snap)
        step(env, assist, total, act.action)
    ram = env.get_ram()
    snap = read_snapshot(ram)
    return {
        "success": bool(getattr(nav, "success", False)),
        "nav": nav.report() if hasattr(nav, "report") else {},
        "final": dump(snap, ram),
        "trail": trail,
        "candle_0x065B": int(read_u8(ram, ADDR_CANDLE)),
        "rupees": int(snap.rupees),
    }


def farm_rupees(env, assist, total, need: int = 60, budget: int = 18000) -> dict:
    """Chase enemies / rupee drops; bounce to a neighbor to respawn."""
    start_r = int(read_snapshot(env.get_ram()).rupees)
    if start_r >= need:
        return {"ok": True, "start": start_r, "end": start_r, "frames": 0, "note": "already"}
    log = []
    bounce_dir = "RIGHT"
    empty = 0
    farm_sc = read_snapshot(env.get_ram()).screen
    for i in range(budget):
        ram = env.get_ram()
        snap = read_snapshot(ram)
        rupees = int(snap.rupees)
        if rupees >= need:
            log.append({"f": total[0], "rupees": rupees, "note": "hit_need"})
            break
        if snap.mode == 17:
            break
        if snap.level != 0:
            # accidentally entered cave/dungeon — back out
            step(env, assist, total, nes_action("DOWN"))
            continue
        if snap.transitioning or snap.mode not in (PLAY_MODE, 8):
            step(env, assist, total, nes_idle_action())
            continue
        if snap.screen != farm_sc:
            # drifted — try to return
            empty += 1
            if empty % 80 == 0:
                bounce_dir = {"RIGHT": "LEFT", "LEFT": "RIGHT", "UP": "DOWN", "DOWN": "UP"}[
                    bounce_dir
                ]
            step(env, assist, total, nes_action(bounce_dir if empty % 2 == 0 else "DOWN"))
            if empty > 200:
                farm_sc = snap.screen
                empty = 0
            continue

        drops = [
            o
            for o in snap.objects
            if 1 <= o.slot <= 12 and o.type_id in RUPEE_TYPES
        ]
        enemies = [
            o
            for o in snap.objects
            if 1 <= o.slot <= 12
            and o.type_id not in ENEMY_SKIP
            and o.type_id not in RUPEE_TYPES
            and 40 < o.y < 220
            and 8 < o.x < 248
        ]
        target = None
        if drops:
            target = min(drops, key=lambda o: abs(o.x - snap.link_x) + abs(o.y - snap.link_y))
            empty = 0
        elif enemies:
            target = min(enemies, key=lambda o: abs(o.x - snap.link_x) + abs(o.y - snap.link_y))
            empty = 0
        else:
            empty += 1

        if target is not None:
            dx = target.x - snap.link_x
            dy = target.y - snap.link_y
            if abs(dx) >= abs(dy) and abs(dx) > 3:
                d = "RIGHT" if dx > 0 else "LEFT"
            elif abs(dy) > 3:
                d = "DOWN" if dy > 0 else "UP"
            else:
                d = "RIGHT" if dx >= 0 else "LEFT"
            act = swing_action(
                total[0], d, "farm", period=FARM_SWING_PERIOD, hold=FARM_SWING_HOLD
            )
            step(env, assist, total, act.action)
        elif empty > 90:
            # leave and return to respawn
            d = bounce_dir
            step(env, assist, total, nes_action(d))
            if empty > 160:
                bounce_dir = "LEFT" if bounce_dir == "RIGHT" else "RIGHT"
                empty = 0
                log.append({"f": total[0], "rupees": rupees, "bounce": bounce_dir, "sc": f"0x{snap.screen:02x}"})
                print("FARM_BOUNCE", log[-1], flush=True)
        else:
            # patrol mid band
            if abs(snap.link_y - 141) > 8:
                d = "DOWN" if snap.link_y < 141 else "UP"
            else:
                d = "RIGHT" if (total[0] // 40) % 2 == 0 else "LEFT"
            act = swing_action(
                total[0], d, "farm_patrol", period=FARM_SWING_PERIOD, hold=FARM_SWING_HOLD
            )
            step(env, assist, total, act.action)
        if i % 400 == 0:
            print("FARM", "R", rupees, "sc", hex(snap.screen), "f", total[0], flush=True)
    ram = env.get_ram()
    snap = read_snapshot(ram)
    rec = {
        "ok": int(snap.rupees) >= need,
        "start": start_r,
        "end": int(snap.rupees),
        "need": need,
        "screen": f"0x{snap.screen:02x}",
        "xy": [snap.link_x, snap.link_y],
        "log": log[-12:],
    }
    print("FARM_DONE", rec["ok"], rec["start"], "->", rec["end"], flush=True)
    return rec


def try_enter_cave(env, assist, total, xs: tuple[int, ...] = (48, 80, 112, 144, 176)) -> dict:
    """Hunt a cave mouth by walking UP at several x stands."""
    snap0 = read_snapshot(env.get_ram())
    sc0 = snap0.screen
    tries = []
    for x in xs:
        walk_axis(env, assist, total, "y", 141, max_f=200)
        walk_axis(env, assist, total, "x", x, max_f=200)
        walk_axis(env, assist, total, "y", 85, max_f=200)
        push_dir(env, assist, total, "UP", frames=160)
        idle(env, assist, total, 16)
        extra = 0
        while extra < 80:
            snap = read_snapshot(env.get_ram())
            if snap.mode == 11 or snap.level != 0:
                break
            if not snap.transitioning and snap.mode == PLAY_MODE:
                break
            step(env, assist, total, nes_idle_action())
            extra += 1
        snap = read_snapshot(env.get_ram())
        rec = {
            "x": x,
            "mode": snap.mode,
            "level": snap.level,
            "screen": f"0x{snap.screen:02x}",
            "xy": [snap.link_x, snap.link_y],
        }
        tries.append(rec)
        print("CAVE_TRY", rec, flush=True)
        if snap.mode == 11 or snap.level != 0:
            return {"entered": True, "tries": tries, "final": rec}
        # back to mid if still on same OW
        if snap.screen == sc0:
            walk_axis(env, assist, total, "y", 141, max_f=200)
    return {"entered": False, "tries": tries}


def buy_in_cave(env, assist, total, budget: int = 900) -> dict:
    """Reuse candle-shop buy geometry: UP to y≈149, RIGHT to x≈152, touch."""
    start = inv(env.get_ram())
    for i in range(budget):
        ram = env.get_ram()
        snap = read_snapshot(ram)
        candle = int(read_u8(ram, ADDR_CANDLE))
        if candle:
            return {"ok": True, "frames": i, "start": start, "end": inv(ram)}
        if snap.mode != 11:
            return {"ok": False, "reason": f"left_cave_mode_{snap.mode}", "end": inv(ram)}
        if i < 80 and snap.link_y > 200:
            step(env, assist, total, nes_idle_action())
            continue
        if snap.link_y > 150:
            step(env, assist, total, nes_action("UP"))
        elif snap.link_x < 152:
            step(env, assist, total, nes_action("RIGHT"))
        else:
            step(env, assist, total, nes_action("UP"))
    return {"ok": False, "reason": "buy_timeout", "end": inv(env.get_ram())}


def main() -> dict:
    configure_headless()
    RECORDINGS_DIR.mkdir(parents=True, exist_ok=True)
    commands = [
        "PYTHONPATH=nes:. uv run python nes/zelda_i/scripts/_probe_l5_candle.py  "
        "# Level5EntranceFromL4 SOUTH OW 0x0B, farm>=60, buy candle 0x5E/0x0C, infinite-life"
    ]
    env = None
    total = [1]
    report: dict = {
        "ok": False,
        "status_claim": None,
        "pokes": False,
        "commands": commands,
        "from_state": STATE_EXIT,
        "shop_price": CANDLE_SHOP_PRICE,
    }
    try:
        env, assist, obs = open_env(STATE_EXIT)
        idle(env, assist, total, 16)
        report["start"] = inv(env.get_ram())
        print("START", report["start"], flush=True)

        report["exit"] = exit_l5(env, assist, total)
        shot(env, assist, total, "l5_candle_ow0b")
        if not report["exit"]["on_overworld"]:
            report["reason"] = "exit_failed"
            write_json_report(RECORDINGS_DIR / "l5_candle.json", report)
            return report

        # Probe 0x0B exits. LEFT/UP re-enter L5 (known). Try EAST (0x0C) and SOUTH (0x1B).
        probes = []
        snap = read_snapshot(env.get_ram())
        if snap.screen == OW_L5:
            walk_axis(env, assist, total, "x", 200, max_f=250)
            probes.append(probe_dir(env, assist, total, "RIGHT", frames=260))
        report["probes_0b"] = probes
        shot(env, assist, total, "l5_candle_after_east")

        snap = read_snapshot(env.get_ram())
        mountain = None
        if snap.level == 0 and snap.screen == OW_SHOP_0C:
            print("REACHED_0C mountain shop screen", flush=True)
            farm = farm_rupees(env, assist, total, need=CANDLE_SHOP_PRICE, budget=16000)
            report["farm_0c"] = farm
            cave = try_enter_cave(env, assist, total)
            report["cave_0c"] = cave
            if cave.get("entered") and read_snapshot(env.get_ram()).mode == 11:
                mountain = buy_in_cave(env, assist, total)
                report["buy_0c"] = mountain
        report["after_0c_attempt"] = inv(env.get_ram())

        ram = env.get_ram()
        candle = int(read_u8(ram, ADDR_CANDLE))
        if candle:
            report["ok"] = True
            report["got_candle"] = True
            report["via"] = "mountain_0x0C"
        else:
            # Walk 0x0B/current → 0x5E using existing candle-shop controller + custom hops.
            snap = read_snapshot(env.get_ram())
            hops = FROM_0B_TO_5E
            if snap.screen != OW_L5:
                # drop hops until we find a hop whose target we can still use
                targets = [h.target for h in hops]
                if snap.screen in targets:
                    hops = hops[targets.index(snap.screen) + 1 :]
                elif snap.screen == 0x5E:
                    hops = ()
            print("WALK_5E from", hex(snap.screen), "hops", [hex(h.target) for h in hops], flush=True)
            nav = OverworldToCandleShopController(
                hops=hops if hops else CANDLE_SHOP_HOPS,
                maze_waypoints=LEVEL8_5C_MAZE_WAYPOINTS,
                maze_hop_pred=is_5c_maze_hop,
                enter_cave=False,
                buy_candle=False,
                max_frames=40000,
            )
            if not hops:
                nav.hop_index = len(nav.hops)
                from zelda_i.level8_overworld import CandleShopNavPhase

                nav.phase = CandleShopNavPhase.DONE
                nav.success = True
            walk = drive_nav(env, assist, total, nav, 40000, "TO5E")
            report["walk_5e"] = {
                "success": walk["success"],
                "trail": walk["trail"],
                "final": walk["final"]["inventory"] if walk.get("final") else None,
                "nav_phase": (walk.get("nav") or {}).get("phase"),
                "nav_notes": (walk.get("nav") or {}).get("notes", [])[-12:],
            }
            shot(env, assist, total, "l5_candle_at5e")

            # Farm on whatever OW screen we have (prefer 0x4A / 0x5E / 0x59).
            snap = read_snapshot(env.get_ram())
            if snap.level == 0 and int(snap.rupees) < CANDLE_SHOP_PRICE:
                # If we reached 0x4A-ish, use HeartFarm chase on that screen first.
                farm = farm_rupees(env, assist, total, need=CANDLE_SHOP_PRICE, budget=20000)
                report["farm_5e"] = farm

            snap = read_snapshot(env.get_ram())
            if snap.level == 0 and snap.screen != OW_SHOP_5E:
                # one more hop attempt from here
                nav2 = OverworldToCandleShopController(
                    hops=FROM_0B_TO_5E,
                    maze_waypoints=LEVEL8_5C_MAZE_WAYPOINTS,
                    maze_hop_pred=is_5c_maze_hop,
                    enter_cave=False,
                    buy_candle=False,
                    max_frames=20000,
                )
                # skip completed prefix
                targets = [h.target for h in FROM_0B_TO_5E]
                if snap.screen in targets:
                    nav2.hops = FROM_0B_TO_5E[targets.index(snap.screen) + 1 :]
                walk2 = drive_nav(env, assist, total, nav2, 20000, "TO5E2")
                report["walk_5e_retry"] = {
                    "success": walk2["success"],
                    "trail": walk2["trail"],
                    "final": walk2["final"]["inventory"] if walk2.get("final") else None,
                }

            # Enter 0x5E cave and buy if we can afford it.
            snap = read_snapshot(env.get_ram())
            if snap.level == 0 and snap.screen == OW_SHOP_5E:
                nav_buy = OverworldToCandleShopController(
                    hops=(),
                    enter_cave=True,
                    buy_candle=True,
                    max_frames=8000,
                )
                nav_buy.hop_index = 0
                from zelda_i.level8_overworld import CandleShopNavPhase

                nav_buy.phase = CandleShopNavPhase.DOOR
                nav_buy.hops = (ScreenHop(OW_SHOP_5E, "UP", align_x=112),)
                buy = drive_nav(env, assist, total, nav_buy, 8000, "BUY5E")
                report["buy_5e"] = {
                    "success": buy["success"],
                    "candle_0x065B": buy["candle_0x065B"],
                    "rupees": buy["rupees"],
                    "final": buy["final"]["inventory"] if buy.get("final") else None,
                    "nav": buy.get("nav"),
                }
                if not buy["candle_0x065B"]:
                    # fallback geometry buy if already in cave
                    if read_snapshot(env.get_ram()).mode == 11:
                        report["buy_5e_fallback"] = buy_in_cave(env, assist, total)

        ram = env.get_ram()
        snap = read_snapshot(ram)
        candle = int(read_u8(ram, ADDR_CANDLE))
        report["final"] = inv(ram)
        report["got_candle"] = candle == 1
        report["ok"] = candle == 1
        report["whistle_0x065C"] = int(read_u8(ram, ADDR_WHISTLE))
        report["frames_total"] = total[0]
        shot(env, assist, total, "l5_candle_final")

        if candle == 1:
            path = write_state_bytes(
                state_path(GAME_DIR, GAME, "Level5HasCandle"),
                env.em.get_state(),
            )
            write_state_provenance(
                path,
                source_state_path=(
                    GAME_DIR / "custom_integrations" / GAME / f"{STATE_EXIT}.state"
                ),
                request={
                    "segment": "Level5HasCandle",
                    "predecessor_entry": True,
                    "start_state": STATE_EXIT,
                    "key_poke": False,
                    "door_poke": False,
                    "bomb_count_poke": False,
                    "selected_item_poke": False,
                    "via": "OW exit 0x0B, farm rupees, buy blue candle",
                },
                selected_trial={
                    "success": True,
                    "frames": total[0],
                    "candle_0x065B": 1,
                    "rupees": int(snap.rupees),
                    "screen": snap.screen,
                    "level": snap.level,
                    "mode": snap.mode,
                },
                natural_entry=False,
            )
            report["checkpoint"] = "Level5HasCandle"

            # Return to L5 if we are on OW.
            if snap.level == 0:
                # leave cave first
                if snap.mode == 11:
                    push_dir(env, assist, total, "DOWN", frames=240)
                    extra = 0
                    while extra < 200:
                        snap = read_snapshot(env.get_ram())
                        if snap.mode == PLAY_MODE and snap.level == 0:
                            idle(env, assist, total, 16)
                            break
                        step(env, assist, total, nes_idle_action())
                        extra += 1
                snap = read_snapshot(env.get_ram())
                hops = FROM_5E_TO_L5
                targets = [h.target for h in hops]
                if snap.screen in targets:
                    hops = hops[targets.index(snap.screen) + 1 :]
                elif snap.screen == 0x4A:
                    hops = LEVEL5_PATH_HOPS
                ret = OverworldToLevel5Controller(
                    hops=hops if hops else LEVEL5_PATH_HOPS,
                    require_dungeon=True,
                    max_frames=40000,
                )
                back = drive_nav(env, assist, total, ret, 40000, "BACKL5")
                report["return_l5"] = {
                    "success": back["success"],
                    "trail": back["trail"],
                    "final": back["final"]["inventory"] if back.get("final") else None,
                }
                snap = read_snapshot(env.get_ram())
                if snap.level == 5:
                    path2 = write_state_bytes(
                        state_path(GAME_DIR, GAME, "Level5CandleInside"),
                        env.em.get_state(),
                    )
                    write_state_provenance(
                        path2,
                        source_state_path=(
                            GAME_DIR / "custom_integrations" / GAME / "Level5HasCandle.state"
                        ),
                        request={
                            "segment": "Level5CandleInside",
                            "start_state": "Level5HasCandle",
                            "key_poke": False,
                            "via": "return Lost Hills after candle buy",
                        },
                        selected_trial={
                            "success": True,
                            "frames": total[0],
                            "candle_0x065B": 1,
                            "room": snap.screen,
                            "level": 5,
                        },
                        natural_entry=False,
                    )
                    report["checkpoint_inside"] = "Level5CandleInside"
                    shot(env, assist, total, "l5_candle_inside")

        report["frames_total"] = total[0]
        write_json_report(RECORDINGS_DIR / "l5_candle.json", report)
        return report
    finally:
        if env is not None:
            env.close()


if __name__ == "__main__":
    r = main()
    print("CMD", r.get("commands"))
    print("OK", r.get("ok"), "CANDLE", (r.get("final") or {}).get("candle_0x065B"))
    print("RUPEES", (r.get("final") or {}).get("rupees"))
    print("WHISTLE", r.get("whistle_0x065C"))
    print("CHECKPOINT", r.get("checkpoint"), r.get("checkpoint_inside"))
    print("VIA", r.get("via") or r.get("buy_5e") or r.get("buy_0c"))
    print("STATUS_CLAIM", r.get("status_claim"))
    print("POKES", r.get("pokes"))
