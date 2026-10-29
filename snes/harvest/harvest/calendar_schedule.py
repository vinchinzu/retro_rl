"""Festival dates, shop hours, carpenter notes, and measured girl locations.

Festival dates follow bank_82 ``WeatherTomorrow`` ($828CF9): the forecast byte
is written the night before, so the festival is the next calendar day.
Forced festivals are Star Night and Y2+ New Year (measured house-exit warp;
see ``FestivalEscapeTask``). Shop hours are door-flag checks in bank_80
``Unk_Table13`` ($80B6F5), applied in ``CODE_809B88``. Girl rows are filled
only from a live probe; unmeasured season, hour, or weather cells stay empty.
"""

from __future__ import annotations

from dataclasses import dataclass

from harvest.planner.day_phase_calendar import (
    festival_name_for_date,
    forecast_code_for_date,
)

# Weekday 0 is Sunday and 6 is Saturday (NightReset wraps 0..6; new game starts at 1).
SUNDAY = 0
SATURDAY = 6
HOUR_CLOSE = 0x11  # door checks compare hour with 17:00

# Unk_Table13 byte 3. Reject path is CODE_809C14.
FLAG_CLOSED_FROM_17 = 0x02
FLAG_OPEN_FROM_17 = 0x04
FLAG_WEEKEND_ONLY = 0x08
FLAG_CLOSED_SATURDAY_OR_WINTER_D10 = 0x10
FLAG_CLOSED_SUNDAY = 0x20
FLAG_CLOSED_IN_STORM = 0x80
STORM_WEATHER_BIT = 0x0010
RAIN_WEATHER_BIT = 0x0002
SNOW_WEATHER_BIT = 0x0008

# Index, dest tilemap, flags. Dest names are Maps_Graphics.asm. Index 28
# (dest 0x2B, MapUnknown2B) uses shop hours and is not a labeled carpenter door.
DOOR_FLAGS: dict[str, tuple[int, int, int]] = {
    "mayor": (18, 0x18, 0x22),
    "church": (19, 0x1B, 0x02),
    "flower": (20, 0x1C, 0x32),
    "bar": (21, 0x1E, 0x24),
    "restaurant": (22, 0x20, 0x32),
    "tool": (23, 0x22, 0x32),
    "animal": (24, 0x24, 0x32),
    "animal_side": (25, 0x24, 0x32),
    "witch": (26, 0x25, 0x32),
    "map_2b": (28, 0x2B, 0x32),
    "house_exit": (9, 0x00, 0x81),
    "summit": (31, 0x31, 0x41),
}

# Plan "shops": the 0x32 storefronts. Bar is evening-only. map_2b is unnamed.
SHOP_NAMES = ("flower", "restaurant", "tool", "animal", "witch")

# Dialogue text id 0x128 (DATA16_B7BD58): "We are carpenters."
# PlayerActionPointerTable at $81D593. Facing an NPC stores this index in
# $096E (bank_82 $82AEBA) and the handler loads that girl's heart field.
GIRL_ACTION_INDEX = {
    "maria": 0x26,  # CODE_00DC5E reads !hearts_maria
    "nina": 0x2D,  # DATA8_00DDAC reads !hearts_nina
    "ellen": 0x33,  # DATA8_00E2D9 reads !hearts_ellen
    "ann": 0x36,  # DATA8_00E394 reads !hearts_ann
    "eve": 0x3F,  # DATA8_00E52C reads !hearts_eve
}
ACTION_TO_GIRL = {index: name for name, index in GIRL_ACTION_INDEX.items()}

CARPENTER_INTRO_TEXT_ID = 0x128
# CODE_83D175 loads CC script 0x16 at Saturday 06:00 under extra flag gates.
# That is a cutscene, not a weekly carpenter door hour.
SATURDAY_MORNING_CC_INDEX = 0x16

FORCED_FESTIVAL_KEYS = frozenset({(3, 24)})  # Winter D24 Star Night, every year


@dataclass(frozen=True)
class FestivalDay:
    name: str
    season: int
    day: int
    year: int
    forecast_code: int
    forced: bool


@dataclass(frozen=True)
class GirlSighting:
    """One live contact. ``year`` is the 0-based RAM year."""

    name: str
    sample: str
    year: int
    season: int
    day: int
    weekday: int
    hour: int
    minute: int
    weather_flags: int
    tilemap: int
    pixel_x: int
    pixel_y: int
    sprite: int
    text_id: int


@dataclass(frozen=True)
class CarpenterSighting:
    sample: str
    year: int
    season: int
    day: int
    weekday: int
    hour: int
    minute: int
    weather_flags: int
    tilemap: int
    pixel_x: int
    pixel_y: int
    sprite: int
    text_id: int


def festival_on(year: int, season: int, day: int) -> FestivalDay | None:
    """Festival on this date, or None. ``year`` is 0-based, as in RAM."""

    name = festival_name_for_date(season, day, year)
    if name is None:
        return None
    forced = (int(season), int(day)) in FORCED_FESTIVAL_KEYS or (
        name == "New Year" and int(year) >= 1
    )
    return FestivalDay(
        name=name,
        season=int(season),
        day=int(day),
        year=int(year),
        forecast_code=int(forecast_code_for_date(year, season, day)),
        forced=forced,
    )


def door_refuses(
    flags: int,
    *,
    weekday: int,
    hour: int,
    season: int,
    day: int,
    weather_flags: int,
) -> bool:
    """True when bank_80 ``CODE_809B88`` rejects this door."""

    bits = int(flags)
    if bits & FLAG_CLOSED_FROM_17 and int(hour) >= HOUR_CLOSE:
        return True
    if bits & FLAG_OPEN_FROM_17 and int(hour) < HOUR_CLOSE:
        return True
    if bits & FLAG_WEEKEND_ONLY and int(weekday) not in (SUNDAY, SATURDAY):
        return True
    if bits & FLAG_CLOSED_SATURDAY_OR_WINTER_D10:
        if int(season) == 3 and int(day) == 0x0A:
            return True
        if int(weekday) == SATURDAY:
            return True
    if bits & FLAG_CLOSED_SUNDAY and int(weekday) == SUNDAY:
        return True
    if bits & FLAG_CLOSED_IN_STORM and int(weather_flags) & STORM_WEATHER_BIT:
        return True
    return False


def venue_open(
    name: str,
    *,
    weekday: int,
    hour: int,
    season: int,
    day: int,
    weather_flags: int = 0,
) -> bool:
    _index, _dest, flags = DOOR_FLAGS[name]
    return not door_refuses(
        flags,
        weekday=weekday,
        hour=hour,
        season=season,
        day=day,
        weather_flags=weather_flags,
    )


def shop_open(name: str, **kwargs: int) -> bool:
    if name not in SHOP_NAMES:
        raise KeyError(name)
    return venue_open(name, **kwargs)


def weather_class(weather_flags: int) -> str:
    flags = int(weather_flags)
    if flags & STORM_WEATHER_BIT:
        return "storm"
    if flags & SNOW_WEATHER_BIT:
        return "snow"
    if flags & RAIN_WEATHER_BIT:
        return "rain"
    return "clear"


def _load_girl_sightings() -> tuple[GirlSighting, ...]:
    """One live row per girl per sample. Missing or empty JSON stays empty."""

    import json
    from pathlib import Path

    path = Path(__file__).resolve().parents[1] / "recordings" / "pr021_calendar_probe.json"
    if not path.is_file():
        return ()
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return ()
    names = {"maria", "ann", "nina", "ellen", "eve"}
    samples = {"weekday", "saturday", "sunday"}
    rows: list[GirlSighting] = []
    seen: set[tuple[str, str]] = set()
    for raw in payload.get("sightings") or []:
        name = str(raw.get("name", ""))
        sample = str(raw.get("sample", ""))
        key = (sample, name)
        if name not in names or sample not in samples or key in seen:
            continue
        seen.add(key)
        rows.append(
            GirlSighting(
                name=name,
                sample=sample,
                year=int(raw["year"]),
                season=int(raw["season"]),
                day=int(raw["day"]),
                weekday=int(raw["weekday"]),
                hour=int(raw["hour"]),
                minute=int(raw["minute"]),
                weather_flags=int(raw["weather_flags"]),
                tilemap=int(raw["tilemap"]),
                pixel_x=int(raw["pixel_x"]),
                pixel_y=int(raw["pixel_y"]),
                sprite=int(raw["sprite"]),
                text_id=int(raw.get("text_id", 0)),
            )
        )
    return tuple(rows)


# Filled from recordings/pr021_calendar_probe.json after a live run.
# Empty until that probe names all five girls on the three sample days.
GIRL_SIGHTINGS: tuple[GirlSighting, ...] = _load_girl_sightings()
CARPENTER_SIGHTINGS: tuple[CarpenterSighting, ...] = ()


def girl_location(
    name: str,
    *,
    season: int,
    weekday: int,
    hour: int,
    weather_flags: int,
) -> GirlSighting | None:
    """Measured cell for this season, weekday, hour, and weather class.

    Returns None for any axis the probe did not walk. Hours match the
    clock hour recorded at contact.
    """

    want = weather_class(weather_flags)
    for row in GIRL_SIGHTINGS:
        if row.name != name:
            continue
        if row.season != int(season) or row.weekday != int(weekday):
            continue
        if row.hour != int(hour):
            continue
        if weather_class(row.weather_flags) != want:
            continue
        return row
    return None


def carpenter_availability() -> dict[str, object]:
    """What the ROM and the probe actually show about the carpenters."""

    return {
        "intro_text_id": CARPENTER_INTRO_TEXT_ID,
        "shop_door": None,
        "map_2b_uses_shop_hours": True,
        "map_2b_is_carpenter": False,
        "saturday_0600_cc_index": SATURDAY_MORNING_CC_INDEX,
        "sightings": CARPENTER_SIGHTINGS,
    }


def _probe_main() -> None:
    import argparse
    import json
    import sys
    from pathlib import Path

    parser = argparse.ArgumentParser(description="Live girl-location probe")
    parser.add_argument("--trials", type=int, default=1)
    args = parser.parse_args()
    if args.trials != 1:
        print(f"FAILED trials={args.trials} only 1 is run", flush=True)
        sys.exit(1)
    try:
        report = run_live_probe()
    except Exception as exc:
        print(f"FAILED {type(exc).__name__}: {exc}", flush=True)
        raise
    out = Path(__file__).resolve().parents[1] / "recordings" / "pr021_calendar_probe.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    girls = report.get("girls_per_sample") or {}
    ok = all(len(names) == 5 for names in girls.values()) and len(girls) == 3
    print(
        f"{'DONE' if ok else 'FAILED'} frames={report.get('frames')} "
        f"girls={girls} json={out}",
        flush=True,
    )
    sys.exit(0 if ok else 1)


def run_live_probe() -> dict:
    """Boot the town pin three times. A girl sprite in the object table counts.

    Copies of Y1_D22_Town_Eve stay in town. Church enter/exit reloads the
    hour-6 scripts. The east gate is left by walking right from (728, 424).
    """

    import traceback

    import numpy as np
    from retro_harness import TaskStatus, WorldState

    from harvest.core.harvest_state import HarvestStateDocument, weekday_for_date
    from harvest.core.npc_catalog import ROMANCE_NPCS, game_objects
    from harvest.core.ram_catalog import read_ram_value
    from harvest.maps.map_routes import ROUTES, densify_waypoints
    from harvest.maps.map_types import Waypoint
    from harvest.paths import STATES_DIR
    from harvest.planner.tasks.multi_nav import MultiMapNavTask
    from harvest.runtime.retro_setup import make_harvest_env
    from harvest.tasks.nav import get_pos_from_ram, make_action

    # pr021_calendar_probe.log npc_candidates, named from the event-script
    # sprite at that pixel. Weekday: maria 0x0230, nina 0x024B, ellen 0x023C.
    # Saturday: ann 0x0226, ellen 0x023C. The same base, within 8, is the
    # live anim frame. Other girls use the 0x1A base from bank B3.
    stands = {
        "weekday": (
            ("maria", 0x04, 312, 280, 0x0230),
            ("nina", 0x04, 520, 264, 0x024B),
            ("ellen", 0x04, 648, 904, 0x023C),
            ("eve", 0x04, 392, 504, 0x0248),
            ("ann", 0x22, 152, 408, 0x022C),
        ),
        "saturday": (
            ("ann", 0x04, 376, 888, 0x0226),
            ("ellen", 0x04, 536, 792, 0x023C),
            ("maria", 0x1B, 72, 408, 0x0222),
            ("eve", 0x10, 504, 216, 0x0245),
            ("nina", 0x29, 88, 424, 0x024C),
        ),
        "sunday": (
            ("maria", 0x1B, 193, 136, 0x0224),
            ("ann", 0x1B, 216, 408, 0x0228),
            ("ellen", 0x1B, 72, 408, 0x023E),
            ("nina", 0x10, 40, 200, 0x024C),
            ("eve", 0x10, 672, 216, 0x0255),
        ),
    }
    log_sprite_names: dict[int, str] = {}
    for sprite in (0x0230, 0x0231):
        log_sprite_names[sprite] = "maria"
    for sprite in range(0x024B, 0x0251):
        log_sprite_names[sprite] = "nina"
    for sprite in range(0x023C, 0x0242):
        log_sprite_names[sprite] = "ellen"
    for sprite in range(0x0226, 0x022C):
        log_sprite_names[sprite] = "ann"

    samples = (
        ("weekday", "LN_mgr-calendar_whatif_town_wd", 22, 1),
        ("saturday", "LN_mgr-calendar_whatif_town_sa", 20, 6),
        ("sunday", "LN_mgr-calendar_whatif_town_su", 21, 0),
    )

    def mint(name: str, day: int) -> None:
        dest = STATES_DIR / f"{name}.state"
        if dest.exists() or dest.is_symlink():
            print(f"PIN_KEEP {name}", flush=True)
            return
        doc = HarvestStateDocument.load("Y1_D22_Town_Eve")
        doc.set_calendar_date(game_year=1, season="spring", day=day)
        doc.set_clock(hour=6, minute=0, second=0, time_running=0)
        doc.set_scalar_value("weather_tomorrow", 0)
        doc.save_as(dest)
        print(f"MINTED {dest.name}", flush=True)

    def check_pin(name: str, day: int, weekday: int) -> None:
        doc = HarvestStateDocument.load(name)
        got_day = doc.scalar_value("day")
        got_wd = doc.scalar_value("weekday")
        expect = weekday_for_date(game_year=1, season=0, day=day)
        weather = doc.mutable_state.read_u16(0x196)
        if (
            doc.scalar_value("year") != 0
            or doc.scalar_value("season") != 0
            or got_day != day
            or got_wd != weekday
            or expect != weekday
            or doc.scalar_value("hour") != 6
            or doc.scalar_value("time_running") != 0
            or doc.scalar_value("weather_tomorrow") != 0
            or doc.scalar_value("tilemap") != 0x04
            or weather != 0x80E1
            or doc.scalar_value("player_x") != 681
            or doc.scalar_value("player_y") != 54
        ):
            raise RuntimeError(
                f"{name} pin year={doc.scalar_value('year')} "
                f"day={got_day} wd={got_wd} expect={expect} "
                f"hour={doc.scalar_value('hour')} "
                f"run={doc.scalar_value('time_running')} "
                f"tom={doc.scalar_value('weather_tomorrow')} "
                f"map={doc.scalar_value('tilemap')} weather=0x{weather:04X} "
                f"xy=({doc.scalar_value('player_x')},{doc.scalar_value('player_y')})"
            )
        print(
            f"PIN_OK {name} f68=0x{doc.scalar_value('event_flags_1f68'):04X} "
            f"f6c=0x{doc.scalar_value('family_event_flags'):04X}",
            flush=True,
        )

    for _sample, name, day, weekday in samples:
        mint(name, day)
        check_pin(name, day, weekday)

    frames_total = 0
    sightings: list[dict] = []
    day_logs: list[dict] = []
    per_sample: dict[str, list[str]] = {}
    active = {"found": {}, "sample": "", "reloaded": False}
    seen_obj: set[tuple[str, int, int]] = set()

    def clock_of(ram: np.ndarray) -> dict:
        keys = ("year", "season", "weekday", "day", "hour", "minute", "tilemap")
        row = {key: int(read_ram_value(ram, key)) for key in keys}
        row["weather_flags"] = int(ram[0x196]) | (int(ram[0x197]) << 8)
        pos = get_pos_from_ram(ram)
        row["x"] = int(pos.x)
        row["y"] = int(pos.y)
        return row

    def hook(frame: int, ram: np.ndarray) -> None:
        clk = clock_of(ram)
        tilemap = clk["tilemap"]
        if tilemap == 0x1B:
            active["reloaded"] = True
        if not active["reloaded"] or int(clk["hour"]) >= 17:
            return
        sample = active["sample"]
        found: dict = active["found"]
        for obj in game_objects(ram):
            if obj.is_player or obj.kind != "npc_candidate":
                continue
            token = (sample, tilemap, obj.sprite_table_idx)
            if token not in seen_obj:
                seen_obj.add(token)
                print(
                    f"OBJ {sample} sprite=0x{obj.sprite_table_idx:04X} "
                    f"map=0x{tilemap:02X} px={obj.pixel}",
                    flush=True,
                )
            best_name = ""
            best_key = None
            for name, map_id, px, py, base in stands[sample]:
                if name in found or tilemap != map_id:
                    continue
                dist = max(abs(obj.pixel[0] - px), abs(obj.pixel[1] - py))
                delta = abs(obj.sprite_table_idx - base)
                if dist > 56 or delta > 8:
                    continue
                key = (dist, delta)
                if best_key is None or key < best_key:
                    best_key = key
                    best_name = name
            if not best_name:
                continue
            text_id = int(read_ram_value(ram, "dialog_text_id"))
            snap = {
                "name": best_name,
                "sample": sample,
                "year": clk["year"],
                "season": clk["season"],
                "day": clk["day"],
                "weekday": clk["weekday"],
                "hour": clk["hour"],
                "minute": clk["minute"],
                "weather_flags": clk["weather_flags"],
                "tilemap": tilemap,
                "pixel_x": int(obj.pixel[0]),
                "pixel_y": int(obj.pixel[1]),
                "sprite": int(obj.sprite_table_idx),
                "text_id": text_id,
            }
            found[best_name] = snap
            sightings.append(snap)
            print(
                f"LOCATED {sample} {best_name} map=0x{tilemap:02X} "
                f"px=({obj.pixel[0]},{obj.pixel[1]}) "
                f"{clk['hour']:02d}:{clk['minute']:02d} "
                f"sprite=0x{obj.sprite_table_idx:04X}",
                flush=True,
            )

    def drive(env, task, state, budget: int):
        nonlocal frames_total
        frame, ram, info, obs = state
        task.reset(WorldState(frame, ram, info, obs))
        status = TaskStatus.RUNNING
        reason = "budget"
        for _ in range(budget):
            frame += 1
            frames_total += 1
            result = task.step(WorldState(frame, ram, info, obs))
            action = make_action()
            if result.action is not None and result.action.action is not None:
                action = result.action.action
            obs, _reward, _terminated, _truncated, info = env.step(action)
            ram = np.asarray(env.get_ram(), dtype=np.uint8)
            hook(frame, ram)
            if result.status != TaskStatus.RUNNING:
                status = result.status
                reason = result.reason or ""
                break
            if frames_total % 1000 == 0:
                clk = clock_of(ram)
                print(
                    f"FRAME {frames_total} map=0x{clk['tilemap']:02X} "
                    f"({clk['x']},{clk['y']}) {clk['hour']:02d}:{clk['minute']:02d} "
                    f"found={sorted(active['found'])}",
                    flush=True,
                )
        return (frame, ram, info, obs), status, reason

    def idle(env, state, count: int):
        nonlocal frames_total
        frame, ram, info, obs = state
        for _ in range(count):
            frame += 1
            frames_total += 1
            obs, _reward, _terminated, _truncated, info = env.step(make_action())
            ram = np.asarray(env.get_ram(), dtype=np.uint8)
            hook(frame, ram)
        return (frame, ram, info, obs)

    def tile_of(state) -> int:
        return int(read_ram_value(state[1], "tilemap"))

    def hops(tilemap: int, points: list[tuple[int, int]], exit_direction: str | None = None):
        if (744, 424) in points:
            raise RuntimeError("east-gate landing is not a hop")
        dense = densify_waypoints(
            [
                Waypoint(tilemap=tilemap, target_px=(int(x), int(y)), radius=14)
                for x, y in points
            ]
        )
        if exit_direction is not None:
            dense[-1] = dense[-1]._replace(
                is_exit=True, exit_direction=exit_direction, radius=8
            )
        return dense

    def go(env, state, label: str, waypoints, budget: int):
        if not waypoints:
            return state
        if tile_of(state) != waypoints[0].tilemap:
            print(
                f"SKIP {active['sample']} {label} map=0x{tile_of(state):02X} "
                f"want=0x{waypoints[0].tilemap:02X}",
                flush=True,
            )
            return state
        state, status, reason = drive(
            env,
            MultiMapNavTask(waypoints=list(waypoints), timeout=budget),
            state,
            budget,
        )
        state = idle(env, state, 180)
        print(
            f"{active['sample']} {label} {status} {reason} "
            f"map=0x{tile_of(state):02X} found={sorted(active['found'])}",
            flush=True,
        )
        return state

    church = hops(
        0x04,
        [
            (680, 120),
            (680, 200),
            (680, 280),
            (520, 280),
            (376, 280),
            (376, 200),
            (358, 150),
        ],
        "up",
    )
    # 0xC6 is (408, 856). Two 0xFF tiles sit south of it; exit up off the
    # path, same inset as the flower door. (392, 856) is the west facade.
    tool_plaza = hops(
        0x04,
        [(376, 280), (560, 280), (680, 280), (684, 360), (688, 430)],
    )
    tool_tail = list(ROUTES["d1_town_to_ann"][1:])
    tool_tail += hops(0x04, [(408, 914), (408, 888)], "up")
    tool_tail[-1] = tool_tail[-1]._replace(radius=16)
    # West off the east gate along y=424. Not north from (744, 424).
    gate_west = hops(
        0x04,
        [(728, 424), (692, 424), (684, 400), (684, 340), (684, 288)],
    )
    # Proven east exit. Right from (728, 424), not north from (744, 424).
    gate = hops(
        0x04,
        [
            (376, 280),
            (560, 280),
            (684, 288),
            (684, 340),
            (684, 400),
            (692, 424),
            (728, 424),
        ],
        "right",
    )
    path = hops(0x0C, [(80, 128), (132, 128), (132, 70), (132, 30)], "up")
    # Path left exit, open face (40, 128), back onto town.
    back = hops(0x0C, [(80, 128), (40, 128)], "left")
    cave = []
    for wp in ROUTES["mountain_entry_to_first_berry"]:
        cave.append(wp)
        if wp.target_px == (144, 448):
            break
    cave.append(
        Waypoint(
            tilemap=0x10,
            target_px=(166, 430),
            radius=10,
            is_exit=True,
            exit_direction="up",
        )
    )

    def boot(state_name: str):
        env = make_harvest_env(state_name, render_mode=None, lean_step=True)
        obs, info = env.reset()
        ram = np.asarray(env.get_ram(), dtype=np.uint8)
        print(f"BOOT {state_name} {clock_of(ram)}", flush=True)
        return env, (0, ram, info, obs)

    for sample, state_name, day, weekday in samples:
        env = None
        try:
            active["found"] = {}
            active["sample"] = sample
            active["reloaded"] = False
            env, state = boot(state_name)
            clk = clock_of(state[1])
            if clk["day"] != day or clk["weekday"] != weekday or clk["tilemap"] != 0x04:
                raise RuntimeError(f"{sample} boot calendar {clk}")
            state = go(env, state, "church", church, 9000)
            state = go(env, state, "church_exit", ROUTES["d1_maria_to_town"], 6000)
            # Event01 child 6 rolls RNG once per town load. A zero skips Eve
            # for that visit. Leave and re-enter before 16:00 for another roll.
            if sample == "weekday" and "eve" not in active["found"]:
                for roll in range(5):
                    if "eve" in active["found"]:
                        break
                    if int(clock_of(state[1])["hour"]) >= 16:
                        break
                    state = go(env, state, f"gate{roll}", gate, 4000)
                    state = go(env, state, f"back{roll}", back, 4000)
            if sample == "weekday" and "ann" not in active["found"]:
                px = int(clock_of(state[1])["x"])
                prefix = gate_west + tool_plaza if px > 500 or px < 80 else tool_plaza
                state = go(env, state, "tool", prefix + tool_tail, 7000)
            if sample in ("saturday", "sunday") and (
                "eve" not in active["found"] or "nina" not in active["found"]
            ):
                state = go(env, state, "gate", gate, 8000)
                state = go(env, state, "path", path, 8000)
            if sample == "saturday" and "nina" not in active["found"]:
                state = go(env, state, "cave", cave, 24000)
            missing = [name for name in ROMANCE_NPCS if name not in active["found"]]
            per_sample[sample] = sorted(active["found"])
            day_logs.append(
                {
                    "sample": sample,
                    "state": state_name,
                    "found": sorted(active["found"]),
                    "missing": missing,
                    "end": clock_of(state[1]),
                }
            )
            print(f"DAY {sample} found={sorted(active['found'])} missing={missing}", flush=True)
        except Exception:
            traceback.print_exc()
            missing = [name for name in ROMANCE_NPCS if name not in active["found"]]
            per_sample.setdefault(sample, sorted(active["found"]))
            day_logs.append(
                {
                    "sample": sample,
                    "state": state_name,
                    "found": sorted(active["found"]),
                    "missing": missing,
                    "error": "exception",
                }
            )
            print(f"DAY {sample} EXC found={sorted(active['found'])} missing={missing}", flush=True)
        finally:
            if env is not None:
                env.close()

    live_names = {
        f"0x{row['sprite']:04X}": row["name"] for row in sightings
    }
    return {
        "frames": frames_total,
        "log_sprite_names": {
            f"0x{sprite:04X}": name for sprite, name in sorted(log_sprite_names.items())
        },
        "sprite_names": live_names,
        "days": day_logs,
        "girls_per_sample": per_sample,
        "sightings": sightings,
        "carpenter": [],
    }

if __name__ == "__main__":
    _probe_main()
