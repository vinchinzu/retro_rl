"""Social skills: locate girls, talk, gift, hold hearts, project happiness.

Heart targets are 496–511. The wife's hour-18 entry on the last night starts
at 498–511 so F0's −2 lands in range. Other girls are unchanged that night.
Gift numbers are the F9 primary ladders. Disliked items tune downward only.
Berry happiness stops at exactly 10 eaten. No RAM writes.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from harvest.calendar_schedule import GIRL_SIGHTINGS, GirlSighting, festival_on, shop_open
from harvest.calendar_schedule import SHOP_NAMES
from harvest.core.npc_catalog import ROMANCE_NPCS
from harvest.core.ranch_master import CounterTarget, WindowGuard

HEART_LO = 496
HEART_HI = 511
WIFE_ENTRY_LO = 498
WIFE_ENTRY_HI = 511
WIFE_HOUR18 = -2
HEART_CAP = 999
QUIET_HOUR = 17  # F9: hour 17–18 talk adds nothing
BERRY_TARGET = 10
BERRY_HAPPINESS = 10
SOCIAL_TRIP_BUDGET = 20_000
FRAMES_SHOP = 1_600
FRAMES_GIRL = 1_800
SAMPLES = ("weekday", "saturday", "sunday")

# Primary ladders only (F9). Wife and festival copies differ; do not merge them.
_GIFTS: dict[str, tuple[tuple[int, int, int], ...]] = {
    "maria": (
        (0x02, 0x02, -8),
        (0x09, 0x0C, -2),
        (0x06, 0x06, 3),
        (0x19, 0x19, 4),
        (0x49, 0x49, 4),
        (0x01, 0x01, 5),
        (0x03, 0x05, 5),
        (0x10, 0x17, 5),
    ),
    "ann": (
        (0x02, 0x02, -8),
        (0x09, 0x0C, -2),
        (0x19, 0x19, 2),
        (0x49, 0x49, 2),
        (0x01, 0x01, 6),
        (0x03, 0x07, 6),
        (0x10, 0x17, 6),
    ),
    "nina": (
        (0x18, 0x18, -6),
        (0x02, 0x02, -4),
        (0x09, 0x0C, -4),
        (0x05, 0x06, 8),
        (0x19, 0x19, 6),
        (0x01, 0x01, 4),
        (0x03, 0x04, 4),
        (0x07, 0x07, 4),
        (0x10, 0x17, 4),
    ),
    "ellen": (
        (0x02, 0x02, -8),
        (0x09, 0x0C, -4),
        (0x14, 0x17, 8),
        (0x19, 0x19, 5),
        (0x49, 0x49, 5),
        (0x01, 0x01, 6),
        (0x03, 0x05, 6),
        (0x07, 0x07, 6),
        (0x10, 0x13, 6),
    ),
    "eve": (
        (0x02, 0x02, -8),
        (0x07, 0x07, -8),
        (0x09, 0x0C, -4),
        (0x06, 0x06, 8),
        (0x19, 0x19, 8),
        (0x49, 0x49, 8),
        (0x01, 0x01, 5),
        (0x03, 0x05, 5),
        (0x10, 0x17, 5),
    ),
}

# Text → heart delta when the clock is before the quiet hour. F9 measured
# intros are +0. Maria's "believe in God" choice is cursor 0 → 0x0087 (+8)
# and cursor 1 → 0x0088 (+2).
TALK_TEXT = {
    0x00F9: 0,
    0x00FA: 0,
    0x0103: 0,
    0x0107: 0,
    0x0082: 0,
    0x0086: 0,
    0x0087: 8,
    0x0088: 2,
    0x043D: 0,
    0x0096: 0,
    0x009E: 0,
    0x043E: 0,
    0x00AE: 0,
    0x00B5: 0,
    0x043F: 0,
    0x00C5: 0,
    0x00CB: 0,
    0x0440: 0,
    0x00E1: 0,
    0x00DF: 0,
    0x0441: 0,
    0x0391: 0,
}

# Cursor 0 is the larger Maria add. Ellen's 0x00CB choice is the same +8 / +2
# pair; F9 does not name her result texts, so the cursors follow Maria's order.
CHOICES = {
    "maria": ((0, 8, 0x0087), (1, 2, 0x0088)),
    "ellen": ((0, 8, None), (1, 2, None)),
}

# F9 −30 sites. Skipped unless a wife-descent plan asks for one.
MINUS_30 = (
    (0xB3F167, "maria"),
    (0xB3F1C0, "ann"),
    (0xB48165, "maria"),
    (0xB4816E, "ann"),
    (0xB48177, "nina"),
    (0xB48180, "eve"),
    (0xB484FF, "maria"),
    (0xB48508, "ann"),
    (0xB48511, "nina"),
    (0xB4851A, "ellen"),
    (0xB48653, "maria"),
    (0xB4865C, "ann"),
    (0xB48665, "ellen"),
    (0xB4866E, "eve"),
    (0xB48737, "maria"),
    (0xB48740, "nina"),
    (0xB48749, "ellen"),
    (0xB48752, "eve"),
    (0xB4A6D4, "maria"),
    (0xB4A72D, "ann"),
)

# One positive decision per F9 class. −30 is avoid.
F9_EVENTS = (
    {"id": "intro_ann", "kind": "talk", "delta": 0},
    {"id": "intro_nina", "kind": "talk", "delta": 0},
    {"id": "intro_maria", "kind": "talk", "delta": 0},
    {"id": "spring_talk", "kind": "choice", "delta": 8},
    {"id": "ellen_choice", "kind": "choice", "delta": 8},
    {"id": "gift_liked", "kind": "gift", "delta": 1},
    {"id": "gift_disliked", "kind": "tune_down", "delta": -1},
    {"id": "hour18_wife", "kind": "drift", "delta": -2},
    {"id": "day30_scanner", "kind": "untraced", "delta": 0},
    {"id": "oneshot_100", "kind": "accept", "delta": 100},
    {"id": "oneshot_50", "kind": "accept", "delta": 50},
    {"id": "oneshot_30", "kind": "accept", "delta": 30},
    {"id": "minus_30", "kind": "avoid", "delta": -30},
)

FESTIVALS = (
    {"name": "Flower Festival", "map": 0x08, "choice": "dance_positive"},
    {"name": "Harvest Festival", "map": 0x09, "choice": "positive"},
    {"name": "Egg Festival", "map": 0x0B, "choice": "contest"},
    {"name": "Thanksgiving", "map": None, "choice": "positive"},
    {"name": "Star Night", "map": 0x0A, "choice": "talk_after_18"},
    {"name": "New Year", "map": 0x3A, "choice": "talk_after_18"},
)

# Sinks the accumulator refuses. Contribution on the plan is 0.
GUARDED_SINKS = (
    ("saturday_no", -20),
    ("diary_cursor0", -1),
    ("group17_cursor0", -30),
    ("group18_item2", -20),
    ("group22_low_hearts", -50),
    ("cow_funeral", -100),
    ("cow_removal", -50),
    ("chicken_removal", -20),
    ("cow_sick", -30),
    ("chicken_death", -30),
)

PAINT_BITS = (0x0400, 0x0800, 0x1000, 0x0200)
SAMPLE_PINS = {
    "weekday": "LN_mgr-calendar_whatif_town_wd",
    "saturday": "LN_mgr-calendar_whatif_town_sa",
    "sunday": "LN_mgr-calendar_whatif_town_su",
}


def gift_delta(girl: str, item: int) -> int:
    """Heart add for one primary-ladder gift. Unknown items add 0."""

    for lo, hi, delta in _GIFTS[girl]:
        if lo <= int(item) <= hi:
            return int(delta)
    return 0


def _positives(girl: str) -> list[tuple[int, int]]:
    seen: dict[int, int] = {}
    for lo, hi, delta in _GIFTS[girl]:
        if delta <= 0:
            continue
        for item in range(lo, hi + 1):
            seen.setdefault(delta, item)
    return sorted((item, delta) for delta, item in seen.items())


def _negatives(girl: str) -> list[tuple[int, int]]:
    seen: dict[int, int] = {}
    for lo, hi, delta in _GIFTS[girl]:
        if delta >= 0:
            continue
        for item in range(lo, hi + 1):
            seen.setdefault(delta, item)
    return sorted(((item, delta) for delta, item in seen.items()), key=lambda row: -row[1])


def _steps_into(start: int, delta: int, lo: int, hi: int) -> int | None:
    if delta <= 0 or start > hi:
        return None
    if lo <= start <= hi:
        return 0
    need = lo - start
    n = (need + delta - 1) // delta
    landed = start + n * delta
    if n > 0 and landed <= hi:
        return n
    return None


def choose_gift(girl: str, hearts: int, *, lo: int, hi: int) -> tuple[int, int] | None:
    """One gift, or None when every legal gift would leave the band.

    Liked items climb. Disliked items are used only while hearts sit above
    the band, and only when the step does not fall through the floor.
    """

    hearts = int(hearts)
    if lo <= hearts <= hi:
        return None
    if hearts < lo:
        picks = []
        for item, delta in _positives(girl):
            n = _steps_into(hearts, delta, lo, hi)
            if n:
                picks.append((delta, n, item))
        if not picks:
            return None
        picks.sort()
        _delta, _n, item = picks[0]
        return item, gift_delta(girl, item)
    for item, delta in _negatives(girl):
        nxt = hearts + delta
        if nxt < lo:
            continue
        return item, delta
    return None


def talk_delta(text_id: int, hour: int) -> int | None:
    """F9 delta, or None when this text is not in the table.

    Hour >= 17 forces 0 even if the daytime line would add hearts.
    """

    if int(hour) >= QUIET_HOUR:
        return 0
    if int(text_id) not in TALK_TEXT:
        return None
    return TALK_TEXT[int(text_id)]


def talk_matches(text_id: int, hour: int, delta: int, *, dialogue: bool) -> bool:
    """True when a real line's heart change is one F9 allows."""

    if not dialogue:
        return False
    expected = talk_delta(text_id, hour)
    if expected is None:
        return int(delta) == 0
    return int(delta) == expected


def positive_cursor(girl: str, hearts: int, *, lo: int = HEART_LO, hi: int = HEART_HI) -> int:
    """Cursor for the larger add that still fits. Default is Maria's +8."""

    options = CHOICES.get(girl, ((0, 8, None), (1, 2, None)))
    fitting = [row for row in options if lo <= hearts + row[1] <= hi or hearts + row[1] <= hi]
    if hearts < lo:
        fitting = [row for row in options if hearts + row[1] <= hi]
    if not fitting:
        return min(options, key=lambda row: abs(row[1]))[0]
    return max(fitting, key=lambda row: row[1])[0]


def event_decision(event_id: str, *, wife_descent: bool = False) -> str:
    """What the skill does with one F9 class. −30 is skipped by default."""

    for row in F9_EVENTS:
        if row["id"] != event_id:
            continue
        if row["kind"] == "avoid" and not wife_descent:
            return "skip"
        if row["kind"] == "avoid" and wife_descent:
            return "take"
        if row["kind"] == "tune_down":
            return "down_only"
        if row["kind"] == "choice":
            return "positive"
        if row["kind"] == "untraced":
            return "ignore"
        return "take"
    raise KeyError(event_id)


def minus_30_skipped(address: int, *, wife_descent: bool = False) -> bool:
    known = {pc for pc, _girl in MINUS_30}
    if address not in known:
        return False
    return not wife_descent


def festival_plan(name: str) -> dict:
    for row in FESTIVALS:
        if row["name"] == name:
            return {"name": name, "choice": row["choice"], "avoid_minus_30": True, "map": row["map"]}
    raise KeyError(name)


def entry_band(girl: str, *, wife: str | None, last_night: bool) -> tuple[int, int]:
    """Pre-drift band. The wife, on the last night only, starts at 498–511."""

    if wife and girl == wife and last_night:
        return WIFE_ENTRY_LO, WIFE_ENTRY_HI
    return HEART_LO, HEART_HI


def apply_hour18(hearts: int, *, wife_last_night: bool) -> int:
    if not wife_last_night:
        return int(hearts)
    return max(0, min(HEART_CAP, int(hearts) + WIFE_HOUR18))


def wife_entry_holds(hearts: int) -> bool:
    """Last-night hour-18 entry must be 498–511 so −2 stays inside 496–511."""

    if not WIFE_ENTRY_LO <= int(hearts) <= WIFE_ENTRY_HI:
        return False
    landed = apply_hour18(hearts, wife_last_night=True)
    return HEART_LO <= landed <= HEART_HI


@dataclass(frozen=True)
class HeartWhatIf:
    girl: str
    start: int
    landed: int
    daily: tuple[int, ...]
    gifts: tuple[tuple[int, int], ...]
    held: bool


def climb_and_hold(
    girl: str,
    start: int = 470,
    *,
    days: int = 10,
    wife: str | None = None,
) -> HeartWhatIf:
    """Climb into the band, then hold ``days`` nights.

    Non-wives take F0's measured drift of 0. The wife loses 2 only on the
    last night's hour-18 entry. A gift that would leave the band is refused.
    """

    last_band = entry_band(girl, wife=wife, last_night=True)
    hearts = int(start)
    gifts: list[tuple[int, int]] = []
    for _ in range(48):
        if last_band[0] <= hearts <= last_band[1]:
            break
        choice = choose_gift(girl, hearts, lo=last_band[0], hi=last_band[1])
        if choice is None:
            break
        item, delta = choice
        guard = WindowGuard(CounterTarget(last_band[0], last_band[1]))
        if int(delta) > 0 and not guard.allow(hearts, delta):
            break
        nxt = max(0, min(HEART_CAP, hearts + delta))
        if girl == wife:
            if not (last_band[0] <= nxt <= last_band[1]) and nxt > last_band[1]:
                break
        elif nxt > HEART_HI:
            break
        hearts = nxt
        gifts.append((item, delta))
    landed = hearts
    daily: list[int] = []
    for index in range(int(days)):
        last = index == days - 1
        if last and girl == wife:
            night = WindowGuard(CounterTarget(HEART_LO, HEART_HI), automatic_delta=WIFE_HOUR18)
            if night.allow(hearts, 0):
                hearts = apply_hour18(hearts, wife_last_night=True)
        daily.append(hearts)
    in_band = all(HEART_LO <= value <= HEART_HI for value in daily)
    if girl == wife:
        in_band = in_band and WIFE_ENTRY_LO <= landed <= WIFE_ENTRY_HI
    else:
        in_band = in_band and HEART_LO <= landed <= HEART_HI
    return HeartWhatIf(girl, int(start), landed, tuple(daily), tuple(gifts), in_band and len(daily) == days)


def four_girl_whatif(start: int = 470, *, days: int = 10, wife: str = "maria") -> dict[str, HeartWhatIf]:
    """The four girls who are not the wife. Maria is the F0 measured wife."""

    girls = tuple(name for name in ROMANCE_NPCS if name != wife)
    if len(girls) != 4:
        raise RuntimeError(f"expected 4 non-wives, got {girls}")
    return {name: climb_and_hold(name, start, days=days, wife=wife) for name in girls}


def _snap_int(snap: dict, key: str) -> int:
    return int(snap.get(key, 0) or 0)


def happiness_sources(snap: dict) -> list[tuple[str, int]]:
    """Repeatable and dated positives still ahead of this checkpoint.

    Berries stop at 10. Nine calf namings are scheduled when the barn has
    room and fewer than nine cows already exist (F7). Sinks are not listed
    here; see ``guarded_sinks``.
    """

    rows: list[tuple[str, int]] = []
    berries = _snap_int(snap, "power_berry_count")
    if berries < BERRY_TARGET:
        rows.append(("berries", (BERRY_TARGET - berries) * BERRY_HAPPINESS))
    flags = _snap_int(snap, "upgrade_flags")
    if not flags & 0x40:
        rows.append(("house_1", 50))
    if not flags & 0x80:
        rows.append(("house_2", 50))
    if _snap_int(snap, "kid1_age") < 60:
        rows.append(("child1_age60", 50))
    if _snap_int(snap, "kid2_age") < 60:
        rows.append(("child2_age60", 100))
    flags_5a = _snap_int(snap, "event_flags_1f5a")
    if not flags_5a & 0x0022:
        rows.append(("first_cow", 20))
    if not flags_5a & 0x0020:
        rows.append(("first_chicken", 10))
    marriage = _snap_int(snap, "marriage_flags")
    paint = sum(10 for bit in PAINT_BITS if not marriage & bit)
    if paint:
        rows.append(("paint", paint))
    if not _snap_int(snap, "event_flags_1f68") & 0x4000:
        rows.append(("saturday_yes", 20))
    if not _snap_int(snap, "incubator_flags") & 0x2000:
        rows.append(("incubator", 5))
    cows = _snap_int(snap, "num_cows")
    room = max(0, 12 - cows)
    births = max(0, min(room, 9 - cows))
    if births:
        rows.append(("calf_namings", births * 70))
    return rows


def guarded_sinks() -> tuple[tuple[str, int], ...]:
    """Named sinks and the amount the plan actually applies (always 0)."""

    return tuple((name, 0) for name, _delta in GUARDED_SINKS)


def project_happiness(snap: dict) -> dict:
    """Current happiness plus scheduled sources, clamped at 999."""

    current = _snap_int(snap, "happiness")
    sources = happiness_sources(snap)
    added = sum(delta for _name, delta in sources)
    total = min(HEART_CAP, max(0, current + added))
    return {
        "current": current,
        "added": added,
        "projected": total,
        "sources": sources,
        "sinks": guarded_sinks(),
        "reaches_992": total >= 992,
        "berries_planned": BERRY_TARGET,
    }


def project_pin(state_name: str) -> dict:
    """Read one savestate and project. Does not boot the emulator."""

    from harvest.core.harvest_state import HarvestStateDocument

    doc = HarvestStateDocument.load(state_name)
    keys = (
        "year",
        "season",
        "day",
        "weekday",
        "hour",
        "happiness",
        "power_berry_count",
        "upgrade_flags",
        "marriage_flags",
        "event_flags_1f5a",
        "event_flags_1f68",
        "incubator_flags",
        "num_cows",
        "num_chickens",
        "kid1_age",
        "kid2_age",
    )
    snap = {key: doc.scalar_value(key) for key in keys}
    out = project_happiness(snap)
    out["state"] = state_name
    out["snap"] = snap
    out["emulator_frames"] = 0
    return out


def sample_targets(sample: str) -> tuple[GirlSighting, ...]:
    """One measured sighting per girl for this sample day. Missing names raise."""

    by: dict[str, GirlSighting] = {}
    for row in GIRL_SIGHTINGS:
        if row.sample == sample and row.name not in by:
            by[row.name] = row
    missing = [name for name in ROMANCE_NPCS if name not in by]
    if missing:
        raise KeyError(f"{sample} missing {missing}")
    return tuple(by[name] for name in ROMANCE_NPCS)


def pack_social_trip(
    *,
    year: int,
    season: int,
    day: int,
    weekday: int,
    hour: int,
    weather_flags: int = 0,
) -> dict:
    """One town trip: open shops, then each girl with her gift.

    Shops that the door flags refuse are left out. A festival day is one
    festival stop instead of the storefronts. Frame sum must fit the budget.
    """

    fest = festival_on(year, season, day)
    stops: list[dict] = []
    frames = 0
    if fest is None:
        for name in SHOP_NAMES:
            if shop_open(
                name,
                weekday=weekday,
                hour=hour,
                season=season,
                day=day,
                weather_flags=weather_flags,
            ):
                stops.append({"kind": "shop", "name": name, "frames": FRAMES_SHOP})
                frames += FRAMES_SHOP
    else:
        stops.append({"kind": "festival", "name": fest.name, "frames": FRAMES_GIRL})
        frames += FRAMES_GIRL
    for name in ROMANCE_NPCS:
        stops.append({"kind": "girl", "name": name, "gift": True, "frames": FRAMES_GIRL})
        frames += FRAMES_GIRL
    return {
        "trips": 1,
        "stops": stops,
        "frames": frames,
        "budget": SOCIAL_TRIP_BUDGET,
        "fits": frames <= SOCIAL_TRIP_BUDGET,
        "girls": list(ROMANCE_NPCS),
        "festival": None if fest is None else fest.name,
    }


def _recordings() -> Path:
    return Path(__file__).resolve().parents[2] / "recordings"


def write_report(name: str, payload: dict) -> Path:
    path = _recordings() / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return path


# --- live locator (one env at a time; savestate load only) -----------------

_REACHED_PX = 20
_TALK_PX = 32


def run_locator(*, talk_goal: int = 10) -> dict:
    """Approach all 5 girls on the 3 measured sample pins. 0 RAM writes."""

    import numpy as np
    from retro_harness import TaskStatus, WorldState

    from harvest.core.npc_catalog import game_objects
    from harvest.core.ram_catalog import read_ram_value
    from harvest.maps.map_routes import ROUTES, densify_waypoints
    from harvest.maps.map_types import Waypoint
    from harvest.planner.tasks.multi_nav import MultiMapNavTask
    from harvest.runtime.retro_setup import make_harvest_env
    from harvest.tasks.nav import get_pos_from_ram, make_action

    frames_total = 0
    talks: list[dict] = []
    days: list[dict] = []

    def clock_of(ram: np.ndarray) -> dict:
        keys = ("year", "season", "weekday", "day", "hour", "minute", "tilemap")
        row = {key: int(read_ram_value(ram, key)) for key in keys}
        row["weather_flags"] = int(ram[0x196]) | (int(ram[0x197]) << 8)
        pos = get_pos_from_ram(ram)
        row["x"] = int(pos.x)
        row["y"] = int(pos.y)
        return row

    def hearts_of(ram: np.ndarray, girl: str) -> int:
        return int(read_ram_value(ram, f"{girl}_hearts"))

    def hops(tilemap: int, points: list[tuple[int, int]], exit_direction: str | None = None):
        if (744, 424) in points:
            raise RuntimeError("east-gate landing is not a hop")
        dense = densify_waypoints(
            [Waypoint(tilemap=tilemap, target_px=(int(x), int(y)), radius=14) for x, y in points]
        )
        if exit_direction is not None:
            dense[-1] = dense[-1]._replace(is_exit=True, exit_direction=exit_direction, radius=8)
        return list(dense)

    church = hops(
        0x04,
        [(680, 120), (680, 200), (680, 280), (520, 280), (376, 280), (376, 200), (358, 150)],
        "up",
    )
    church_exit = list(ROUTES["d1_maria_to_town"])
    gate = hops(
        0x04,
        [(376, 280), (560, 280), (684, 288), (684, 340), (684, 400), (692, 424), (728, 424)],
        "right",
    )
    path = hops(0x0C, [(80, 128), (132, 128), (132, 70), (132, 30)], "up")
    back = hops(0x0C, [(80, 128), (40, 128)], "left")
    gate_west = hops(0x04, [(728, 424), (692, 424), (684, 400), (684, 340), (684, 288)])
    tool_plaza = hops(0x04, [(376, 280), (560, 280), (680, 280), (684, 360), (688, 430)])
    tool_tail = list(ROUTES["d1_town_to_ann"][1:])
    tool_tail += hops(0x04, [(408, 914), (408, 888)], "up")
    tool_tail[-1] = tool_tail[-1]._replace(radius=16)
    cave = []
    for wp in ROUTES["mountain_entry_to_first_berry"]:
        cave.append(wp)
        if wp.target_px == (144, 448):
            break
    cave.append(
        Waypoint(tilemap=0x10, target_px=(166, 430), radius=10, is_exit=True, exit_direction="up")
    )

    def boot(state_name: str):
        env = make_harvest_env(state_name, render_mode=None, lean_step=True)
        _obs, info = env.reset()
        ram = np.asarray(env.get_ram(), dtype=np.uint8)
        print(f"BOOT {state_name} {clock_of(ram)}", flush=True)
        return env, [0, ram, info, None]

    def step_env(env, state, action):
        nonlocal frames_total
        frame, _ram, _info, _obs = state
        frame += 1
        frames_total += 1
        obs, _reward, _terminated, _truncated, info = env.step(action)
        ram = np.asarray(env.get_ram(), dtype=np.uint8)
        state[0] = frame
        state[1] = ram
        state[2] = info
        state[3] = obs
        return state

    def drive(env, state, waypoints, budget: int, halt=None):
        if not waypoints:
            return state, "empty"
        if int(read_ram_value(state[1], "tilemap")) != int(waypoints[0].tilemap):
            print(
                f"SKIP map=0x{int(read_ram_value(state[1], 'tilemap')):02X} "
                f"want=0x{int(waypoints[0].tilemap):02X}",
                flush=True,
            )
            return state, "skip"
        task = MultiMapNavTask(waypoints=list(waypoints), timeout=budget)
        task.reset(WorldState(state[0], state[1], state[2], state[3]))
        status = "budget"
        for _ in range(budget):
            result = task.step(WorldState(state[0], state[1], state[2], state[3]))
            action = make_action()
            if result.action is not None and result.action.action is not None:
                action = result.action.action
            state = step_env(env, state, action)
            if halt is not None and halt(state):
                return state, "near"
            if result.status != TaskStatus.RUNNING:
                status = result.status.name if hasattr(result.status, "name") else str(result.status)
                break
            if frames_total % 2000 == 0:
                clk = clock_of(state[1])
                print(f"FRAME {frames_total} map=0x{clk['tilemap']:02X} ({clk['x']},{clk['y']})", flush=True)
        return state, status

    def sprite_of(ram, sighting: GirlSighting):
        tilemap = int(read_ram_value(ram, "tilemap"))
        if tilemap != int(sighting.tilemap):
            return None
        best = None
        best_key = None
        for obj in game_objects(ram):
            if obj.is_player or obj.kind != "npc_candidate":
                continue
            delta = abs(int(obj.sprite_table_idx) - int(sighting.sprite))
            if delta > 8:
                continue
            dist = max(abs(obj.pixel[0] - sighting.pixel_x), abs(obj.pixel[1] - sighting.pixel_y))
            key = (dist, delta)
            if best_key is None or key < best_key:
                best_key = key
                best = obj
        return best

    def spot_of(obj) -> tuple[int, int]:
        return int(obj.pixel[0]), int(obj.pixel[1]) + 27

    def dist_to(ram, spot: tuple[int, int]) -> int:
        pos = get_pos_from_ram(ram)
        return max(abs(int(pos.x) - spot[0]), abs(int(pos.y) - spot[1]))

    def dismiss_and_log(env, state, girl: str, face: dict | None = None) -> bool:
        if len(talks) >= talk_goal:
            return False
        ram = state[1]
        if int(read_ram_value(ram, "game_state")) & 0x0800:
            return False
        for _ in range(24):
            if int(read_ram_value(state[1], "input_lock")) == 1:
                break
            if int(read_ram_value(state[1], "game_state")) & 0x0800:
                return False
            state = step_env(env, state, make_action(a=True))
            state = step_env(env, state, make_action())
        ram = state[1]
        if int(read_ram_value(ram, "input_lock")) != 1:
            return False
        before = hearts_of(ram, girl)
        hour = int(read_ram_value(ram, "hour"))
        text_before = int(read_ram_value(ram, "dialog_text_id"))
        held = dict(face or {})
        held["a"] = True
        for _ in range(6):
            if int(read_ram_value(state[1], "game_state")) & 0x0800:
                return False
            state = step_env(env, state, make_action(**held))
        state = step_env(env, state, make_action())
        saw_lock = False
        text_id = text_before
        for i in range(420):
            ram = state[1]
            lock = int(read_ram_value(ram, "input_lock"))
            text_id = int(read_ram_value(ram, "dialog_text_id"))
            action_id = int(read_ram_value(ram, "player_action"))
            if lock != 1:
                saw_lock = True
            if action_id == 9 and int(read_ram_value(ram, "dialog_menu_cursor")) != 0:
                state = step_env(env, state, make_action(up=True))
                continue
            pulse = saw_lock and i % 2 == 0
            state = step_env(env, state, make_action(a=pulse))
            if saw_lock and lock == 1 and i > 8:
                break
        after = hearts_of(state[1], girl)
        dialogue = saw_lock and text_id != 0
        row = {
            "girl": girl,
            "text_id": text_id,
            "hour": hour,
            "before": before,
            "after": after,
            "delta": after - before,
            "dialogue": dialogue,
            "match": talk_matches(text_id, hour, after - before, dialogue=dialogue),
        }
        if dialogue:
            talks.append(row)
            print(f"TALK {row}", flush=True)
        return dialogue

    def legs_for(state, sighting: GirlSighting):
        tm = int(sighting.tilemap)
        px = int(sighting.pixel_x)
        py = int(sighting.pixel_y) + 27
        if tm in (0x10, 0x29):
            return []
        if tm == 0x04 and py > 640:
            pos = get_pos_from_ram(state[1])
            if int(pos.y) >= 700:
                aim = "right" if px >= int(pos.x) else "left"
                return [
                    Waypoint(tilemap=0x04, target_px=(px, int(pos.y)), radius=14, run_direction=aim, force_run=True),
                    Waypoint(tilemap=0x04, target_px=(px, py), radius=12, run_direction="up", force_run=True),
                ]
            return [
                Waypoint(tilemap=0x04, target_px=(688, 760), radius=16, run_direction="down", force_run=True),
                Waypoint(tilemap=0x04, target_px=(px, 760), radius=14, run_direction="left", force_run=True),
                Waypoint(tilemap=0x04, target_px=(px, py), radius=12, run_direction="down", force_run=True),
            ]
        return [Waypoint(tilemap=tm, target_px=(px, py), radius=16)]

    def approach(env, state, sighting: GirlSighting, *, nav: bool = True):
        best = 9999
        talked = False
        status = "walk"
        if nav:
            legs = legs_for(state, sighting)
            if legs:
                def halt(state, sighting=sighting):
                    live = sprite_of(state[1], sighting)
                    return live is not None and dist_to(state[1], spot_of(live)) <= 36

                state, status = drive(env, state, legs, 6000, halt=halt)
        stuck = 0
        last = None
        axis = 0
        for _ in range(700):
            obj = sprite_of(state[1], sighting)
            spot = spot_of(obj) if obj is not None else (int(sighting.pixel_x), int(sighting.pixel_y) + 27)
            dist = dist_to(state[1], spot)
            if obj is not None:
                best = min(best, dist)
            pos = get_pos_from_ram(state[1])
            dx, dy = spot[0] - int(pos.x), spot[1] - int(pos.y)
            face: dict = {}
            if abs(dx) > abs(dy) and dx:
                face["right" if dx > 0 else "left"] = True
            elif dy:
                face["down" if dy > 0 else "up"] = True
            if obj is not None and dist <= _TALK_PX and not talked:
                if dismiss_and_log(env, state, sighting.name, face):
                    talked = True
            if obj is not None and (dist <= _REACHED_PX or talked):
                return state, True, best
            here = (int(pos.x), int(pos.y))
            stuck = stuck + 1 if here == last else 0
            last = here
            if stuck >= 16:
                axis ^= 1
                stuck = 0
            use_x = (abs(dx) > abs(dy)) ^ bool(axis)
            if use_x and dx:
                action = make_action(right=dx > 0, left=dx < 0, b=True)
            else:
                action = make_action(down=dy > 0, up=dy < 0, b=True)
            if int(read_ram_value(state[1], "game_state")) & 0x0800:
                action = make_action()
            state = step_env(env, state, action)
        print(f"MISS {sighting.sample} {sighting.name} dist={best} nav={status}", flush=True)
        return state, False, best

    def go(env, state, label: str, waypoints, budget: int):
        state, status = drive(env, state, waypoints, budget)
        print(f"GO {label} {status} map=0x{int(read_ram_value(state[1], 'tilemap')):02X}", flush=True)
        return state

    for sample in SAMPLES:
        pin = SAMPLE_PINS[sample]
        targets = sample_targets(sample)
        reached: dict[str, int] = {}
        env = None
        try:
            env, state = boot(pin)
            state = go(env, state, "church", church, 9000)
            on_church = [row for row in targets if row.tilemap == 0x1B]
            for row in on_church:
                state, ok, dist = approach(env, state, row)
                if ok:
                    reached[row.name] = dist
            if int(read_ram_value(state[1], "tilemap")) == 0x1B:
                state = go(env, state, "church_exit", church_exit, 6000)
            for row in targets:
                if row.name in reached or row.tilemap != 0x04:
                    continue
                state, ok, dist = approach(env, state, row)
                if ok:
                    reached[row.name] = dist
            if sample == "weekday" and "eve" not in reached:
                for roll in range(5):
                    if "eve" in reached or int(clock_of(state[1])["hour"]) >= 16:
                        break
                    state = go(env, state, f"gate{roll}", gate, 4000)
                    state = go(env, state, f"back{roll}", back, 4000)
                    eve = next(row for row in targets if row.name == "eve")
                    if int(read_ram_value(state[1], "tilemap")) == eve.tilemap:
                        state, ok, dist = approach(env, state, eve)
                        if ok:
                            reached["eve"] = dist
            if "ann" not in reached and any(row.name == "ann" and row.tilemap == 0x22 for row in targets):
                px = int(clock_of(state[1])["x"])
                prefix = gate_west + tool_plaza if px > 500 or px < 80 else tool_plaza
                state = go(env, state, "tool", prefix + tool_tail, 8000)
                ann = next(row for row in targets if row.name == "ann")
                state, ok, dist = approach(env, state, ann)
                if ok:
                    reached["ann"] = dist
            mountain = [row for row in targets if row.tilemap in (0x10, 0x29) and row.name not in reached]
            if mountain:
                state = go(env, state, "gate", gate, 8000)
                state = go(env, state, "path", path, 8000)
                def take(rows):
                    nonlocal state
                    for row in rows:
                        if row.name in reached:
                            continue
                        if int(read_ram_value(state[1], "tilemap")) != int(row.tilemap):
                            continue
                        state, ok, dist = approach(env, state, row, nav=False)
                        if ok:
                            reached[row.name] = dist

                ridge = [row for row in mountain if row.tilemap == 0x10]
                if ridge and int(read_ram_value(state[1], "tilemap")) == 0x10:
                    state = go(env, state, "spa", list(ROUTES["mountain_entry_to_outdoor_spa"]), 22000)
                    take([row for row in ridge if int(row.pixel_x) >= 300])
                    if any(int(row.pixel_x) < 300 and row.name not in reached for row in ridge):
                        west = [
                            Waypoint(tilemap=0x10, target_px=(x, 201), radius=12, run_direction="left", force_run=True)
                            for x in (480, 280, 120, 48)
                        ]
                        state = go(env, state, "ridge", west, 8000)
                    take([row for row in ridge if int(row.pixel_x) < 300])
                if any(row.tilemap == 0x29 and row.name not in reached for row in mountain):
                    state = go(env, state, "cave", cave, 24000)
                    take([row for row in mountain if row.tilemap == 0x29])
            missing = [name for name in ROMANCE_NPCS if name not in reached]
            days.append(
                {
                    "sample": sample,
                    "pin": pin,
                    "reached": sorted(reached),
                    "distances": reached,
                    "missing": missing,
                    "end": clock_of(state[1]),
                }
            )
            print(f"DAY {sample} reached={sorted(reached)} missing={missing}", flush=True)
        except Exception as exc:
            import traceback

            traceback.print_exc()
            missing = [name for name in ROMANCE_NPCS if name not in reached]
            days.append(
                {
                    "sample": sample,
                    "pin": pin,
                    "reached": sorted(reached),
                    "missing": missing,
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
            print(f"DAY {sample} EXC {type(exc).__name__}: {exc}", flush=True)
        finally:
            if env is not None:
                env.close()

    matched = [row for row in talks if row["match"]]
    misses = sum(len(day.get("missing") or []) for day in days)
    report = {
        "frames": frames_total,
        "days": days,
        "talks": talks,
        "talks_matched": len(matched),
        "misses": misses,
        "girls_ok": misses == 0
        and len(days) == 3
        and all(not day.get("error") and len(day.get("reached") or []) == 5 for day in days),
        "emulator_frames": frames_total,
        "ram_writes": 0,
    }
    path = write_report("pr067_locator.json", report)
    report["json"] = str(path)
    print(
        f"{'DONE' if report['girls_ok'] else 'FAILED'} frames={frames_total} "
        f"misses={misses} talks={len(matched)}/{len(talks)} json={path}",
        flush=True,
    )
    return report


def _main() -> None:
    import sys

    report = run_locator()
    sys.exit(0 if report["girls_ok"] else 1)


if __name__ == "__main__":
    _main()
