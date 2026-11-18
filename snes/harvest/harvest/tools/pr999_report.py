"""Campaign progress report for a Perfect Rancher run JSON.

``python -m harvest.tools.pr999_report <run.json>`` prints term gaps against
the milestone deadlines, window states, and the remaining collectible chances.
The day-journal writer calls :func:`journal_census` so a checkpoint journal
records counted tiles, the gap to 2,950, and the farm-worktree linter's
uncovered list. The game-owned linter is shared with farm cover tasks.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import yaml

from harvest.core.animal_status import COW_SLOT_COUNT, count_chicken_slots, existing_cow_slots
from harvest.core.carry import carry_pair_items
from harvest.core.shipping_credit import SHIPPED_CROP_FIELDS as SHIPPED_COUNTERS
from harvest.core.harvest_state import (
    CURRENT_MAP_ARRAY_ADDR,
    FARM_MAP_ARRAY_ADDR,
    HarvestStateDocument,
    is_farm_development_tile,
)
from harvest.core.ram_catalog import (
    LIVE_RAM_WRAM_OFFSET,
    WRAM_SNAPSHOT_SIZE,
    read_animal_slot_field,
    read_ram_u16,
    read_ram_u8,
    read_ram_value,
)
from harvest.core.tile_catalog import MAP_TILE_COUNT, Tool
from harvest.runtime.rom_tools import wram_offset
from harvest.planner.livestock_econ import (
    SHED_ROW_2_ADDR,
    SHED_ROW_2_BRUSH_BIT,
    SHED_ROW_2_MILKER_BIT,
    ranch_gates,
)
from harvest.planner.tasks.ending import is_ending_date
from retro_harness import WorldState

# Talk +1 and brush +3. F3: 200 on the latest morning is 50 cared days from 0.
COW_CARE_PER_DAY = 4
COW_HAPPINESS_TARGET = 200
# $7F1F70 bit $0004 is the clock arm. It has no catalog key.
CLOCK_ARM_ADDR = 0x11F70
CLOCK_ARM_BIT = 0x0004
CLOCK_SCORE_BIT = 0x4000
TURTLE_BIT = 0x1000
INDOOR_ARM_BIT = 0x2000
SEASONS = ("Spring", "Summer", "Fall", "Winter")
DAYS_PER_SEASON = 30
SEASONS_PER_YEAR = 4

SEED_BAGS = ("grass", "corn", "tomato", "potato", "turnip")

# Designed one-shots. The three duplication edges have no flag of their own.
DESIGNED_BERRY_FLAGS = {
    "berry_field_1": ("upgrade_flags", 0x0800),
    "berry_field_2": ("upgrade_flags", 0x1000),
    "berry_stump": ("upgrade_flags", 0x0400),
    "berry_statue": ("upgrade_flags", 0x0200),
    "berry_cave": ("romance_event_flags", 0x0040),
    "berry_indoor": ("family_event_flags", 0x4000),
    "berry_goddess": ("romance_event_flags", 0x0020),
}

TERM_MILESTONES = {
    "money": ("money_hold",),
    "cows": ("cow_slot0",),
    "chickens": ("hen",),
    "power_berries": ("power_berry_n",),
    "maria": ("wife_hearts_d29",),
    "ann": ("wife_hearts_d29",),
    "nina": ("wife_hearts_d29",),
    "ellen": ("wife_hearts_d29",),
    "eve": ("wife_hearts_d29",),
    "tomatoes": ("crop_tomato",),
    "corn": ("crop_corn",),
    "potatoes": ("crop_potato",),
    "turnips": ("crop_turnip",),
    "happiness": ("happiness",),
    "house": ("house_upgrade_1", "house_level_2"),
    "children": ("child_1", "child_2"),
    "marriage": ("wife_bit",),
    "clock": ("clock_arm", "clock_gift"),
    "turtle": ("turtle_shell",),
    "development": ("cover",),
    "cow_happiness": tuple(f"cow_feedable_{slot:02d}" for slot in range(1, 12)),
    "total": (),
}

_MILESTONES: dict[str, Any] | None = None


def farm_layout_path() -> Path:
    """Game-owned layout linter, available in ordinary checkouts."""
    return Path(__file__).resolve().parents[1] / "maps" / "farm_final_layout.py"


def farm_layout():
    """Use the same layout linter as cover tasks and pin inspection."""
    from harvest.maps import farm_final_layout

    return farm_final_layout


def milestones_path() -> Path:
    return Path(__file__).resolve().parents[2] / "docs" / "perfect_rancher" / "milestones.yaml"


def load_milestones() -> dict[str, Any]:
    global _MILESTONES
    if _MILESTONES is None:
        _MILESTONES = yaml.safe_load(milestones_path().read_text())
    return _MILESTONES


def milestone_rows() -> list[dict[str, Any]]:
    document = load_milestones()
    return [row for row in document.get("milestones", ()) if isinstance(row, dict)]


def milestones_by_id() -> dict[str, dict[str, Any]]:
    return {str(row["id"]): row for row in milestone_rows() if "id" in row}


def calendar_index(year: int, season: int, day: int) -> int:
    """Y1 Spring D1 morning is 0. Y3 Summer D30 morning is 299."""

    return int(year) * (SEASONS_PER_YEAR * DAYS_PER_SEASON) + int(season) * DAYS_PER_SEASON + (int(day) - 1)


def format_date(year: int, season: int, day: int) -> str:
    name = SEASONS[int(season)] if 0 <= int(season) < len(SEASONS) else str(season)
    return f"Y{int(year) + 1} {name} D{int(day)}"


def quota_index(quota: Mapping[str, Any]) -> int:
    """Milestone quotas use a 1-based year. RAM year 0 is Y1."""

    season = str(quota["season"]).strip().lower()
    season_id = SEASONS.index(season.capitalize()) if season.capitalize() in SEASONS else SEASONS.index(season.title())
    return calendar_index(int(quota["year"]) - 1, season_id, int(quota["day"]))


def _wram_image(ram: np.ndarray) -> np.ndarray:
    """Save-state WRAM image the linter indexes, from a live or saved RAM blob."""

    if len(ram) > WRAM_SNAPSHOT_SIZE:
        image = np.array(ram[LIVE_RAM_WRAM_OFFSET : LIVE_RAM_WRAM_OFFSET + WRAM_SNAPSHOT_SIZE], copy=True)
        # Catalog reads tilemap with live_offset=False, outside the mirror.
        image[0x22] = int(read_ram_value(ram, "tilemap", raw=True)) & 0xFF
        return image
    return ram


def _map_slice(image: np.ndarray, address: int) -> tuple[int, ...]:
    start = wram_offset(address)
    end = start + MAP_TILE_COUNT
    if end > len(image):
        raise ValueError(f"farm map slice {start:#x}:{end:#x} exceeds RAM {len(image)}")
    return tuple(int(tile) & 0xFF for tile in image[start:end])


def live_tiles_for_ram(ram: np.ndarray) -> tuple[int, str, tuple[int, ...]]:
    """``(tilemap, source, tiles)`` using the linter's live-versus-persistent rule."""

    layout = farm_layout()
    image = _wram_image(ram)
    tilemap = int(image[0x22]) & 0xFF
    persistent = _map_slice(image, FARM_MAP_ARRAY_ADDR)
    visible = _map_slice(image, CURRENT_MAP_ARRAY_ADDR)
    if tilemap not in layout.FARM_TILEMAP_IDS:
        return tilemap, "persistent", persistent
    tiles = tuple(
        int(layout.live_tile(tilemap, visible[index], persistent[index])) & 0xFF
        for index in range(MAP_TILE_COUNT)
    )
    return tilemap, "live", tiles


def development_census(ram: np.ndarray) -> dict[str, Any]:
    """Counted tiles, the signed gap to 2,950, and the linter's uncovered cells."""

    layout = farm_layout()
    tilemap, source, tiles = live_tiles_for_ram(ram)
    target = layout.final_layout()
    todos = layout.lint_tiles(tiles, target)
    counted = sum(1 for tile in tiles if is_farm_development_tile(int(tile)))
    minimum = int(layout.MIN_COUNTED_TILES)
    uncovered = [
        {
            "x": int(todo.x),
            "y": int(todo.y),
            "live": int(todo.live),
            "target": int(todo.target),
            "steps": list(todo.steps),
        }
        for todo in todos
    ]
    ledger = layout.terrain_ledger(
        tiles,
        cover=target.cover,
        planted_grass=int(read_ram_value(ram, "planted_grass")),
        need=minimum,
    )
    return {
        "counted_tiles": counted,
        "minimum_tiles": minimum,
        "gap_to_2950": minimum - counted,
        "uncovered": uncovered,
        "uncovered_count": len(uncovered),
        "source": source,
        "tilemap": tilemap,
        "linter": "harvest.maps.farm_final_layout",
        "ledger": ledger.as_dict(),
        "ledger_line": ledger.line(),
    }


def cow_rows(ram: np.ndarray) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for slot in range(COW_SLOT_COUNT):
        status = int(read_animal_slot_field(ram, "cow", slot, "status_raw"))
        exists = bool(status & 0x01)
        happiness = int(read_animal_slot_field(ram, "cow", slot, "happiness")) if exists else 0
        rows.append(
            {
                "slot": slot,
                "exists": exists,
                "happiness": happiness,
                "status": status,
            }
        )
    return rows


def collectible_flags(ram: np.ndarray) -> dict[str, int]:
    flags = {
        key: int(read_ram_value(ram, key, raw=True))
        for key in (
            "upgrade_flags",
            "romance_event_flags",
            "family_event_flags",
            "incubator_flags",
            "power_berry_count",
        )
    }
    flags["event_flags_1f70"] = int(read_ram_u16(ram, CLOCK_ARM_ADDR))
    return flags


def economy_snapshot(ram: np.ndarray) -> dict[str, Any]:
    """Compact animal and wallet facts for one journal checkpoint. Read-only.

    ``bin_gold`` is the shipping bin value (paid out at the night reset, so it
    is 0 on a morning read). ``shipped`` is the ROM's lifetime crop counters.
    ``chickens`` is ``num_chickens``: adults, chicks and incubating eggs.
    ``hens`` is the adult slot count, the number ``fed_chickens`` must reach.
    ``brush`` and ``milker`` are true on the shed shelf or in the carry pair.
    ``house_size`` is left out: mid-run it is a dialogue step counter.
    """

    carried = carry_pair_items(ram)
    shelf = read_ram_u8(ram, SHED_ROW_2_ADDR)
    from harvest.planner.power_berry_calendar import berry_receipts
    return {
        "power_berries": berry_receipts(ram),
        "money": int(read_ram_value(ram, "money")),
        "bin_gold": int(read_ram_value(ram, "shipping_money")),
        "cows": int(read_ram_value(ram, "num_cows")),
        "chickens": int(read_ram_value(ram, "num_chickens")),
        "hens": count_chicken_slots(ram)[0],
        "planted_grass": int(read_ram_value(ram, "planted_grass")),
        "hay": int(read_ram_value(ram, "stored_grass")),
        "cow_feed": int(read_ram_value(ram, "cow_feed")),
        "chicken_feed": int(read_ram_value(ram, "chicken_feed")),
        "fed_cows": int(read_ram_value(ram, "fed_cows_n")),
        "fed_chickens": int(read_ram_value(ram, "fed_chickens_n")),
        "egg_available": int(read_ram_value(ram, "egg_available")),
        "brush": bool(int(Tool.BRUSH) in carried or shelf & SHED_ROW_2_BRUSH_BIT),
        "milker": bool(int(Tool.MILKER) in carried or shelf & SHED_ROW_2_MILKER_BIT),
        "seeds": {kind: int(read_ram_value(ram, f"{kind}_seeds")) for kind in SEED_BAGS},
        "shipped": {kind: int(read_ram_value(ram, key)) for kind, key in SHIPPED_COUNTERS.items()},
    }


def _guarded(read, ram: np.ndarray) -> dict[str, Any]:
    """A reporting field must not end a 40-minute run: record the error instead."""

    try:
        return read(ram)
    except Exception as exc:
        return {"error": repr(exc)}


def _guarded_economy(ram: np.ndarray) -> dict[str, Any]:
    return _guarded(economy_snapshot, ram)


def day_end_economy(
    phase: str, held: dict[str, Any] | None, ram: np.ndarray
) -> dict[str, Any] | None:
    """Pre-sleep snapshot: cleared when a day plan starts, taken at the first
    end-of-work phase (wait_shipping, return_home, rescue, sleep)."""

    if phase == "plan_day":
        return None
    return held if held is not None else _guarded_economy(ram)


def journal_census(ram: np.ndarray, eod: dict[str, Any] | None = None) -> dict[str, Any]:
    """Fields the day journal stores for one checkpoint morning.

    ``economy`` is the morning read (same moment as ``money``). ``gates`` is
    :func:`livestock_econ.ranch_gates` at that read: why no animal or tool was
    bought. ``eod`` is the pre-sleep read, or ``None`` when the day ended inside
    the day plan.
    """

    return {
        "development_census": development_census(ram),
        "cows": cow_rows(ram),
        "collectible_flags": collectible_flags(ram),
        "economy": _guarded_economy(ram),
        "gates": _guarded(ranch_gates, ram),
        "eod": eod,
    }


def happiness_floor(latest_index: int, index: int) -> int:
    """Lowest happiness that can still be 200 on ``latest_index`` at +4 a day."""

    remaining = int(latest_index) - int(index)
    if remaining <= 0:
        return COW_HAPPINESS_TARGET
    return max(0, COW_HAPPINESS_TARGET - COW_CARE_PER_DAY * remaining)


def _last_journal(run: Mapping[str, Any]) -> dict[str, Any]:
    journal = run.get("day_journal") or []
    if not journal:
        return {}
    last = journal[-1]
    return last if isinstance(last, dict) else {}


def run_date(run: Mapping[str, Any]) -> dict[str, Any]:
    end = run.get("end") if isinstance(run.get("end"), dict) else {}
    last = _last_journal(run)
    year = end.get("year", (run.get("start") or {}).get("year", 0) if isinstance(run.get("start"), dict) else 0)
    season = end.get("season", last.get("end_season", 0))
    day = end.get("day", last.get("end_day", 1))
    index = calendar_index(int(year), int(season), int(day))
    return {
        "year": int(year),
        "season": int(season),
        "day": int(day),
        "index": index,
        "label": format_date(int(year), int(season), int(day)),
    }


def _pin_facts(pin: str) -> dict[str, Any] | None:
    try:
        document = HarvestStateDocument.load(pin)
    except Exception:
        return None
    ram = document.ram_array()
    return {"cows": cow_rows(ram), "collectible_flags": collectible_flags(ram)}


def resolve_facts(run: Mapping[str, Any]) -> dict[str, Any]:
    """Prefer JSON cows and flags. Fall back to the named state pin, read-only."""

    last = _last_journal(run)
    cows = run.get("cows")
    if cows is None:
        cows = last.get("cows")
    flags = run.get("collectible_flags")
    if flags is None:
        flags = last.get("collectible_flags")
    if cows is None or flags is None:
        pin = run.get("state")
        loaded = _pin_facts(str(pin)) if pin else None
        if loaded is not None:
            if cows is None:
                cows = loaded["cows"]
            if flags is None:
                flags = loaded["collectible_flags"]
    return {
        "cows": list(cows or []),
        "collectible_flags": dict(flags or {}),
    }


def _flag_set(flags: Mapping[str, Any], key: str, bit: int) -> bool:
    try:
        return bool(int(flags.get(key, 0)) & bit)
    except (TypeError, ValueError):
        return False


def eaten_count(run: Mapping[str, Any], flags: Mapping[str, Any]) -> int:
    last = _last_journal(run)
    ledger = last.get("score_ledger") if isinstance(last.get("score_ledger"), dict) else {}
    terms = ledger.get("terms") if isinstance(ledger.get("terms"), dict) else {}
    berries = terms.get("power_berries") if isinstance(terms.get("power_berries"), dict) else None
    if berries is not None and "raw" in berries:
        return int(berries["raw"])
    if "power_berry_count" in flags:
        return int(flags["power_berry_count"])
    return 0


def term_gap_rows(run: Mapping[str, Any], date: Mapping[str, Any]) -> list[dict[str, Any]]:
    last = _last_journal(run)
    ledger = last.get("score_ledger") if isinstance(last.get("score_ledger"), dict) else {}
    terms = ledger.get("terms") if isinstance(ledger.get("terms"), dict) else {}
    by_id = milestones_by_id()
    index = int(date["index"])
    rows: list[dict[str, Any]] = []
    ordered = list(terms) or list(TERM_MILESTONES)
    for name in ordered:
        term = terms.get(name) if isinstance(terms.get(name), dict) else {}
        deadlines = []
        for mid in TERM_MILESTONES.get(str(name), ()):
            milestone = by_id.get(mid)
            if milestone is None:
                continue
            latest = int(milestone["latest_index"])
            deadlines.append(
                {
                    "id": mid,
                    "date": milestone.get("latest"),
                    "latest_index": latest,
                    "slack": latest - index,
                    "committed": milestone.get("date"),
                    "committed_index": int(milestone.get("committed_index", latest)),
                }
            )
            for quota in milestone.get("quotas") or []:
                q_index = quota_index(quota)
                raw = int(term.get("raw", 0) or 0)
                if index >= q_index and raw < int(quota["shipped"]):
                    deadlines.append(
                        {
                            "id": mid,
                            "date": f"Y{quota['year']} {quota['season']} D{quota['day']}",
                            "quota_shipped": int(quota["shipped"]),
                            "raw": raw,
                            "short": int(quota["shipped"]) - raw,
                            "slack": q_index - index,
                        }
                    )
        rows.append(
            {
                "term": name,
                "gap": term.get("gap"),
                "value": term.get("value"),
                "max": term.get("max"),
                "window_state": term.get("window_state"),
                "raw": term.get("raw"),
                "deadlines": deadlines,
            }
        )
    return rows


def window_rows(run: Mapping[str, Any]) -> list[dict[str, Any]]:
    last = _last_journal(run)
    ledger = last.get("score_ledger") if isinstance(last.get("score_ledger"), dict) else {}
    terms = ledger.get("terms") if isinstance(ledger.get("terms"), dict) else {}
    rows = []
    for name, term in terms.items():
        if not isinstance(term, dict):
            continue
        rows.append(
            {
                "term": name,
                "window_state": term.get("window_state"),
                "raw": term.get("raw"),
                "lo": term.get("lo"),
                "hi": term.get("hi"),
                "gap": term.get("gap"),
            }
        )
    return rows


def _berry_open(milestone: Mapping[str, Any], index: int, owned: bool) -> bool:
    if owned:
        return False
    return index <= int(milestone["latest_index"])


def berry_report(run: Mapping[str, Any], date: Mapping[str, Any], flags: Mapping[str, Any]) -> dict[str, Any]:
    by_id = milestones_by_id()
    target_row = by_id["power_berry_n"]
    target = int(target_row.get("target", load_milestones()["target"]["power_berry_n"]))
    designed_rows = [row for row in milestone_rows() if row.get("term") == "power_berry" and row.get("edge") == "designed"]
    dup_rows = [row for row in milestone_rows() if row.get("term") == "power_berry" and row.get("edge") == "duplication"]
    eaten = eaten_count(run, flags)
    index = int(date["index"])
    owned_ids: list[str] = []
    chances: list[dict[str, Any]] = []
    for row in designed_rows:
        mid = str(row["id"])
        key, bit = DESIGNED_BERRY_FLAGS[mid]
        owned = _flag_set(flags, key, bit)
        if owned:
            owned_ids.append(mid)
        open_chance = _berry_open(row, index, owned)
        chances.append(
            {
                "id": mid,
                "edge": "designed",
                "owned": owned,
                "open": open_chance,
                "earliest": row.get("earliest"),
                "latest": row.get("latest"),
                "latest_index": int(row["latest_index"]),
                "slack": int(row["latest_index"]) - index,
            }
        )
    designed_owned = len(owned_ids)
    for row in dup_rows:
        mid = str(row["id"])
        deps = [str(item) for item in row.get("depends_on") or []]
        dep_open = all(
            (dep in owned_ids) or _berry_open(by_id[dep], index, dep in owned_ids)
            for dep in deps
            if dep in by_id
        )
        window_open = index <= int(row["latest_index"])
        # Room stops at 10. A later eat fails $83F097.
        open_chance = window_open and dep_open and eaten < target
        chances.append(
            {
                "id": mid,
                "edge": "duplication",
                "owned": False,
                "open": open_chance,
                "earliest": row.get("earliest"),
                "latest": row.get("latest"),
                "latest_index": int(row["latest_index"]),
                "slack": int(row["latest_index"]) - index,
                "depends_on": deps,
            }
        )
    return {
        "designed": len(designed_rows),
        "designed_note": int(target_row.get("designed", len(designed_rows))),
        "eaten": eaten,
        "target": target,
        "room": max(0, target - eaten),
        "owned": owned_ids,
        "unassigned_eats": max(0, eaten - designed_owned),
        "chances": chances,
        "remaining": [row["id"] for row in chances if row["open"]],
    }


def collectible_rows(date: Mapping[str, Any], flags: Mapping[str, Any], berries: Mapping[str, Any]) -> list[dict[str, Any]]:
    by_id = milestones_by_id()
    index = int(date["index"])
    clock = by_id["clock_gift"]
    arm = by_id["clock_arm"]
    turtle = by_id["turtle_shell"]
    clock_owned = _flag_set(flags, "incubator_flags", CLOCK_SCORE_BIT)
    arm_set = _flag_set(flags, "event_flags_1f70", CLOCK_ARM_BIT)
    arm_latest = int(arm.get("calendar_latest_index", arm["latest_index"]))
    clock_open = (not clock_owned) and index <= int(clock["latest_index"]) and (arm_set or index <= arm_latest)
    turtle_owned = _flag_set(flags, "family_event_flags", TURTLE_BIT)
    turtle_left = []
    if not turtle_owned:
        for chance in turtle.get("chances") or []:
            if index <= int(chance):
                turtle_left.append(int(chance))
    indoor_arm = _flag_set(flags, "family_event_flags", INDOOR_ARM_BIT)
    return [
        {
            "id": "clock",
            "owned": clock_owned,
            "arm": arm_set,
            "open": clock_open,
            "latest": clock.get("latest"),
            "chances_left": 1 if clock_open else 0,
        },
        {
            "id": "turtle_shell",
            "owned": turtle_owned,
            "open": bool(turtle_left),
            "latest": turtle.get("latest"),
            "chances_left": len(turtle_left),
            "chance_indexes": turtle_left,
        },
        {
            "id": "festival_indoor_arm",
            "owned": indoor_arm,
            "open": (not indoor_arm) and index <= int(by_id["festival_indoor_arm"]["latest_index"]),
            "latest": by_id["festival_indoor_arm"].get("latest"),
            "chances_left": 0 if indoor_arm or index > int(by_id["festival_indoor_arm"]["latest_index"]) else 1,
        },
        {
            "id": "power_berry",
            "owned_count": len(berries["owned"]),
            "designed": berries["designed"],
            "eaten": berries["eaten"],
            "target": berries["target"],
            "chances_left": len(berries["remaining"]),
            "open": bool(berries["remaining"]),
        },
    ]


def _cow_milestone_for(cow: Mapping[str, Any], feedable: list[dict[str, Any]], used: set[str]) -> str | None:
    named = cow.get("milestone")
    if named:
        return str(named)
    slot = int(cow.get("slot", -1))
    if slot == 0:
        return "cow_slot0"
    if not cow.get("exists"):
        return None
    for row in feedable:
        mid = str(row["id"])
        if mid not in used:
            return mid
    return None


def lagging_cows(cows: list[Mapping[str, Any]], date: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Flag a cow that exists and cannot reach 200 by its latest morning."""

    by_id = milestones_by_id()
    index = int(date["index"])
    feedable = [by_id[mid] for mid in TERM_MILESTONES["cow_happiness"] if mid in by_id]
    flagged: list[dict[str, Any]] = []
    used: set[str] = set()
    for cow in cows:
        if not cow.get("exists"):
            continue
        mid = _cow_milestone_for(cow, feedable, used)
        if mid is None or mid in used:
            continue
        used.add(mid)
        milestone = by_id.get(mid)
        if milestone is None or milestone.get("feedable") is False:
            continue
        floor = happiness_floor(int(milestone["latest_index"]), index)
        happiness = int(cow.get("happiness", 0))
        if happiness < floor:
            flagged.append(
                {
                    "id": mid,
                    "slot": int(cow.get("slot", -1)),
                    "happiness": happiness,
                    "needed": floor,
                    "latest": milestone.get("latest"),
                    "latest_index": int(milestone["latest_index"]),
                    "reason": "happiness",
                }
            )
    for mid, milestone in by_id.items():
        if not str(mid).startswith("cow_"):
            continue
        if mid in used:
            continue
        if index < int(milestone.get("committed_index", 0)):
            continue
        flagged.append(
            {
                "id": mid,
                "slot": milestone.get("slot"),
                "happiness": None,
                "needed": None,
                "latest": milestone.get("date"),
                "latest_index": int(milestone.get("latest_index", 0)),
                "reason": "missing",
            }
        )
    return flagged


def build_report(run: Mapping[str, Any]) -> dict[str, Any]:
    date = run_date(run)
    facts = resolve_facts(run)
    flags = facts["collectible_flags"]
    berries = berry_report(run, date, flags)
    last = _last_journal(run)
    return {
        "state": run.get("state"),
        "date": date,
        "term_gaps": term_gap_rows(run, date),
        "window_states": window_rows(run),
        "collectibles": collectible_rows(date, flags, berries),
        "berry_chances": berries,
        "lagging_cows": lagging_cows(list(facts["cows"]), date),
        "development_census": last.get("development_census"),
    }


def format_report(report: Mapping[str, Any]) -> str:
    date = report["date"]
    lines = [
        f"PR-999 report  date={date['label']}  index={date['index']}  state={report.get('state')}",
        "term gaps",
    ]
    for row in report["term_gaps"]:
        deadline = "none"
        if row["deadlines"]:
            parts = []
            for item in row["deadlines"]:
                if "short" in item:
                    parts.append(
                        f"{item['id']} quota {item['date']} short={item['short']} raw={item['raw']}/{item['quota_shipped']}"
                    )
                else:
                    committed = item.get("committed")
                    plan = f" committed={committed}" if committed and committed != item["date"] else ""
                    parts.append(f"{item['id']}@{item['date']} slack={item['slack']}{plan}")
            deadline = "; ".join(parts)
        lines.append(
            f"  {row['term']} gap={row['gap']} value={row['value']}/{row['max']} "
            f"window={row['window_state']} deadlines={deadline}"
        )
    lines.append("window states")
    for row in report["window_states"]:
        lines.append(
            f"  {row['term']} {row['window_state']} raw={row['raw']} lo={row['lo']} hi={row['hi']}"
        )
    lines.append("remaining collectible chances")
    for row in report["collectibles"]:
        lines.append(
            f"  {row['id']} owned={row.get('owned', row.get('owned_count'))} "
            f"open={row['open']} chances_left={row['chances_left']} latest={row.get('latest')}"
        )
    berries = report["berry_chances"]
    lines.append(
        f"berry chances  designed={berries['designed']}  eaten={berries['eaten']}  "
        f"target={berries['target']}  room={berries['room']}"
    )
    if int(berries["eaten"]) != int(berries["target"]):
        lines.append(
            f"  eaten {berries['eaten']} must land on {berries['target']}; "
            "any other count fails $83F097"
        )
    else:
        lines.append(f"  eaten is {berries['target']}; another berry fails $83F097")
    for chance in berries["chances"]:
        state = "open" if chance["open"] else "closed"
        lines.append(
            f"  {chance['id']} {chance['edge']} {state} "
            f"earliest={chance['earliest']} latest={chance['latest']} slack={chance['slack']}"
        )
    lines.append("lagging cows")
    if not report["lagging_cows"]:
        lines.append("  none")
    for cow in report["lagging_cows"]:
        lines.append(
            f"  FLAG lagging cow {cow['id']} slot={cow['slot']} "
            f"happiness={cow['happiness']} needed={cow['needed']} "
            f"by {cow['latest']} reason={cow['reason']}"
        )
    census = report.get("development_census")
    if isinstance(census, dict) and "counted_tiles" in census:
        lines.append(
            f"development census  counted={census['counted_tiles']} "
            f"gap_to_2950={census['gap_to_2950']} uncovered={census['uncovered_count']}"
        )
        if census.get("ledger_line"):
            lines.append(f"  {census['ledger_line']}")
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m harvest.tools.pr999_report")
    parser.add_argument("run_json", type=Path)
    args = parser.parse_args(argv)
    run = json.loads(args.run_json.read_text())
    sys.stdout.write(format_report(build_report(run)))
    return 0


# Run journals and progress reports share this reporting owner.

def _date_fields(ram) -> dict[str, int]:
    return {
        "year": int(read_ram_value(ram, "year")),
        "season": int(read_ram_value(ram, "season")),
        "day": int(read_ram_value(ram, "day")),
        "hour": int(read_ram_value(ram, "hour")),
        "minute": int(read_ram_value(ram, "minute")),
        "tilemap": int(read_ram_value(ram, "tilemap")),
        "money": int(read_ram_value(ram, "money")),
        "num_cows": int(read_ram_value(ram, "num_cows")),
        "num_chickens": int(read_ram_value(ram, "num_chickens")),
        "cow_slots": sum(1 << slot for slot in existing_cow_slots(ram)),
        "planted_grass": int(read_ram_value(ram, "planted_grass")),
        "stored_grass": int(read_ram_value(ram, "stored_grass")),
        "grass_seeds": int(read_ram_value(ram, "grass_seeds")),
        "stamina": int(read_ram_value(ram, "stamina")),
        "marriage_flags": int(read_ram_value(ram, "marriage_flags")),
        "stored_wood": int(read_ram_value(ram, "stored_wood")),
        "upgrade_flags": int(read_ram_value(ram, "upgrade_flags")),
        "house_build_nights": int(read_ram_value(ram, "development_rate")),
        "ann_hearts": int(read_ram_value(ram, "ann_hearts")),
        "eve_hearts": int(read_ram_value(ram, "eve_hearts")),
        "maria_hearts": int(read_ram_value(ram, "maria_hearts")),
        "nina_hearts": int(read_ram_value(ram, "nina_hearts")),
        "ellen_hearts": int(read_ram_value(ram, "ellen_hearts")),
        "wife_pregnancy": int(read_ram_value(ram, "wife_pregnancy")),
        "tool_selected": int(read_ram_value(ram, "tool_selected")),
        "shed_row_3": read_ram_u8(ram, 0x11F02),
    }

def _summarize_journal(journal: list[dict]) -> dict:
    phase_success: dict[str, int] = {}
    phase_skip: dict[str, int] = {}
    phase_no_work: dict[str, int] = {}
    phase_failure: dict[str, int] = {}
    water_deltas: list[dict] = []
    establish_deltas: list[dict] = []
    harvest_deltas: list[dict] = []
    total_shipped = 0
    total_harvested = 0
    total_planted = 0
    for row in journal:
        try:
            total_shipped += int(row.get("shipped_count") or 0)
            total_harvested += int(row.get("harvested_count") or 0)
            total_planted += int(row.get("establish_planted") or 0)
        except Exception:
            pass
        for result in row.get("phase_results") or []:
            name = str(result.get("phase", "?"))
            status = str(result.get("status", ""))
            reason = str(result.get("reason") or "")
            if status == "success":
                phase_success[name] = phase_success.get(name, 0) + 1
            elif status == "skipped":
                phase_skip[name] = phase_skip.get(name, 0) + 1
            elif status == "no_work":
                phase_no_work[name] = phase_no_work.get(name, 0) + 1
            elif status in {"failure", "blocked"}:
                phase_failure[name] = phase_failure.get(name, 0) + 1
            if name == "CROP_WATER" and "watered=" in reason:
                water_deltas.append( { "plan_day": row.get("plan_day"), "status": status, "reason": reason, } )
            if name == "CROP_ESTABLISH" and "planted=" in reason:
                establish_deltas.append( { "plan_day": row.get("plan_day"), "status": status, "reason": reason, } )
            if name == "HARVEST_ROUTE":
                harvest_deltas.append(
                    {
                        "plan_day": row.get("plan_day"),
                        "status": status,
                        "reason": reason,
                        "shipped_count": result.get("shipped_count"),
                        "harvested_count": result.get("harvested_count"),
                    }
                )
    final_money = journal[-1].get("money") if journal else None
    harvest_phases_present = bool( phase_success.get("HARVEST_ROUTE") or harvest_deltas )
    if total_planted <= 0:
        for row in establish_deltas:
            reason = str(row.get("reason") or "")
            if "planted=" not in reason:
                continue
            try:
                n = int(reason.split("planted=")[1].split()[0].rstrip(","))
            except Exception:
                n = 0
            if n > 0:
                total_planted += n
    establish_nonzero = total_planted > 0
    try:
        money_ok = final_money is not None and int(final_money) > 100
    except Exception:
        money_ok = False
    return {
        "overnights": len(journal),
        "phase_success_counts": phase_success,
        "phase_skip_counts": phase_skip,
        "phase_no_work_counts": phase_no_work,
        "phase_failure_counts": phase_failure,
        "crop_water_deltas": water_deltas,
        "crop_establish_deltas": establish_deltas,
        "harvest_deltas": harvest_deltas,
        "total_shipped": total_shipped,
        "total_harvested": total_harvested,
        "total_planted": total_planted,
        "final_money": final_money,
        "harvest_phases_present": harvest_phases_present,
        "crop_establish_nonzero": establish_nonzero,
        "gate_a_economy_ok": bool(money_ok and harvest_phases_present),
    }

def _year_report_fields(planner: object) -> dict[str, object]:
    journal = list(getattr(planner, "day_journal", ()) or ())
    return {
        "days_completed": getattr(planner, "_days_completed", None),
        "day_failures": list(getattr(planner, "day_failures", ()) or ()),
        "day_journal": journal,
        "journal_summary": _summarize_journal(journal),
        "floor_report": getattr(planner, "floor_report", None),
    }

def _gate_b_full_ok(
    *,
    power_on: bool,
    until_ending: bool,
    end_key: tuple[int, int, int],
    total_shipped: int,
    total_harvested: int,
    mid_run_state_loads: int | None,
) -> bool:
    if not power_on or until_ending or mid_run_state_loads is None:
        return False
    try:
        loads = int(mid_run_state_loads)
        shipped = int(total_shipped)
        harvested = int(total_harvested)
    except (TypeError, ValueError):
        return False
    if loads != 0:
        return False
    year, season, day = end_key
    if ( year != 0 or season != 1 or end_key <= (0, 0, 30) or is_ending_date(year, season, day) ):
        return False
    return shipped > 0 or harvested > 0

def _crop_survival_report(ram: np.ndarray) -> dict:
    from harvest.planner.day_plan_status import is_farm_tilemap
    from harvest.tasks.crop_planter import count_crop_survival
    from harvest.core.tile_catalog import ADDR_TILEMAP
    tilemap = int(ram[ADDR_TILEMAP]) if ADDR_TILEMAP < len(ram) else -1
    if not is_farm_tilemap(tilemap):
        return {
            "farm_map_loaded": False,
            "tilemap": tilemap,
            "note": "crop tiles only visible on farm metatile map; house end-state is inconclusive",
        }
    counts = count_crop_survival(ram)
    return { "farm_map_loaded": True, "tilemap": tilemap, **counts, "alive": int(counts.get("crop", 0)) > 0, }

def _d2_spine_checkpoint_evidence(task: object, world: WorldState) -> dict[str, object]:
    fields = _date_fields(world.ram)
    phase_results = list(getattr(task, "_last_day_phase_results", ()) or ())
    successful = { str(row.get("phase")) for row in phase_results if str(row.get("status")) == "success" }
    potato_seeds = int(read_ram_value(world.ram, "potato_seeds"))
    ready = bool(
        fields["season"] == 0
        and fields["day"] == 2
        and fields["hour"] >= 17
        and getattr(task, "_phase", None) == "return_home"
        and "MOUNTAIN_BERRY" in successful
        and "BUY_SEEDS" in successful
        and "WAIT_FARM_SHIPPING" in successful
    )
    return {
        "ready": ready,
        "date": fields,
        "planner_phase": getattr(task, "_phase", None),
        "successful_phases": sorted(successful),
        "potato_seeds": potato_seeds,
        "mountain_grape_shipped": "MOUNTAIN_BERRY" in successful,
        "potato_purchase_complete": "BUY_SEEDS" in successful,
        "shipping_dialogue_cleared": "WAIT_FARM_SHIPPING" in successful,
    }


if __name__ == "__main__":
    raise SystemExit(main())
