"""Campaign milestone solver and season-pin lineage.

Latest-start dates come from the F-gate deadlines. Slack is
``latest_index - committed_index`` in mornings (Y1 Spring D1 = 0).
A what-if shift moves a commitment by that many days and recomputes
dependents. Power berries are exactly 10: the seven designed one-shots
plus three duplication-edge eats. Any other count fails ``$83F097``.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Mapping, Optional, Sequence

import numpy as np

from harvest.core.ram_catalog import read_ram_value
from harvest.core.scene import classify_scene_from_ram, morning_scene_ready
from harvest.paths import GAME_DIR, PROJECT_DIR
from harvest.planner.day_plan_status import read_world_ymd
from harvest.planner.livestock_econ import (
    D30_INDEX,
    calendar_ymd,
    shop_open_on,
)

SEASONS = ("Spring", "Summer", "Fall", "Winter")
MILESTONES_PATH = PROJECT_DIR / "docs" / "perfect_rancher" / "milestones.yaml"
SCORE_TARGET = 999
ENDING = "0x25"
POWER_BERRY_TARGET = 10
DESIGNED_BERRIES = 7
DUPLICATION_BERRIES = 3
D29_INDEX = 298  # Y3 Summer D29 morning; D30 is too late to eat or ship
WIFE_LATEST = 188  # Y2 Fall D9, so child 2 is already set on D29
CHILD1_LAG = 20
CHILD2_LAG = 110
CHILD2_AFTER_CHILD1 = 90
COW_CARE_MORNING = 50  # first 06:00 at >= 200 is 50 mornings after the buy
LAST_FEEDABLE_LATEST = D30_INDEX - COW_CARE_MORNING  # 249, Y3 Spring D10
FEEDABLE_COWS = 11
HEN_NIGHTS = 110  # 11 birds, 10 nights each (F5 egg-then-chick loop)
CLOCK_ARM_WINDOW = 59  # Y1 Summer D30, year 0 and still before fall
POTATO_TURNIP_DEADLINE = 269  # Y3 Spring D30
CROP_WINDOW = (496, 511)
CROP_TARGET = 504
BAG_TILES = 9
SPRING_BAGS = 56
SUMMER_BAGS = 8
SUMMER_HARVESTS = 7
GROW = {"turnip": 4, "potato": 6}
COVER_MIN_TILES = 2950
HOUSE_NIGHTS = 4
INDOOR_ARM_EARLIEST = 79  # Y1 Fall D20
TURTLE_LATEST = 199  # Y2 Fall D20
SaveState = Callable[[Path], Path]


def format_index(index: int) -> str:
    """Calendar label. Indexes before day 0 stay numeric."""
    if index < 0:
        return f"before 0 ({index})"
    year, season, day = calendar_ymd(index)
    return f"Y{year} {SEASONS[season]} D{day}"


def season_pin_name(year_ram: int, season: int) -> str:
    """``PR_Y1_Summer_D1``. ``year_ram`` is the 0-based RAM year."""
    season_i = int(season)
    if season_i not in range(4):
        raise ValueError(f"season out of range: {season}")
    return f"PR_Y{int(year_ram) + 1}_{SEASONS[season_i]}_D1"


def parent_run_hash(
    *,
    power_on: bool,
    source_sha256: Optional[str],
    start_key: tuple[int, int, int],
) -> str:
    """Identity of the run that wrote a season pin. Not the pin bytes."""
    text = (
        f"power_on={int(bool(power_on))}\n"
        f"source={source_sha256 or '-'}\n"
        f"start={int(start_key[0])},{int(start_key[1])},{int(start_key[2])}\n"
    )
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class Milestone:
    id: str
    term: str
    committed_index: int
    latest_index: int
    earliest_index: int
    slack_days: int
    depends_on: tuple[str, ...] = ()
    note: str = ""
    data: dict = field(default_factory=dict)

    def as_dict(self) -> dict:
        row = {
            "id": self.id,
            "term": self.term,
            "date": format_index(self.committed_index),
            "committed_index": self.committed_index,
            "earliest_index": self.earliest_index,
            "earliest": format_index(self.earliest_index),
            "latest_index": self.latest_index,
            "latest": format_index(self.latest_index),
            "slack_days": self.slack_days,
            "depends_on": list(self.depends_on),
            "note": self.note,
        }
        row.update(self.data)
        return row


def _ms(
    ident: str,
    term: str,
    committed: int,
    latest: int,
    earliest: int,
    *,
    depends_on: Sequence[str] = (),
    note: str = "",
    **data: object,
) -> Milestone:
    return Milestone(
        id=ident,
        term=term,
        committed_index=int(committed),
        latest_index=int(latest),
        earliest_index=int(earliest),
        slack_days=int(latest) - int(committed),
        depends_on=tuple(depends_on),
        note=note,
        data=dict(data),
    )


def solve_milestones(what_if: Optional[Mapping[str, int]] = None) -> list[Milestone]:
    """Apply dated what-ifs to the one campaign table, without inventing receipts.

    This is a constraint model. The YAML's commitment and evidence status are
    authoritative; purchased cows, births or house flags are never inferred
    from the economy's independent paper ledger.
    """
    raw = load_milestone_rows()
    base = {str(row["id"]): int(row["committed_index"]) for row in raw}
    if len(base) != len(raw):
        raise ValueError("duplicate milestone id")
    shifts = {str(key): int(value) for key, value in (what_if or {}).items()}
    unknown = set(shifts) - set(base)
    if unknown:
        raise ValueError(f"unknown milestone shifts: {sorted(unknown)}")
    anchors = {
        "berry_field_2": "berry_field_1", "berry_dup_cave": "berry_cave",
        "berry_dup_indoor": "berry_indoor", "berry_dup_transition": "berry_field_1",
        "child_1": "wife_bit", "child_2": "wife_bit",
        "child_2_age60": "wife_bit",
    }
    moved: dict[str, int] = {}
    def date(ident: str) -> int:
        if ident not in moved:
            anchor = anchors.get(ident)
            delta = date(anchor) - base[anchor] if anchor in base else 0
            moved[ident] = base[ident] + delta + shifts.get(ident, 0)
        return moved[ident]
    rows = []
    fixed = {"id", "term", "date", "committed_index", "latest_index", "latest",
             "earliest_index", "earliest", "slack_days", "depends_on", "note"}
    for row in raw:
        ident = str(row["id"])
        deps = tuple(row.get("depends_on", ()))
        if set(deps) - set(base):
            raise ValueError(f"{ident}: unknown dependency")
        rows.append(_ms(ident, str(row["term"]), date(ident), int(row["latest_index"]),
                        int(row["earliest_index"]), depends_on=deps,
                        note=str(row.get("note", "")),
                        **{key: value for key, value in row.items() if key not in fixed}))
    return rows


def milestones_by_id(what_if: Optional[Mapping[str, int]] = None) -> dict[str, Milestone]:
    rows = solve_milestones(what_if)
    found: dict[str, Milestone] = {}
    for row in rows:
        if row.id in found:
            raise RuntimeError(f"duplicate milestone {row.id}")
        found[row.id] = row
    return found


def _scalar(value: object) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int) and not isinstance(value, bool):
        return str(value)
    if isinstance(value, (list, tuple)):
        return "[" + ", ".join(_scalar(item) for item in value) + "]"
    text = str(value).replace("\\", "\\\\").replace('"', '\\"')
    return f'"{text}"'


def _dump(value: object, indent: int = 0) -> str:
    pad = "  " * indent
    if isinstance(value, dict):
        lines: list[str] = []
        for key, item in value.items():
            if isinstance(item, list) and item and isinstance(item[0], dict):
                lines.append(f"{pad}{key}:")
                for child in item:
                    lines.append(_dump_flow_item(child, indent + 1))
            elif isinstance(item, dict) and item:
                lines.append(f"{pad}{key}:")
                lines.append(_dump(item, indent + 1))
            else:
                lines.append(f"{pad}{key}: {_scalar(item)}")
        return "\n".join(lines)
    raise TypeError(type(value))


def _dump_flow_item(row: dict, indent: int) -> str:
    pad = "  " * indent
    keys = list(row)
    first, *rest = keys
    lines = [f"{pad}- {first}: {_scalar(row[first])}"]
    inner = "  " * (indent + 1)
    for key in rest:
        item = row[key]
        if isinstance(item, list) and item and isinstance(item[0], dict):
            lines.append(f"{inner}{key}:")
            for child in item:
                lines.append(_dump_flow_item(child, indent + 2))
        else:
            lines.append(f"{inner}{key}: {_scalar(item)}")
    return "\n".join(lines)


def render_milestones(rows: Optional[Sequence[Milestone]] = None) -> str:
    body = {
        "target": {
            "ranch_mastery": SCORE_TARGET,
            "ending": ENDING,
            "power_berry_n": POWER_BERRY_TARGET,
        },
        "index_note": "Y1 Spring D1 morning = 0. Y3 Summer D30 morning = 299. Slack = latest_index - committed_index.",
        "milestones": [row.as_dict() for row in (rows if rows is not None else solve_milestones())],
    }
    return _dump(body) + "\n"


def write_milestones(path: Path = MILESTONES_PATH) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(render_milestones(), encoding="utf-8")
    return path


def morning_index(year0: int, season: int, day: int) -> int:
    """Inverse of ``calendar_ymd``. ``year0`` is the 0-based RAM year."""
    return int(year0) * 120 + int(season) * 30 + int(day) - 1


def load_milestone_rows(path: Path = MILESTONES_PATH) -> list[dict]:
    """Read ``milestones.yaml``. The solver is not a substitute for this file."""
    import yaml

    loaded = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    rows = loaded.get("milestones") if isinstance(loaded, dict) else None
    if not isinstance(rows, list):
        raise ValueError(f"{path} has no milestone list")
    return rows


def day_intents_for_morning(
    year0: int,
    season: int,
    day: int,
    *,
    rows: Optional[Sequence[Mapping]] = None,
) -> list[dict]:
    """Commitments whose date is this morning, in file order."""
    index = morning_index(year0, season, day)
    chosen: list[dict] = []
    for row in rows if rows is not None else load_milestone_rows():
        if int(row.get("committed_index", -1)) != index:
            continue
        chosen.append(dict(row))
    return chosen


# These rows select existing Skills. Unimplemented recipes are deferrals;
# they never turn a paper milestone into a successful production task.
MILESTONE_SKILLS = {
    "pasture_ready": ("grass", {"target_tiles": 32, "bag_limit": 4}),
    "hen": ("cow_purchase", {"species": "chicken"}),
    "cow_slot0": ("cow_purchase", {"species": "cow"}),
    "clock_arm": ("home_upgrade", {"goal": "clock_arm"}),
    "house_upgrade_1": ("home_upgrade", {"goal": "order"}),
    "clock_gift": ("home_upgrade", {"goal": "clock_gift"}),
}


def milestone_phases(intents: Sequence[Mapping], *, completed: Sequence[str] = (),
                     storm: bool = False) -> tuple[list, list]:
    """Resolve commitments before advisor selection into Skills or deferrals."""
    from harvest.planner.day_phase_types import PhaseKind, PhaseSpec
    from harvest.planner.day_phase_registry import PHASE_TASK_BUILDERS
    from harvest.planner.day_plan_decision import DeferredPlan

    phases, deferred = [], []
    done = set(completed)
    for intent in intents:
        ident = str(intent["id"])
        if ident in done:
            continue
        recipe = MILESTONE_SKILLS.get(ident)
        if ident.startswith("cow_feedable_"):
            recipe = ("cow_purchase", {"species": "cow"})
        kind, params = recipe or ("campaign_unimplemented", {})
        params = {**params, "intent_id": ident, "milestone": dict(intent)}
        spec = PhaseSpec(f"MILESTONE_{ident.upper()}", kind, params,
                         failure_policy="opportunistic")
        missing = [dep for dep in intent.get("depends_on", ()) if dep not in done]
        reason = None
        if int(intent.get("runtime_failures", 0)) >= 3:
            reason = "three_runtime_failures:tighten the native Skill probe before retrying"
        elif int(intent.get("attempt_index", -1)) > int(intent.get("latest_index", D29_INDEX)):
            reason = "deadline_missed:revise this campaign constraint"
        elif storm:
            reason = "storm:reconsider on the next playable morning"
        elif missing:
            reason = f"dependency_unverified:{','.join(missing)}"
        elif not isinstance(spec.kind, PhaseKind) or spec.kind not in PHASE_TASK_BUILDERS:
            reason = f"native_skill_missing:{ident}:implement and prove its acquisition Skill"
        if reason:
            deferred.append(DeferredPlan.from_phase(spec, reason,
                            retry="after_dependency" if missing else "after_native_skill_proof"))
        else:
            if spec.kind == PhaseKind.COW_PURCHASE:
                from harvest.planner.day_phase_catalog import animal_purchase_phases
                for phase in animal_purchase_phases(str(params["species"])):
                    recipe_params = {**phase.params, "campaign_block": "animals"}
                    if phase.kind == PhaseKind.COW_PURCHASE:
                        recipe_params.update(params)
                    phases.append(PhaseSpec(f"MILESTONE_{ident.upper()}_{phase.phase}", phase.kind,
                                           recipe_params, failure_policy="opportunistic",
                                           contract=phase.contract))
            else:
                phases.append(spec)
    return phases, deferred


def intent_ram_snapshot(ram) -> dict:
    """Read immutable campaign receipts; this function never writes RAM."""
    from harvest.core.ram_catalog import read_animal_slot_field, read_ram_u16
    names = ("power_berry_count", "max_stamina", "num_cows", "num_chickens",
             "money", "planted_grass", "upgrade_flags", "family_event_flags", "romance_event_flags", "incubator_flags")
    return {**{name: int(read_ram_value(ram, name)) for name in names},
            "clock_request_flags": int(read_ram_u16(ram, 0x11F70)),
            "cow_slots": tuple(slot for slot in range(12)
                               if read_animal_slot_field(ram, "cow", slot, "status_raw") & 1)}


def verify_intent_receipt(spec, before: Mapping, after: Mapping) -> bool:
    """Require the ROM postcondition, independent of a Skill's SUCCESS string."""
    ident = spec.params.get("intent_id", "")
    if ident.startswith("berry_"):
        return (after["power_berry_count"] == before["power_berry_count"] + 1
                and after["max_stamina"] == before["max_stamina"] + 10
                and after["power_berry_count"] <= POWER_BERRY_TARGET)
    if ident.startswith("cow_"):
        return (after["num_cows"] == before["num_cows"] + 1
                and len(set(after.get("cow_slots", ())) - set(before.get("cow_slots", ()))) == 1)
    if ident == "hen":
        return after["num_chickens"] == before["num_chickens"] + 1
    if ident == "pasture_ready":
        return after["planted_grass"] >= int(spec.params.get("milestone", {}).get("minimum_tiles", 32))
    if ident == "clock_arm":
        return bool(after["clock_request_flags"] & 0x0004)
    if ident == "clock_gift":
        return bool(after["incubator_flags"] & 0x4000)
    # Unspecified postconditions may never clear an intent.
    return False


# Daily arbiter. Clock stops at 18:00 (ADR 0003): cover and hoeing are the
# post-18 lane, and a hot-spring soak refills stamina without moving the clock.
# Daytime accounts are phase caps and CostModel figures, not new measurements.
BLOCK_ORDER = ("deadline", "animals", "crops", "social", "cover", "hoeing")
_DAY_BLOCKS = frozenset({"deadline", "animals", "crops", "social"})
_CLOSE_NAMES = frozenset({"RETURN_HOME", "GO_TO_SLEEP", "READY_TO_GO_HOME"})
_CLOSE_KINDS = frozenset({"return_home", "sleep", "ready_to_go_home"})
_DEADLINE_NAMES = frozenset(
    {"MOUNTAIN_BERRY", "BUY_SEEDS", "BUY_SEEDS_WINDOW", "NAV_FARM_EXIT", "D2_FARM_CLEAR"}
)
_DEADLINE_KINDS = frozenset({"mountain_berry", "shop_buy", "deadline", "wait_until_time", "home_upgrade", "power_berry"})
_ANIMAL_KINDS = frozenset(
    {
        "cow_chores",
        "cow_purchase",
        "coop_chores",
        "pickup_chicken",
        "drop_chicken",
        "chicken_sale_followup",
        "chicken_sale_request",
        "chicken_sale_event",
        "ensure_animal_tools",
    }
)
_CROP_NAMES = frozenset(
    {
        "NAV_CROP",
        "HARVEST_ROUTE",
        "CROP_WATER",
        "CROP_ESTABLISH",
        "REPAIR_CROPS",
        "ENSURE_CROP_SEEDS",
        "ENSURE_WATERING_CAN",
        "DYNAMIC_OUTDOOR_PLAN",
    }
)
_CROP_KINDS = frozenset({"crop", "harvest", "ensure_seed", "dynamic_outdoor_plan"})
_SOCIAL_NAMES = frozenset(
    {"EVE_TALK_LOOP", "SUNDAY_CHURCH", "SUNDAY_MOUNTAIN", "ATTEND_FESTIVAL", "TOWN_EXPLORE"}
)
_COVER_KINDS = frozenset({"grass", "fence_clear", "hot_spring"})
# Slot watchdogs: cow_task 480 care / 720 milk; coop_layout 480 egg nav + 480 feed.
_COW_SLOT_ACCOUNT = 480 * 3 + 720
_BARN_ACCOUNT = 800 + 120 + 100
_HEN_ACCOUNT = 480 + 480
_COOP_ACCOUNT = 800
# F9 flower ladder, one town trip (MASTER_PLAN §6), not a per-girl journey.
_GIRL_TALK_FRAMES = 109
# Account for one post-18 hoe or seed swing. Not a measured anim length.
_POST18_TILE_FRAMES = 30
# MASTER_PLAN §6 upper estimate for a full-scale day, daytime blocks only.
_DAY_ENVELOPE = 60_000
FULL_SCALE_DAY = {
    "cows": 12,
    "hens": 12,
    "crop_plots": 6,
    "girls": 5,
    "deadline_jobs": 1,
    "cover_tiles": 120,
    "hoe_tiles": 120,
    "max_stamina": 100 + POWER_BERRY_TARGET * 10,
    "day_budget": _DAY_ENVELOPE,
}


def _kind_name(phase: object) -> str:
    kind = getattr(phase, "kind", "")
    return str(getattr(kind, "value", kind))


def block_of(phase: object) -> Optional[str]:
    """Work block for one phase. None means travel that sticks to a neighbor."""
    name = str(getattr(phase, "phase", ""))
    kind = _kind_name(phase)
    params = getattr(phase, "params", None) or {}
    if params.get("campaign_block") in BLOCK_ORDER:
        return params["campaign_block"]
    if name in _CLOSE_NAMES or kind in _CLOSE_KINDS:
        return "close"
    if name in _DEADLINE_NAMES or kind in _DEADLINE_KINDS:
        return "deadline"
    if kind in _ANIMAL_KINDS or any(tok in name for tok in ("CHICKEN", "COW", "COOP", "BARN")):
        return "animals"
    if name in _CROP_NAMES or kind in _CROP_KINDS:
        return "crops"
    if name in _SOCIAL_NAMES or kind == "eve_talk_loop":
        return "social"
    if name in {"CLEAR_FIELD", "POST18_HOE"} or kind == "clear_field":
        return "hoeing"
    if name in {"PLANT_PASTURE", "CUT_FODDER", "POST18_COVER"} or kind in _COVER_KINDS:
        return "cover"
    if isinstance(params, dict) and params.get("lane") == "post_18":
        return "hoeing" if "HOE" in name else "cover"
    return None


def _tagged(phases: Sequence[object]) -> list[tuple[str, object]]:
    """Inherit the previous block so a barn exit does not jump ahead."""
    tagged: list[tuple[str, object]] = []
    prev: Optional[str] = None
    for phase in phases:
        tag = block_of(phase)
        if tag is None:
            tag = prev if prev is not None else "prefix"
        tagged.append((tag, phase))
        if tag not in {"prefix", "close"}:
            prev = tag
    return tagged


def apply_block_order(phases: Sequence[object], *, storm: bool = False) -> list:
    """Deadline, animals, crops, social, then the post-18 cover and hoe lane.

    A leading exit stays in front. Sleep stays last. Storm days are not reordered:
    the farmer cannot leave the house.
    """
    if storm or not phases:
        return list(phases)
    tagged = _tagged(phases)
    ordered = [phase for tag, phase in tagged if tag == "prefix"]
    for name in BLOCK_ORDER:
        ordered.extend(phase for tag, phase in tagged if tag == name)
    ordered.extend(phase for tag, phase in tagged if tag == "close")
    return ordered


def _phase_account(phase: object) -> int:
    contract = getattr(phase, "contract", None)
    estimate = getattr(contract, "estimated_frames", None) if contract is not None else None
    if estimate:
        return int(estimate)
    params = getattr(phase, "params", None) or {}
    timeout = params.get("timeout") if isinstance(params, dict) else None
    return int(timeout or 0)


def _frames_from_results(results: Sequence[Mapping], phases: Sequence[object] = ()) -> dict[str, int]:
    """Step stamps are cumulative. The delta is that phase's frames."""
    rows = sorted(
        (row for row in results if row.get("phase") is not None),
        key=lambda row: int(row.get("step") or 0),
    )
    spent: dict[str, int] = {}
    planned_tags = {str(getattr(phase, "phase", "")): tag for tag, phase in _tagged(phases)}
    prev = 0
    for row in rows:
        step = int(row.get("step") or prev)
        delta = int(row["frames"]) if "frames" in row else max(0, step - prev)
        prev = step
        tag = planned_tags.get(str(row.get("phase"))) or block_of(
            type("_P", (), {"phase": row.get("phase"), "kind": row.get("kind"), "params": {}})())
        if tag is None or tag in {"prefix", "close"}:
            continue
        spent[tag] = spent.get(tag, 0) + delta
    return spent


def arbitration_journal(phases: Sequence[object], results: Sequence[Mapping] = ()) -> dict:
    """Per-block frame account, plus frames already spent when a day ran."""
    tagged = _tagged(apply_block_order(phases))
    spent = _frames_from_results(results, phases)
    blocks = []
    for name in BLOCK_ORDER:
        group = [phase for tag, phase in tagged if tag == name]
        account = sum(_phase_account(phase) for phase in group)
        blocks.append(
            {
                "block": name,
                "lane": "day" if name in _DAY_BLOCKS else "post_18",
                "account": int(account),
                "frames": int(spent.get(name, 0)),
                "phases": [str(getattr(phase, "phase", "")) for phase in group],
                "hot_spring_refills": 0,
            }
        )
    return {"order": list(BLOCK_ORDER), "blocks": blocks,
            "frame_source": "native_frame_clock" if results and all("frames" in row for row in results)
                            else "task_step_estimate"}


def _whatif_units(spec: Mapping, cost: object) -> list[tuple[str, str, list[tuple[int, int]]]]:
    from harvest.core.stamina import SWING_STAMINA_COST

    cows = int(spec["cows"])
    hens = int(spec["hens"])
    plots = int(spec["crop_plots"])
    animals: list[tuple[int, int]] = []
    if cows:
        animals.append((_BARN_ACCOUNT, 0))
        animals.extend((_COW_SLOT_ACCOUNT, 0) for _ in range(cows))
    if hens:
        animals.append((_COOP_ACCOUNT, 0))
        animals.extend((_HEN_ACCOUNT, 0) for _ in range(hens))
    crop_units: list[tuple[int, int]] = []
    for index in range(plots):
        if index == 0:
            frames = int(cost.water_ring_f) + int(cost.harvest_ring_f)
        else:
            frames = int(cost.water_ring_marginal_f) + int(cost.harvest_ring_marginal_f)
        crop_units.append((frames, 0))
    social: list[tuple[int, int]] = []
    girls = int(spec["girls"])
    if girls:
        social.append((int(cost.shop_roundtrip_f), 0))
        social.extend((_GIRL_TALK_FRAMES, 0) for _ in range(girls))
    swing = ( _POST18_TILE_FRAMES, int(SWING_STAMINA_COST) )
    return [
        ("deadline", "day", [(int(cost.grape_first_f), 0)] * int(spec["deadline_jobs"])),
        ("animals", "day", animals),
        ("crops", "day", crop_units),
        ("social", "day", social),
        ("cover", "post_18", [swing] * int(spec["cover_tiles"])),
        ("hoeing", "post_18", [swing] * int(spec["hoe_tiles"])),
    ]


def _spend_units(
    units: Sequence[tuple[int, int]],
    *,
    lane: str,
    budget_left: int,
    stamina: int,
    max_stamina: int,
    spa_frames: int,
    unlimited: bool,
) -> tuple[int, int, int, int]:
    frames = 0
    refills = 0
    done = 0
    stam = int(stamina)
    for unit_frames, unit_stam in units:
        extra = 0
        next_stam = stam
        if unit_stam and next_stam < unit_stam:
            extra = int(spa_frames)
            next_stam = int(max_stamina)
            if next_stam < unit_stam:
                break
        cost = extra + int(unit_frames)
        if not unlimited and lane == "day" and frames + cost > int(budget_left):
            break
        if extra:
            refills += 1
        stam = max(0, next_stam - int(unit_stam)) if unit_stam else next_stam
        frames += cost
        done += 1
    return frames, refills, done, stam


def run_whatif_day(demand: Optional[Mapping] = None) -> dict:
    """Run one full-scale day on paper. Every block spends its frame account.

    Post-18 cover and hoeing ignore the clock. A swing that stamina cannot
    pay waits for one hot-spring refill (CostModel.spa_refill_f) and continues.
    Loads are not used. Nothing here writes RAM.
    """
    from harvest.planner.spring_opt import CostModel

    spec = dict(FULL_SCALE_DAY)
    if demand:
        spec.update(demand)
    cost = CostModel()
    max_stam = int(spec["max_stamina"])
    budget = int(spec["day_budget"])
    units = _whatif_units(spec, cost)
    accounts: dict[str, int] = {}
    stam = max_stam
    for name, lane, block_units in units:
        frames, _refills, _done, stam = _spend_units(
            block_units,
            lane=lane,
            budget_left=budget,
            stamina=stam if lane == "post_18" else max_stam,
            max_stamina=max_stam,
            spa_frames=int(cost.spa_refill_f),
            unlimited=True,
        )
        accounts[name] = frames
        if lane != "post_18":
            stam = max_stam
    blocks = []
    stam = max_stam
    remaining = budget
    for name, lane, block_units in units:
        frames, refills, done, stam = _spend_units(
            block_units,
            lane=lane,
            budget_left=remaining,
            stamina=stam if lane == "post_18" else max_stam,
            max_stamina=max_stam,
            spa_frames=int(cost.spa_refill_f),
            unlimited=lane == "post_18",
        )
        if lane == "day":
            remaining -= frames
            stam = max_stam
        blocks.append(
            {
                "block": name,
                "lane": lane,
                "account": int(accounts[name]),
                "frames": int(frames),
                "units": len(block_units),
                "done": int(done),
                "hot_spring_refills": int(refills),
            }
        )
    day_frames = sum(int(row["frames"]) for row in blocks if row["lane"] == "day")
    return {
        "order": list(BLOCK_ORDER),
        "day_budget": budget,
        "day_frames": day_frames,
        "within_envelope": day_frames <= budget,
        "evidence_kind": "model",
        "emulator_frames": 0,
        "blocks": blocks,
    }


class _DayTrace:
    """Rolling hash of each env step's action and the WRAM it produced."""

    def __init__(self) -> None:
        self._hasher = hashlib.sha256()
        self.frames = 0

    def add(self, action: object, ram: object) -> None:
        act = np.ascontiguousarray([] if action is None else action, dtype=np.uint8).tobytes()
        raw = np.ascontiguousarray(ram, dtype=np.uint8).tobytes()
        self._hasher.update(len(act).to_bytes(4, "little"))
        self._hasher.update(act)
        self._hasher.update(len(raw).to_bytes(8, "little"))
        self._hasher.update(raw)
        self.frames += 1

    def digest(self) -> str:
        return self._hasher.hexdigest()


def _carry_planner(planner: object) -> dict:
    """Counters the savestate does not hold and that change the next action."""
    carried: dict = {}
    if getattr(planner, "perfect_rancher", False):
        carried["campaign"] = {"pending": dict(planner._pending_intents),
                               "completed": sorted(planner._completed_intents)}
    for name in ("_step_count", "_cow_route_failures", "_cow_failure_season"):
        if hasattr(planner, name):
            carried[name] = int(getattr(planner, name))
    for name in ("_was_storm_yesterday", "_child_pressed"):
        if hasattr(planner, name):
            carried[name] = bool(getattr(planner, name))
    floor = getattr(planner, "floor", None)
    if floor is not None and hasattr(floor, "_started_at"):
        late = getattr(floor, "_late_since", None)
        carried["floor"] = {
            "_started_at": int(floor._started_at),
            "_late_since": None if late is None else int(late),
            "_tripped": bool(getattr(floor, "_tripped", False)),
            "rescues": int(getattr(floor, "rescues", 0)),
        }
    guard = getattr(planner, "scene_guard", None)
    if guard is not None and hasattr(guard, "_quiet"):
        carried["scene_guard"] = {
            "_quiet": int(guard._quiet),
            "_active": int(guard._active),
            "_gave_up": bool(guard._gave_up),
            "_name_press_step": int(guard._name_press_step),
        }
    return carried


def _resync_loaded_pin(env: object, loaded: dict, ram) -> None:
    """Put the core back on the pin after stable-retro's blank load frame."""
    path = loaded.get("path") if isinstance(loaded, dict) else None
    if env is None or not path:
        return
    import gzip

    with gzip.open(path, "rb") as handle:
        data = handle.read()
    if hasattr(env, "load_start_state"):
        env.load_start_state(data)
    else:
        env.em.set_state(data)
    fresh = np.ascontiguousarray(env.get_ram(), dtype=np.uint8)
    ram[...] = fresh


def _restore_planner(planner: object, carried: object) -> None:
    if not isinstance(carried, dict):
        return
    campaign = carried.get("campaign")
    if isinstance(campaign, dict) and getattr(planner, "perfect_rancher", False):
        planner._pending_intents = dict(campaign.get("pending", {}))
        planner._completed_intents = set(campaign.get("completed", ()))
    for name in ("_step_count", "_cow_route_failures", "_cow_failure_season"):
        if name in carried and hasattr(planner, name):
            setattr(planner, name, int(carried[name]))
    for name in ("_was_storm_yesterday", "_child_pressed"):
        if name in carried and hasattr(planner, name):
            setattr(planner, name, bool(carried[name]))
    floor_row = carried.get("floor")
    floor = getattr(planner, "floor", None)
    if isinstance(floor_row, dict) and floor is not None and hasattr(floor, "_started_at"):
        floor._started_at = int(floor_row.get("_started_at", 0))
        late = floor_row.get("_late_since")
        floor._late_since = None if late is None else int(late)
        floor._tripped = bool(floor_row.get("_tripped", False))
        floor.rescues = int(floor_row.get("rescues", 0))
    guard_row = carried.get("scene_guard")
    guard = getattr(planner, "scene_guard", None)
    if isinstance(guard_row, dict) and guard is not None and hasattr(guard, "_quiet"):
        guard._quiet = int(guard_row.get("_quiet", 0))
        guard._active = int(guard_row.get("_active", 0))
        guard._gave_up = bool(guard_row.get("_gave_up", False))
        guard._name_press_step = int(guard_row.get("_name_press_step", -(10**9)))


class SeasonPinSession:
    """Write season or daily morning pins and remember the parent run.

    The pin is the morning the new season's day plan starts (settle already
    finished, day task not stepped). The trace is every env step after that
    until :meth:`finish`. A ``--state PR_*`` resume hashes the same span.
    """

    def __init__(
        self,
        *,
        save: bool,
        power_on: bool,
        source_sha256: Optional[str],
        start_key: tuple[int, int, int],
        state_name: str,
        save_dir: Path,
        daily: bool = False,
        prefix: str = "PR_",
    ) -> None:
        self.save = bool(save)
        self.daily = bool(daily)
        self.prefix = prefix
        self.save_dir = Path(save_dir)
        self.start_key = (int(start_key[0]), int(start_key[1]), int(start_key[2]))
        self.parent_hash = parent_run_hash(
            power_on=power_on,
            source_sha256=source_sha256,
            start_key=self.start_key,
        )
        self._pins: list[dict] = []
        self._open: Optional[dict] = None
        self._saved: set[tuple[int, int, int]] = set()
        self.trace: Optional[_DayTrace] = None
        self.loaded: Optional[dict] = None
        path = self.save_dir / f"{state_name}.lineage.json"
        if not power_on and state_name and path.is_file():
            self.loaded = json.loads(path.read_text(encoding="utf-8"))
            # Frozen runs and copied checkouts move; resync the state beside
            # the lineage we actually opened, not its historical pathname.
            self.loaded["path"] = str(self.save_dir / f"{state_name}.state")

    def before_step(self, planner: object, ram, frame: int, save_state: SaveState, env: object = None, *, terminal_morning: bool = False) -> None:
        if self.loaded is not None and self.trace is None:
            _resync_loaded_pin(env, self.loaded, ram)
            _restore_planner(planner, self.loaded.get("planner"))
            self.trace = _DayTrace()
        if not self.save:
            return
        year, season, day = read_world_ymd(ram)
        key = (int(year), int(season), int(day))
        if (day != 1 and not self.daily) or key == self.start_key or key in self._saved:
            return
        if not terminal_morning and getattr(planner, "_phase", None) != "plan_day":
            return
        if not terminal_morning and getattr(planner, "_current_task", None) is not None:
            return
        hour = int(read_ram_value(ram, "hour"))
        if not morning_scene_ready(classify_scene_from_ram(ram), hour):
            return
        name = f"{self.prefix}Y{year + 1}_{SEASONS[season]}_D{day}"
        path = self.save_dir / f"{name}.state"
        save_state(path)
        from harvest.planner.power_berry_calendar import berry_receipts
        record = {
            "pin": name,
            "path": str(path),
            "parent_run_hash": self.parent_hash,
            "saved_frame": int(frame),
            "date": {"year": key[0], "season": key[1], "day": key[2]},
            "power_berries": berry_receipts(ram),
            "planner": _carry_planner(planner),
            "next_day": None,
        }
        self._pins.append(record)
        self._open = record
        self._saved.add(key)
        if self.trace is None:
            self.trace = _DayTrace()
        self._write_lineage(record)
        print(
            f"[RUN] {'morning' if self.daily else 'season'} pin {name} frame={frame} parent={self.parent_hash[:12]}",
            flush=True,
        )

    def note(self, action: object, ram: object) -> None:
        if self.trace is not None:
            self.trace.add(action, ram)

    def finish(self) -> dict:
        measured = None
        if self.trace is not None:
            measured = {"frames": self.trace.frames, "sha256": self.trace.digest()}
            if self._open is not None:
                self._open["next_day"] = measured
                self._write_lineage(self._open)
            self.trace = None
        expect = None if self.loaded is None else self.loaded.get("next_day")
        match = None
        if isinstance(expect, dict) and measured is not None:
            match = (
                int(expect.get("frames", -1)) == int(measured["frames"])
                and str(expect.get("sha256", "")) == measured["sha256"]
            )
        return {
            "parent_run_hash": self.parent_hash,
            "loaded_lineage": self.loaded,
            "pins": list(self._pins),
            "next_day": measured,
            "next_day_match": match,
        }

    def _write_lineage(self, record: dict) -> None:
        path = self.save_dir / f"{record['pin']}.lineage.json"
        path.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")


def season_pin_session(args: object, start_key: tuple[int, int, int], source_sha256: Optional[str]):
    """None unless this run saves morning pins or resumes a pin with lineage."""
    power_on = bool(getattr(args, "power_on", False))
    state_name = "" if power_on else str(getattr(args, "state", "") or "")
    daily = bool(getattr(args, "save_day_pins", False))
    save = bool(getattr(args, "save_season_pins", False)) or daily
    has_lineage = bool(state_name) and (GAME_DIR / f"{state_name}.lineage.json").is_file()
    if not save and not state_name.startswith("PR_") and not has_lineage:
        return None
    return SeasonPinSession(
        save=save,
        power_on=power_on,
        source_sha256=source_sha256,
        start_key=tuple(start_key),
        state_name=state_name,
        save_dir=GAME_DIR,
        daily=daily,
        prefix=str(getattr(args, "pin_prefix", "PR_")),
    )
