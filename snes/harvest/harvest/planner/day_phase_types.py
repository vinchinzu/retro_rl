"""Shared phase data types for day planning."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

from harvest.planner.livestock_econ import HENS_KEPT


class PhaseKind(StrEnum):
    """Typed phase kinds for day-plan task construction."""

    EXIT = "exit"
    FARM_BUILDING_EXIT = "farm_building_exit"
    FARM_EXIT = "farm_exit"
    NAV = "nav"
    RECORDED = "recorded"
    RECORDED_SLICE = "recorded_slice"
    RECORDED_TRANSITION = "recorded_transition"
    CROSS_MAP = "cross_map"
    SHOP_BUY = "shop_buy"
    DIRECTIONAL_TRANSITION = "directional_transition"
    MULTI_NAV = "multi_nav"
    MOUNTAIN_BERRY = "mountain_berry"
    POWER_BERRY = "power_berry"
    ENSURE_TOOL = "ensure_tool"
    ENSURE_ANIMAL_TOOLS = "ensure_animal_tools"
    ENSURE_SEED = "ensure_seed"
    DEADLINE = "deadline"
    WAIT_UNTIL_TIME = "wait_until_time"
    HARVEST = "harvest"
    CLEAR_FIELD = "clear_field"
    FENCE_CLEAR = "fence_clear"
    COOP_CHORES = "coop_chores"
    COW_CHORES = "cow_chores"
    COW_ROUNDUP = "cow_roundup"
    COW_PURCHASE = "cow_purchase"
    HEN_DELIVERY = "hen_delivery"
    CROP = "crop"
    GRASS = "grass"
    HOT_SPRING = "hot_spring"
    RETURN_HOME = "return_home"
    SLEEP = "sleep"
    READY_TO_GO_HOME = "ready_to_go_home"
    EVE_TALK_LOOP = "eve_talk_loop"
    FLOWER_FESTIVAL = "flower_festival"
    HARVEST_HAPPINESS = "harvest_happiness"
    PROPOSE_EVE = "propose_eve"
    GIVE_WIFE = "give_wife"
    PICKUP_CHICKEN = "pickup_chicken"
    DROP_CHICKEN = "drop_chicken"
    CHICKEN_SALE_FOLLOWUP = "chicken_sale_followup"
    CHICKEN_SALE_REQUEST = "chicken_sale_request"
    CHICKEN_SALE_EVENT = "chicken_sale_event"
    PROPOSE = "propose"
    COURT = "court"
    FEATHER_BUY = "feather_buy"
    HOUSE_WOOD = "house_wood"
    HOME_UPGRADE = "home_upgrade"
    DYNAMIC_OUTDOOR_PLAN = "dynamic_outdoor_plan"
    CHURCH_COURT = "church_court"
    TOWN_COURT = "town_court"


# Last-resort stall backstop for every phase without a watchdog of its own. The day floor is off by design (owner,
# 2026-11-02: no per-day cap), so a phase that loops silently ran forever: Y2 Spring D8 of ``compose6`` sat in
# ENSURE_ANIMAL_TOOLS for 2M frames. The longest ordinary phase in a finished Year 3 run is 31k frames (COVER_FIELD),
# so 120k leaves ample room for real work; the phases below run long on purpose and keep no backstop.
PHASE_BACKSTOP_FRAMES = 120_000
BACKSTOP_EXEMPT_PHASES = frozenset({"COVER_FIELD", "COVER_TILL", "EVE_TALK_LOOP", "HARVEST_HAPPINESS"})


def phase_backstop_frames(spec: "PhaseSpec") -> int:
    """Frames after which a phase with no watchdog fails; 0 for the day-ending and deliberately long phases."""
    if spec.phase in BACKSTOP_EXEMPT_PHASES or spec.params.get("ends_day"):
        return 0
    return max(PHASE_BACKSTOP_FRAMES, 2 * int(spec.params.get("timeout") or 0))


# Phases whose job is to go and fetch the tool they declare. Their
# ``required_tools`` is a postcondition, so a missing tag must not skip them.
ACQUIRE_TOOL_KINDS = frozenset(
    {
        PhaseKind.ENSURE_TOOL,
        PhaseKind.ENSURE_ANIMAL_TOOLS,
        PhaseKind.ENSURE_SEED,
    }
)

SKIP_MAP_LOCK_KINDS = frozenset(
    {
        PhaseKind.RECORDED,
        PhaseKind.RECORDED_SLICE,
        PhaseKind.RECORDED_TRANSITION,
        PhaseKind.CROSS_MAP,
        PhaseKind.SHOP_BUY,
        PhaseKind.MULTI_NAV,
        PhaseKind.MOUNTAIN_BERRY,
        PhaseKind.POWER_BERRY,
        PhaseKind.ENSURE_TOOL,
        PhaseKind.ENSURE_ANIMAL_TOOLS,
        PhaseKind.ENSURE_SEED,
        PhaseKind.CLEAR_FIELD,
        PhaseKind.GRASS,
        PhaseKind.FENCE_CLEAR,
        PhaseKind.COW_PURCHASE,
        PhaseKind.HEN_DELIVERY,
        PhaseKind.EVE_TALK_LOOP,
        PhaseKind.FLOWER_FESTIVAL,
        PhaseKind.PROPOSE_EVE,
        PhaseKind.HOT_SPRING,
        PhaseKind.PROPOSE,
        PhaseKind.COURT,
        PhaseKind.FEATHER_BUY,
        PhaseKind.CHURCH_COURT,
        PhaseKind.TOWN_COURT,
    }
)


def coerce_phase_kind(kind: PhaseKind | str) -> PhaseKind | str:
    """Return a :class:`PhaseKind` when known; preserve unknown strings for tests."""
    if isinstance(kind, PhaseKind):
        return kind
    try:
        return PhaseKind(kind)
    except ValueError:
        return kind


@dataclass(frozen=True)
class TaskContract:
    """Declarative preconditions / estimates for a phase or skill.

    Optional on PhaseSpec so advisors and tests can validate proposals without
    executing them. Empty fields mean "no contract declared."

    Soft checks live in :func:`evaluate_task_contract` — contracts document
    intent and gate advisor rewrites; they do not hard-abort builders yet.
    """

    required_ram: tuple[str, ...] = ()
    required_maps: tuple[int, ...] = ()
    required_tools: tuple[str, ...] = ()
    estimated_frames: int | None = None
    failure_modes: tuple[str, ...] = ()

    def to_jsonable(self) -> dict[str, Any]:
        return {
            "required_ram": list(self.required_ram),
            "required_maps": list(self.required_maps),
            "required_tools": list(self.required_tools),
            "estimated_frames": self.estimated_frames,
            "failure_modes": list(self.failure_modes),
        }

    def is_empty(self) -> bool:
        return (
            not self.required_ram
            and not self.required_maps
            and not self.required_tools
            and self.estimated_frames is None
            and not self.failure_modes
        )

    @classmethod
    def from_mapping(cls, data: dict[str, Any] | None) -> "TaskContract":
        if not data:
            return cls()
        return cls(
            required_ram=tuple(str(x) for x in data.get("required_ram", ()) or ()),
            required_maps=tuple(int(x) for x in data.get("required_maps", ()) or ()),
            required_tools=tuple(str(x) for x in data.get("required_tools", ()) or ()),
            estimated_frames=(
                int(data["estimated_frames"])
                if data.get("estimated_frames") is not None
                else None
            ),
            failure_modes=tuple(str(x) for x in data.get("failure_modes", ()) or ()),
        )


def evaluate_task_contract(
    contract: TaskContract,
    *,
    tilemap: int | None = None,
    tools: tuple[str, ...] | list[str] | set[str] | frozenset[str] | None = None,
    ram: Any = None,
) -> tuple[bool, tuple[str, ...]]:
    """Soft pre-check for advisors, probes, and unit tests.

    Returns ``(ok, reasons)``. Missing optional observation inputs skip that
    clause (e.g. no ``tilemap`` → map requirements are not evaluated). Does not
    execute the phase and never mutates RAM.
    """
    if contract.is_empty():
        return True, ()

    reasons: list[str] = []

    if contract.required_maps and tilemap is not None:
        have = int(tilemap)
        if have not in contract.required_maps:
            need = ",".join(f"0x{m:02X}" for m in contract.required_maps)
            reasons.append(f"map_mismatch:have=0x{have:02X}:need={need}")

    if contract.required_tools and tools is not None:
        have = {str(t).lower() for t in tools}
        for tool in contract.required_tools:
            if str(tool).lower() not in have:
                reasons.append(f"missing_tool:{tool}")

    if contract.required_ram:
        # Lazy import keeps day_phase_types free of numpy at module import time
        # for pure type/serialization unit tests.
        from harvest.core.ram_catalog import field_spec, read_ram_value

        for name in contract.required_ram:
            try:
                field_spec(name)
            except KeyError:
                reasons.append(f"unknown_ram_field:{name}")
                continue
            if ram is not None:
                try:
                    read_ram_value(ram, name)
                except Exception as exc:  # pragma: no cover - defensive
                    reasons.append(f"ram_unreadable:{name}:{type(exc).__name__}")

    return (not reasons), tuple(reasons)


# Contract tool tags ↔ carry item IDs. Seed bags share the generic "seed" tag.
_TOOL_TAG_BY_ITEM_ID: dict[int, str] = {
    0x01: "sickle",
    0x02: "hoe",
    0x03: "hammer",
    0x04: "axe",
    0x0E: "milker",
    0x0F: "brush",
    0x10: "watering_can",
}
_SEED_ITEM_IDS = frozenset({0x05, 0x06, 0x07, 0x08, 0x0C})


def tool_tags_from_ram(ram: Any) -> tuple[str, ...]:
    """Map the two-slot carry pair to contract tool tags (``hoe``, ``seed``, …)."""
    from harvest.core.carry import carry_pair_items

    tags: set[str] = set()
    for item in carry_pair_items(ram):
        item_id = int(item)
        if item_id in _SEED_ITEM_IDS:
            tags.add("seed")
        tag = _TOOL_TAG_BY_ITEM_ID.get(item_id)
        if tag:
            tags.add(tag)
    return tuple(sorted(tags))


def tilemap_from_ram(ram: Any) -> int | None:
    """Best-effort live tilemap for contract preflight."""
    from harvest.core.ram_catalog import field_spec, read_ram_value

    try:
        return int(read_ram_value(ram, "tilemap", raw=True))
    except Exception:
        try:
            addr = field_spec("tilemap").address
            return int(ram[addr]) if addr < len(ram) else None
        except Exception:
            return None


def preflight_phase_contract(
    phase: "PhaseSpec",
    *,
    ram: Any = None,
    tilemap: int | None = None,
    tools: tuple[str, ...] | list[str] | set[str] | frozenset[str] | None = None,
) -> dict[str, Any]:
    """Soft contract preflight for probes, advisors, and day-plan diagnostics.

    Never aborts execution. Empty contracts report ``ok=True`` with
    ``empty=True``. When ``ram`` is provided, tools and tilemap are inferred
    unless explicitly overridden.
    """
    contract = phase.contract
    inferred_tools = tools
    if inferred_tools is None and ram is not None:
        inferred_tools = tool_tags_from_ram(ram)
    inferred_tilemap = tilemap
    if inferred_tilemap is None and ram is not None:
        inferred_tilemap = tilemap_from_ram(ram)

    ok, reasons = evaluate_task_contract(
        contract,
        tilemap=inferred_tilemap,
        tools=inferred_tools,
        ram=ram,
    )
    return {
        "phase": phase.phase,
        "kind": str(phase.kind),
        "ok": bool(ok),
        "empty": contract.is_empty(),
        "reasons": list(reasons),
        "tools": list(inferred_tools) if inferred_tools is not None else None,
        "tilemap": (
            int(inferred_tilemap) if inferred_tilemap is not None else None
        ),
        "tilemap_hex": (
            f"0x{int(inferred_tilemap):02X}" if inferred_tilemap is not None else None
        ),
        "contract": None if contract.is_empty() else contract.to_jsonable(),
        "failure_policy": phase.failure_policy,
    }


@dataclass
class PhaseSpec:
    phase: str
    kind: PhaseKind | str
    params: dict[str, Any] = field(default_factory=dict)
    failure_policy: str = "required"
    contract: TaskContract = field(default_factory=TaskContract)

    def __init__(
        self,
        phase: str,
        kind: PhaseKind | str,
        params: dict[str, Any] | None = None,
        failure_policy: str = "required",
        *,
        contract: TaskContract | dict[str, Any] | None = None,
        required_ram: tuple[str, ...] | list[str] | None = None,
        required_maps: tuple[int, ...] | list[int] | None = None,
        required_tools: tuple[str, ...] | list[str] | None = None,
        estimated_frames: int | None = None,
        failure_modes: tuple[str, ...] | list[str] | None = None,
    ) -> None:
        self.phase = phase
        self.kind = coerce_phase_kind(kind)
        self.params = dict(params or {})
        self.failure_policy = failure_policy
        if isinstance(contract, TaskContract):
            base = contract
        elif isinstance(contract, dict):
            base = TaskContract.from_mapping(contract)
        else:
            base = TaskContract()
        # Explicit kwargs override contract fields when provided.
        self.contract = TaskContract(
            required_ram=tuple(required_ram) if required_ram is not None else base.required_ram,
            required_maps=tuple(required_maps) if required_maps is not None else base.required_maps,
            required_tools=tuple(required_tools) if required_tools is not None else base.required_tools,
            estimated_frames=(
                estimated_frames if estimated_frames is not None else base.estimated_frames
            ),
            failure_modes=tuple(failure_modes) if failure_modes is not None else base.failure_modes,
        )


@dataclass(frozen=True)
class DayPlannerPolicy:
    berry_cutoff_hour: int = 15
    berry_exit_cutoff_hour: int = 14
    # HaveLunch is a fixed +20 stamina pulse at 12:00 wherever we stand.
    lunch_hour: int = 12
    lunch_minute: int = 0
    # Flower-shop seed buy window (planning start hour). Berries ship by 15:00;
    # keep shop open long enough that a morning berry run can still buy after.
    buy_seed_hour: int = 12
    late_water_hour: int = 17
    include_chickens: bool = True
    include_cows: bool = True
    # Chores continue when purchases stop (see COW_ROUTE_MAX_FAILURES).
    include_animal_purchases: bool = True
    include_harvest: bool = True
    include_field_clear: bool = True
    include_bush_clear: bool = True
    include_spring_till: bool = True
    include_watering: bool = True
    include_planting: bool = True
    include_pasture: bool = True
    include_mushroom_gather: bool = True
    # Early-spring money: mountain berries + ship. Default on so D2+ does not
    # thrash CLEAR_FIELD all day with empty pockets (live power-on feedback).
    include_berry_run: bool = True
    include_shop_run: bool = True
    include_end_day: bool = True
    include_spa: bool = True
    include_chicken_sales: bool = True
    # Ship eggs instead of incubating them while ``livestock_econ.hatch_allowed``
    # is shut. Every adult past ``max_adult_chickens`` is sold and goes unfed.
    hold_flock_growth: bool = False
    max_adult_chickens: int = HENS_KEPT
    chicken_sale_cutoff_hour: int = 10
    # Festival and Sunday event planning
    include_festivals: bool = True
    # The only festival macro is a whole-day Spring recording from the house; it
    # replays blind inputs from anywhere else (Fall D12 burned 20k frames doing
    # nothing). Festival days close the shops but the bot does not attend.
    attend_festivals: bool = False
    # The church macro replays a Spring farm recording blind; in Winter it walked
    # into the path horse event and stranded the run. Nothing needs it.
    include_sunday_church: bool = False
    include_sunday_mountain: bool = False
    # Seasonal default crop preferences
    spring_crop: str = "potato"
    summer_crop: str = "corn"
    fall_crop: str = "eggplant"
    # None = derive from calendar season via crop_planner.
    seed_purchase_recording: str | None = None

    def lunch_clock(self):
        from harvest.core.game_clock import ClockTime

        return ClockTime(self.lunch_hour, self.lunch_minute)


def day_planner_policy_for_season(
    season: int | str,
    base: DayPlannerPolicy | None = None,
    *,
    year: int = 0,
) -> DayPlannerPolicy:
    """Adjust a day-plan policy for the calendar: planting vs off-season.

    ``year`` is the ROM year byte (0 = Year 1).
    """
    from harvest.planner.forage_calendar import forage_for_season
    from harvest.planner.crop_planner import (
        SEASON_FALL,
        SEASON_WINTER,
        is_crop_planting_season,
        normalize_season,
        seed_purchase_recording_for_season,
    )

    policy = base or DayPlannerPolicy()
    season_id = normalize_season(season)
    planting = is_crop_planting_season(season_id)
    recording = policy.seed_purchase_recording
    if recording is None and planting:
        recording = seed_purchase_recording_for_season(season_id)
    # Fall and Winter: the ROM has no fall seed, growing crops that are
    # watered in Fall die (NightlyFarmTilesCheck DECs odd growing ids) and
    # freeze in Winter, and CLEAR_FIELD's lift_only failure repeats daily. The
    # Fall keeps animal care and mushroom shipping; neither needs crop water
    # or a whole-field clear. Winter keeps animal care and sleep.
    off_season = season_id in (SEASON_FALL, SEASON_WINTER)
    winter = season_id == SEASON_WINTER
    return DayPlannerPolicy(
        berry_cutoff_hour=policy.berry_cutoff_hour,
        berry_exit_cutoff_hour=policy.berry_exit_cutoff_hour,
        lunch_hour=policy.lunch_hour,
        lunch_minute=policy.lunch_minute,
        buy_seed_hour=policy.buy_seed_hour,
        late_water_hour=policy.late_water_hour,
        include_chickens=policy.include_chickens,
        include_cows=policy.include_cows,
        include_animal_purchases=policy.include_animal_purchases,
        include_harvest=policy.include_harvest,
        # Summer pockets clear their own planting area. A whole-field pass
        # spends stamina that the pasture and repeat-harvest corn need.
        include_field_clear=policy.include_field_clear and season_id == 0,
        include_bush_clear=policy.include_bush_clear and season_id != SEASON_WINTER,
        include_spring_till=policy.include_spring_till and season_id == 0,
        include_watering=policy.include_watering and not off_season,
        include_planting=planting and policy.include_planting,
        include_pasture=policy.include_pasture,
        include_mushroom_gather=policy.include_mushroom_gather and season_id == SEASON_FALL,
        # Forage by season (``forage_calendar``): grapes in Spring, the green fruit
        # in Summer; Fall mushrooms have their own phase and Winter has none. The
        # Summer picks that came back "unverified after 3 picks" and stranded the
        # farmer (run summer5 D16) went to the grape stand, which is empty then.
        include_berry_run=(
            policy.include_berry_run
            and forage_for_season(season_id) is not None
            and season_id != SEASON_FALL
        ),
        # Winter keeps town trips: the path is tilemap 0x0F and the one-off
        # horse scene is met by ``MEET_HORSE`` (``day_phase_calendar``).
        include_shop_run=policy.include_shop_run,
        include_end_day=policy.include_end_day,
        include_spa=policy.include_spa and not winter,
        include_chicken_sales=policy.include_chicken_sales,
        hold_flock_growth=policy.hold_flock_growth,
        max_adult_chickens=policy.max_adult_chickens,
        chicken_sale_cutoff_hour=policy.chicken_sale_cutoff_hour,
        include_festivals=policy.include_festivals,
        attend_festivals=policy.attend_festivals,
        include_sunday_church=policy.include_sunday_church,
        include_sunday_mountain=policy.include_sunday_mountain,
        spring_crop=policy.spring_crop,
        summer_crop=policy.summer_crop,
        fall_crop=policy.fall_crop,
        seed_purchase_recording=recording if planting else None,
    )


__all__ = [
    "DayPlannerPolicy",
    "PhaseKind",
    "PhaseSpec",
    "ACQUIRE_TOOL_KINDS",
    "SKIP_MAP_LOCK_KINDS",
    "TaskContract",
    "coerce_phase_kind",
    "day_planner_policy_for_season",
    "evaluate_task_contract",
]
