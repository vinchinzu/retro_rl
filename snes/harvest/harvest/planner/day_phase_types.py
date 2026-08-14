"""Shared phase data types for day planning."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any


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
    BERRY_SHIP = "berry_ship"
    MOUNTAIN_BERRY = "mountain_berry"
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
    COW_PURCHASE = "cow_purchase"
    CROP = "crop"
    HOT_SPRING = "hot_spring"
    RETURN_HOME = "return_home"
    SLEEP = "sleep"
    READY_TO_GO_HOME = "ready_to_go_home"
    EVE_TALK_LOOP = "eve_talk_loop"
    PICKUP_CHICKEN = "pickup_chicken"
    DROP_CHICKEN = "drop_chicken"
    CHICKEN_SALE_FOLLOWUP = "chicken_sale_followup"
    CHICKEN_SALE_REQUEST = "chicken_sale_request"
    CHICKEN_SALE_EVENT = "chicken_sale_event"
    DYNAMIC_OUTDOOR_PLAN = "dynamic_outdoor_plan"


SKIP_MAP_LOCK_KINDS = frozenset(
    {
        PhaseKind.RECORDED,
        PhaseKind.RECORDED_SLICE,
        PhaseKind.RECORDED_TRANSITION,
        PhaseKind.CROSS_MAP,
        PhaseKind.SHOP_BUY,
        PhaseKind.MULTI_NAV,
        PhaseKind.BERRY_SHIP,
        PhaseKind.MOUNTAIN_BERRY,
        PhaseKind.ENSURE_TOOL,
        PhaseKind.ENSURE_ANIMAL_TOOLS,
        PhaseKind.ENSURE_SEED,
        PhaseKind.CLEAR_FIELD,
        PhaseKind.FENCE_CLEAR,
        PhaseKind.COW_PURCHASE,
        PhaseKind.EVE_TALK_LOOP,
        PhaseKind.HOT_SPRING,
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
    # Flower-shop seed buy window (planning start hour). Berries ship by 15:00;
    # keep shop open long enough that a morning berry run can still buy after.
    buy_seed_hour: int = 12
    late_water_hour: int = 17
    include_chickens: bool = True
    include_cows: bool = True
    include_harvest: bool = True
    include_field_clear: bool = True
    include_watering: bool = True
    include_planting: bool = True
    # Early-spring money: mountain berries + ship. Default on so D2+ does not
    # thrash CLEAR_FIELD all day with empty pockets (live power-on feedback).
    include_berry_run: bool = True
    include_shop_run: bool = True
    include_end_day: bool = True
    include_chicken_sales: bool = True
    max_adult_chickens: int = 2
    chicken_sale_cutoff_hour: int = 10
    # None = derive from calendar season via crop_planner.
    seed_purchase_recording: str | None = None


def day_planner_policy_for_season(
    season: int | str,
    base: DayPlannerPolicy | None = None,
) -> DayPlannerPolicy:
    """Adjust a day-plan policy for spring/summer planting vs fall/winter.

    HM SNES has no fall/winter field crops, so planting and seed shopping stop
    after summer while harvest/water remain gated by live crop tiles.
    """
    from harvest.planner.crop_planner import (
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
    return DayPlannerPolicy(
        berry_cutoff_hour=policy.berry_cutoff_hour,
        berry_exit_cutoff_hour=policy.berry_exit_cutoff_hour,
        buy_seed_hour=policy.buy_seed_hour,
        late_water_hour=policy.late_water_hour,
        include_chickens=policy.include_chickens,
        include_cows=policy.include_cows,
        include_harvest=policy.include_harvest,
        include_field_clear=policy.include_field_clear,
        include_watering=policy.include_watering,
        include_planting=planting and policy.include_planting,
        include_berry_run=policy.include_berry_run,
        include_shop_run=policy.include_shop_run,
        include_end_day=policy.include_end_day,
        include_chicken_sales=policy.include_chicken_sales,
        max_adult_chickens=policy.max_adult_chickens,
        chicken_sale_cutoff_hour=policy.chicken_sale_cutoff_hour,
        seed_purchase_recording=recording if planting else None,
    )


__all__ = [
    "DayPlannerPolicy",
    "PhaseKind",
    "PhaseSpec",
    "SKIP_MAP_LOCK_KINDS",
    "TaskContract",
    "coerce_phase_kind",
    "day_planner_policy_for_season",
    "evaluate_task_contract",
]
