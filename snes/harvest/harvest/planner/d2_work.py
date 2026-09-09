"""Spring D2 work — one D2_FARM_CLEAR Tactic after BUY_SEEDS.

Grape → shop → CLEAR_PLOT → plant 8 → water 8 → leftover smash.
Lift is section-first: weeds then fences then stones in one quadrant
before walking to the next. Hammer/rocks then axe/stumps still chain
by chunk after hands work. ``next_d2_spec`` is the live order.
"""

from __future__ import annotations

from dataclasses import dataclass, fields
from enum import StrEnum
from typing import List, Sequence

from retro_harness import ActionResult, TaskResult, TaskStatus, WorldState

from harvest.core.stamina import Stamina
from harvest.core.task_progress import GOAL_STALL_FRAMES, MOTION_STALL_FRAMES
from harvest.core.tile_catalog import LARGE_ROCK_DAMAGE_TILES, Tool
from harvest.planner.d2_farm_chunks import (
    EXHAUSTIVE,
    FARM_CHUNK_BOUNDS,
    FARM_CHUNK_ORDER,
    chunk_of_tile,
    resolve_chunks,
)
from harvest.planner.day_phase_catalog import (
    CROP_ESTABLISH_PHASE,
    ENSURE_CROP_SEEDS_PHASE,
    ENSURE_WATERING_CAN_PHASE,
    NAV_CROP_PHASE,
)
from harvest.planner.day_phase_stamina import coerce_stamina, full_restore_spa_phase
from harvest.maps.map_config import WEST_PLANT_POCKET_BOUNDS
from harvest.planner.day_phase_types import PhaseSpec


# Crop establishment targets are separate from the evening debris quotas.
D2_TARGETS = {
    "plant": 8,
    "water": 8,
}
D2_SEASON = 0  # Spring in the Harvest Moon RAM calendar.
D2_DAY = 2

D2_LEFTOVER_PHASE_NAMES = (
    "HOT_SPRING_STAMINA",
    "CLEAR_BUSHES",
    "CLEAR_FENCES",
    "CLEAR_STONES",
    "ENSURE_HAMMER",
    "CLEAR_ROCKS",
    "ENSURE_AXE",
    "CLEAR_STUMPS",
)
_QUEUED_LEFTOVER = frozenset(("D2_FARM_CLEAR",) + D2_LEFTOVER_PHASE_NAMES) - {
    "HOT_SPRING_STAMINA"
}
_SPA_RETRY_PHASES = frozenset({"CLEAR_ROCKS", "CLEAR_STUMPS"})
_SECTION_KEY = {
    "bushes": "weeds",
    "fences": "fences",
    "stones": "stones",
    "rocks": "large_rocks",
    "stumps": "stumps",
}


class D2FarmOutcome(StrEnum):
    COMPLETE = "complete"
    WORK_REMAINING = "work_remaining"
    TEMPORARILY_UNOBSERVABLE = "temporarily_unobservable"
    BLOCKED = "blocked"


@dataclass(frozen=True)
class D2FarmStatus:
    season: int
    day: int
    planted: int
    wet: int
    weeds: int
    fences: int
    stones: int
    small_rocks: int
    large_rocks: int
    trees_or_stumps: int
    damaged_boulder: bool
    stamina: Stamina
    hands_clear: bool
    farm_map_loaded: bool
    animating: bool
    shipped_before_17: bool
    hour: int
    minute: int
    tilemap: int
    input_stable: bool
    settled: bool
    outcome: D2FarmOutcome
    reason: str = ""
    pocket_needs_clear: bool = False
    potato_seeds: int = 0
    weeds_by_chunk: tuple[int, ...] = ()
    fences_by_chunk: tuple[int, ...] = ()
    stones_by_chunk: tuple[int, ...] = ()
    rocks_by_chunk: tuple[int, ...] = ()
    stumps_by_chunk: tuple[int, ...] = ()

    @property
    def is_complete(self) -> bool:
        return self.outcome == D2FarmOutcome.COMPLETE

    @property
    def stumps(self) -> int:
        """Compatibility alias; ``trees_or_stumps`` is the final contract name."""
        return self.trees_or_stumps

    def to_record(self) -> dict[str, object]:
        """Serialize every terminal clause for a final D2 evidence report."""
        row: dict[str, object] = {}
        for item in fields(self):
            if item.name == "stamina":
                continue
            value = getattr(self, item.name)
            if item.name == "outcome":
                value = value.value
            elif item.name == "stumps_by_chunk":
                row["trees_or_stumps_by_chunk"] = list(value)
                continue
            elif isinstance(value, tuple):
                value = list(value)
            row[item.name] = value
        return row


def _required_clear(
    phase: str,
    params: dict,
    *,
    required_tools: Sequence[str] = (),
    estimated_frames: int = 8000,
    failure_modes: Sequence[str] = (
        "timeout_budget",
        "tool_missing",
        "stamina_low",
        "debris_remaining",
    ),
) -> PhaseSpec:
    return PhaseSpec(
        phase,
        "clear_field",
        params,
        failure_policy="required",
        required_maps=(0x00,),
        required_tools=tuple(required_tools),
        estimated_frames=estimated_frames,
        failure_modes=tuple(failure_modes),
    )


def pocket_clear_phase() -> PhaseSpec:
    """Lift weeds/stones on the 3x3+stands. Hands off via plot_ring."""
    return PhaseSpec(
        "CLEAR_PLOT",
        "clear_field",
        {
            "timeout": 7000,
            "fetch_tools": False,
            "prefer_lift_for_weeds": True,
            "prefer_lift_for_stones": True,
            "farm_bounds": WEST_PLANT_POCKET_BOUNDS,
            "priority": ["weed", "stone"],
            "handoff": "plot_ring",
        },
        failure_policy="optional",
        required_maps=(0x00,),
        estimated_frames=5000,
        failure_modes=("timeout_budget", "pocket_sealed"),
    )


def pocket_water_phase() -> PhaseSpec:
    """8-ring water from the untilled notch. Can pass, not the plant pair."""
    return PhaseSpec(
        "CROP_WATER",
        "crop",
        {
            "work_mode": "pocket",
            "refill_bounds": (3, 10, 62, 60),
            "min_wet": D2_TARGETS["water"],
        },
        failure_policy="optional",
        required_maps=(0x00,),
        required_tools=("watering_can",),
        estimated_frames=6000,
        failure_modes=("empty_can", "refill_fail", "dry_ring", "precheck_tool_success"),
    )


def ensure_hammer_phase() -> PhaseSpec:
    return PhaseSpec(
        "ENSURE_HAMMER",
        "ensure_tool",
        {"tool_id": int(Tool.HAMMER)},
        failure_policy="optional",
        required_tools=("hammer",),
        estimated_frames=8000,
        failure_modes=("shelf_miss", "carry_full"),
    )


def ensure_axe_phase() -> PhaseSpec:
    return PhaseSpec(
        "ENSURE_AXE",
        "ensure_tool",
        {"tool_id": int(Tool.AXE)},
        failure_policy="optional",
        required_tools=("axe",),
        estimated_frames=8000,
        failure_modes=("shelf_miss", "carry_full"),
    )


def bush_clear_phase(*, farm_bounds=None, chunk: str | None = None) -> PhaseSpec:
    """Lift remaining weeds in bounds (or the whole farm)."""
    return _required_clear(
        "CLEAR_BUSHES",
        _with_chunk(
            {
                "timeout": 0,
                "fetch_tools": False,
                "prefer_lift_for_weeds": True,
                "prefer_lift_for_stones": True,
                "priority": ["weed"],
                "quota": {"weeds": EXHAUSTIVE},
                "handoff": "quota",
            },
            farm_bounds=farm_bounds,
            chunk=chunk,
        ),
        estimated_frames=100000,
    )


def fence_dump_phase(*, farm_bounds=None, chunk: str | None = None) -> PhaseSpec:
    """Lift remaining fence posts in bounds and dump them in a pond."""
    return PhaseSpec(
        "CLEAR_FENCES",
        "fence_clear",
        _with_chunk(
            {
                "timeout": 0,
                "max_fences": None,
                "corridor_only": False,
                "pond_dump": True,
                "max_steps_per_fence": 2800,
                "max_failures": 20,
                "debris_types": ["fence"],
            },
            farm_bounds=farm_bounds,
            chunk=chunk,
        ),
        failure_policy="required",
        required_maps=(0x00,),
        estimated_frames=200000,
        failure_modes=("timeout_budget", "no_reachable_fence"),
    )


def _with_chunk(params: dict, *, farm_bounds=None, chunk: str | None = None) -> dict:
    out = dict(params)
    if farm_bounds is not None:
        out["farm_bounds"] = tuple(int(v) for v in farm_bounds)
    if chunk is not None:
        out["chunk"] = chunk
    return out


def stone_pond_phase(*, farm_bounds=None, chunk: str | None = None) -> PhaseSpec:
    """Lift remaining stones in bounds (or the whole farm) and dump in a pond.

    After_Stumps (axe selected, hoe backpack) still lifts; do not stow first.
    """
    return PhaseSpec(
        "CLEAR_STONES",
        "fence_clear",
        _with_chunk(
            {
                "timeout": 0,
                "max_fences": None,
                "corridor_only": False,
                "pond_dump": True,
                "max_steps_per_fence": 2800,
                "max_failures": 60,
                "debris_types": ["stone"],
            },
            farm_bounds=farm_bounds,
            chunk=chunk,
        ),
        failure_policy="required",
        required_maps=(0x00,),
        estimated_frames=400000,
        failure_modes=("timeout_budget", "no_reachable_fence"),
    )


def rock_clear_phase(*, farm_bounds=None, chunk: str | None = None) -> PhaseSpec:
    """Hammer every remaining large 2×2 boulder in bounds."""
    return _required_clear(
        "CLEAR_ROCKS",
        _with_chunk(
            {
                "timeout": 0,
                "fetch_tools": False,
                "prefer_lift_for_weeds": True,
                "prefer_lift_for_stones": False,
                "priority": ["rock"],
                "quota": {"large_rocks": EXHAUSTIVE},
                "handoff": "quota",
            },
            farm_bounds=farm_bounds,
            chunk=chunk,
        ),
        required_tools=("hammer",),
        estimated_frames=400000,
    )


def stump_clear_phase(*, farm_bounds=None, chunk: str | None = None) -> PhaseSpec:
    """Axe every remaining stump in bounds. Axe replaces the hammer in carry."""
    return _required_clear(
        "CLEAR_STUMPS",
        _with_chunk(
            {
                "timeout": 0,
                "fetch_tools": False,
                "priority": ["stump"],
                "quota": {"stumps": EXHAUSTIVE},
                "handoff": "quota",
            },
            farm_bounds=farm_bounds,
            chunk=chunk,
        ),
        required_tools=("axe",),
        estimated_frames=400000,
    )


def should_spa_retry(
    phase: str,
    reason: str | None,
    stamina: Stamina | int | None,
    *,
    include_spa: bool,
) -> bool:
    """Insert spa+retry when a smash phase stops on stamina, not aim."""
    if not include_spa or phase not in _SPA_RETRY_PHASES:
        return False
    if "stamina_low" not in (reason or ""):
        return False
    stam = coerce_stamina(stamina)
    return stam is not None and not stam.can_finish_multi_hit()


def needs_spa_before_next_smash(
    just_finished: str,
    stamina: Stamina | int | None,
    *,
    include_spa: bool,
    remaining_phases: Sequence[str],
) -> bool:
    """Soak before the next 2×2 smash when the last rocks/stumps chunk drained us."""
    if not include_spa or just_finished not in _SPA_RETRY_PHASES:
        return False
    if not any(name in remaining_phases for name in _SPA_RETRY_PHASES):
        return False
    stam = coerce_stamina(stamina)
    return stam is not None and not stam.can_finish_multi_hit()


def leftover_chain_decision(
    phase: str,
    status: TaskStatus | str | None,
    reason: str | None,
    stamina: Stamina | int | None,
    remaining_phases: Sequence[str],
    *,
    include_spa: bool = True,
) -> str:
    if status is None:
        return "abort"
    text = status.value if isinstance(status, TaskStatus) else str(status)
    if text == TaskStatus.SUCCESS.value:
        if needs_spa_before_next_smash(
            phase,
            stamina,
            include_spa=include_spa,
            remaining_phases=remaining_phases,
        ):
            return "insert_spa"
        return "continue"
    if should_spa_retry(phase, reason, stamina, include_spa=include_spa):
        return "spa_retry"
    if "partial_clear" in str(reason or ""):
        return "continue"
    if "field_clear cleared=0" in str(reason or ""):
        return "continue"
    return "abort"


def _shipped_before_17(ram, journal) -> bool:
    """True only for an explicit Spring D2 bin-deposit event before 17:00."""
    for row in journal or ():
        if not isinstance(row, dict):
            continue
        deposit = row.get("shipping_deposit")
        if not isinstance(deposit, dict):
            continue
        if (
            int(deposit.get("season", -1)) == D2_SEASON
            and int(deposit.get("day", -1)) == D2_DAY
            and (int(deposit.get("hour", 24)), int(deposit.get("minute", 60))) < (17, 0)
            and int(deposit.get("shipping_money_after", 0))
            > int(deposit.get("shipping_money_before", 0))
        ):
            return True
    return False


def observe_d2_farm(ram, journal=None) -> D2FarmStatus:
    """Pure D2 farm observation: debris, pocket crops, shipping, map lock."""
    from harvest.core.ram_catalog import read_ram_value
    from harvest.core.tile_catalog import ADDR_INPUT_LOCK, ADDR_TILEMAP
    from harvest.maps.farm_pond import WEST_POCKET_PLANT_CENTER
    from harvest.planner.tasks.transitions import hands_are_clear
    from harvest.tasks.crop_skills import count_ring_planted, count_ring_wet
    from harvest.tasks.farm_clear_quota import classify_target, farm_map_loaded
    from harvest.tasks.farm_ops import TileScanner

    loaded = farm_map_loaded(ram)
    lock = int(ram[ADDR_INPUT_LOCK]) if ADDR_INPUT_LOCK < len(ram) else 1
    animating = lock != 1
    tilemap = int(ram[ADDR_TILEMAP]) if ADDR_TILEMAP < len(ram) else 0
    season = int(read_ram_value(ram, "season") or 0)
    day = int(read_ram_value(ram, "day") or 0)
    hour = int(read_ram_value(ram, "hour") or 0)
    minute = int(read_ram_value(ram, "minute") or 0)
    stam = Stamina.from_ram(ram)
    planted = count_ring_planted(ram, WEST_POCKET_PLANT_CENTER) if loaded else 0
    wet = count_ring_wet(ram, WEST_POCKET_PLANT_CENTER) if loaded else 0
    potato_seeds = int(read_ram_value(ram, "potato_seeds") or 0)
    hands = hands_are_clear(ram)
    shipped = _shipped_before_17(ram, journal)

    weeds = fences = stones = small_rocks = large_rocks = stumps = 0
    pocket = False
    damaged = False
    by_w = [0, 0, 0, 0]
    by_f = [0, 0, 0, 0]
    by_s = [0, 0, 0, 0]
    by_r = [0, 0, 0, 0]
    by_u = [0, 0, 0, 0]
    idx = {name: i for i, name in enumerate(FARM_CHUNK_ORDER)}
    x0, y0, x1, y1 = WEST_PLANT_POCKET_BOUNDS
    if loaded:
        for target in TileScanner().scan(ram):
            key = classify_target(int(target.tile_id), target.debris_type)
            tx, ty = target.tile
            if key == "weeds":
                weeds += 1
                by_w[idx[chunk_of_tile(tx, ty)]] += 1
                if x0 <= tx <= x1 and y0 <= ty <= y1:
                    pocket = True
            elif key == "fences":
                fences += 1
                by_f[idx[chunk_of_tile(tx, ty)]] += 1
            elif key == "stones":
                stones += 1
                by_s[idx[chunk_of_tile(tx, ty)]] += 1
                if x0 <= tx <= x1 and y0 <= ty <= y1:
                    pocket = True
            elif key == "small_rocks":
                small_rocks += 1
            elif key == "large_rocks":
                large_rocks += 1
                by_r[idx[chunk_of_tile(tx, ty)]] += 1
            elif key == "stumps":
                stumps += 1
                by_u[idx[chunk_of_tile(tx, ty)]] += 1
            if int(target.tile_id) in LARGE_ROCK_DAMAGE_TILES:
                damaged = True

    input_stable = lock == 1
    settled = loaded and input_stable and not animating
    if not loaded or not input_stable:
        outcome, why = D2FarmOutcome.TEMPORARILY_UNOBSERVABLE, (
            "stale_farm_map" if not loaded else "input_unstable"
        )
    elif (
        season == D2_SEASON
        and day == D2_DAY
        and planted >= D2_TARGETS["plant"]
        and wet >= D2_TARGETS["water"]
        and weeds == fences == stones == small_rocks == large_rocks == stumps == 0
        and not damaged
        and hands
        and shipped
        and input_stable
        and settled
    ):
        outcome, why = D2FarmOutcome.COMPLETE, ""
    else:
        outcome, why = D2FarmOutcome.WORK_REMAINING, "work_remaining"

    return D2FarmStatus(
        season=season,
        day=day,
        planted=planted,
        wet=wet,
        weeds=weeds,
        fences=fences,
        stones=stones,
        small_rocks=small_rocks,
        large_rocks=large_rocks,
        trees_or_stumps=stumps,
        damaged_boulder=damaged,
        stamina=stam,
        hands_clear=hands,
        farm_map_loaded=loaded,
        animating=animating,
        shipped_before_17=shipped,
        hour=hour,
        minute=minute,
        tilemap=tilemap,
        input_stable=input_stable,
        settled=settled,
        outcome=outcome,
        reason=why,
        pocket_needs_clear=pocket,
        potato_seeds=potato_seeds,
        weeds_by_chunk=tuple(by_w),
        fences_by_chunk=tuple(by_f),
        stones_by_chunk=tuple(by_s),
        rocks_by_chunk=tuple(by_r),
        stumps_by_chunk=tuple(by_u),
    )


def confirm_d2_complete(previous, current) -> bool:
    return (
        previous is not None
        and current is not None
        and previous.outcome == D2FarmOutcome.COMPLETE
        and current.outcome == D2FarmOutcome.COMPLETE
        and previous.settled
        and current.settled
    )


def _live_chunks(smash: Sequence[str], counts: Sequence[int]) -> list[str]:
    mapping = dict(zip(FARM_CHUNK_ORDER, counts)) if counts else {}
    return [name for name in smash if mapping.get(name, 0) > 0]


def _count_in(counts: Sequence[int], name: str) -> int:
    mapping = dict(zip(FARM_CHUNK_ORDER, counts)) if counts else {}
    return int(mapping.get(name, 0))


def _lift_next(status: D2FarmStatus, smash: Sequence[str], section: str):
    """Hands work stays in one quadrant: weeds, then fences, then stones."""
    want_w = section in {"all", "bushes"}
    want_f = section in {"all", "fences"}
    want_s = section in {"all", "stones"}
    if not (want_w or want_f or want_s):
        return None
    for name in smash:
        if want_w and _count_in(status.weeds_by_chunk, name) > 0:
            return bush_clear_phase(farm_bounds=FARM_CHUNK_BOUNDS[name], chunk=name)
        if want_f and _count_in(status.fences_by_chunk, name) > 0:
            return fence_dump_phase(farm_bounds=FARM_CHUNK_BOUNDS[name], chunk=name)
        if want_s and _count_in(status.stones_by_chunk, name) > 0:
            return stone_pond_phase(farm_bounds=FARM_CHUNK_BOUNDS[name], chunk=name)
    return None


def _crop_next(last_phase: str) -> PhaseSpec:
    if last_phase == "ENSURE_CROP_SEEDS":
        return NAV_CROP_PHASE
    if last_phase in {"NAV_CROP", "CROP_ESTABLISH"}:
        return CROP_ESTABLISH_PHASE
    return ENSURE_CROP_SEEDS_PHASE


def _water_next(last_phase: str) -> PhaseSpec:
    if last_phase in {"ENSURE_WATERING_CAN", "CROP_WATER"}:
        return pocket_water_phase()
    return ENSURE_WATERING_CAN_PHASE


def _smash_next(status, live, last, include_spa, ensure, ensure_phase, clear_phase, builder):
    if not live:
        return None
    if include_spa and not status.stamina.can_finish_multi_hit() and last != "HOT_SPRING_STAMINA":
        return full_restore_spa_phase()
    if last not in {ensure, clear_phase}:
        return ensure_phase()
    name = live[0]
    return builder(farm_bounds=FARM_CHUNK_BOUNDS[name], chunk=name)


def next_d2_spec(
    status: D2FarmStatus,
    *,
    include_spa: bool = True,
    section: str = "all",
    chunk: str | Sequence[str] = "all",
    last_phase: str = "",
    plot_attempted: bool = False,
) -> PhaseSpec | None:
    """Next mandatory D2 child spec, or None when ready to verify."""
    if status.outcome == D2FarmOutcome.TEMPORARILY_UNOBSERVABLE:
        return None
    smash = resolve_chunks(chunk)
    if section == "all":
        if (
            status.pocket_needs_clear
            and not plot_attempted
            and last_phase != "CLEAR_PLOT"
        ):
            return pocket_clear_phase()
        if status.planted < D2_TARGETS["plant"] and status.potato_seeds > 0:
            return _crop_next(last_phase)
        if status.planted >= D2_TARGETS["plant"] and status.wet < D2_TARGETS["water"]:
            return _water_next(last_phase)
    lift = _lift_next(status, smash, section)
    if lift is not None:
        return lift
    if section in {"all", "rocks"}:
        spec = _smash_next(
            status, _live_chunks(smash, status.rocks_by_chunk), last_phase,
            include_spa, "ENSURE_HAMMER", ensure_hammer_phase, "CLEAR_ROCKS",
            rock_clear_phase,
        )
        if spec is not None:
            return spec
    if section in {"all", "stumps"}:
        return _smash_next(
            status, _live_chunks(smash, status.stumps_by_chunk), last_phase,
            include_spa, "ENSURE_AXE", ensure_axe_phase, "CLEAR_STUMPS",
            stump_clear_phase,
        )
    return None


def d2_farm_clear_phase() -> PhaseSpec:
    return PhaseSpec(
        "D2_FARM_CLEAR",
        "clear_field",
        {"timeout": 0, "section": "all", "chunk": "all"},
        failure_policy="required",
        required_maps=(0x00,),
        estimated_frames=400000,
        failure_modes=(
            "stamina_low",
            "debris_remaining",
            "tool_missing",
            "stale_farm_map",
            "blocked",
        ),
    )


def d2_post_shop_work_phases() -> List[PhaseSpec]:
    """One required D2_FARM_CLEAR Tactic after BUY_SEEDS."""
    return [d2_farm_clear_phase()]


def leftover_already_queued(remaining: Sequence[str]) -> bool:
    return bool(set(remaining) & _QUEUED_LEFTOVER)


def _section_done(status: D2FarmStatus, section: str, chunk: str) -> bool:
    if status.outcome == D2FarmOutcome.TEMPORARILY_UNOBSERVABLE:
        return False
    if not status.farm_map_loaded or status.animating or not status.hands_clear:
        return False
    if section == "all":
        return status.is_complete
    key = _SECTION_KEY.get(section)
    if key is None:
        return status.is_complete
    by = {
        "bushes": status.weeds_by_chunk,
        "fences": status.fences_by_chunk,
        "stones": status.stones_by_chunk,
        "rocks": status.rocks_by_chunk,
        "stumps": status.stumps_by_chunk,
    }.get(section)
    if chunk not in (None, "all", "") and by:
        idx = dict(zip(FARM_CHUNK_ORDER, range(4))).get(chunk)
        if idx is not None and by[idx] > 0:
            return False
        return not (section == "rocks" and status.damaged_boulder)
    return int(getattr(status, key, 0)) <= 0 and not (
        section == "rocks" and status.damaged_boulder
    )


def _debris_row(st: D2FarmStatus | None) -> dict:
    if st is None:
        return {}
    return {
        "weeds": st.weeds,
        "stones": st.stones,
        "large_rocks": st.large_rocks,
        "stumps": st.stumps,
        "fences": st.fences,
    }


class D2FarmClearTactic:
    """Thin stepper: observe → next_d2_spec → build_phase_task → settle."""

    name = "d2_farm_clear"

    def __init__(
        self, *, section="all", chunk="all", include_spa=True, ctx=None, evidence=None
    ) -> None:
        self.section = section
        self.chunk = chunk
        self.include_spa = include_spa
        self._ctx = ctx
        self.journal: list[dict] = list(evidence or [])
        self.farm_status: D2FarmStatus | None = None
        self._prev = self._child = self._spec = self._retry = self._pending = None
        self._fails: dict[tuple, int] = {}
        self._plot_attempted = False
        self._step = self._motion_at = self._goal_at = self._unobs = 0
        self._motion_key = self._goal_key = self._last_phase = ""

    @classmethod
    def from_spec(cls, ctx, spec: PhaseSpec) -> "D2FarmClearTactic":
        p = spec.params or {}
        return cls(
            section=str(p.get("section") or "all"),
            chunk=str(p.get("chunk") or "all"),
            include_spa=bool(p.get("include_spa", True)),
            ctx=ctx,
        )

    def reset(self, world: WorldState) -> None:
        self._child = self._spec = self._retry = self._pending = self._prev = self.farm_status = None
        self._fails.clear()
        self._plot_attempted = False
        self._step = self._unobs = 0
        self._motion_key = self._goal_key = self._last_phase = ""

    def can_start(self, world: WorldState) -> bool:
        return True

    @property
    def current_task(self):
        return self._child

    def set_evidence(self, evidence: Sequence[dict]) -> None:
        """Seed prior same-day facts needed by the terminal contract."""
        self.journal = [dict(row) for row in evidence]

    @property
    def step_count(self) -> int:
        return self._step

    def progress_snapshot(self):
        from harvest.core.task_progress import ProgressSnapshot, task_progress_snapshot

        st, spec = self.farm_status, self._spec
        details = []
        if st is not None:
            details.extend(
                (k, getattr(st, k))
                for k in ("weeds", "stones", "large_rocks", "stumps", "fences", "planted", "wet")
            )
        if spec is not None:
            details += [("child", spec.phase), ("chunk", (spec.params or {}).get("chunk"))]
        return ProgressSnapshot(
            task_name=self.__class__.__name__,
            phase_text=spec.phase if spec is not None else (st.outcome.value if st else ""),
            step_count=self._step,
            details=tuple(details),
            child=task_progress_snapshot(self._child) if self._child is not None else None,
        )

    def _observe(self, world: WorldState) -> D2FarmStatus:
        self.farm_status = observe_d2_farm(world.ram, self.journal)
        return self.farm_status

    def _snap(self, prefix: str, st: D2FarmStatus | None = None) -> str:
        st, spec, bits = st or self.farm_status, self._spec, [prefix]
        if spec is not None:
            bits.append(f"target={spec.phase}")
            chunk = (spec.params or {}).get("chunk")
            if chunk:
                bits.append(f"chunk={chunk}")
        if st is not None:
            bits += [
                f"debris=w{st.weeds}/f{st.fences}/s{st.stones}/r{st.large_rocks}/u{st.stumps}",
                f"stamina={st.stamina.current}/{st.stamina.maximum}",
                f"carry_clear={st.hands_clear}",
            ]
        return " ".join(bits)

    def _idle(self, reason: str, ram=None) -> TaskResult:
        from harvest.tasks.farm_clear_quota import yard_load_action
        from harvest.tasks.nav import make_action

        walk = reason == "stale_farm_map" and ram is not None
        action = yard_load_action(ram) if walk else make_action()
        return TaskResult(status=TaskStatus.RUNNING, action=ActionResult(action), reason=reason)

    def _blocked(self, prefix: str, st: D2FarmStatus | None = None) -> TaskResult:
        return TaskResult(status=TaskStatus.BLOCKED, reason=self._snap(prefix, st))

    def _navigation_motion_key(self, world: WorldState):
        """Liveness key while navigating. Tool-swing stays planted on purpose."""
        from harvest.core.task_progress import task_progress_snapshot
        from harvest.tasks.nav import get_pos_from_ram

        child = self._child
        if child is None:
            return None
        snapshot = task_progress_snapshot(child)
        phase = (snapshot.phase_text if snapshot is not None else "").lower()
        name = str(getattr(child, "name", "")).lower()
        navigating = phase in {"navigate", "navigating", "navigation"}
        navigating = navigating or name == "nav" or name.startswith("nav_")
        if not navigating:
            return None

        details = dict(snapshot.details) if snapshot is not None else {}
        pos = get_pos_from_ram(world.ram)
        return (
            (pos.x, pos.y),
            details.get("target", getattr(child, "_target_tile", None)),
            details.get("approach", getattr(child, "_approach_tile", None)),
        )

    def _record_navigation_stall(self, status: D2FarmStatus) -> TaskResult:
        """Stop a required phase visibly; it must never become a skipped chunk."""
        spec = self._spec
        chunk = (spec.params or {}).get("chunk") if spec is not None else None
        reason = self._snap("navigation motion stall", status)
        self.journal.append({
            "phase": spec.phase if spec is not None else "",
            "status": TaskStatus.BLOCKED.value,
            "reason": reason,
            "chunk": chunk,
            "watchdog": "navigation_motion_stall",
            "debris_before": _debris_row(status),
            "debris_after": _debris_row(status),
        })
        return TaskResult(status=TaskStatus.BLOCKED, reason=reason)

    def _watchdogs(self, world: WorldState, status: D2FarmStatus) -> TaskResult | None:
        from harvest.core.carry import backpack_tool, selected_tool

        if self._child is not None and getattr(self._child, "name", "") != "hot_spring_stamina":
            motion = self._navigation_motion_key(world)
            if motion is None:
                # A cleared/stationary tool sequence is not a movement stall.
                self._motion_key, self._motion_at = "", self._step
            elif motion != self._motion_key:
                self._motion_key, self._motion_at = motion, self._step
            elif self._step - self._motion_at >= MOTION_STALL_FRAMES:
                return self._record_navigation_stall(status)
        goal = (
            status.weeds, status.fences, status.stones, status.large_rocks, status.stumps,
            status.planted, status.wet, status.stamina.current,
            int(selected_tool(world.ram)), int(backpack_tool(world.ram)),
        )
        if goal != self._goal_key:
            self._goal_key, self._goal_at = goal, self._step
        elif self._step - self._goal_at >= GOAL_STALL_FRAMES:
            return self._blocked("goal stall", status)
        return None

    def _start(self, spec: PhaseSpec, world: WorldState) -> TaskResult:
        from harvest.planner.day_phase_registry import TaskBuildContext, build_phase_task

        task = build_phase_task(self._ctx or TaskBuildContext(), spec, world)
        if task is None:
            return TaskResult(status=TaskStatus.FAILURE, reason=self._snap(f"no task for {spec.phase}"))
        task.reset(world)
        self._child, self._spec = task, spec
        result = task.step(world)
        if result.status == TaskStatus.RUNNING:
            return result
        return self._after(result, world)

    def _after(self, result: TaskResult, world: WorldState) -> TaskResult:
        spec, before = self._spec, self.farm_status
        status = self._observe(world)
        if spec is not None:
            self.journal.append({
                "phase": spec.phase, "status": result.status.value, "reason": result.reason or "",
                "chunk": (spec.params or {}).get("chunk"),
                "spa": spec.phase == "HOT_SPRING_STAMINA",
                "debris_before": _debris_row(before), "debris_after": _debris_row(status),
            })
        self._child = None
        last = spec.phase if spec is not None else ""
        self._last_phase = last
        if last == "CLEAR_PLOT" and result.status == TaskStatus.SUCCESS:
            self._plot_attempted = True
        if spec is not None and spec.phase == "HOT_SPRING_STAMINA":
            if result.status != TaskStatus.SUCCESS:
                return self._blocked(f"spa failed: {result.reason or result.status.value}", status)
            retry, self._retry = self._retry, None
            self._spec = None
            return self._queue_next(retry) if retry is not None else self._idle("advance")
        remaining = []
        if last in _SPA_RETRY_PHASES and not _section_done(
            status, self.section, self.chunk
        ):
            remaining = ["CLEAR_ROCKS", "CLEAR_STUMPS"]
        decision = leftover_chain_decision(
            last, result.status, result.reason, status.stamina, remaining,
            include_spa=self.include_spa,
        )
        if decision == "spa_retry" and spec is not None:
            self._retry = spec
            return self._queue_next(full_restore_spa_phase())
        if decision == "insert_spa":
            return self._queue_next(full_restore_spa_phase())
        if decision == "continue" or result.status == TaskStatus.SUCCESS:
            self._spec = None
            return self._idle("advance")
        chunk = (spec.params or {}).get("chunk") if spec is not None else None
        key = (last, chunk)
        self._fails[key] = self._fails.get(key, 0) + 1
        if chunk and self._fails[key] >= 2:
            reason = (
                f"required chunk failed {self._fails[key]} times: "
                f"{last} chunk={chunk}; {result.reason or result.status.value}"
            )
            self.journal.append({
                "phase": last,
                "status": TaskStatus.BLOCKED.value,
                "reason": reason,
                "chunk": chunk,
                "watchdog": "required_chunk_failure",
                "debris_before": _debris_row(status),
                "debris_after": _debris_row(status),
            })
            return self._blocked(reason, status)
        return self._blocked(f"blocked: {result.reason or result.status.value}", status)

    def _queue_next(self, spec: PhaseSpec) -> TaskResult:
        self._pending = spec
        return self._idle("queued")

    def _select(self, world: WorldState, status: D2FarmStatus) -> TaskResult:
        done = _section_done(status, self.section, self.chunk)
        if done:
            settled = (
                confirm_d2_complete(self._prev, status)
                if self.section == "all"
                else (self._prev is not None and _section_done(self._prev, self.section, self.chunk))
            )
            if settled:
                return TaskResult(status=TaskStatus.SUCCESS, reason="d2 farm clear complete")
            self._prev = status
            return self._idle("settle")
        pending, self._pending = self._pending, None
        spec = pending or next_d2_spec(
            status, include_spa=self.include_spa, section=self.section,
            chunk=self.chunk, last_phase=self._last_phase,
            plot_attempted=self._plot_attempted,
        )
        if spec is None:
            self._prev = status
            return self._idle("waiting verification")
        return self._start(spec, world)

    def step(self, world: WorldState) -> TaskResult:
        from harvest.core.shipping_credit import shipping_scene_needs_dismiss
        from harvest.tasks.primitives import dismiss_dialogue_result

        self._step += 1
        if shipping_scene_needs_dismiss(world.ram):
            return dismiss_dialogue_result(
                world.frame, buttons=("a",), pulse_every=2, reason="shipping scene"
            )
        status = self._observe(world)
        stall = self._watchdogs(world, status)
        if stall is not None:
            return stall
        if self._child is not None:
            # Shed/spa fetches must keep stepping; yard idle is only for
            # choosing the next farm child while the map is unloaded.
            result = self._child.step(world)
            return result if result.status == TaskStatus.RUNNING else self._after(result, world)
        if status.outcome == D2FarmOutcome.TEMPORARILY_UNOBSERVABLE:
            self._unobs += 1
            if self._unobs >= GOAL_STALL_FRAMES:
                return self._blocked("stale_farm_map", status)
            return self._idle(status.reason or "temporarily_unobservable", world.ram)
        self._unobs = 0
        return self._select(world, status)


__all__ = [
    "D2_LEFTOVER_PHASE_NAMES", "D2_TARGETS", "D2FarmClearTactic", "D2FarmOutcome",
    "D2FarmStatus", "bush_clear_phase", "confirm_d2_complete", "d2_farm_clear_phase",
    "d2_post_shop_work_phases", "ensure_axe_phase", "ensure_hammer_phase",
    "fence_dump_phase", "leftover_already_queued", "leftover_chain_decision",
    "needs_spa_before_next_smash", "next_d2_spec", "observe_d2_farm",
    "pocket_clear_phase", "pocket_water_phase", "rock_clear_phase",
    "should_spa_retry", "stone_pond_phase", "stump_clear_phase",
]
