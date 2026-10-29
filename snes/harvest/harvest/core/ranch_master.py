"""Ranch Master evaluation matching SNES bank_83 RanchMasteryCalculator."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Mapping

import numpy as np

from harvest.core.ram_catalog import (
    COW_SLOT_COUNT,
    field_spec,
    ram_index,
    read_animal_slot_field,
    read_ram_value,
)


RANCH_MASTER_SCORE_THRESHOLD = 200
PERFECT_SCORE_THRESHOLD = 999
MONEY_SCORE_CAP = 78
# LDA.L !moneyL at $83F26D is a 16-bit load of the 24-bit money word.
MONEY_LOW_MASK = 0xFFFF
# min(78, moneyL >> 7) first hits 78 at 78 << 7. Low word is the whole window.
MONEY_WINDOW_LO = MONEY_SCORE_CAP << 7
MONEY_WINDOW_HI = MONEY_LOW_MASK
# AddMoney saturates at 999999 ($83B1C9, CMP #$423F / SBC #$0F).
MONEY_RAW_CAP = 0x0F423F
# $7EA4E6 farm_map_array. Same tiles CalculateFarmDevelopment counts.
FARM_MAP_WRAM = 0xA4E6
FARM_MAP_TILES = 64 * 64
FARM_DEVELOPMENT_DIVISOR = 0x0127
# First set marriage bit is the wife. Hour-18 house entry subtracts 2.
WIFE_BITS = (
    ("maria", 0x0001),
    ("ann", 0x0002),
    ("nina", 0x0004),
    ("ellen", 0x0008),
    ("eve", 0x0010),
)
WIFE_DRIFT_BLOCK_BIT = 0x0002
WIFE_DRIFT_HEART_FLOOR = 200
WIFE_DRIFT_DELTA = -2
HEART_CLAMP = 999
# Score window is (raw & 0x1FF) in 496..511. Crops use the same mask.
NINE_BIT_WINDOW = (496, 511)
# $83F097 accepts power_berry_N only when it is exactly 10.
BERRY_COUNT_TARGET = 10
HAPPINESS_WINDOW = (992, 999)
HERD_COUNT_TARGET = 12
SCORE_INDEX = 0x20
TERMINAL_INDEX = 0x25
# Text_Peddler_BlueFeather1, bank_83 pointer table id.
BLUE_FEATHER_TEXT_ID = 0x01C3
# Event 0x46 JumpifBetweenValue on $7F1F54 ($B5A90E). Inclusive bands.
RATING_TEXT_BANDS = (
    (0, 199, 0x0483),
    (200, 399, 0x0488),
    (400, 599, 0x0489),
    (600, 799, 0x048A),
    (800, 999, 0x048B),
)


@dataclass
class RanchMasterBreakdown:
    """Sub-score breakdown of RanchMasteryCalculator ($83F26D)."""

    total_score: int
    is_ranch_master: bool
    rating_title: str
    rating_text_id: int

    money_score: int
    cows_score: int
    chickens_score: int
    power_berries_score: int
    bachelorettes_score: int
    shipped_crops_score: int
    happiness_score: int
    house_upgrades_score: int
    marriage_score: int
    children_score: int
    special_items_score: int
    ranch_development_score: int
    cow_happiness_score: int

    bachelorette_details: dict[str, int] = field(default_factory=dict)
    crop_ship_details: dict[str, int] = field(default_factory=dict)
    cow_happiness_details: list[int] = field(default_factory=list)
    missing_milestones: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def rating_text_id_for_score(score: int) -> int:
    """ROM rating line for a score already capped at 999."""
    capped = min(PERFECT_SCORE_THRESHOLD, max(0, int(score)))
    for low, high, text_id in RATING_TEXT_BANDS:
        if low <= capped <= high:
            return text_id
    return RATING_TEXT_BANDS[0][2]


def _raw(ram: np.ndarray, key: str, default: int = 0) -> int:
    try:
        return int(read_ram_value(ram, key, raw=True))
    except Exception:
        return default


def _cow_happiness(ram: np.ndarray) -> tuple[int, list[int]]:
    """12-slot loop at $83F4xx: existence bit 0, happiness byte +4, min(25, >> 3)."""
    details = [0] * COW_SLOT_COUNT
    total = 0
    try:
        for slot in range(COW_SLOT_COUNT):
            status = int(read_animal_slot_field(ram, "cow", slot, "status_raw"))
            if not status & 0x01:
                continue
            happiness = int(read_animal_slot_field(ram, "cow", slot, "happiness"))
            points = min(25, happiness >> 3)
            details[slot] = points
            total += points
    except Exception:
        return 0, []
    return total, details


def evaluate_ranch_mastery_score(ram: np.ndarray) -> RanchMasterBreakdown:
    """Evaluate Ranch Mastery from RAM, matching bank_83 RanchMasteryCalculator."""
    money_low = _raw(ram, "money_raw") & MONEY_LOW_MASK
    money_score = min(MONEY_SCORE_CAP, money_low >> 7)

    num_cows = _raw(ram, "num_cows")
    cows_score = num_cows * 3

    num_chickens = _raw(ram, "num_chickens")
    chickens_score = num_chickens * 3

    max_stam = _raw(ram, "max_stamina", 100)
    power_berries_score = max(0, (max_stam - 100) >> 1)

    bachelorette_details: dict[str, int] = {}
    bachelorettes_score = 0
    for name in ("maria", "ann", "nina", "ellen", "eve"):
        girl_score = (_raw(ram, f"{name}_hearts") & 0x01FF) >> 4
        bachelorette_details[name] = girl_score
        bachelorettes_score += girl_score

    crop_ship_details: dict[str, int] = {}
    shipped_crops_score = 0
    for crop_name, ram_field in (
        ("tomatoes", "shipped_tomatoes"),
        ("corn", "shipped_corn"),
        ("potatoes", "shipped_potatoes"),
        ("turnips", "shipped_turnips"),
    ):
        crop_score = (_raw(ram, ram_field) & 0x01FF) >> 4
        crop_ship_details[crop_name] = crop_score
        shipped_crops_score += crop_score

    happiness_score = _raw(ram, "happiness") >> 5

    upg_flags = _raw(ram, "upgrade_flags")
    house_upgrades_score = 0
    if upg_flags & 0x0040:
        house_upgrades_score += 16
    if upg_flags & 0x0080:
        house_upgrades_score += 16

    fam_flags = _raw(ram, "incubator_flags")
    children_score = 0
    if fam_flags & 0x0008:
        children_score += 16
    if fam_flags & 0x0004:
        children_score += 16

    marriage_score = 32 if (_raw(ram, "marriage_flags") & 0x001F) else 0

    special_items_score = 0
    if fam_flags & 0x4000:
        special_items_score += 22
    if _raw(ram, "family_event_flags") & 0x1000:
        special_items_score += 21

    ranch_development_score = _raw(ram, "ranch_development") >> 1

    cow_happiness_score, cow_happiness_details = _cow_happiness(ram)

    raw_total = (
        money_score
        + cows_score
        + chickens_score
        + power_berries_score
        + bachelorettes_score
        + shipped_crops_score
        + happiness_score
        + house_upgrades_score
        + marriage_score
        + children_score
        + special_items_score
        + ranch_development_score
        + cow_happiness_score
    )
    total_score = min(PERFECT_SCORE_THRESHOLD, raw_total)
    rating_text_id = rating_text_id_for_score(total_score)

    missing_milestones: list[str] = []
    if money_score < MONEY_SCORE_CAP:
        missing_milestones.append(f"Money: {money_score}/{MONEY_SCORE_CAP} pts (reach ~99,840 G)")
    if num_cows < 12:
        missing_milestones.append(f"Cows: {num_cows}/12 cows ({cows_score}/36 pts)")
    if num_chickens < 12:
        missing_milestones.append(f"Chickens: {num_chickens}/12 chickens ({chickens_score}/36 pts)")
    if power_berries_score < 50:
        missing_milestones.append(f"Power Berries: {power_berries_score}/50 pts (collect 10 berries)")
    if marriage_score == 0:
        missing_milestones.append(
            f"Marriage: not married (0/32 pts, propose with text 0x{BLUE_FEATHER_TEXT_ID:04X})"
        )
    if house_upgrades_score < 32:
        missing_milestones.append(f"House Upgrades: {house_upgrades_score}/32 pts (build Super House L3)")
    if children_score < 32:
        missing_milestones.append(f"Children: {children_score}/32 pts (have 2 children)")
    if special_items_score < 43:
        missing_milestones.append(f"Special Items: {special_items_score}/43 pts (buy clock & turtle shell)")
    if ranch_development_score < 50:
        missing_milestones.append(f"Pasture / Development: {ranch_development_score}/50 pts (grow grass pasture)")

    return RanchMasterBreakdown(
        total_score=total_score,
        is_ranch_master=total_score >= RANCH_MASTER_SCORE_THRESHOLD,
        rating_title=f"0x{rating_text_id:04X}",
        rating_text_id=rating_text_id,
        money_score=money_score,
        cows_score=cows_score,
        chickens_score=chickens_score,
        power_berries_score=power_berries_score,
        bachelorettes_score=bachelorettes_score,
        shipped_crops_score=shipped_crops_score,
        happiness_score=happiness_score,
        house_upgrades_score=house_upgrades_score,
        marriage_score=marriage_score,
        children_score=children_score,
        special_items_score=special_items_score,
        ranch_development_score=ranch_development_score,
        cow_happiness_score=cow_happiness_score,
        bachelorette_details=bachelorette_details,
        crop_ship_details=crop_ship_details,
        cow_happiness_details=cow_happiness_details,
        missing_milestones=missing_milestones,
    )


def counter_window_state(raw: int, lo: int, hi: int) -> str:
    """``below`` / ``in_window`` / ``overshoot`` for an inclusive counter target."""
    if int(raw) > int(hi):
        return "overshoot"
    if int(raw) < int(lo):
        return "below"
    return "in_window"


def nine_bit_window_state(raw: int, lo: int = NINE_BIT_WINDOW[0], hi: int = NINE_BIT_WINDOW[1]) -> str:
    """Hearts and shipped crops: in window when ``(raw & 0x1FF)`` is ``lo..hi``."""
    residue = int(raw) & 0x01FF
    if lo <= residue <= hi:
        return "in_window"
    if int(raw) > hi:
        return "overshoot"
    return "below"


def money_window_state(money_raw: int) -> str:
    """Low word in ``[9984, 65535]`` scores 78. A wrapped low word is overshoot."""
    low = int(money_raw) & MONEY_LOW_MASK
    if low >= MONEY_WINDOW_LO:
        return "in_window"
    if int(money_raw) > MONEY_LOW_MASK:
        return "overshoot"
    return "below"


def _term_row(
    value: int,
    maximum: int,
    *,
    raw: int,
    lo: int,
    hi: int,
    window_state: str,
) -> dict[str, Any]:
    return {
        "value": int(value),
        "max": int(maximum),
        "gap": int(maximum) - int(value),
        "window_state": window_state,
        "raw": int(raw),
        "lo": int(lo),
        "hi": int(hi),
    }


def _write_raw(ram: np.ndarray, key: str, value: int) -> None:
    spec = field_spec(key)
    idx = ram_index(ram, spec.address, live_offset=spec.live_offset)
    width = {"u8": 1, "u16": 2, "u24": 3}[spec.kind]
    if idx + width > len(ram):
        return
    stored = int(value)
    for offset in range(width):
        ram[idx + offset] = (stored >> (8 * offset)) & 0xFF


def count_development_tiles(ram: np.ndarray) -> int:
    """Tiles ``CalculateFarmDevelopment`` counts: fence ``0x05/0x06`` or ``>= 0x1D``, below ``0xA0``."""
    base = ram_index(ram, FARM_MAP_WRAM)
    end = base + FARM_MAP_TILES
    if end > len(ram):
        return 0
    tiles = ram[base:end]
    counted = (tiles < 0xA0) & ((tiles == 0x05) | (tiles == 0x06) | (tiles >= 0x1D))
    return int(np.count_nonzero(counted))


def _flag_points(raw_points: int, maximum: int) -> dict[str, Any]:
    return _term_row(
        raw_points,
        maximum,
        raw=raw_points,
        lo=maximum,
        hi=maximum,
        window_state=counter_window_state(raw_points, maximum, maximum),
    )


def score_ledger(ram: np.ndarray) -> dict[str, Any]:
    """Per-term ``value``, ``max``, ``gap``, and ``window_state`` from the oracle."""
    breakdown = evaluate_ranch_mastery_score(ram)
    money_raw = _raw(ram, "money_raw")
    berry_count = _raw(ram, "power_berry_count")
    happiness = _raw(ram, "happiness")
    dev_byte = _raw(ram, "ranch_development")
    rows: list[dict[str, Any]] = [
        _term_row(
            breakdown.money_score,
            MONEY_SCORE_CAP,
            raw=money_raw,
            lo=MONEY_WINDOW_LO,
            hi=MONEY_WINDOW_HI,
            window_state=money_window_state(money_raw),
        ),
        _term_row(
            breakdown.cows_score,
            HERD_COUNT_TARGET * 3,
            raw=_raw(ram, "num_cows"),
            lo=HERD_COUNT_TARGET,
            hi=HERD_COUNT_TARGET,
            window_state=counter_window_state(_raw(ram, "num_cows"), HERD_COUNT_TARGET, HERD_COUNT_TARGET),
        ),
        _term_row(
            breakdown.chickens_score,
            HERD_COUNT_TARGET * 3,
            raw=_raw(ram, "num_chickens"),
            lo=HERD_COUNT_TARGET,
            hi=HERD_COUNT_TARGET,
            window_state=counter_window_state(
                _raw(ram, "num_chickens"), HERD_COUNT_TARGET, HERD_COUNT_TARGET
            ),
        ),
        _term_row(
            breakdown.power_berries_score,
            50,
            raw=berry_count,
            lo=BERRY_COUNT_TARGET,
            hi=BERRY_COUNT_TARGET,
            window_state=counter_window_state(berry_count, BERRY_COUNT_TARGET, BERRY_COUNT_TARGET),
        ),
    ]
    for name in ("maria", "ann", "nina", "ellen", "eve"):
        hearts = _raw(ram, f"{name}_hearts")
        rows.append(
            _term_row(
                breakdown.bachelorette_details[name],
                31,
                raw=hearts,
                lo=NINE_BIT_WINDOW[0],
                hi=NINE_BIT_WINDOW[1],
                window_state=nine_bit_window_state(hearts),
            )
        )
    for name in ("tomatoes", "corn", "potatoes", "turnips"):
        shipped = _raw(ram, f"shipped_{name}")
        rows.append(
            _term_row(
                breakdown.crop_ship_details[name],
                31,
                raw=shipped,
                lo=NINE_BIT_WINDOW[0],
                hi=NINE_BIT_WINDOW[1],
                window_state=nine_bit_window_state(shipped),
            )
        )
    rows.extend(
        (
            _term_row(
                breakdown.happiness_score,
                31,
                raw=happiness,
                lo=HAPPINESS_WINDOW[0],
                hi=HAPPINESS_WINDOW[1],
                window_state=counter_window_state(happiness, *HAPPINESS_WINDOW),
            ),
            _flag_points(breakdown.house_upgrades_score, 32),
            _flag_points(breakdown.children_score, 32),
            _flag_points(breakdown.marriage_score, 32),
            _flag_points(22 if _raw(ram, "incubator_flags") & 0x4000 else 0, 22),
            _flag_points(21 if _raw(ram, "family_event_flags") & 0x1000 else 0, 21),
            _term_row(
                breakdown.ranch_development_score,
                50,
                raw=dev_byte,
                lo=100,
                hi=101,
                window_state=counter_window_state(dev_byte, 100, 101),
            ),
            _flag_points(breakdown.cow_happiness_score, 300),
        )
    )
    names = (
        "money",
        "cows",
        "chickens",
        "power_berries",
        "maria",
        "ann",
        "nina",
        "ellen",
        "eve",
        "tomatoes",
        "corn",
        "potatoes",
        "turnips",
        "happiness",
        "house",
        "children",
        "marriage",
        "clock",
        "turtle",
        "development",
        "cow_happiness",
    )
    if len(names) != len(rows):
        raise RuntimeError(f"ledger terms {len(rows)} != names {len(names)}")
    for name, row in zip(names, rows):
        row["term"] = name
    uncapped = sum(int(row["value"]) for row in rows)
    total = _term_row(
        breakdown.total_score,
        PERFECT_SCORE_THRESHOLD,
        raw=uncapped,
        lo=PERFECT_SCORE_THRESHOLD,
        hi=PERFECT_SCORE_THRESHOLD,
        window_state=counter_window_state(
            breakdown.total_score, PERFECT_SCORE_THRESHOLD, PERFECT_SCORE_THRESHOLD
        ),
    )
    total["term"] = "total"
    rows.append(total)
    return {
        "rows": rows,
        "terms": {str(row["term"]): row for row in rows},
        "total_score": breakdown.total_score,
        "rating_text_id": breakdown.rating_text_id,
        "rating_title": breakdown.rating_title,
        "uncapped_total": uncapped,
    }


def _loaded_wife(marriage_flags: int) -> str | None:
    for name, bit in WIFE_BITS:
        if int(marriage_flags) & bit:
            return name
    return None


def apply_f0_d30_drift(ram: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
    """Project a state through the F0 D29-evening → index ``0x20`` writers.

    Legal sleep re-enters the house at hour ``0x12``. That path subtracts 2
    from the loaded wife when ``$7F1F6E`` bit ``0x0002`` is clear and her
    hearts are at least 200. Money gains the shipping pile (``AddMoney``).
    ``ranch_development`` becomes ``min(100, tiles * 10 // 295)``.
    """
    out = np.array(ram, copy=True)
    wife = _loaded_wife(_raw(out, "marriage_flags"))
    hearts_delta = 0
    blocked = _raw(out, "incubator_flags") & WIFE_DRIFT_BLOCK_BIT
    if wife is not None and not blocked:
        hearts = _raw(out, f"{wife}_hearts")
        if hearts >= WIFE_DRIFT_HEART_FLOOR:
            drifted = max(0, min(HEART_CLAMP, hearts + WIFE_DRIFT_DELTA))
            _write_raw(out, f"{wife}_hearts", drifted)
            hearts_delta = drifted - hearts
    money = min(MONEY_RAW_CAP, _raw(out, "money_raw") + _raw(out, "shipping_money_raw"))
    shipping_added = money - _raw(out, "money_raw")
    _write_raw(out, "money_raw", money)
    _write_raw(out, "shipping_money_raw", 0)
    tiles = count_development_tiles(out)
    dev_byte = min(100, tiles * 10 // FARM_DEVELOPMENT_DIVISOR)
    _write_raw(out, "ranch_development", dev_byte)
    return out, {
        "model": "F0",
        "wife": wife,
        "wife_hearts_delta": hearts_delta,
        "money_raw": money,
        "shipping_added": shipping_added,
        "tiles": tiles,
        "development_byte": dev_byte,
    }


def morning_score_ledger(ram: np.ndarray) -> dict[str, Any]:
    """Morning ledger plus the F0 projection of this RAM to the D30 score."""
    ledger = score_ledger(ram)
    projected, drift = apply_f0_d30_drift(ram)
    projected_ledger = score_ledger(projected)
    ledger["d30_projection"] = {
        "model": "F0",
        "drift": drift,
        "total_score": projected_ledger["total_score"],
        "rating_text_id": projected_ledger["rating_text_id"],
        "rows": projected_ledger["rows"],
        "terms": projected_ledger["terms"],
    }
    return ledger


def ending_ranch_report(
    *,
    ram_at_0x20: np.ndarray | None,
    ram_at_0x25: np.ndarray | None,
) -> dict[str, Any]:
    """Oracle breakdown and ROM ``ranch_mastery`` at indexes ``0x20`` and ``0x25``.

    Inputs are read at ``0x20`` (the calculator's frame). The RAM score cell
    can still be the pre-store value there; ``0x25`` is the terminal cell.
    """

    def _one(ram: np.ndarray | None) -> dict[str, Any] | None:
        if ram is None:
            return None
        oracle = evaluate_ranch_mastery_score(ram)
        return {
            "ranch_mastery": _raw(ram, "ranch_mastery"),
            "oracle": oracle.to_dict(),
            "rating_text_id": oracle.rating_text_id,
        }

    at_20 = _one(ram_at_0x20)
    at_25 = _one(ram_at_0x25)
    primary = at_20 if at_20 is not None else at_25
    return {
        "oracle": None if primary is None else primary["oracle"],
        "rating_text_id": None if primary is None else primary["rating_text_id"],
        "rating_text_id_0x20": None if at_20 is None else at_20["rating_text_id"],
        "rating_text_id_0x25": None if at_25 is None else at_25["rating_text_id"],
        "ranch_mastery_0x20": None if at_20 is None else at_20["ranch_mastery"],
        "ranch_mastery_0x25": None if at_25 is None else at_25["ranch_mastery"],
        "at_0x20": at_20,
        "at_0x25": at_25,
    }


class EndingIndexCapture:
    """First RAM snapshot at ending index ``0x20`` and at ``0x25``."""

    def __init__(self) -> None:
        self.ram_at_0x20: np.ndarray | None = None
        self.ram_at_0x25: np.ndarray | None = None

    def observe(self, ram: np.ndarray, scene_index: int) -> bool:
        changed = False
        index = int(scene_index)
        if index == SCORE_INDEX and self.ram_at_0x20 is None:
            self.ram_at_0x20 = np.array(ram, copy=True)
            changed = True
        if index >= TERMINAL_INDEX and self.ram_at_0x25 is None:
            self.ram_at_0x25 = np.array(ram, copy=True)
            changed = True
        return changed

    def report(self) -> dict[str, Any]:
        return ending_ranch_report(ram_at_0x20=self.ram_at_0x20, ram_at_0x25=self.ram_at_0x25)


@dataclass(frozen=True)
class CounterTarget:
    """Inclusive ``[lo, hi]`` for one RAM counter."""

    lo: int
    hi: int

    def __post_init__(self) -> None:
        if int(self.lo) > int(self.hi):
            raise ValueError(f"counter target lo {self.lo} > hi {self.hi}")


# High side is 10. Passing 10 fails the best-ending check at ``$83F097``.
BERRY_COUNTER_TARGET = CounterTarget(BERRY_COUNT_TARGET, BERRY_COUNT_TARGET)


@dataclass(frozen=True)
class WindowGuard:
    """Refuse an action whose landing, including F0's remaining automatic delta, exceeds ``hi``."""

    target: CounterTarget
    automatic_delta: int = 0

    def landing(self, current: int, action_delta: int) -> int:
        return int(current) + int(action_delta) + int(self.automatic_delta)

    def allow(self, current: int, action_delta: int) -> bool:
        return self.landing(current, action_delta) <= int(self.target.hi)

    def allow_measured(self, current: int, action: str, deltas: Mapping[str, int]) -> bool:
        """``deltas`` are measured per-action changes. An unknown action is refused."""
        if action not in deltas:
            return False
        return self.allow(current, int(deltas[action]))

    def window_state(self, current: int) -> str:
        projected = int(current) + int(self.automatic_delta)
        return counter_window_state(projected, int(self.target.lo), int(self.target.hi))


def berry_window_guard() -> WindowGuard:
    """Power-berry counter. No night writer. High side is 10."""
    return WindowGuard(BERRY_COUNTER_TARGET, automatic_delta=0)
