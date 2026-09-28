"""Ranch Master evaluation and score breakdown matching SNES ROM bank_83.asm."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from harvest.core.ram_catalog import read_ram_value, read_ram_u8, read_ram_u16
from harvest.core.animal_probe import cow_slot_snapshots


RANCH_MASTER_SCORE_THRESHOLD = 200
PERFECT_SCORE_THRESHOLD = 999


@dataclass
class RanchMasterBreakdown:
    """Exact sub-score breakdown of the Ranch Mastery evaluation from ROM bank_83."""

    total_score: int
    is_ranch_master: bool
    rating_title: str

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

    # Detailed sub-maps
    bachelorette_details: Dict[str, int] = field(default_factory=dict)
    crop_ship_details: Dict[str, int] = field(default_factory=dict)
    cow_happiness_details: List[int] = field(default_factory=list)
    missing_milestones: List[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def evaluate_ranch_mastery_score(ram: np.ndarray) -> RanchMasterBreakdown:
    """Evaluate Ranch Mastery score directly from RAM, matching bank_83.asm."""
    # 1. Money: !moneyL >> 7, capped at 78
    try:
        money_raw = int(read_ram_value(ram, "money_raw", raw=True))
    except Exception:
        money_raw = 0
    money_score = min(78, money_raw >> 7)

    # 2. Number of cows: cow_N * 3
    try:
        num_cows = int(read_ram_value(ram, "num_cows", raw=True))
    except Exception:
        num_cows = 0
    cows_score = num_cows * 3

    # 3. Number of chickens: chicks_N * 3
    try:
        num_chickens = int(read_ram_value(ram, "num_chickens", raw=True))
    except Exception:
        num_chickens = 0
    chickens_score = num_chickens * 3

    # 4. Power berries / Max stamina: (max_stamina - 100) >> 1
    try:
        max_stam = int(read_ram_value(ram, "max_stamina", raw=True))
    except Exception:
        max_stam = 100
    power_berries_score = max(0, (max_stam - 100) >> 1)

    # 5. Bachelorette hearts: (hearts & 0x01FF) >> 4 for each girl
    bachelorettes = ("maria", "ann", "nina", "ellen", "eve")
    bachelorette_details: Dict[str, int] = {}
    bachelorettes_score = 0
    for name in bachelorettes:
        try:
            h = int(read_ram_value(ram, f"{name}_hearts", raw=True))
        except Exception:
            h = 0
        girl_score = (h & 0x01FF) >> 4
        bachelorette_details[name] = girl_score
        bachelorettes_score += girl_score

    # 6. Shipped crops: (shipped & 0x01FF) >> 4 for each crop
    crops = (
        ("tomatoes", "shipped_tomatoes"),
        ("corn", "shipped_corn"),
        ("potatoes", "shipped_potatoes"),
        ("turnips", "shipped_turnips"),
    )
    crop_ship_details: Dict[str, int] = {}
    shipped_crops_score = 0
    for crop_name, ram_field in crops:
        try:
            shipped = int(read_ram_value(ram, ram_field, raw=True))
        except Exception:
            shipped = 0
        crop_score = (shipped & 0x01FF) >> 4
        crop_ship_details[crop_name] = crop_score
        shipped_crops_score += crop_score

    # 7. Farmer happiness: happiness >> 5
    try:
        happy = int(read_ram_value(ram, "happiness", raw=True))
    except Exception:
        happy = 0
    happiness_score = happy >> 5

    # 8. House upgrades: $7F1F64 & 0x0040 (+16), & 0x0080 (+16)
    try:
        upg_flags = int(read_ram_value(ram, "upgrade_flags", raw=True))
    except Exception:
        upg_flags = 0
    house_upgrades_score = 0
    if upg_flags & 0x0040:
        house_upgrades_score += 16
    if upg_flags & 0x0080:
        house_upgrades_score += 16

    # 9. Children: $7F1F6E & 0x0008 (+16 for child 1), & 0x0004 (+16 for child 2)
    try:
        fam_flags = int(read_ram_value(ram, "incubator_flags", raw=True))
    except Exception:
        fam_flags = 0
    children_score = 0
    if fam_flags & 0x0008:
        children_score += 16
    if fam_flags & 0x0004:
        children_score += 16

    # 10. Marriage: $7F1F66 & 0x001F (+32)
    try:
        marr_flags = int(read_ram_value(ram, "marriage_flags", raw=True))
    except Exception:
        marr_flags = 0
    marriage_score = 32 if (marr_flags & 0x001F) else 0

    # 11. Special items: clock ($7F1F6E & 0x4000: +22), turtle shell ($7F1F6C & 0x1000: +21)
    special_items_score = 0
    if fam_flags & 0x4000:
        special_items_score += 22
    try:
        fam_event_flags = int(read_ram_value(ram, "family_event_flags", raw=True))
    except Exception:
        fam_event_flags = 0
    if fam_event_flags & 0x1000:
        special_items_score += 21

    # 12. Ranch development: ranch_development >> 1
    try:
        ranch_dev = int(read_ram_value(ram, "ranch_development", raw=True))
    except Exception:
        ranch_dev = 0
    ranch_development_score = ranch_dev >> 1

    # 13. Cow happiness: for each existing cow, min(25, cow_happiness >> 3)
    cow_happiness_details: List[int] = []
    cow_happiness_score = 0
    try:
        slots = cow_slot_snapshots(ram)
        for slot in slots:
            h = int(slot.get("happiness") or 0)
            cow_pts = min(25, h >> 3)
            cow_happiness_details.append(cow_pts)
            cow_happiness_score += cow_pts
    except Exception:
        pass

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
    total_score = min(999, raw_total)
    is_ranch_master = total_score >= RANCH_MASTER_SCORE_THRESHOLD

    if total_score >= 900:
        rating_title = "Supreme Ranch Master (Near Perfect)"
    elif total_score >= 500:
        rating_title = "Grand Ranch Master"
    elif total_score >= RANCH_MASTER_SCORE_THRESHOLD:
        rating_title = "Ranch Master"
    elif total_score >= 150:
        rating_title = "Senior Farmer"
    elif total_score >= 100:
        rating_title = "Established Farmer"
    else:
        rating_title = "Novice Rancher"

    missing_milestones: List[str] = []
    if money_score < 78:
        missing_milestones.append(f"Money: {money_score}/78 pts (reach ~99,840 G)")
    if num_cows < 12:
        missing_milestones.append(f"Cows: {num_cows}/12 cows ({cows_score}/36 pts)")
    if num_chickens < 12:
        missing_milestones.append(f"Chickens: {num_chickens}/12 chickens ({chickens_score}/36 pts)")
    if power_berries_score < 50:
        missing_milestones.append(f"Power Berries: {power_berries_score}/50 pts (collect 10 berries)")
    if marriage_score == 0:
        missing_milestones.append("Marriage: not married (0/32 pts, propose with Blue Feather)")
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
        is_ranch_master=is_ranch_master,
        rating_title=rating_title,
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
