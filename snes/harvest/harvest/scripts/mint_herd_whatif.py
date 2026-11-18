#!/usr/bin/env python3
"""Mint a herd research pin: N barn cows, chosen cows sick, medicine in the carry pair.

A what-if pin edits RAM, so it proves a component (a cure, a feed pass, a night), never a route. The
source is a barn pin; slots at or past ``--cows`` are cleared, the listed slots get status sick
(0x20) with the death timer, and the carry pair gets medicine (tool 9) for the cure. Pins are local
(``*.state`` is gitignored); tests skip when one is missing, so this recipe is the record.

    uv run python -m harvest.scripts.mint_herd_whatif \\
        --src LN_repair-herd_whatif_all12_finalnight --cows 5 --sick 1,3 --medicine selected \\
        --out LN_herd_whatif_c5_sick2

Names minted for the ladder tests (all from ``LN_repair-herd_whatif_all12_finalnight``, hay 300):
``LN_herd_whatif_c{3,5,8}_sick1`` (slot 1 sick, medicine selected, fed), ``LN_herd_whatif_c{3,5,8}``
(healthy, fed), ``LN_herd_whatif_c{3,5,8}_am`` (healthy, unfed: a morning, brush and milker kept) and
``LN_herd_whatif_c{3,5,8}_am_sick1`` (the same morning with slot 1 sick and a bottle on the shed shelf).
"""

from __future__ import annotations

import argparse
from pathlib import Path

from harvest.core.harvest_state import HarvestStateDocument
from harvest.core.animal_status import COW_STATUS_SICK_FLAG
from harvest.core.ram_catalog import field_spec

SHED_ROW_2_ADDR = 0x11F01
PINS = Path(__file__).resolve().parents[2] / "custom_integrations" / "HarvestMoon-Snes"
MEDICINE_TOOL_ID = 9
SICK_TIMER_DAYS = 7  # COW_SICKNESS_START_TIMER: a sick cow dies when it reaches 0


def mint(
    src: str,
    out: str,
    *,
    cows: int,
    sick: tuple[int, ...] = (),
    timer: int = SICK_TIMER_DAYS,
    medicine: str = "none",
    hay: int | None = None,
    happiness: int | None = None,
    tool: int | None = None,
    unfed: bool = False,
    shed_medicine: bool = False,
) -> Path:
    doc = HarvestStateDocument.load(src)
    existing = [slot for slot in range(12) if doc.cows()[slot].status_raw & 0x01]
    if len(existing) < cows:
        raise SystemExit(f"{src} has {len(existing)} cows, cannot keep {cows}")
    keep = existing[:cows]
    for slot in existing:
        if slot not in keep:
            doc.clear_cow_slot(slot)
    for slot in sick:
        if slot not in keep:
            raise SystemExit(f"sick slot {slot} is not among the kept cows {keep}")
        doc.set_cow_field(slot, "status_raw", 0x09 | COW_STATUS_SICK_FLAG)
        doc.set_cow_field(slot, "pregnancy_raw", timer)
    if happiness is not None:
        for slot in keep:
            doc.set_cow_field(slot, "happiness", happiness)
    fed_mask = 0
    for slot in keep:
        fed_mask |= 1 << slot
    if unfed:
        # A morning: the trough flags are cleared by the night pass, so the chores owe a feed, and
        # the clock reads 07:00 so a shop trip is still possible (the source pin is 18:00).
        fed_mask = 0
        doc.set_clock(hour=7, minute=0, second=0)
    doc.set_scalar_value("fed_cows_flags", fed_mask)
    doc.set_scalar_value("fed_cows_n", 0 if unfed else len(keep))
    doc.set_scalar_value("num_cows", len(keep))
    if hay is not None:
        doc.set_scalar_value("stored_grass", hay)
    selected = {"selected": (MEDICINE_TOOL_ID, 0), "backpack": (0, MEDICINE_TOOL_ID), "none": None}[medicine]
    if tool is not None:
        selected = (tool, 0)
    if selected is not None:
        doc.set_scalar_value("tool_selected", selected[0])
        doc.set_scalar_value("tool_backpack", selected[1])
    assert field_spec("tool_selected") and field_spec("tool_backpack")
    if shed_medicine:
        # Row 2 of the shed shelf (SHED_ROW_2_ADDR): bit 0x01 is the medicine bottle.
        row = int(doc.mutable_state.read_u8(SHED_ROW_2_ADDR))
        doc.mutable_state.write_u8(SHED_ROW_2_ADDR, row | 0x01)
    return doc.save_as(PINS / f"{out}.state")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--src", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--cows", type=int, required=True)
    parser.add_argument("--sick", default="", help="comma list of slots made sick")
    parser.add_argument("--timer", type=int, default=SICK_TIMER_DAYS)
    parser.add_argument("--medicine", choices=("selected", "backpack", "none"), default="none")
    parser.add_argument("--tool", type=int, default=None, help="carry-pair selected tool id (overrides --medicine)")
    parser.add_argument("--unfed", action="store_true", help="clear the fed flags: the chores owe a feed")
    parser.add_argument("--shed-medicine", action="store_true", help="put a bottle on the shed shelf (carry pair untouched)")
    parser.add_argument("--hay", type=int, default=None)
    parser.add_argument("--happiness", type=int, default=None)
    args = parser.parse_args(argv)
    sick = tuple(int(item) for item in args.sick.split(",") if item.strip())
    path = mint(
        args.src, args.out, cows=args.cows, sick=sick, timer=args.timer,
        medicine=args.medicine, hay=args.hay, happiness=args.happiness, tool=args.tool, unfed=args.unfed, shed_medicine=args.shed_medicine,
    )
    print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
