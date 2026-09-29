# Plan: Harvest Moon (SNES)

Future work only. Proven facts: [STATUS.md](STATUS.md).
Structure: [PLANNING_STACK.md](PLANNING_STACK.md).
Tracker: `bd ready -l harvest -l spine`. One living residual:
[tasks/rr-20w-run16-defects.md](tasks/rr-20w-run16-defects.md).

## Next

The power-on run now reaches the ROM ending with a working farm (STATUS,
2026-09-29). Epic `rr-bu62`; `bd ready -l harvest`.

1. Economy on a power-on farm (`rr-bu62.25`): the working-farm ending run
   holds $5,730, far below the pin chain ($29,650). Daily hand clear is off
   for power-on because a lifted boulder wedged the wild north yard; a real
   disposal target would let it back on (`rr-bu62.26`). Then Gate B
   (`rr-5in`).
2. Decide the ending contract (manifest `ending_definition` is TBD) before
   promoting maturity: credits alone, or year 3 credits with a score.
3. Pre-sleep dog release (`rr-bu62.21`): the stuck 0x0800 flag is only
   cleared after 12 failed sleeps and a rescue (~40k frames when it happens).
4. Animal-shop nav that changes the wallet and the animal count (`rr-4zvl`).
   A menu, or a return to the origin map, is not a buy. Do not RAM-poke
   livestock and call it purchased. See
   [livestock_purchase_plan.md](livestock_purchase_plan.md).
5. Marriage, then the published video (M8 records by replaying the tape from
   power-on). Ranch master and 999 are not required.
6. Spring economy, still open: a third pocket ring only after hoe, water and
   harvest stands are already in the search (do not record that walk); NW
   bush clear stalls on the stone at `(5, 5)` (`Skipping lift thrash`), pin
   `--section bushes --chunk nw`.

## Stack

Crop and coop composers are still large. Merge into the day-plan composer
or delete. Soft max about 1000 lines. No sibling extract. Detail stays in
PLANNING_STACK. Closed fixture checks stay in STATUS. Do not reopen them
here.
