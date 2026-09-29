# Plan: Harvest Moon (SNES)

Future work only. Proven facts: [STATUS.md](STATUS.md).
Structure: [PLANNING_STACK.md](PLANNING_STACK.md).
Tracker: `bd ready -l harvest -l spine`. One living residual:
[tasks/rr-20w-run16-defects.md](tasks/rr-20w-run16-defects.md).

## Next

Credits already finish. The 2026-09-29 power-on `--until-ending` run reaches
Year 3 Summer D30 (2,175,350 frames, $5,730, one rescue) with daily field
clear forced off. That farm is a berry trip plus a small potato pocket.
The follow-up fixes and seasonal animal schedule pass the complete Harvest
suite (1550 passed, 30 skipped), and the visitor/barn ROM continuation feeds
both animals on 7/7 mornings. Evidence stays in the living residual.
The next sitting proves a short window. It does not start with another
`--until-ending`.

1. Turn power-on daily `CLEAR_FIELD` back on (`rr-bu62.27`, done):
   `run_to_day2` daily field clear re-enabled for `--power-on` and
   `--end-of-spring`. Early clear days verified on a HEADLESS power-on run
   through Day 7 morning (5 overnights, 143,156 frames) ending with held
   item 0 and clean return-home on every day, including the D6 north-yard
   disposal.
2. Score Gate B (`rr-5in`) on a summer-entry stop:
   `--power-on --until-season 1 --until-day 1`. `gate_b_full_ok` is that
   stop with a crop shipped or harvested and `mid_run_state_loads == 0`.
   The Year 3 ending does not set the flag. The out JSON takes
   `day_journal`, `day_failures`, `days_completed`, and `floor_report` from
   the year planner.
3. Spring yield (`rr-bu62.25`): the defaults path reached Summer D1 at
   $3,220 with 3 harvests, under the berry-only $3,600. A single missed
   water stand now continues, and a day whose armed stands all skipped
   stays a partial or a failure. The live miss to explain is a stand that
   waters nothing (`dry_crops=16 watered=0`).
4. Prove the seasonal pasture → cow → chicken → fall mushroom schedule from
   power-on through its real spring predecessor. The purchase flag is enabled
   after live receipts; the summer policy starts checking on D14 and preserves
   feed plus a 300G reserve. The pin window buys later as crop income arrives.
   Improve the remaining seed-carry/harvest misses before another ending run.
   The visitor/barn timing is fixed; Fall D3 cow talking still times out
   (`rr-bu62.30`) after feeding succeeds.
   Buy animal tools and measure milk output in a later window. See
   [livestock_purchase_plan.md](livestock_purchase_plan.md).
5. Name-entry tilemap `0x5F` (`rr-bu62.18`, and the open half of
   `rr-bu62.26`): the one rescue on the farm ending, Y2 Spring D17 at
   `(0, 0)`.
6. Dog proof (`rr-bu62.21`): sleep and rescue already share the `0x0800`
   check, whether or not `0x0080` is set, and only `DogReleaseTask` may
   press A beside the dog. Close it when one stuck morning releases before
   the twelve failed bed presses.
7. The ending contract stays the owner's call: credits alone, or year 3
   credits with a score. Marriage waits on that call and on a live animal.
   Ranch master and 999 stay out.

Spring D2 section work is unchanged and still open: hoe, water, and harvest
stands in the search before a third pocket ring; the northwest bush clear
stalls on the stone at `(5, 5)`.

## Stack

Crop and coop composers are still large. Merge into the day-plan composer
or delete. Soft max about 1000 lines. No sibling extract. After the
working-tree edits, `multi_day_planner.py` is 989 lines,
`day_plan_phases.py` is 936, and `home_return.py` is 758. Detail stays in
PLANNING_STACK. Closed fixture checks stay in STATUS. Do not reopen them
here.
