# F10 feasibility verdict

Audited 2026-09-30, PR-019 (`rr-ux6x.7.11`). This combines existing F0–F9
research, current code and reports. New emulator frames: 0. The audit does
not amend ADR 0007 or promote a product result.

## Verdict: conditional ceiling, campaign not ready

Seven designed berries leave max stamina 170 and contribute
`(170 - 100) >> 1 = 35`, fifteen below the table maximum of 50.
With every other ordinary term maxed, the sum is **984**:

```
78 + 36 + 36 + 35 + 155 + 124 + 31 + 32 + 32 + 32 + 22 + 21 + 50 + 300 = 984
```

That is an upper bound under those assumptions, not a proved reachable
natural maximum. The previous verdict counted unproved cow happiness,
clock acquisition and an incompatible happiness plan as achieved terms.
Exactly ten native berry eats would restore the arithmetic total to 999;
the remaining gates would still have to clear. Heavy campaign integration
does not have a feasibility go-ahead.

## Evidence that changes the result

| Gate | Established | Missing or contradicted |
|------|-------------|-------------------------|
| F0/F1, wife | `$B5805A` subtracts 2 on hour-18 house entry. Married Maria's poison mushroom `0x02` / text `0x03B0` repeats at −6; 25 offers take 660→510. 510→508 survives. | F1 forced her into player range. PR-080 must approach the native wife script. A pre-entry 496 becomes 494. |
| F2, development | The clear mask has 2,968 walkable grass candidates; 2,950 scores development 100. | Geometry and isolated planting do not establish final coverage. The 18-tile margin can be lost to the final-night hurricane. PR-057/059 are open. |
| F3, herd | Bought day-0 happiness 0. Talk +1, brush +3; one fed slot-1 night held happiness 1 at 1,404 frames. Model reaches 200 after 50 care days. | Bought slot 0 has no matching feed stall, peaks at 88 and declines. A full herd includes slot 0. Born-calf escape (inherit 100, care within 14+21 protected nights) has no native naming/final-night proof. PR-048 is mandatory. Fifty days were not replayed. |
| F4, family | Latest decoded wife bit for two children already set by Y3 Summer D29 is Y2 Fall D9; child 2 is +110 nights. A live gate rejected 448 hearts. | Dated marriage, naming and pregnancy route remains PR-077–079. Children existing by D29 does not imply child 2 has reached age 60. |
| F5, money | Paper ledger ends at 620,000G using decoded prices and selected receipts. | It buys 12 cows, no potions/births. F7 needs nine namings, and slot 0 needs replacement. Milk/egg curves, 750 lumber, frames, losses and happiness repairs must share one budget. PR-014 reopened. |
| F6, berries | Seven designed sources; eat `$81CAAB` adds 10 stamina and increments uncapped count. Best-ending `$83F097` accepts exactly 10. | Death spawns, off-map respawns and transition edge were decoded, not executed. `rr-ux6x.16.7` must prove native 7→10 and guard against 11. Extra berries are not assumed slack. |
| F6, clock/turtle | Clock store and summer arm decoded. Fall D20 prize supplies turtle or indoor-berry arm. | No reached quote-request writer found. Clock +22 is conditional on PR-084. Both prize wins need runtime proof and a recovery policy. |
| F7, happiness | Naming +70, child age-60 bonuses +50/+100 and other sources/sinks decoded. | Wife at index 188 gives child 2 at 298; its age-60 +100 is after ending index 299. Claimed seven-berry weak floor 1000 becomes 900 before other unbudgeted sources; ten berries give 930. Nine births conflict with F5's full purchased barn. PR-016/073 reopened. |
| F8/F9, windows | Accepted bin drops increment immediately; three live potato drops measured 0→3. Maria/Ann flower deltas match ladders. | Four-girl ten-day hold report has `emulator_frames: 0`. Native talk/gift/window and crop scale proofs remain required. |

The score oracle agrees with ROM scoring on six saves including the 999
showcase. That validates the formula; the showcase is not campaign evidence.
`recordings/pr021_calendar_probe.json` now has all five girls on weekday,
Saturday and Sunday (15 sightings, 9,648 frames). The older empty-sighting
handoff is superseded; native interaction skills remain open.

## Execution and evidence gates

PR-023 is reopened: the planner chooses `auto_day_plan_decision` before
reading milestone intents, then prints and journals them. The 55,961-frame
power-on report through D5 demonstrates annotations, not berry execution.

PR-024 is reopened: `campaign_composer.run_whatif_day` explicitly runs on
paper and reports configured score/count values without a ROM.
PR-045/047/050 use synthetic RAM or model objects; PR-072's social report
has zero emulator frames. Reopen those runtime proofs; retain model tests.

PR-098 stays open: Clean power-on ending is 3,476,992 frames, 0 RAM writes
and 0 loads, versus required 2,175,350. Difference +1,301,642 (+59.84%);
new report has 139 bad days. Old report has no day journal, so no first
divergence is established. Four original size targets are at/below 1,000
lines, but `campaign_composer.py` is 1,292 and `multi_day_planner.py` 1,029.

Some domain evidence exists only in uncommitted lane worktrees. Main and
those branches shared `8d6c3602` at audit entry. Lane evidence cannot prove
merged-main behavior. Integrate and check bounded natural-entry windows
before another long campaign run.

## Binding deadlines for the corrected plan

Index 0 is Y1 Spring D1; index 299 is the Y3 Summer D30 tally morning.
These are decoded/model constraints, not a completed route:

- Clock arm by Y1 Summer D30; gift four mornings after acceptance while
  second-house bit is clear. Request trigger remains open.
- Turtle/indoor arm: Y1/Y2 Fall D20 before 15:00. Do not assume recovery
  after a missed second prize.
- Wife bit by Y2 Fall D9 for children present by D29; house level 2 and
  hearts ≥450/≥650 at pregnancy checks. To count child 2's age-60 bonus,
  marry substantially earlier or budget another happiness source.
- Feedable cow buy by Y3 Spring D10 for the modeled 50-morning path.
  Slot 0 needs its own native replacement/birth/care timeline.
- Before D29 hour-18 entry: wife 498–511; other girls 496–511; shipped
  counts 496–511 modulo 512. Guard additional event sinks separately.
- At tally: happiness ≥992, ≥2,950 tiles after night/weather, money
  99,840G–655,350G after shipping, twelve cows at ≥200, twelve chickens,
  and required flags. Exact-ten route must retain count 10/stamina 200.

## Owner decisions and follow-up

ADR 0007 and the earlier PR-005 closure record duplication allowed,
savestate loads allowed, target 999, and 984 accepted if ten cannot be
reached. The current handoff calls decisions parked, while CONTEXT and
PR-094/096 still require zero loads. PR-005 is reopened to reconcile these
sources. This audit preserves the ADR and makes no ruling.

Use the revised [local plan](../../plan.md) for the queue: native born-slot
proof; compatible F7/F5 and PR-022; dispatch PR-023; clock and exact-ten
edge; live domain/arbiter evidence; bounded lane integration. A future F10
update can replace conditional bounds with a demonstrated feasible campaign.
