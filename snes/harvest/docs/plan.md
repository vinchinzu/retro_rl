# Plan: Harvest Moon (SNES)

Future work. Verified product results: [STATUS.md](STATUS.md). Structure:
[PLANNING_STACK.md](PLANNING_STACK.md). Campaign detail:
[perfect_rancher/MASTER_PLAN.md](perfect_rancher/MASTER_PLAN.md), with real
bead IDs in [BEAD_PLAN.md](perfect_rancher/BEAD_PLAN.md).
Queue: `bd ready -l harvest -l rancher999`.

## Next: repair feasibility and execution before a campaign soak

The 2026-09-30 audit found newer work than the manager handoff: the original
100 beads had 36 closed, F5 and F10 files existed, and Wave 1 had produced
lane code. Eleven closures were reopened for incompatible plans, missing
runtime evidence, or conflicting contract text. The calendar report now has
all five girls on each of three sample days (15 sightings, 9,648 frames).
That closes the locator-data probe; native location and interaction remain
PR-067–069.

The [F10 verdict](perfect_rancher/research/F10_verdict.md) is conditional.
Seven designed berries contribute 35 stamina points, so **984 is the score
table ceiling if every other term can be maxed**. Neither 984 nor 999 has a
verified campaign route. Exactly ten berries would restore the missing 15
points, but would not resolve the herd, clock, happiness, or execution gaps.

| Priority | Blocker and bead | Required next evidence |
|----------|------------------|------------------------|
| P0 | Slot 0 and breeding: PR-048 `rr-ux6x.11.6`; naming PR-079 `rr-ux6x.15.4` | F3 says a bought slot-0 cow has no matching feed stall and peaks at 88. Prove a native replacement by a calf inheriting 100, its care and final-night survival, and a dated herd/birth plan. `rr-bu62.18`/`.26` block naming. |
| P0 | Incompatible economy and happiness: F5 `rr-ux6x.7.6`, F7 `rr-ux6x.7.8` | F5 buys 12 cows and pays for no births; F7 needs nine namings. The late wedding puts child 2's age-60 bonus after the ending. Build one dated plan including births, milk, feed, potions, lumber, losses and happiness repair. 620,000G is the old model result. |
| P0 | Exactly-ten berries: `rr-ux6x.16.7` | Live native 7→10 acquisition, stamina 200, suppression flags and a stop guard; preserve ten through the final sleep. F6 decoded edges but executed none. PR-082/083 collect the seven designed berries. |
| P0 | Clock trigger: PR-084 `rr-ux6x.16.3` | Reach the quote-request writer through inputs, arm by Y1 Summer D30, and receive the first-upgrade gift before house level 2. F6's store is decoded; the request route is missing. Upgrade 2 now waits on this proof. |
| P0 | Campaign dispatch: PR-022/023 `rr-ux6x.8.2`/`.8.3` | Replace the invalid herd/berry milestones, then turn due intents into tasks with RAM receipts or explicit deferrals. Current code reads intents after choosing daily phases and logs them. |
| P0 | Contract reconciliation: PR-005 `rr-ux6x.4` | ADR 0007 records duplication and savestate loads as allowed, target 999 with 984 accepted as a fallback. The handoff calls those decisions parked; CONTEXT and final-run beads still say zero loads. Resolve the conflict without inventing a new owner ruling. |
| P0 | Structure and behavior: PR-098 `rr-ux6x.18.1` | Proof finished at 3,476,992 frames, +1,301,642 (+59.84%) over 2,175,350, with 139 bad days. Compare matching policies in a short trace; the old report lacks a day journal. `campaign_composer.py` is 1,292 lines and `multi_day_planner.py` 1,029. Keep the frame-equality acceptance. |

Immediate ready work is a bounded clock-request probe (`rr-ux6x.16.3`),
the exactly-ten research probe, or a short PR-098 comparison. Claim one.
For the herd branch, fix the name-entry prerequisites, then PR-048; feed
that result and F7 into F5, and F5 into the milestone solver. Do not repeat
an ending soak to diagnose these local gaps.

## Runtime proofs and integration

PR-024 (arbiter), PR-045 (12-slot targeting), PR-047 (cow window), PR-050
(medicine/weather), PR-072 (four-girl windows), and PR-073 (happiness
projection) are reopened. The arbiter explicitly simulates a day on paper;
animal tests synthesize RAM or modify model objects; the social reports say
`emulator_frames: 0`. Keep those checks as model coverage and obtain the
runtime evidence their acceptance requires. A closed research or unit-test
bead is not a natural-entry route proof.

Lane branches and main were all at `8d6c3602` at audit entry. Some closures
refer to uncommitted FARM, ANIMALS and SOCIAL worktree files absent from
main. Integrate one lane at a time with its evidence and tracker, then run
a bounded ROM window on the merged code. Only then advance PR-089–095
through natural checkpoints and the final run under the reconciled load rule.

## Constraints the revised calendar must preserve

- Wife hearts: F1 measured married Maria 660→510 in 25 poison-mushroom
  offers (item `0x02`, text `0x03B0`), with her position forced into range.
  PR-080 must approach the real wife script. F0's hour-18 entry subtracts 2:
  target 498–511 before it; 510→508 is safe, 496→494 is not.
- Two children already present on Y3 Summer D29: wife bit by Y2 Fall D9,
  with house and pregnancy heart floors satisfied. That is a children
  deadline, not a guarantee of either child's age-60 happiness bonus.
- Farm geometry: 2,968 walkable cover candidates, 2,950 needed, 18 tiles
  of slack. PR-057/059 must preserve coverage after the final night;
  a hurricane can destroy grass after the last work morning.
- Cows: a feedable purchased cow reaches 200 after 50 care days in the
  model; latest buy Y3 Spring D10 for the D30 morning read. Only one
  feedable night was measured. Slot 0 needs a separate birth timeline.
- Girls and crops: four non-wife girls and all four shipped counts must
  remain in 496–511 (crop counts modulo 512). Happiness must read ≥992;
  money must be in 99,840G–655,350G after the overnight credit.
- Turtle/indoor-berry arm: Y1 and Y2 Fall D20 before 15:00. Missed wins
  require an explicit recovery policy. Above ten berries fails the
  best-ending equality check even if the mastery sum caps at 999.

## Existing farm spine

Keep short-window work in
[tasks/rr-20w-run16-defects.md](tasks/rr-20w-run16-defects.md): summer-entry
Gate B (`rr-5in`), weak spring yield (`rr-bu62.25`), cow talk timeout
(`rr-bu62.30`), crop stand weeds (`rr-bu62.8`), name-entry warps
(`rr-bu62.18`/`.26`), dog release (`rr-bu62.21`/`.23`), and the money
visitor (`rr-bu62.14`). Purchase success must include a new initialized
cow slot after naming, not only wallet/count changes. The seasonal purchase
recipe remains [livestock_purchase_plan.md](livestock_purchase_plan.md).

The earlier 2,175,350-frame working-farm credits proof remains in STATUS.
The audit changes planning and evidence claims; it does not promote a
Perfect result or replace the verified product baseline.
