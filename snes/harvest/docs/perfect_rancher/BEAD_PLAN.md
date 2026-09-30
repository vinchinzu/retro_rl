# Perfect Rancher (999): 100-bead plan

**Filed under `rr-ux6x`.** Original plan keys `PR-001..PR-100` now have
real IDs printed under each heading. The live tracker owns current status.
The 2026-09-30 audit reopened eleven closures and added exactly-ten
research `rr-ux6x.16.7`; audit task `rr-ux6x.19` tracks this update. Strategy, score formula and the evidence tags
([ROM], [SCAN], [MEAS], [EXT], [TBD]) are defined in
[MASTER_PLAN.md](MASTER_PLAN.md). Every bead below assumes a sub-agent has
read that file and `snes/harvest/AGENTS.md` first.

## Historical filing recipe

1. Create PR-001 first, then each epic with `--parent <PR-001 id>`. Then
   create each task with `--parent <its epic id>`.
2. Use these fields:
   - `-t epic|task|decision`, `-p <P>`.
   - `-l harvest,rancher999,<kind>`. Kinds: `research`, `meta`, `compose`,
     `stabilize`, `status`.
   - Add `spine` only where it is listed below. That keeps
     `bd ready -l harvest -l spine` for the serial product path.
   - `--acceptance "<Done when>"`.
   - `-d "plan key PR-0NN; snes/harvest/docs/perfect_rancher/BEAD_PLAN.md"`.
3. Add each `Deps` entry with `bd dep add <this> <dep>`. `rr-…` IDs are
   existing beads.
4. Then run `bd export -o .beads/issues.jsonl`, and commit it together with
   these docs.

Working queue: `bd ready -l harvest -l rancher999`.

## Rules for every sub-agent

- Work in your lane's worktree (PR-006 kit). Do not write to `bd` or edit
  `.beads/`: report back and the orchestrator updates the bead. Never run
  `git add -A`.
- Clean only on route code: no `LiveRamEditor` or `--ram-set`. RAM-edited
  pins are research-only and are named `LN_<lane>_whatif_*`.
- Run ROM jobs detached (`setsid nohup`, `--trials 1`). Give every report its
  JSON path, frame count and the Clean audit block.
- Research output goes to `snes/harvest/docs/perfect_rancher/research/<file>.md`,
  with decomp line citations and probe JSON paths.
- When the ROM contradicts the plan, stop. Write down the contradiction and
  hand back. Do not code around it.

Agent brief template (the orchestrator fills this in when spawning):

```
Bead PR-0NN (<rr id>): <title>. Lane <LANE>, worktree <path>, branch harvest/pr999-<lane>.
Read: snes/harvest/AGENTS.md, docs/perfect_rancher/MASTER_PLAN.md §<n>, BEAD_PLAN.md#pr-0NN,
and the research verdicts it depends on: <files>.
Do: <Do>. Done when: <Done when>.
Constraints: Clean route code; no bd writes; lane-prefixed pins; detached ROM runs.
Return: changed files, test command + result, ROM report paths, frames, open questions.
```

## Audited execution order, 2026-09-30

Read [../plan.md](../plan.md) and the [F10 audit](research/F10_verdict.md).
The original queue had 36 closed beads at audit entry, beyond the manager
handoff's 13. Wave 1 has lane work, but model tests were treated as live
proofs and those closures are reopened. Do not gate a natural campaign on
those tests. The calendar-data probe has fifteen live sightings now.

| Stage | Entry condition | Work |
|-------|-----------------|------|
| Feasibility repair | now, bounded probes | PR-084 clock request, native exact-ten edge, PR-098 short behavior comparison; naming prerequisites then mandatory PR-048 slot-0 birth path |
| Compatible plan | herd/naming and dated happiness sources | Reopened F7/PR-016 and F5/PR-014, then PR-022; preserve clock-before-house-2, final-night cover, children and window constraints |
| Runtime skills | accepted dated tasks | Reopened PR-023 dispatch, PR-024 full-scale ROM day, PR-045/047/050 animals, PR-072/073 social; remaining crop, home and collectible skills |
| Integration | lane code and native receipts available | Commit and integrate one lane at a time; bounded ROM windows on main, then PR-089/090/091 natural checkpoints |
| Finish | term gaps zero, PR-005 reconciled, PR-098 passes | PR-092/093 audit and endgame, PR-094/095 final proof and reproduction, PR-096 publication |

PR-005 records an ADR ruling but is reopened because the handoff and
zero-load acceptance text conflict with it. No new owner ruling is inferred.
The old `milestones.yaml` is marked as a model awaiting revision. Exactly
seven berries imply a conditional ceiling of 984; ten berries alone do not
establish 999. Critical blockers include bought slot 0, no birth budget,
late child-age happiness, the clock trigger and milestone execution.

---

## Root

### PR-001 — Harvest: Perfect Rancher, Clean power-on to a 999 ranch rating
Tracker: `rr-ux6x`.
`epic · P1 · CAMPAIGN · harvest,rancher999,spine · Deps: none (related: rr-bu62)`
Done when: PR-095 is closed, STATUS shows the 999 run, and the manifest
ending contract says Perfect.

### PR-002 — Record the Perfect contract (ADR 0007)
Tracker: `rr-ux6x.1`.
`decision · P0 · ORACLE · meta · Deps: none · W0`
Do: Write ADR 0007: Perfect = RAM `ranch_mastery` 999 at ending `0x25`,
from one Clean continuous power-on run. Add the Perfect rung after Finish in
`CONTEXT.md`, and rewrite plan.md item 7 ("999 stays out"). Set the harvest
manifest ending contract, then run `docs/generate_game_matrix.py`.
Done when: `uv run pytest tests/test_docs.py -q` passes and the owner
approves the wording.

### PR-003 — Validate the score oracle against the ROM
Tracker: `rr-ux6x.2`.
`task · P0 · ORACLE · stabilize · Deps: none · W0`
Do: Fix these in `harvest/core/ranch_master.py`:
- The money term reads the low 16 bits of `moneyL`, not 24-bit `money_raw`.
- Replace the invented rating titles and the "Blue Feather" text with ROM
  text ids.

Add unit tests for every term: the 9-bit wrap, the money wrap, all 12 cow
slots, and the 999 cap. On the ROM, run `harvest.tools.ending_probe` on at
least 6 states:
- `LN_final_poweron_ending`
- the three showcase saves (unmarried, married, 999 preset)
- two mixed what-if pins

Assert oracle == RAM `ranch_mastery` after index `0x20` on each.
Done when: 6/6 agree and `tests/test_ranch_master.py` passes.

### PR-004 — Score ledger in every run report
Tracker: `rr-ux6x.3`.
`task · P1 · ORACLE · compose · Deps: PR-003, PR-009 · W1`
Do: Each morning, write per-term `{value, max, gap, window_state}` into
`day_journal`. Add the D30 projection using the F0 drift model. Extend the
`ending` block with the oracle breakdown, the ROM `ranch_mastery` at
`0x20`/`0x25`, and the rating text ids.
Done when: a `--power-on --until-day 3` JSON has ledger rows, and the
berry-only ending replays with a score of 9.

### PR-005 — Owner ruling: glitch levers, RNG lookahead, fallback target
Tracker: `rr-ux6x.4`.
`decision · P0 · orchestrator · meta · Deps: PR-019 · W1`
Do: Reconcile ADR 0007's recorded duplication/savestate/fallback rulings
with the latest owner handoff and zero-load text in CONTEXT, PR-094 and
PR-096. Exactly ten eaten berries preserves the equality check; above ten
fails it. The score-table ceiling with seven is conditional 984, not an
already feasible fallback run.
Done when: ADR, CONTEXT and final-run acceptance agree on all three rulings,
including load lineage/reproduction if loads remain allowed. Preserve the
recorded ruling until the owner resolves the conflict.

### PR-006 — Sub-agent lane kit
Tracker: `rr-ux6x.5`.
`task · P0 · orchestrator · meta · Deps: none · W0`
Do: Add `snes/harvest/scripts/pr999_lane.sh <lane>`. It creates worktree
`rr_harvest_pr999_<lane>` on branch `harvest/pr999-<lane>`, with symlinks for
`roms`, `.venv`, the pins and the tapes, and an `info/exclude` entry. Add a
detached run launcher (`setsid nohup`, `wait`) and a pytest wrapper with
the lane `PYTHONPATH`. Add a 3-line trap note to the harvest AGENTS.md.
Done when: two lanes are created, and each runs a pin probe and the harvest
suite from its own code.

### PR-007 — Research what-if pin set (research-only)
Tracker: `rr-ux6x.6`.
`task · P1 · ORACLE · meta · Deps: PR-003 · W0`
Do: Use `state_preset_builder` to build `LN_whatif_*` pins:
- married with house level 3
- 12 cows with mixed happiness
- 12 chickens
- girls at 480 hearts
- the Y3 Summer D20 near-perfect state
- a fully covered farm

List them in `perfect_rancher/research/whatif_pins.md`, marked as never
route evidence.
Done when: each pin loads and the oracle reports the expected per-term
values.

---

## E1: Feasibility gates (ROM lane)

### PR-008 — Epic: Feasibility gates F0–F10
Tracker: `rr-ux6x.7`.
`epic · P0 · ROM · research · Deps: PR-003`
Done when: PR-019 posts the verdict.

### PR-009 — F0: end-of-game drift census
Tracker: `rr-ux6x.7.1`.
`task · P0 · ROM · research · Deps: PR-007 · W0`
Do: From `LN_ending_Y3S29_eve` and a near-perfect D29 what-if pin, sleep
through to index `0x20`. Record every score input before and after: hearts,
the 12 cow bytes, the farm tile census, money, happiness and flags. Decode
the writers in NightReset, the D29 6:00 event (`0A/5`) and `CODE_83F4D8`.
Correct the Maria drift to −2 at `$B5805A`; target the post-drift window.
Done when: `research/F0_drift.md` gives a per-term drift model and a ledger
projection formula, with probe JSON.

### PR-010 — F1: the wife's hearts path (≥ 650, then 496–511)
Tracker: `rr-ux6x.7.2`.
`task · P0 · ROM · research · Deps: PR-007 · W0`
Do: Decode the trigger behind every negative `0x41` heart site [SCAN]: gift
item, dialogue choice, festival or event. For each, find whether it repeats,
how often per day, and whether it still applies when married. Decode the
wife's automatic heart changes after marriage. Probe: a married wife at 660
reaches 496–511 through the chosen source.
Done when: `research/F1_wife_hearts.md` holds a probed recipe, or a proof
that it is impossible and what the natural maximum becomes.

### PR-011 — F2: development coverage (2,950 of 2,974)
Tracker: `rr-ux6x.7.3`.
`task · P0 · ROM · research · Deps: PR-007 · W0`
Do: Take the tile-id census of a cleared farm. Answer:
- Which tiles can be covered by grass, fence or crop?
- Can the farmer walk on each counted type?
- What does cover cost per tile (seed price and pattern, lumber per fence,
  frames)?
- Do all grass stages stay at or above `0x1D` after cutting?
- What do hurricanes, season changes and winter do to counted tiles?

Draft a layout with at most 24 uncovered tiles and paths to the house, bin,
barn, coop, shed and exit.
Done when: `research/F2_development.md` and the layout draft prove
≥ 2,950 with access.

### PR-012 — F3: herd and flock feasibility
Tracker: `rr-ux6x.7.4`.
`task · P0 · ROM · research · Deps: PR-007 · W0`
Do: Measure the cow happiness byte (slot `+4`):
- start value when bought and when born
- deltas for talk, brush, milk, feed and time outdoors
- decay, sickness and death rules

Also cover barn capacity, the miracle potion (price, gestation, calf to
adult), whether `cow_N` counts calves, coop capacity (are more than 12
chickens possible?), incubation time and chick growth.
Done when: `research/F3_herd.md` has the latest dates for "cow k reaches
200", backed by 2 probes (a bought cow's day-0 value and one day of full care).

### PR-013 — F4: family timeline
Tracker: `rr-ux6x.7.5`.
`task · P0 · ROM · research · Deps: PR-007 · W0`
Do: Decode the writers of the `$7F1F66` wife bits: the marriage requirements
(house level, hearts, event or item) and the flow. Decode:
- `wife_pregnancy` counter semantics (the `≥ 20` check)
- gestation length
- child flags and ages in `$7F1F6E`
- the spacing between child 1 and child 2
- the naming screens

Compute the latest marriage date that still gives two children by Y3 Summer
D29.
Done when: `research/F4_family.md` has the dated timeline and one
pregnancy step probed from a what-if pin.

### PR-014 — F5: economy model over the whole game
Tracker: `rr-ux6x.7.6`.
`task · P0 · ROM · research · Deps: PR-011, PR-012, PR-013 · W0`
Do: Build the price table from ROM tables and shop probes: house upgrades,
brush, milker, sprinkler, grass seed, fodder, chicken feed, miracle potion,
medicine, and tomato and corn seed. Measure milk size against cow happiness,
and egg output. Extend `livestock_econ` and `spring_opt` into a 299-day cash
ledger.
Done when: `research/F5_economy.md` and the ledger output show a money path
that ends between 99,840G and 655,350G on D30 under compatible F2/F3/F4/F7
plans, including native slot-0 replacement, births, lumber and repairs.

### PR-015 — F6: collectibles (10 berries, clock, turtle shell)
Tracker: `rr-ux6x.7.7`.
`task · P0 · ROM · research · Deps: PR-007 · W0`
Do: Enumerate every power-berry source (object spawn sites, the
`$7F1F64`/`$7F1F5C` flags) with its trigger, dates, RNG and number of
chances. Record the seven designed sources and the three missing eats; do not
assume ten one-shots. Decode the sources of the clock
(`$7F1F6E & 0x4000`) and the turtle shell (`$7F1F6C & 0x1000`). Note the
duplication glitch mechanics without using them.
Done when: `research/F6_collectibles.md` gives an acquisition plan per item
with earliest and latest dates. Closed research is not route proof; the
quote-request trigger and exactly-ten native acquisition still need their
own live probes.

### PR-016 — F7: happiness to 992 or more
Tracker: `rr-ux6x.7.8`.
`task · P0 · ROM · research · Deps: none · W0`
Do: Decode the `AddPlayerHappiness` callers and the `0x41` sites on
`$7F1F33` [SCAN] into a trigger table. Rank the repeatable positive sources
and list the sinks to avoid (Saturday no, dialogue −1s, collapse).
Done when: `research/F7_happiness.md` has a dated plan to reach ≥ 992 by
Y3 Summer D29, with each bonus before tally and barn space/birth/repair
costs compatible with F5. Child 2 age 60 cannot be counted after the ending.

### PR-017 — F8: shipped-count semantics
Tracker: `rr-ux6x.7.9`.
`task · P0 · ROM · research · Deps: none · W0`
Do: Decode the `$7F1F5E` bit 0 gate around `$818140`. Probe:
- Does dropping N potatoes in the bin add N to the counter?
- Is it counted at drop or at the overnight sale?
- Do the peddler and the horse saddlebag count?
- Does milk or anything else touch the four counters?

Done when: `research/F8_shipping.md` and the probe JSON exist.

### PR-018 — F9: heart table per girl
Tracker: `rr-ux6x.7.10`.
`task · P0 · ROM · research · Deps: none · W0`
Do: For each of the 5 girls, list heart sources and sinks with triggers:
talk (including any daily cap), liked and disliked gifts, festival dances,
birthdays, choice events and automatic changes, plus her schedule. Measure
talk and gift deltas live on 2 girls.
Done when: `research/F9_hearts.md` exists, with measured deltas matching
the decoded ones.

### PR-019 — F10: feasibility verdict
Tracker: `rr-ux6x.7.11`.
`task · P0 · ROM · research,meta · Deps: PR-009..PR-018 · W0 end`
Do: Combine F0–F9 into one verdict. Is 999 reachable naturally? If not,
what is the natural maximum, and which term is short? Where is the real
slack (chickens above 12, berries above 10)? Name the binding deadlines.
Done when: `research/F10_verdict.md` exists, MASTER_PLAN §3 and §4 are
updated, and PR-005 is triggered if needed.

---

## E2: Campaign scheduler

### PR-020 — Epic: milestone calendar, arbiter, windows, checkpoints
Tracker: `rr-ux6x.8`.
`epic · P1 · CAMPAIGN · compose · Deps: PR-019`

### PR-021 — Calendar and schedule data
Tracker: `rr-ux6x.8.1`.
`task · P1 · CAMPAIGN · compose · Deps: none · W0`
Do: Build the data for festivals (dates, forced or not), shop hours and
closures from the ROM/Calendar data, carpenter availability, and each
girl's location by season, weekday, hour and weather. Sources:
`npc_catalog`, the ROM, and live probes.
Done when: the data module and tests exist, and a live probe locates all 5
girls on 3 sample days.

### PR-022 — Backward milestone solver → `milestones.yaml`
Tracker: `rr-ux6x.8.2`.
Audit: rebuild the bought-slot-0/no-birth milestones using accepted F5/F7
and live native edge recipes; existing tests establish arithmetic only.
`task · P0 · CAMPAIGN · compose · Deps: PR-019, PR-014 · W1`
Do: From the F-gate numbers, solve latest-start dates for every term: herd
purchases, the marriage chain, crop quotas per season, the cover window,
berry chances. Emit dated commitments with their slack.
Done when: solver tests pass, and moving a what-if milestone 10 days
reports the slack change correctly. Commitments must cover a compatible
herd/birth/happiness/cash plan and label every unproved native edge.

### PR-023 — Year planner consumes the milestones
Tracker: `rr-ux6x.8.3`.
Audit: current intent integration only logs; dispatch and outcome receipts
are required before closing this runtime task.
`task · P0 · CAMPAIGN · compose · Deps: PR-022 · W2`
Do: `MultiDayPlannerTask` picks day intents from `milestones.yaml`. This
merges into the existing composer; no new runner.
Done when: a bounded power-on journal shows due milestones dispatched into
actual tasks, with RAM success receipts or explicit failure/defer outcomes.
Logging a commitment after daily phase selection is insufficient.

### PR-024 — Daily arbiter with frame accounts
Tracker: `rr-ux6x.8.4`.
`task · P1 · CAMPAIGN · compose · Deps: PR-023 · W2`
Do: Order the day's blocks: deadline-critical, then animals, then crops,
then social, then cover. Give every block a frame account, and add the
post-18:00 lane (cover, hoeing) with hot-spring refills.
Done when: arbitration unit tests pass, and a live full-scale ROM what-if
day runs every block with measured per-block frames and native receipts.
`run_whatif_day` paper accounts are model coverage only.

### PR-025 — Window-controller framework
Tracker: `rr-ux6x.8.5`.
`task · P1 · ORACLE · compose · Deps: PR-004, PR-009 · W1`
Do: A generic target `[lo, hi]` for a RAM counter. It uses measured
per-action deltas and the projected automatic changes (F0) up to D30, and a
hard guard refuses any action that would overshoot.
Done when: unit tests with synthetic deltas pass, and at least 2 controllers
(PR-032, PR-072) use it.

### PR-026 — Natural checkpoint ladder
Tracker: `rr-ux6x.8.6`.
`task · P1 · CAMPAIGN · stabilize · Deps: none · W1`
Do: Add `run_to_day2 --save-season-pins`, which writes
`PR_Y{n}_{Season}_D1` pins during a continuous run and records the parent
run hash. `--state PR_*` resumes and reports that lineage.
Done when: a resume from the Y1 Summer D1 pin reproduces the continuous
run's next day frame for frame.

### PR-027 — Campaign progress report
Tracker: `rr-ux6x.8.7`.
`task · P2 · ORACLE · meta · Deps: PR-004, PR-022 · W2`
Do: `python -m harvest.tools.pr999_report <run.json>` prints the term gaps
against the milestone deadlines, the window states, and the remaining
collectible chances.
Done when: it runs on a Y1 checkpoint JSON and flags a what-if lagging cow.

---

## E3: Economy, crops and shipping windows (FARM lane)

### PR-028 — Epic: crops to four shipped windows and money to its window
Tracker: `rr-ux6x.9`.
`epic · P1 · FARM · compose · Deps: PR-019`

### PR-029 — Crop quota planner
Tracker: `rr-ux6x.9.1`.
`task · P1 · FARM · compose · Deps: PR-017, PR-022 · W1`
Do: Set per-season targets so each crop lands at 496–511. Potato and
turnip must close by Y3 Spring D30. Tomato and corn must close by Y3
Summer D29. Feed the targets into `spring_opt` and the summer plan.
Done when: the quota table is in the milestones, and tests cover
overshoot-proof targets.

### PR-030 — Field scale-up: many plots per crop
Tracker: `rr-ux6x.9.2`.
`task · P1 · FARM · compose · Deps: rr-bu62.25 · W1`
Do: Plant, water, harvest and ship several 3×3 plots per cycle. Fix stands
that water nothing (`dry_crops=16 watered=0`).
Done when: a pin A/B shows N plots per cycle with frames, and a natural
spring checkpoint meets the Y1 spring quota.

### PR-031 — Watering at scale: sprinkler
Tracker: `rr-ux6x.9.3`.
`task · P2 · FARM · compose · Deps: PR-014 · W1`
Do: Buy the sprinkler (unlocked Summer D20+ or Y2 [ROM]) and add a use
skill, with can-refill routing for plots it does not reach.
Done when: a summer pin waters every plot with fewer frames than the can
alone.

### PR-032 — Shipped-count window controller
Tracker: `rr-ux6x.9.4`.
`task · P0 · FARM · compose · Deps: PR-025, PR-017 · W1`
Do: Add a per-crop stop rule in the harvest and ship phases. Once a crop is
in its window, surplus stays unharvested or goes out as a gift, never into
the bin.
Done when: a what-if pin at 490 potatoes ends at 496–511 after a harvest
day, and unit tests cover it.

### PR-033 — Bulk shopping skill
Tracker: `rr-ux6x.9.5`.
`task · P1 · FARM · compose · Deps: none · W1`
Do: Buy several bags or units of seed, grass seed and feed in one visit.
Verify every purchase by wallet change plus stock change, not by the menu.
Done when: one visit buys 5 bags on a pin, with receipts in the JSON.

### PR-034 — Money ledger and end window
Tracker: `rr-ux6x.9.6`.
`task · P1 · FARM · compose · Deps: PR-014, PR-004, rr-bu62.14 · W1`
Do: Forecast every income line (crops, milk, eggs, forage) and spend. Hold
money at or below 600k. Handle the Y2 2,000G visitor. Check the end window
99,840G–655,350G with the D29 overnight credit included.
Done when: the forecast matches a Y1 checkpoint run within 5%, and a
what-if test covers the window guard.

### PR-035 — Produce shipping at scale
Tracker: `rr-ux6x.9.7`.
`task · P2 · FARM · compose · Deps: PR-046, PR-038 · W2`
Do: Put milk, eggs and forage into the bin loop. Measure bin capacity per
day and handle the 5pm ShippingScene with many items (pulse A).
Done when: a full-scale what-if day ships 12 milk, 12 eggs and crops, and
the overnight credit matches.

---

## E4: Chickens (ANIMALS lane)

### PR-036 — Epic: 12 chickens held to D30
Tracker: `rr-ux6x.10`.
`epic · P1 · ANIMALS · compose · Deps: PR-012`

### PR-037 — Flock growth to capacity
Tracker: `rr-ux6x.10.1`.
`task · P1 · ANIMALS · compose · Deps: PR-012 · W1`
Do: Buy the first chicken (1,000G [MEAS]) or take the free source. Run the
incubation loop (egg, incubator, chick, adult) until 12, or until the
capacity F3 found.
Done when: a pin run goes from 1 to at least 4 chickens Clean, and the
growth rate matches F3.

### PR-038 — Chicken feed supply and eggs
Tracker: `rr-ux6x.10.2`.
`task · P1 · ANIMALS · compose · Deps: PR-033 · W1`
Do: Keep a feed stock target, feed 12 birds daily, and route each egg to
ship or keep for incubation.
Done when: a 7-day pin window with 12 hens has 0 unfed days.

### PR-039 — Coop chores at 12
Tracker: `rr-ux6x.10.3`.
`task · P2 · ANIMALS · stabilize · Deps: PR-037 · W2`
Do: Audit `CoopChoresTask` frames and pathing in a full coop.
Done when: at most N frames per day with 12 hens (N set by the arbiter),
and 0 stalls over 7 days.

### PR-040 — Flock safety
Tracker: `rr-ux6x.10.4`.
`task · P1 · ANIMALS · stabilize · Deps: PR-037 · W2`
Do: Never leave a bird outdoors in a storm, and never sell or lose one.
Add weather gates.
Done when: a storm-day what-if keeps 12 of 12.

### PR-041 — Flock count guard
Tracker: `rr-ux6x.10.5`.
`task · P2 · ANIMALS · compose · Deps: PR-025 · W2`
Do: Stop incubating at the target. Raise a ledger alarm on any drop.
Done when: a what-if at 12 incubates nothing more.

---

## E5: Cows (ANIMALS lane)

### PR-042 — Epic: 12 cows, every one at 200 or more on D30
Tracker: `rr-ux6x.11`.
`epic · P1 · ANIMALS · compose · Deps: PR-012`

### PR-043 — Cow purchase at scale
Tracker: `rr-ux6x.11.1`.
`task · P1 · ANIMALS · compose · Deps: PR-012, PR-022 · W1`
Do: Buy cows to the milestone count and dates, respecting the delivery
gates (grass ≥ 16 × animals, weather, one a day) and the cash reserve.
Done when: a pin window buys 3 cows on schedule, each with wallet/count
receipts and a new initialized slot after naming. Buying all twelve is not
the current herd/birth plan.

### PR-044 — Cow tools: brush and milker
Tracker: `rr-ux6x.11.2`.
`task · P1 · ANIMALS · compose · Deps: PR-014 · W1`
Do: Buy both tools and add use skills, checking the ownership bits.
Done when: one pin day brushes and milks one cow, the milk is shipped, and
the happiness delta is recorded.

### PR-045 — Per-cow tracking and targeting
Tracker: `rr-ux6x.11.3`.
Audit: synthetic RAM slot targeting is not a live twelve-cow proof.
`task · P1 · ANIMALS · compose · Deps: PR-012 · W1`
Do: Map each slot to its sprite so a named cow can be targeted, starting
with the lowest happiness (extend `cow_target`).
Done when: in a 12-cow what-if, every slot is targeted once, verified by
slot deltas.

### PR-046 — Daily cow care loop at 12
Tracker: `rr-ux6x.11.4`.
`task · P1 · ANIMALS · compose · Deps: rr-bu62.30, PR-044, PR-045 · W2`
Do: Feed, talk, brush, milk and ship milk for 12 cows within the arbiter's
frame budget.
Done when: 7 days on a 12-cow what-if with all cows cared for daily, 0
timeouts, and frames logged.

### PR-047 — Cow happiness controller
Tracker: `rr-ux6x.11.5`.
Audit: `controller_day` model objects do not measure a ROM happiness gap.
`task · P0 · ANIMALS · compose · Deps: PR-025, PR-046 · W2`
Do: Project each cow to ≥ 200 by D30 and escalate care for laggards. Raise
a ledger alarm when a cow cannot make it in time.
Done when: in a what-if with one cow 20 behind, the controller closes the
gap.

### PR-048 — Mandatory breeding and born-slot-0 path
Tracker: `rr-ux6x.11.6`.
`task · P0 · ANIMALS · compose · Deps: PR-079, PR-012 · repair`
Do: Prove a native way to free slot 0 and name a calf inheriting 100
happiness. Schedule protected care, the final sleep, all required births,
maternal quality, barn space, miracle potions and slot turnover. Coordinate
with F5/F7 rather than filling the barn with purchases.
Done when: a live born-slot-0 probe reaches ≥200 at the D30 tally without
RAM slot copies, and a dated herd/birth budget supplies twelve existing
cows at ≥200 and the required happiness sources.

### PR-049 — Fodder pipeline for 12
Tracker: `rr-ux6x.11.7`.
`task · P1 · ANIMALS · compose · Deps: PR-054 · W2`
Do: Keep a hay stock of at least 12 a day, including the winter stockpile,
and cut grass at scale.
Done when: a winter what-if with 12 cows has 0 unfed days for 30 days.

### PR-050 — Herd health and weather
Tracker: `rr-ux6x.11.8`.
Audit: pure medicine/storm formulas are not a cure or a live overnight.
`task · P1 · ANIMALS · stabilize · Deps: PR-046 · W2`
Do: Detect sickness and give medicine. Set the in/out policy per weather.
Never lose a cow.
Done when: a sick-cow what-if is cured, and a storm day keeps 12 of 12.

---

## E6: Farm development to 2,950 tiles (FARM lane)

### PR-051 — Epic: development 100 on D30
Tracker: `rr-ux6x.12`.
`epic · P1 · FARM · compose · Deps: PR-011`

### PR-052 — Full clear
Tracker: `rr-ux6x.12.1`.
`task · P1 · FARM · compose · Deps: PR-011, PR-065 · W1`
Do: Remove every weed, stone, stump and large rock. The farm census then
shows only coverable ids.
Done when: after a pin run, the census has 0 debris ids.

### PR-053 — Final layout file and linter
Tracker: `rr-ux6x.12.2`.
`task · P0 · FARM · compose · Deps: PR-011 · W1`
Do: Write the target tile map: ≥ 2,950 counted tiles, with paths to the
house, bin, barn, coop, shed and exit. Add a linter that diffs the live farm
map into a per-tile to-do list.
Done when: the linter runs on 3 pins, and the layout passes a BFS access
check.

### PR-054 — Grass cover skill at scale
Tracker: `rr-ux6x.12.3`.
`task · P1 · FARM · compose · Deps: PR-033, PR-053 · W1`
Do: Till and seed large areas in the post-18:00 lane, with hot-spring
stamina refills.
Done when: one pin evening covers at least N tiles (target from F2), with
tiles/frame logged.

### PR-055 — Fence placement skill
Tracker: `rr-ux6x.12.4`.
`task · P2 · FARM · compose · Deps: PR-062, PR-053 · W2`
Do: Place fences (lumber → `0x05`) wherever F2 says fences beat grass.
Done when: a pin places a 10-tile fence run and the census counts it.

### PR-056 — Navigation over a covered farm
Tracker: `rr-ux6x.12.5`.
`task · P1 · FARM · stabilize · Deps: PR-053, rr-bu62.8 · W1`
Do: Add walkable sets for grass, crop and fence ids, and BFS across a fully
covered farm.
Done when: a covered-farm what-if completes all chores with 0 nav stalls.

### PR-057 — Storm and season resilience
Tracker: `rr-ux6x.12.6`.
`task · P1 · FARM · stabilize · Deps: PR-009, PR-054 · W2`
Do: Detect counted tiles lost to a hurricane or season change, and
re-cover them.
Done when: a hurricane what-if recovers to the pre-storm count.

### PR-058 — Daily development census in the ledger
Tracker: `rr-ux6x.12.7`.
`task · P2 · ORACLE · compose · Deps: PR-004, PR-053 · W1`
Do: Log counted tiles, the gap to 2,950, and the linter's uncovered list.
Done when: the census appears in the journal of a checkpoint run.

### PR-059 — Final fill window (Y3 Summer D20–D29)
Tracker: `rr-ux6x.12.8`.
`task · P0 · FARM · compose · Deps: PR-054, PR-057 · W4`
Do: Finish the cover and hold it. The D29 projection after drift must be
≥ 2,950.
Done when: the Y3 Summer pin run reaches ≥ 2,950 at index `0x20`.

---

## E7: House, lumber and tools (HOME lane)

### PR-060 — Epic: both house upgrades and the tools the plan needs
Tracker: `rr-ux6x.13`.
`epic · P1 · HOME · compose · Deps: PR-019`

### PR-061 — Carpenter research and probe
Tracker: `rr-ux6x.13.1`.
`task · P0 · HOME · research · Deps: none · W0`
Do: Document the upgrade flow: dialogs, cost, lumber needed, build days,
and the flag bits (`$7F1F64` `0x40`/`0x80`).
Done when: `research/carpenter.md` exists, with one upgrade order probed
from a funded what-if pin.

### PR-062 — Lumber pipeline
Tracker: `rr-ux6x.13.2`.
`task · P1 · HOME · compose · Deps: PR-061 · W1`
Do: Chop stumps on the farm and the mountain. Track the lumber count and
cap, with stock targets for the upgrades and fences.
Done when: a pin run gathers the upgrade-1 lumber Clean.

### PR-063 — House upgrade 1
Tracker: `rr-ux6x.13.3`.
`task · P1 · HOME · compose · Deps: PR-061, PR-062 · W2`
Do: Order upgrade 1 and complete it, Clean.
Done when: flag `0x40` is set on a natural pin, with a wallet and lumber
receipt.

### PR-064 — House upgrade 2 and the Saturday event
Tracker: `rr-ux6x.13.4`.
`task · P1 · HOME · compose · Deps: PR-063 · W2`
Do: Order upgrade 2 and complete it (flag `0x80`). Answer the Y2+ Saturday
yes/no event with the +20 happiness choice.
Done when: both are verified on a natural pin.

### PR-065 — Tool upgrades
Tracker: `rr-ux6x.13.5`.
`task · P2 · HOME · compose · Deps: PR-011 · W1`
Do: Get the upgraded or golden tools that F2 and F3 need (large-rock
hammer, faster hoe or sickle). Research the source, then add the skill.
Done when: each tool is owned on a pin, and the tiles/frame gain is
measured.

---

## E8: Social: girls, festivals, happiness (SOCIAL lane)

### PR-066 — Epic: four girls in window, the wife on her path, happiness ≥ 992
Tracker: `rr-ux6x.14`.
`epic · P1 · SOCIAL · compose · Deps: PR-018, PR-016`

### PR-067 — Girl locator
Tracker: `rr-ux6x.14.1`.
`task · P1 · SOCIAL · compose · Deps: PR-021 · W1`
Do: Find and approach each girl from the live schedule model.
Done when: all 5 girls are reached on 3 sample days, 0 misses.

### PR-068 — Talk skill with heart read-back
Tracker: `rr-ux6x.14.2`.
`task · P1 · SOCIAL · compose · Deps: PR-067, PR-018 · W1`
Do: Talk, dismiss the text, and verify the heart delta in RAM, following
the daily-cap rules.
Done when: 10 talks log deltas that match F9.

### PR-069 — Gift skill
Tracker: `rr-ux6x.14.3`.
`task · P1 · SOCIAL · compose · Deps: PR-068 · W1`
Do: Get each girl's liked items and deliver them. Disliked items are used
only for downward tuning.
Done when: +gift and −gift deltas are verified on 2 girls.

### PR-070 — Festival skills
Tracker: `rr-ux6x.14.4`.
`task · P1 · SOCIAL · compose · Deps: PR-021, PR-018, PR-016 · W1`
Do: Take part in the flower, harvest, egg, thanksgiving, Star Night and New
Year festivals with the right choices (for example the dance partner) for
hearts and happiness.
Done when: each festival is passed on a pin with the expected deltas.

### PR-071 — Girl events and birthdays
Tracker: `rr-ux6x.14.5`.
`task · P2 · SOCIAL · compose · Deps: PR-018 · W2`
Do: Detect scripted choices and take the positive option. Avoid the −30
sites unless the wife-descent plan calls for one.
Done when: every event in F9 is handled on pins.

### PR-072 — Hearts window controller (four girls)
Tracker: `rr-ux6x.14.6`.
Audit: `pr072_hearts_whatif.json` has zero emulator frames; measure ten
real nights and the four non-wife windows on the ROM.
`task · P0 · SOCIAL · compose · Deps: PR-025, PR-068, PR-069 · W2`
Do: Bring each non-wife girl to 496–511 and hold her there through D30
after drift.
Done when: a what-if from 470 lands in the window and stays there for 10
days.

### PR-073 — Happiness accumulator
Tracker: `rr-ux6x.14.7`.
Audit: projection must use compatible dated F5/F7 sources, excluding any
child-age bonus after tally and any births a full barn cannot receive.
`task · P1 · SOCIAL · compose · Deps: PR-016, PR-025 · W2`
Do: Schedule the repeatable positive sources to reach ≥ 992, and guard the
sinks.
Done when: the ledger projects ≥ 992 by D29 on a Y2 checkpoint.

### PR-074 — Social trip packing
Tracker: `rr-ux6x.14.8`.
`task · P2 · CAMPAIGN · compose · Deps: PR-024, PR-067 · W2`
Do: Combine each town trip (shop, girls, gifts) inside the frame budget.
Done when: a full-scale day does all its social work in one trip.

---

## E9: Marriage and family (SOCIAL lane)

### PR-075 — Epic: married, two children, the wife in window
Tracker: `rr-ux6x.15`.
`epic · P1 · SOCIAL · compose,spine · Deps: PR-013, PR-010`

### PR-076 — Wife selection
Tracker: `rr-ux6x.15.1`.
`decision · P0 · SOCIAL · meta · Deps: PR-010, PR-013, PR-018 · W1`
Do: Pick the wife by her F1 recipe, schedule proximity, gift supply and
event risk.
Done when: `research/wife_choice.md` exists and the owner acknowledges it.

### PR-077 — Marriage skill
Tracker: `rr-ux6x.15.2`.
`task · P1 · SOCIAL · compose,spine · Deps: PR-064, PR-076, PR-068 · W3`
Do: Meet the requirements, then run the proposal and wedding flow, Clean.
Done when: a wife bit is set on a natural pin at or before the F4 deadline.

### PR-078 — Pregnancy gates
Tracker: `rr-ux6x.15.3`.
`task · P1 · SOCIAL · compose,spine · Deps: PR-077, PR-013 · W3`
Do: Hold the wife's hearts at ≥ 450, then ≥ 650, at the checks, and track
the `wife_pregnancy` counter against the milestones.
Done when: both pregnancies start on schedule on natural pins.

### PR-079 — Name-entry skill and births
Tracker: `rr-ux6x.15.4`.
`task · P0 · SOCIAL · compose,spine · Deps: rr-bu62.18, rr-bu62.26 · W1`
Do: Handle the tilemap `0x5F` screen for a baby, a calf or a horse, then
return control.
Done when: child flags `0x08`/`0x04` are set through births on pins, and
the `0x5F` warp is fixed.

### PR-080 — The wife's descent into the window
Tracker: `rr-ux6x.15.5`.
`task · P0 · SOCIAL · compose,spine · Deps: PR-010, PR-078, PR-025 · W4`
Do: After the child-2 check, apply the F1 recipe down to 496–511 and hold
it through drift.
Done when: a natural Y3 pin has the wife at 496–511 on D30.

---

## E10: Collectibles (HOME lane)

### PR-081 — Epic: 10 berries, clock, turtle shell
Tracker: `rr-ux6x.16`.
`epic · P1 · HOME · compose · Deps: PR-015`

### PR-082 — Field and object berries
Tracker: `rr-ux6x.16.1`.
`task · P1 · HOME · compose · Deps: PR-015 · W1`
Do: Cover the field digs (1/64 per hoe on bare ground, 2 for life), the
chicken statue, the mountain trunk, and the other non-festival sources F6
names. Verify each by flag.
Done when: every non-festival berry is obtained on pins, with its flag and
`power_berry_N` receipt.

### PR-083 — Festival-bound berries
Tracker: `rr-ux6x.16.2`.
`task · P1 · HOME · compose · Deps: PR-015, PR-070 · W2`
Do: Win or perform whatever F6 names for each festival berry, with a retry
plan for later years.
Done when: each one is obtained on a pin for its festival.

### PR-084 — Clock
Tracker: `rr-ux6x.16.3`.
Audit: P0. Prove native request → summer quote arm → first-upgrade gift;
a preset request/arm bit does not establish the request route.
`task · P0 · HOME · compose · Deps: PR-015 · repair`
Do: Get the clock from the source F6 found.
Done when: `$7F1F6E & 0x4000` is set on a pin, Clean.

### PR-085 — Turtle shell
Tracker: `rr-ux6x.16.4`.
`task · P1 · HOME · compose · Deps: PR-015 · W1`
Do: Get the turtle shell from the source F6 found.
Done when: `$7F1F6C & 0x1000` is set on a pin, Clean.

### PR-086 — Collectible tracker in the ledger
Tracker: `rr-ux6x.16.5`.
`task · P2 · ORACLE · compose · Deps: PR-004, PR-015 · W1`
Do: Track the flags owned and the chances left per item, with dates.
Done when: the report lists the remaining berry chances on a checkpoint.

### Audit addition — native exactly-ten berry proof
Tracker: `rr-ux6x.16.7`.
`task · P0 · HOME/ROM · research · Deps: PR-015`
Do: Measure native 7→10 through an off-map respawn or transition recipe,
with no live RAM writes, and add a refusal guard before an eleventh eat.
Done when: a ROM report records exact count 10, stamina 200, source flags,
frames/intervention counters and both values surviving final sleep. Research
presets stay dev-only; natural predecessor integration remains a final gate.

### PR-087 — Dog hugs ≥ 100 (best-ending stretch)
Tracker: `rr-ux6x.16.6`.
`task · P3 · HOME · compose · Deps: rr-bu62.21, rr-bu62.23 · W3`
Do: A safe pick-up skill that avoids the `0x0800` trap, with a count
guard.
Done when: 100 pick-ups on a pin with 0 stuck mornings.

---

## E11: Integration and finish (CAMPAIGN lane, serial)

### PR-088 — Epic: natural checkpoints to a 999 continuous run
Tracker: `rr-ux6x.17`.
`epic · P1 · CAMPAIGN · compose,spine · Deps: PR-020`

### PR-089 — Y1 natural checkpoint
Tracker: `rr-ux6x.17.1`.
`task · P1 · CAMPAIGN · compose,spine · Deps: PR-023, PR-026, PR-030, PR-037, PR-043, PR-063 · W2`
Do: Run from power-on to Y2 Spring D1 and meet every Y1 milestone, with
season pins saved.
Done when: the ledger versus the milestones is green, with 0 loads and 0
writes.

### PR-090 — Y2 natural checkpoint
Tracker: `rr-ux6x.17.2`.
`task · P1 · CAMPAIGN · compose,spine · Deps: PR-089, PR-077, PR-078, PR-046, PR-038 · W3`
Do: From the natural Y2 pin to Y3 Spring D1: married, child 1, herd and
flock on track.
Done when: the milestones are green, with 0 loads inside the run.

### PR-091 — Y3 Spring checkpoint
Tracker: `rr-ux6x.17.3`.
`task · P1 · CAMPAIGN · compose,spine · Deps: PR-090, PR-032, PR-079 · W4`
Do: Close the potato and turnip windows, get child 2, and keep the cows on
track, through to Y3 Summer D1.
Done when: the milestones are green.

### PR-092 — D29 audit procedure
Tracker: `rr-ux6x.17.4`.
`task · P0 · CAMPAIGN · compose,spine · Deps: PR-009, PR-025, PR-027 · W3`
Do: On the evening of D29, compute the projected D30 score after drift, list
the last-day actions, and report which terms fall short if the projection is
below 999.
Done when: the audit's projection matches the ROM on a near-perfect
what-if.

### PR-093 — Y3 Summer endgame from the natural pin
Tracker: `rr-ux6x.17.5`.
`task · P0 · CAMPAIGN · compose,spine · Deps: PR-091, PR-059, PR-072, PR-080, PR-092 · W4`
Do: Run from Y3 Summer D1 to the ending with every window held.
Done when: the oracle shows 999 and RAM `ranch_mastery` is 999 at `0x25`.

### PR-094 — Continuous power-on run to a 999 ending
Tracker: `rr-ux6x.17.6`.
Audit: also blocked by PR-098, PR-005 and `rr-ux6x.16.7`. Reconcile the
zero-load acceptance below with ADR 0007 before accepting the final run.
`task · P0 · CAMPAIGN · compose,spine · Deps: PR-093, PR-100 · W4`
Do: Run `--power-on --until-ending` with no pins.
Done when: 999 at `0x25`, with 0 initial or mid-run loads, 0 RAM writes and
no assists.

### PR-095 — Reproduction and STATUS promotion
Tracker: `rr-ux6x.17.7`.
`task · P0 · CAMPAIGN · status,spine · Deps: PR-094 · W4`
Do: Re-run the same commit, expecting the same frame count and score. Then
promote STATUS and the manifest, and update CONTEXT and plan.
Done when: STATUS records both runs and the docs tests pass.

### PR-096 — Published tape
Tracker: `rr-ux6x.17.8`.
Audit: reconcile the zero-load publication wording with PR-005; preserve
reproducible lineage and video requirements under the final owner rule.
`task · P2 · CAMPAIGN · meta · Deps: PR-095, PR-005 · W4`
Do: Replay the run from power-on with 0 loads and export the video in parts
(ADR 0005).
Done when: the parts render and the final part shows the 999 rating.

---

## E12: Gut and speed (GUT lane)

### PR-097 — Epic: structure and speed for 10M+ frame runs
Tracker: `rr-ux6x.18`.
`epic · P1 · GUT · stabilize · Deps: none`

### PR-098 — Structure plan and merges before new domains
Tracker: `rr-ux6x.18.1`.
Audit: final proof is 3,476,992 frames, +59.84%, with 139 bad days.
Keep this open. Check matching policies in a short trace before another
soak, and bring campaign_composer.py (1,292) and multi_day_planner.py
(1,029) back under the soft max through composer merges or deletion.
`task · P0 · GUT · stabilize · Deps: none · W0`
Do: Bring these under the soft max: `run_to_day2.py` (1,460),
`day_plan_orchestrator.py` (1,213), `skills.py` (1,029) and `spring_opt.py`
(1,028). Fix the composer seams for social, family and cover. No sibling
extracts.
Done when: the power-on farm ending frame count is unchanged at 2,175,350
and the harvest suite is green.

### PR-099 — Full-scale day speed
Tracker: `rr-ux6x.18.2`.
`task · P1 · GUT · stabilize · Deps: rr-bu62.12, PR-007 · W1`
Do: Profile a 12-cow, 12-chicken, 5-girl what-if day. Add the BFS, catalog
and scene caches. The output must stay identical.
Done when: full-scale days run at least 2× faster with the same frames and
RAM.

### PR-100 — Long-run robustness
Tracker: `rr-ux6x.18.3`.
`task · P1 · GUT · stabilize · Deps: PR-026, PR-006 · W2`
Do: Add a detached launcher, crash-resume from the latest natural
checkpoint with a lineage check, log rotation, and a memory guard.
Done when: a 10M+ frame run survives, or resumes with its lineage recorded.
