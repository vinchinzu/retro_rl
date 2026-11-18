# Perfect Rancher (999): master plan

Planning and research, audited 2026-09-30. The original 100 beads are filed
under `rr-ux6x`; real IDs and the amended queue are in
[BEAD_PLAN.md](BEAD_PLAN.md). The execution plan is [../plan.md](../plan.md).
Verified product results stay in [../STATUS.md](../STATUS.md).

At audit entry 36 of 100 beads were closed, F10 existed, and Wave 1 lane
work was uncommitted. Eleven closures were reopened for evidence or plan
gaps. New exactly-ten research is `rr-ux6x.16.7`; the audit is
`rr-ux6x.19`. Read the [F10 audit](research/F10_verdict.md) before scheduling
a campaign: 984 is a conditional ceiling, not a proved reachable score.

## Repair update, 2026-09-30

The [current plan](../plan.md) and [F10](research/F10_verdict.md) supersede
historical feasibility claims below. Slot 0 is feedable: merged native
12-cow feeding/night proof preserves all twelve values at 200 in
7,507 frames. Replacement breeding is optional. Native milestone chicken
purchase matches the legacy skill at 4,028 frames. Dispatch is opt-in via
`--perfect-rancher`, uses RAM receipts, and defers unsupported capabilities.
The conditional YAML uses earlier marriage (index 128) and explicitly
unproved happiness repair alternatives. No campaign budget or 999 route
is accepted. Exact-ten, clock, full-scale native care/arbiter and contract
reconciliation remain open. Short structure A/B is identical at 17,538
frames; full-ending timing acceptance remains open.

ROM-verified score contract, 2026-11-02: every term, window edge and the final night's drift (wife -2,
shipping credit, 2,950-tile development edge) is poked and judged by the ROM's calculator in
`tests/test_perfect_ending.py` (details in the [handoff](../tasks/rr-20w-run16-defects.md)). It pins what the
ROM rewards, not that the bot can reach it.

Evidence tags used below:

- **[ROM]**: read from the decomp (`HM-Decomp/src/code_banks/*`).
- **[SCAN]**: raw byte scan of the ROM. These are candidates, not decoded
  triggers.
- **[MEAS]**: measured on the emulator in this repo.
- **[EXT]**: community figure, not yet checked against the ROM.
- **[TBD]**: open research, owned by a bead.

## 0. Summary

- The ending's `RanchMasteryCalculator` (bank_83 `$83F26D`) caps at 999.
  **With every term at its maximum, the terms add up to exactly 999.** There
  is no slack, so a perfect score means every term is maxed on the morning of
  Year 3 Summer D30.
- Three terms use a 9-bit mask (`AND #$01FF`): girl hearts and shipped
  crops. Each girl's hearts must end at **496–511**. Each crop's shipped
  count must end at **496–511 mod 512**. These are window targets, not "as
  high as possible".
- The wife needs hearts **≥ 650** for the second pregnancy [ROM], and her
  hearts must then come back down to 496–511. The ROM scan found negative
  heart deltas for every girl (−2, −4, −6, −8, −30) [SCAN], so the path looks
  possible. F1 measured Maria 660→510 in 25 poison-mushroom offers, with
  her position forced into range. PR-080 still needs the native wife route.
- Farm development needs **2,950 counted tiles**. The farm map has only
  **2,968** cover candidates on the D2 clear pin [MEAS], leaving 18 tiles
  of slack there. The earlier raw map count was 2,974; use the actual pin
  mask when scheduling final coverage. Tilled soil (`0x07/0x08`) does **not** count. Only
  fences (`0x05/0x06`), seeds, crops and grass (`≥ 0x1D`) count [ROM].
- Current baseline: the berry-only power-on ending scores **9**. The oracle
  (`harvest/core/ranch_master.py`) and the ROM agree on 9 for
  `LN_final_poweron_ending` [MEAS]. The working-farm ending (`924e3f11`) was
  not scored.
- Order of work: (1) feasibility gates and a validated score ledger,
  (2) skills per domain in parallel lanes, (3) a milestone calendar that
  schedules backward from D30, (4) year-by-year natural checkpoints,
  (5) two final 999 runs under the reconciled PR-005 load contract.
- Seven designed berries leave stamina at 35; exactly ten requires a
  native edge. Slot 0 feeding is repaired; F5/F7 still need one funded
  happiness strategy. Clock acquisition and the dated happiness floor are
  also open. These are simultaneous feasibility blockers.

## 1. Definition of done

**Perfect** targets `run_to_day2 --power-on --until-ending`, with the final
load rule reconciled by PR-005 before acceptance:

- 0 RAM writes and no assists (ADR 0006). Savestate loads are allowed
  (ADR 0007, owner ruling 2026-09-30). The duplication edge is allowed and
  must land on exactly 10 berries. The target stays 999.
- RAM `ranch_mastery` (`$7F1F54`) equal to **999** after ending index `0x20`,
  captured at the terminal index `0x25`.
- The on-screen ranch-rating text matching that value.
- The run JSON `ending` block carries the per-term breakdown from the
  validated oracle, and the breakdown sums to 999.
- A second run of the same commit gives the same frame count and score (the
  emulator is deterministic).

Contract: [ADR 0007](../adr/0007-perfect-rancher.md) records Perfect after
Finish and an owner amendment allowing duplication and savestate loads,
with target 999 and fallback 984. The current handoff says those decisions
are parked; CONTEXT and PR-094/096 still demand zero loads. PR-005 is
reopened to reconcile the documents. This audit preserves the ADR and does
not infer a new ruling or treat any recorded policy as runtime proof.

## 2. Score anatomy (bank_83 `RanchMasteryCalculator`)

| # | Term | ROM source | Formula | Max | What maxes it |
|---|------|-----------|---------|-----|----------------|
| 1 | Money | `moneyL $7F1F04` (low 16 bits of a 24-bit value in 10G units) | `min(78, moneyL >> 7)` | 78 | **99,840G ≤ money ≤ 655,350G.** The low word wraps above that. |
| 2 | Cows | `cow_N $7F1F0A` | `3 × cow_N` | 36 | 12 cows (the ROM loops over 12 slots) |
| 3 | Chickens | `chicks_N $7F1F0B` | `3 × chicks_N` | 36 | 12 chickens (decoded cap 12; no above-cap slack) |
| 4 | Stamina | `max_stamina $0917` (u8) | `(max − 100) >> 1` | 50 | 10 eaten berries; 7 designed plus 3 native edge eats, each +10 [ROM `$81CAAB`] |
| 5 | Girls ×5 | `hearts_* $7F1F1F..27` | `Σ (h & 0x1FF) >> 4` | 155 | **each girl at 496–511** (hearts clamp to 0..999 through event op `0x41` [ROM `$84B5A9`]) |
| 6 | Crops ×4 | `shipped_* $7F1F4A..50` | `Σ (n & 0x1FF) >> 4` | 124 | **each count at 496–511 mod 512**. The counter is a 16-bit INC with no cap [ROM `$818140..`] |
| 7 | Happiness | `happiness $7F1F33` | `h >> 5` | 31 | happiness ≥ 992 (clamped at 999 by `AddPlayerHappiness`) |
| 8 | House | `$7F1F64 & 0x40 / 0x80` | +16 / +16 | 32 | both house upgrades |
| 9 | Children | `$7F1F6E & 0x08 / 0x04` | +16 / +16 | 32 | two children |
| 10 | Married | `$7F1F66 & 0x1F` | +32 | 32 | married to any girl |
| 11 | Clock | `$7F1F6E & 0x4000` | +22 | 22 | clock owned (request route open, PR-084) |
| 12 | Turtle shell | `$7F1F6C & 0x1000` | +21 | 21 | turtle shell (Fall D20 prize, runtime proof open) |
| 13 | Development | `ranch_development` = `min(100, tiles×10/295)` | `>> 1` | 50 | **≥ 2,950 counted farm tiles** on D30 morning |
| 14 | Cow happiness | cow slot byte `+4`, 12 slots | `Σ min(25, b >> 3)` | 300 | **every one of 12 cows at ≥ 200** |
| | **Total** | | cap 999 | **999** | |

78+36+36+50+155+124+31+32+32+32+22+21+50+300 = **999**.

Consequences:

- **Zero slack.** One cow at 199 happiness, one girl at 495 or 512, one crop
  at 495, or 25 bare tiles costs the perfect score. Every term has a window
  controller and a read-back in the daily ledger.
- **Slack is not established.** F3 caps chickens at 12. F6 has seven
  designed berries. Native extra eats can raise stamina but counts above
  ten fail the best-ending equality check; the recorded ADR targets exactly
  ten. `rr-ux6x.16.7` owns a live acquisition and stop-guard proof.
- **Calculation time.** D30 starts `CalculateFarmDevelopment`; index `0x20`
  runs the mastery sum. D29 shipping credit, night tiles, weather, animals
  and events all matter. F0 measured the wife losing 2 on hour-18 entry
  (`$B5805A`). Wife before that entry 498–511 is safe; 510→508 survives,
  496→494 loses a point. Other girls and crops stayed unchanged on the
  sampled final night.

Best-ending scene (not required for 999, stretch goal) [ROM `$83F075`]:
child 1, happiness ≥ 900, cows ≥ 1, chickens ≥ 1, `power_berry_N == 10`
exactly, money ≥ 10,000G and dog hugs ≥ 100. Every one of these except the
dog hugs is already implied by 999, unless the berry-glitch lever is used.

## 3. Feasibility gates (before heavy building)

Each gate ends with a short
verdict file under `docs/perfect_rancher/research/`, with decomp citations
and probe JSON. Probes may use what-if pins, since ADR 0001 allows scratch
research. Pin results never go into STATUS.

| Gate | Question | Why it can kill 999 | Known now |
|------|----------|---------------------|-----------|
| F0 | What changes between D29 evening and index `0x20` (night pass, the D29 6:00 text, the D30 ending event)? | Windows are measured one night early | The hour-18 house re-entry subtracts 2 from the wife. It does not add 2. Wife evening window is 498–511. Other girls and the crops are unchanged. [F0] |
| F1 | Can the wife reach ≥ 650 for child 2 and then drop to 496–511? Which scripts apply the negative deltas, can they repeat, and is the wife excluded from them? | Wife term max is 30 if she cannot go down (999 & 0x1FF = 487) | Maria poison mushroom −6, measured 660→510. Repeatable outdoor sinks exist for all five wives. [F1] |
| F2 | Can 2,950 of the 2,974 non-structure tiles be covered while the farmer can still walk to the house, bin, barn, coop and shed? What does grass, fence or crop cover cost per tile, can it be walked, and what do storms do to it? | Development tops out at 49 | Clear pin: 2,968 tiles, all walkable grass, term 50, slack 18. A hurricane on the sleep into Summer D30 can still wipe it. [F2] |
| F3 | 12 cows, every one ≥ 200 happiness by D30: start value, gain per action per day, decay, sickness, buy versus breed, calf growth, barn capacity | Last herd purchase date, and daily chore frames | Talk +1, brush +3, one slot-1 fed night +0. Fifty-day threshold is modeled. Native feeding/night survival now passes all twelve slots; daily care remains unproved. PR-048 is an optional breeding strategy. Chickens cap at 12. [F3] |
| F4 | Marriage requirements, the pregnancy counter (`wife_pregnancy ≥ 20`), gestation, child-1-to-child-2 spacing, naming screens → latest marriage date | Two children must exist by D29 | Latest wife bit for both children already set on D29: Y2 Fall D9. Child 2 is +110 nights. Live counter stepped; 448 hearts did not set the flag. [F4] |
| F5 | Economy: total spend (cows, upgrades, cover, seeds, tools, feed) plus ≥ 99,840G at the end, the 2,000G Y2 visitor, milk and egg income curves | Money term | Paper ledger ends at 620,000G but buys all 12 slots and no births/potions. Its happiness repair budget is unproved; PR-014 remains open. [F5] |
| F6 | All 10 power berries: triggers, dates and RNG. Festival berries give 2–3 chances; field berries are 1/64 per hoe on bare ground and capped at 2 | Stamina term | Designed berries are 7, not 10. Stamina term is 35. The Fall D20 prize is the turtle or the indoor-berry arm, not a third berry. [F6] |
| F7 | Happiness ≥ 992: sources and sinks | Happiness term | Weak floor incorrectly includes child 2 age-60 +100 after the late-wedding ending. Seven-berry late case is 900, ten-berry case 930 before other sources. PR-016 reopened. [F7] |
| F8 | Shipped-count semantics: bin drop versus overnight sale, the `$7F1F5E` bit 0 gate, saddlebag and peddler | Crop windows | Each bin drop INCs that crop by 1 when bit 0 is set. Milk and the peddler do not. [F8] |
| F9 | Heart sources and sinks per girl (talk, gifts, festivals, birthdays, choices, wife nightly drift) | Girl windows | Live Maria flower +5, Ann flower +6, both intros +0. Those match the decoded ladders. [F9] |
| F10 | Verdict: is 999 reachable naturally? If not, what is the natural maximum, and which term is short? | Sets the contract | Seven-berry conditional upper bound 984; reachable maximum unproved. Cow/clock/happiness/dispatch gaps remain. [F10] |

[F10](research/F10_verdict.md) blocks a heavy campaign go-ahead. It combines
the gates but qualifies the score ceiling: stamina is short by 15 on the
seven designed berries, and the other maxima have not been shown
simultaneously. Wave 1 has lane implementations and model tests; several
runtime proofs were reopened. Proceed with bounded feasibility probes,
then compatible milestones and native dispatch, before natural checkpoints.

## 4. Calendar strategy (a strawman until E2's solver replaces it)

There are 299 controllable days through Y3 Summer D29. Dates below are
constraints to preserve, not an accepted route. Current `milestones.yaml`
and PR-022 remain open: the conditional dated plan still needs evidence and
its three duplication recipes have no live proof. Rebuild after F5/F7 and
PR-048. Ten berries permit a 999 projection only if every other term also
passes the post-night audit; otherwise report the actual term gaps.

| Phase | Window | Commitments |
|-------|--------|-------------|
| A | Y1 Spring | Existing D2 farm clear and crop loop. Start the potato and turnip counters. Grapes and forage. Stockpile lumber from farm stumps. Field berry 1 can roll from Spring D1 (1/64 per hoe). There is no spring festival berry. Talk to girls only when the trip is already paid for. |
| B | Y1 Summer | Tomato and corn counters. Pasture ≥ 16 × planned animals. Buy cow 1 and chicken 1, then incubate. No summer festival berry. The clock-arm quote is no later than Y1 Summer D30 (year 0, season below 2). F5 pays the first house upgrade on Summer D1; the gift is four mornings later. Sprinkler once unlocked (Summer D20+) [ROM]. |
| C | Y1 Fall | Mushrooms. Y1 Fall D20 before 15:00 is the first festival prize: the turtle, or the indoor-berry arm, at 1/2. Grow the chicken count toward 12. It does not go past 12. Add cows on feedable stalls. Harvest-festival hearts and happiness. |
| D | Y1 Winter | House upgrade 2 can complete here. The clock is only the first-upgrade gift, and only if the summer quote set the arm. Raise the chosen wife's hearts. Pass Star Night and New Year (already handled). |
| E | Y2 Spring (early marriage window) | Choose marriage against both children and the dated happiness sources. Y2 Fall D9 is only the latest child-flags deadline. Big spring crop season. Schedule buy/birth/slot turnover rather than twelve purchased cows. |
| F | Y2 Summer–Fall (**wife bit latest Y2 Fall D9**) | Wife bit on Y2 Fall D9 (index 188), before that night's sleep, so both child flags are already set on Y3 Summer D29. Child 1 is the morning 20 nights later (Y2 Fall D29 on that date), hearts ≥ 450, house bit `$0080` already set. Y2 Fall D20 before 15:00 is the second prize and covers whichever of the turtle and the indoor arm the first win missed. 12 chickens. Milk becomes the money engine. |
| G | Y2 Winter – Y3 Spring | Child 2 is the morning 110 nights after the wife bit (Y3 Summer D29 when that bit is Y2 Fall D9), hearts ≥ 650 on that night. **Latest buy for a feedable cow that reads ≥ 200 at D30 06:00 is Y3 Spring D10** (50 mornings after the buy). Close the potato and turnip windows by the end of Spring, since those crops can only be planted in Spring. Remove every rock and stump from the farm and prepare the grass cover. |
| H | Y3 Summer D1–D29 | Eat berries so `power_berry_N` is exactly 10 by D29. Seven are the designed set; the allowed duplication edge supplies the other three and must not pass 10. Tomato and corn windows. Development ≥ 2,950 after the sleep into D30. A hurricane on that sleep can destroy grass, and there is no D30 work morning (18 tiles of slack on the 2,968 clear mask). Other girls at 496–511. The wife's hour-18 entry starts at 498–511 so the −2 lands in 496–509. Happiness ≥ 992. Happiness sources and namings follow the corrected dated F7/F5 plan. Money in window; old F5 hold 620,000G is a model. **D29 audit reports each post-night gap; berry count alone cannot establish 999 or 984.** |

Binding deadlines from those notes, not from a new run:

- Berries: the seven one-shots, each while its flag is clear, no later than Y3 Summer D29. Field berry 2 is after field berry 1. The indoor berry is a house entry after the Fall D20 arm. The Y3 Summer D30 house entry is the ending and does not spawn it.
- Clock arm: Y1 Summer D30. Clock gift: four mornings after the quote, no later than Y3 Summer D29, and only while house bit `$0080` is still clear. F6 found no reached writer for request bit `$0020`.
- Turtle and the indoor arm: Y1 Fall D20 and Y2 Fall D20, hour below 15. One win is only one of the two.
- Marriage and children: wife bit Y2 Fall D9 for both flags already true on D29. House bit `$0080` and the 450 / 650 heart floors have to hold on the night each flag is tested. F5 has the level-2 bit on Y1 Summer D16.
- Herd: Y3 Spring D10 is the latest feedable-stall buy that reads ≥ 200 on D30 morning. Slot 0 is not that stall.
- Hearts and crops at the D29 hour-18 entry: wife 498–511. The other girls and all four crops stay at 496–511. This night does not write them.
- Happiness: a dated source/sink total ≥992. Nine namings only supply 630;
  do not include an age-60 bonus that occurs after the ending.
- Development: ≥ 2,950 counted tiles after the sleep into Summer D30.
- Money: 99,840G–655,350G on D30 morning. Rebuild F5's budget around the
  the selected breeding or food repair strategy, feed, tools and cover.

## 5. Economy model to repair (PR-014)

[F5](research/F5_economy.md) prices grass 500G/bag, cows 5,000G, hen
1,000G, miracle potion 4,000G, brush 800G, milker 1,800G, sprinkler 2,000G,
houses 5,000G/10,000G and chicken feed 300G/12. Milk S/M/L is
150/250/350G; eggs 50G. These are decoded prices with selected prior
receipts, not a live full-game budget.

Its paper balance 620,000G assumes twelve purchases and zero births. The
conditional repair alternatives require either funded food or barn space,
maternal care and potion spend. Native feeding removes the slot-0
replacement requirement. Late marriage cannot supply child 2's age-60
happiness bonus by D30. Rebuild one dated model covering
these constraints, 750 lumber, losses, milk/egg curves and verified food or
social repairs; then implement the money-window hold (PR-034). Keep final
money 99,840G–655,350G after overnight credit.

## 6. Daily operating model at full scale

- The clock stops at 18:00 [ROM, ADR 0003]. Outdoor work after 18:00 costs
  frames, not game time. Stamina is the per-evening limit, and hot spring
  refills restore it. Development cover and hoeing go in this after-18:00
  lane.
- Shops, the carpenter and the girls keep schedules, and most of them are
  only reachable before 18:00 [TBD]. The daily arbiter packs one town trip
  per day: shopping, girls, and the gift pickup.
- Fixed chore blocks at scale: 12 cows (feed, talk, brush, milk, then ship
  the milk), 12 chickens (feed, eggs, incubator), crops (water or sprinkler,
  harvest, ship to the window), fodder cutting.
- Frame budget: the working-farm run takes about 7.3k frames a day
  (2.17M frames). A full-scale day may take 30–60k frames, so a full run is
  roughly 10–18M frames, or 3–5 hours of wall time. That makes the
  save-point loop (§8) and speed work (E12) necessary, not optional.

## 7. Architecture

Stay inside the existing composer ([../PLANNING_STACK.md](../PLANNING_STACK.md)):
`MultiDayPlannerTask` → `DayPlanTask` → phase registry → skills. The code
rules still apply: soft max about 1,000 LOC, merge into the composer, no
sibling extracts. New modules only for new capabilities (social, family,
development cover), each with an A/B loop.

1. **Score ledger** (validate and extend `harvest/core/ranch_master.py`): the
   per-term value, max, gap, window state and deadline, read from RAM every
   morning and written into `day_journal`. It also projects D30 using the F0
   drift model. It is the single source every window controller reads.
2. **Milestone plan** (E2): a data file (for example `perfect_rancher/milestones.yaml`)
   of dated commitments and latest-start deadlines. PR-022 must revise it
   using a compatible F5/F7 herd plan and proven edges. PR-023 must dispatch
   intents into tasks; the current integration only logs them.
3. **Window controllers**: generic "reach [lo, hi] and hold" logic for
   hearts, shipped counts, money and happiness. They read RAM only (Clean
   allows reads, never writes), use the measured per-action deltas, and
   refuse any action that projects outside the window.
4. **Daily arbiter**: runtime order is deadline-critical, animals, crops,
   social, cover. PR-024 needs a measured full-scale ROM day: its existing
   `run_whatif_day` is a paper simulation, not runtime evidence.
5. **Skills per domain**: chickens and cows at scale, carpenter, lumber and
   fences, grass cover, girls (talk, gifts, festivals), marriage and
   children, name entry (`0x5F`), and the collectibles (berries, clock,
   turtle shell).

## 8. Evidence ladder and loop

1. **Research** (E1 and the domain spikes): what-if pins and the decomp. The
   output is verdict files and numbers. Never STATUS.
2. **Skill A/B**: from a pin, compare two skills and diff RAM and frames. A
   skill is route-ready only after it clears from the real predecessor state
   (root rule: natural entry).
3. **Natural checkpoints**: the power-on continuous run saves season-boundary
   pins `PR_Y{n}_{Season}_D1` (lane-prefixed in worktrees). Iterate from
   the latest natural checkpoint, then re-run continuously from power-on
   every about 10 fixes or at each year boundary. This is the save-point
   loop: continuous run → fix from checkpoint → resume forward → continuous
   run.
4. **Integration soak after every lane merge**, on merged main, before any
   new work (lesson from rr-bu62).
5. **Final**: two reproducible power-on runs scoring 999 under the PR-005
   contract. Resolve the zero-load wording in PR-094/096 before accepting
   a run. Then STATUS, manifest and publication can be promoted.

## 9. Sub-agent orchestration

**Lanes.** Each lane is one worktree on branch `harvest/pr999-<lane>`,
following the parallel-lane recipe: symlink `roms`, `.venv`, the pins and
the tapes; mint lane-prefixed pin names; run ROM jobs detached with
`setsid nohup … --trials 1`.

| Lane | Epics | Needs the emulator | Notes |
|------|-------|--------------------|-------|
| ORACLE | E0, ledger parts of E2 | light | Owns `ranch_master.py`, the ledger and the report schema |
| ROM | E1 | pins only | Research only, no product code. Writes the verdict files |
| FARM | E3, E6 | yes | Crops, shipping windows, cover, clearing |
| ANIMALS | E4, E5 | yes | Chickens, cows, fodder |
| SOCIAL | E8, E9 | yes | Girls, festivals, happiness, marriage, children, name entry |
| HOME | E7, E10 | yes | Carpenter, lumber, tools, berries, clock, turtle shell, dog |
| CAMPAIGN | E2, E11 | heavy | Serial integrator. The only lane that merges to main and runs power-on soaks |
| GUT | E12 | light | Speed and file structure. Output must match frame for frame |

**Rules for every sub-agent:**

- Read the harvest `AGENTS.md`, this file, and the bead's section in
  BEAD_PLAN.md.
- **No `bd` writes and no `.beads/` edits in lanes.** The orchestrator owns
  beads.
- Never `git add -A`. Never write a pin under an existing name, because it
  writes through the symlink into main.
- Clean only: no `LiveRamEditor` or `--ram-set` on any route path. RAM
  edits are allowed only on research pins under `LN_<lane>_whatif_*`, and
  are marked as research.
- Every bead ends with a report (JSON path, frames, Clean audit block) and a
  one-paragraph handoff in the living residual or the bead notes (the
  orchestrator pastes it in).
- Stop and report instead of guessing when a ROM fact contradicts this plan.
  Update the verdict file and the plan section.

**Waves** (details in BEAD_PLAN.md):

- **Audit repair**, now: PR-084 clock, exactly-ten
  `rr-ux6x.16.7`, F7/F5 consistency, PR-022/023 and PR-098. Claim one
  bounded task; this plan does not itself start more agents.
- **W1**, partially implemented: obtain missing ROM skill proofs, then
  integrate lane code and evidence into main with bounded natural-entry
  checks. A model-test closure does not open a natural campaign gate.
- **W2**: Y1 natural checkpoint run with the Y1 milestones, and domain fixes
  from natural pins.
- **W3**: Y2 natural checkpoint (marriage, child 1, herd and flock at 12).
- **W4**: Y3 endgame windows and the D29 audit. Then the continuous 999 run,
  twice. Then STATUS.

## 10. Risks and fallbacks

| Risk | Mitigation |
|------|------------|
| Native wife approach differs from F1 probe | Reproduce −6 offers after walking to the real wife; pre-entry target 498–511 accounts for final −2 |
| Development slack of 18 tiles on the clear pin is lost to a storm | Walkable cover (grass) on paths. Final fill on D25–D29. Storm re-plant skill. Per-day tile census |
| Native herd care timeline unproved | All twelve troughs and one fed night pass; prove acquisition, daily care, ending-night survival and the selected happiness repair strategy before accepting F5 |
| Frame cost of a full-scale day | E12 speed work, the post-18:00 lane, checkpoint iteration |
| RNG-bound berries (field digs, festival wins) | Heavy hoeing gives many rolls. Festival skills for every year. The ledger tracks remaining chances |
| Name-entry screens (calf, horse, baby) | Name-entry skill (rr-bu62.18 generalised) before any breeding or birth |
| Dog `0x0800` A-press trap | rr-bu62.21 and .23 first. The dog-hug stretch goal waits on them |
| Y2 2,000G visitor blocks the road | rr-bu62.14 |
| Overshooting a 16-wide window | Window controllers read the RAM after every action and project the rest of the calendar's automatic changes |

## 11. Existing beads this plan depends on

rr-bu62.25 (spring yield), rr-bu62.30 (cow talk timeout), rr-bu62.18 and
rr-bu62.26 (the `0x5F` name-entry warp), rr-bu62.21 and rr-bu62.23 (dog),
rr-bu62.14 (money visitor), rr-bu62.8 (weeds on stands), rr-bu62.12 (speed),
rr-5in (Gate B). The new beads depend on these IDs. They do not duplicate
them.
