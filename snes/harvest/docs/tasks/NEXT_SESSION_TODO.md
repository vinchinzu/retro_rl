# Next session TODO (rewritten 2026-11-17 evening)

Goal: a recorded **999 Clean** power-on run with no long stalls, and 999 on most seeds, not one lucky one. Nothing is committed or pushed. History: `docs/plan.md`
HANDOFF 2026-11-14 and its 2026-11-17 notes.

## 1. What the 2026-11-17 batch showed (`rdet` + `r41..r47`, lean, `score_herd`, no extras)

ROM score 986-998 (rdet 990, r45 998), no 999. Y1 and Y2 Summers were identical in all eight (corn+tomato 195, then 417), so every difference was Year 3 Summer, which has to
deliver corn 249 + tomato 131 = 380 and delivered 232-366. Missing terms: shipped crops in all eight (1-11 points), development in four (1-2), Eve hearts in one (r47, 492).

Causes (measured in the reports and logs, `recordings/e2e/<tag>_20261117/`):

| Term | Cause | Fix (all Year 3 only, env switch) |
| --- | --- | --- |
| Crops | the shop sells one crop a trip and Summer D1 is a Friday: 4-6 corn bags D1, tomatoes D5-D6; 3 of 12 sites still hold capped, ripe Spring potato/turnip | `HARVEST_Y3_JOINT_BUY` buys both crops in one visit (`BuyShelfItemTask.then`, `run_strategy.joint_summer_bags`) |
| Crops | a failed shed pickup of the can loses the night for every plant (~25 crops); 17 of 592 Y3 Summer calls failed, 16 right after the evening `CUT_FODDER` | `HARVEST_WATER_RETRY` retries a wedged walk twice (`day_phase_stamina.splice_water_retry`); only 3 of 42 retries worked, so the two repeating failures are open bugs (section 3) |
| Development | counted tiles sat flat D9-D21: full stands done, every crop site's 5x5 reserved to D20; harvest days leave no clock for the shop ("shop start deadline 15:00") | `HARVEST_EARLY_COVER`: ragged stands the day no full stand is left, a sown ring's band released (`field_cover.released_cells`, the 3x3 block stays), 60-bag Summer loads from D4 |
| Eve hearts | `GIVE_WIFE` "gift walk stuck" nine mornings: an NPC beside the stand blocks the walker's last 2 px sidestep | `eve_gift.py`: vertical first after 40 still frames, 6 px x slack (ROM-verified from `PR_e2er47_20261117_Y3_Summer_D21`: `eve 504->508`) |
| Cows (found by the previews) | after a good gift the blind house-exit walk (down only) is held by a child on the farmer's column; `EXIT_TO_FARM` times out, the barn feed check fails, 12 cows unfed, cow happiness -25 (ROM 949) | `ExitBuildingTask._sidestep_if_blocked`: after 30 still frames step left, then right, 24 frames each (ROM-verified from `PR_ab_D_r45_s1_Y3_Summer_D18`) |

Extra Summer sites (`HARVEST_Y3_SUMMER_EXTRA_SITES` 4 / 8, tested again on top of the big cover load): crops reach 380 in 12 of 16 previews but development falls 1-3 points short in 10 of 15,
so the mean stays ~996-997. They trade crops for development one for one and stay 0. Previews and numbers: `docs/plan.md` "2026-11-17 evening" (pin previews mean ~996, 999 on one in four).

## 2. Certify (in flight; read it first)

**Started 2026-11-17 22:00** (lean, `score_herd`, the tree of that minute; `frozen_tree.tgz` and `env.txt` in each dir): `cdet_20261117` (deterministic) and `c41..c47_20261117`
(`HARVEST_PHASE_JITTER=90 HARVEST_PHASE_JITTER_FROM_YEAR=2 HARVEST_FAULT_SEED=<41..47> HARVEST_FAULT_RATE=0`), expected done ~03:00-04:00. Read them with
`grep -m1 "Ranch score (ROM" recordings/e2e/c*_20261117/FINAL.md` plus each "Where the missing points are" table. Pin previews of the same code scored ~996 (999 on one in four);
if the batch agrees, the next levers are the open bugs in section 3 and the storm repair, not more switches.

The three Year-3 switches default on (`AGENTS.md` "Year-3 score switches"); `env.txt` in each run dir lists what was set. Power-on batch, lean, eight runs (about 6 h wall together):

```bash
for s in det 41 42 43 44 45 46 47; do
  tag=c${s}_$(date +%Y%m%d); if [ $s = det ]; then J=0; else J=90; fi
  HARVEST_STRATEGY=score_herd VIDEO=0 HARVEST_PHASE_JITTER=$J HARVEST_PHASE_JITTER_FROM_YEAR=2 HARVEST_FAULT_SEED=${s/det/0} HARVEST_FAULT_RATE=0 scripts/e2e_run.sh $tag
done
```

- Control: `cdet` must replay Years 1-2 exactly: diff its `run.log` against `recordings/e2e/rdet_20261117/run.log` with the tag normalised; the first differing line must be Y3 Summer D1's
  `BUY_SEEDS` (the joint buy). If Year 1 moved, a switch leaked into Years 1-2 (the lottery, section 4).
- Bar: at least 7 of 8 at 999 Clean (0 loads, 0 writes). Scorecards: `recordings/e2e/<tag>/FINAL.md`, "Where the missing points are".
- A 999: record it with the same env and seed plus `VIDEO=1` (about 5 h, 27 GB). Before that `diff -r` the live tree against its `frozen_tree.tgz`.
- Then check the recording: `grep -c NAV_WEDGE run.log`, the "Idle frames" section of `FINAL.md`, `ffprobe` every part.

Previews (pin A/B, paired, ranks ideas only; pins land within +-4 points of the power-on run and always show "Player happiness -1" because planner state is not restored): copy
`PR_e2e<tag>_Y3_Summer_D1.{state,lineage.json}` to `AB<tag>_Y3Su1` (rewrite `pin` and `path` in the lineage), then
`HARVEST_STRATEGY=score_herd JITTER=0 RATE=0 FAULTS='' scripts/y23_sweep.sh AB<tag>_Y3Su1 <name> 1` (~1 h; 16 at a time is the machine's limit, 32 only adds waiting).

## 3. Open bugs (found, not fixed)

- **Deterministic can pickup failure** (`ENSURE_WATERING_CAN` "tool 0x15 missing after shelf"): carry pair [sickle, milker], sprinkler bit set in shed row 2 (`$7F1F02 & 0x10`), the farmer at (98,166)
  at the can stand, A pressed three times with `input_lock` 1 and nothing happens. The failure reason now prints selected/backpack/rows. Repro: `PR_ab_A_r47_s1_Y3_Summer_D3` with
  `run_to_day2 --state ... --days 1` (the retry fails the same way, so the night is lost). Probe idea: poke the pair on a live env (`get_ram()` is a copy, use the env's RAM write) and try
  [sickle, none] / [milker, sickle] to learn which pair the ROM refuses. 2.9% of Y3 Summer calls; costs ~25 crops each.
- Capped Spring potato/turnip stay ripe on 1-3 sites through Summer (`pocket_plant_targets` never replants a held ring), so only 9-11 of 12 rings are free on D1. A task that picks and tosses
  them (or ships up to 511) would add 10-25% supply.
- After the corn bags run out, the next `CROP_ESTABLISH` fails once with "select_carry_0x05: 0x05 not in carry pair" and the tomato rings wait for the next morning (harmless, one day of growth).
- `MultiMapNavTask._camp_escape` (Y1 Winter D8, `HARVEST_BARN_SKIP_CAPPED=1` only) stays off unless `HARVEST_CAMP_ESCAPE=1`: turning it on moves Y1 from Spring D4.
- Snow-flower trip (`power_berry.SummitGateTask`) never checks the summit gate; repro pin `PR_chk_kdet_Y1_Winter_D1`. Y1 `TILL_FIELD` grass-planter runs of ~23k f are wall time only.

## 4. Lottery rule (read before touching anything that can move Y1 frames)

Any change that moves a Year-1 frame re-rolls the wedding chain and the berry count (`rec1`'s 999 had a lucky Y1; the `_camp_escape` edit did it again). Every switch above is Year 3 only
(`year >= 2`); keep it that way, and diff a deterministic run's `run.log` against the previous batch's with the tag names normalised: the first differing line is the frame a change moved.

## 5. Disk

`recordings/e2e` about 19 GB (scorecards plus logs), `recordings/y23_sweep` is large (the 2026-11-17 `ab_*` previews are ~100 reports of ~600 MB; keep the `*.FINAL.md`, delete the
`.json`/`.log`). The pins `AB*_Y3Su1` and `PR_ab_*` are in `custom_integrations/HarvestMoon-Snes` (56k files there; nothing reads the old ones).
