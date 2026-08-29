# Plan — Harvest Moon (SNES)

Future work only. Proven facts live in [STATUS.md](STATUS.md).
Structure / API direction: [PLANNING_STACK.md](PLANNING_STACK.md).
Tracker: `bd ready -l harvest -l spine`. Session loop:
`.grok/skills/harvest-session/` (one bead, one living residual, halt-3,
no STATUS from a pin). Immediate session card: `rr-20w.2.3` D2 whole-farm clear.

**Doc consolidation (2026-08-18):** deleted `CODE_QUALITY_REVIEW.md`
(review essay), `bot_architecture_plan.md` (layer ownership folded here
and in PLANNING_STACK), and `MILESTONES.md` (gate table folded below).
Kept STATUS, plan, PLANNING_STACK, INTERACT, ram_map, and recon notes.
Ready work stays in beads — do not recreate a gate board.

## Working board

1. **D2 farm clear (now):** every weed, stone, fence, large rock, and stump
   gone; potatoes planted and watered; goods shipped before 17:00. The hour
   stops at 18; continue until clear. No leftover quotas and no exception for
   the 19 house-row posts.
2. First potato harvest from those plants; no Day09 fixture.
3. Spring → Summer.
4. Animals, bought live.
5. Year 1 done.
6. Marriage.
7. Natural Year 3 credits with a score.
8. Published 10–20 hour video through the end of credits.

The first natural credits run is intermediate: basics and some score, then
refactor and rewrite. Ranch master and 999 are not requirements. All rung
evidence is Clean controller input; RAM/resource pokes are retired. Build
skills, not a frozen tape.

## Bottleneck

`harvest.planner.d2_work` composes the D2 skills after the live grape and seed
purchase. Crop targets are eight planted and watered tiles. Debris completion
is exhaustive, ordered weeds → fences → stones → large rocks → stumps; numeric
leftover quotas are not completion. The next proof is a natural power-on run,
not `Y1_After_Buy_Potato` evidence.

**Already closed (do not re-open as bottleneck):**

- Gate A multi-day economy (Day09 successor money $1260→$3180) — `rr-y8n`
- Ship debris → bin thrash (`rr-9xyy`) — clear_hands before pick
- Natural empty-can fill + thrash stabilizations — `rr-3q27` + kids
- Power-on full D1→D2 shed on `house_size=0` — `rr-bhr`
- Same-day D2 grape→shop→clear→hoe→plant — `rr-20w.1`
- Spring calendar shell D2→Summer (fixture, no income) — historical soak

**Architecture tax on the tip path:** keep monofiles under ~1000 LOC (AGENTS).
`MultiMapNavTask` extracted to `multi_nav.py` (was `navigation.py` 1.3k mono).
Further MultNav residuals (forage interact, corridor densify) land as helpers,
not thrash `if`s in the MultNav step machine. Layer ownership: RAM catalog
and tile/map model stay in `core` / `maps`; domain tasks compose
`tasks/skills.py` instead of growing monofile FSMs.

| Concern | Module(s) |
|---------|-----------|
| MultNav | `multi_nav` (not `navigation.py`) |
| Pond / crop thrash | `pond_*`, `crop_{establish,water_ops,refill,refill_verify,navigate}` |
| Home | `home_return`, `home_sleep`, `home_approach`, `home_recover` |
| Coop / cow | `coop_{layout,feed_ops,egg_ops}`, `cow_*` |
| Maps / routes | `map_config` facade + `map_types` / `farm_pond` / `map_routes` |
| Day plan | `day_plan_orchestrator`, `multi_day_planner`, `day_phase_{catalog,berry,chicken,cow}` |
| D1 / ROM / editor | `town_day1_*`, `rom_*` / `save_state_io` / `map_render`, `editor_*` |

**Farm-bush residual (P3, not D2):** that legacy `SHIP_BERRY` route still
leaves `shipping_money=0` (debris field north of the bush seals the interact).
Spring D2 no longer depends on it: mountain grape + `BuySeedsTask` compose is
Clean (`rr-zmss`). Keep the bush issue scoped to later repeat-forage days.

## Product backlog

- Close Gate B (CROP_WATER refill `rr-3ae8` → power-on soak `rr-5in` / epic `rr-20w`).
- Day-plan soft fails ENSURE_CAN / CROP_WATER after plant (if still red post-B).
- Extract coop feed/collect/ship (`rr-rbk`); cow mono extract (bead).
- Hot-spring stamina gate in day plan (`rr-pzw`) — evening leftover insert
  wired; D2 night grape-corridor farm→spa→farm GREEN 2026-08-21
  + full restore is wired; live pin still needs a spa soak proof.
- Festival/Sunday/rain ordering (`rr-1vc` / Gate C).
- Gift delivery (carry egg to NPC); summer/fall crop rotations.
- M4 natural-entry summer from `Y1_Summer_D1_Morning`.
- Thin D1 handoff (A6) over Nav+Talk skills — product path green, structure open.

## Next acceptance tests

1. ~~Power-on D1→D2 shed `house_size=0`~~ — closed (`rr-bhr`).
2. ~~Natural empty-can + same-day water fixture~~ — closed (`rr-3q27` / thrash kids).
3. ~~Gate A multi-day money > 100 + harvest~~ — closed (`rr-y8n`).
4. **Gate B:** `run_to_day2 --power-on --end-of-spring` → Summer D1, money > 100,
   Clean, no mid-run load (`rr-5in`).
5. Optional after B: rainy/festival phase order (`rr-1vc`); spa when stam low (`rr-pzw`).

## Architecture track (planning trunk)

Ordered structural work — detail in PLANNING_STACK workstreams A1–A8.

| Priority | Item | Notes |
|----------|------|-------|
| Done | A1 Phase contracts on crop/coop/sleep/hot-spring | `evaluate_task_contract` + catalog wiring |
| Done | A2 Skill boundary factories | feed/ship/talk/farm bin in `tasks/skills.py` |
| Partial | A3 Crop close-loop acceptance | Gate A closed; Gate B continuous open |
| **Now** | **Crop mono extract (`rr-ds3`)** | `crop_planter` ~4.9k — P1 arch tax (review) |
| **Now** | **Pathfinder promote (`rr-fjbk`)** | Out of `farm_clearer` |
| Next | A4 Coop skill composition + multi-adult fix | `rr-rbk` — stop growing `coop_task.py` |
| Done | A5 Contract preflight in day-plan probe | Soft notes when map/tool mismatch |
| Later | A6 D1 skill routes from power-on | `rr-7js5` — product path green; structure open |
| Later | A7 Festival + rainy-day distillation | `rr-1vc` |
| Later | A8 Promote Pathfinder / primitives shared | After second game consumer |

### A3 progress (2026-08-01 subagent push)

| Piece | Status |
|-------|--------|
| Day-plan establish → ensure can → water order | **Verified** (unit tests lock sequence + refill_bounds) |
| Same-day water with charged can | **ROM OK**: `Y1_Test_Crops_Planted_Dry` + can=20 → 3/3 wet `0x55` |
| Ship verify without instant money | **Fixed**: bin drop counts ship; money may settle at 5pm |
| Empty-can natural refill | **ROM GREEN closed (rr-3q27)** — dry fixture `can_peak=20` + watered=3 `dry_end=[]` Clean via east→south + F0 + residual crop-walk. return_home re-nav cap. Full Inside_House→Summer income still **rr-20w**. |
| Multi-day growth → harvest → money > $100 | **Gate A CLOSED (rr-y8n)**: multi-day Day09 successor final_money=$3180, HARVEST+CROP_ESTABLISH, mid_run loads=0; farm 5pm wait wired. Full Inside_House→Summer still open (empty-can water / rr-20w). |

**Empty-can refill traps (ROM-mapped 2026-08-01):**

1. `CheckToolSuccess` farm fill only when tile-in-front property is `F0, F9–FD`
   (`ToolAnimationWateringCan` sets can=`0x14`). Prefer
   `REFILL_PREFERRED_WATER_TILES`; F1/F2/F7/F8 do **not** refill — selection
   now preferred-only (no F8 full-path trap).
2. **Main pond F0** ~(31–34,31–33); stands `(32,34)` face up (human recording)
   and `(33,30)` face down (ROM fill 0→20). Band order: pond → south → north.
3. Early west plant pocket is cut off by **y=31 fence wall (0x05, x=11–29)**.
   Clearing one fence opens full BFS to F0. `_start_refill` starts a limited
   `FenceClearLoopTask` when preferred water is unreachable and the wall is up.
4. Shipping F2 pocket remains blacklisted (`BAD_REFILL_STAND_BOUNDS`).

## Domain milestones

### Immediate (close the bottleneck)

- Natural empty-can refill (north stream + pond, not south-only); same-day water
  without RAM can poke.
- ~~Multi-day growth from `Y1_Test_Crops_Planted_Watered` → mature potatoes.~~ **Done** (rr-3v9).
- ~~Harvest + ship + post-5pm money~~ — done (rr-53g).
- ~~Wire Day09 ship path + Gate A multi-day money>$100~~ — done (rr-y8n); successor `run_spring_gate_a_day09.json`.
- Full `Y1_Inside_House`→Summer soak with income (empty-can refill + stable return_home; parent rr-20w / Gate B).
- Power-on → full D1 → D2 with shed pickups on real `house_size=0`.
- Fix `CoopChoresTask` for multi-adult / multi-egg dynamic tiles (Spring 22 case).

### M4 (natural-entry summer)

- Continue from live `Y1_Summer_D1_Morning`.
- Sunday / festival handling (extract from `spring_festival.json`).
- Hot-spring stamina gate already ROM-verified; wire into day-plan when low.

### M5 (domain depth)

- Cow / barn chores (milking, brushing, feeding) distilled from rainy-day + other recordings.
- Rainy-day ordering (barn → coop → crops → town social).
- Multi-seed bags, gift delivery, town NPC loops.
- Stamina / tool management fully closed.

### Long-horizon (campaign)

- Multi-year planner (seasonal crop rotations, animal expansion, marriage flags, ending path).
- Hierarchical planner: day → week/season goals → recovery.
- Power-on → new game → continuous spring month (promote D2 soak).
- Observation class improvement (Bronze → Silver) once the route is stable.

## Active workstreams

| Stream | Focus |
|--------|--------|
| Seed equip | **Done (ROM)**: shed shelf + hoe |
| Hoe/plant | **Done (ROM)**: virgin plan + hoe + plant; near-player fallback |
| Water after plant | `work_mode=establish` then water + can re-fetch; natural refill flaky |
| Hot spring stamina | **ROM natural-entry OK**. Next: day-plan gate when stam low |
| Harvest/ship | `HARVEST_ROUTE` when mature tiles present; money @ 5pm |
| Multi-day soak | `--end-of-spring` with crop income |
| D1 town handoff | Rest auto green; pure power-on path + shed open |
| Skill composition | Factories exist; extract production coop/cow/harvest |
| Phase contracts | **Wired** + probe preflight (A5 done 2026-08-03) |
| Plan advisor | Advisory default; gated apply (`HARVEST_PLAN_LLM_APPLY`) |
| Navigation | `densify_waypoints`; promote Pathfinder later |

## Deferred

- Full multi-year campaign objective contract
- Ending credits path (scene classifier already blocks)
- Unrestricted LLM plan rewrites (only schema-validated optional changes)
- Hard-abort builders on failed contracts (soft evaluate first)

## Infrastructure blockers

- ROM not in git; `retro_setup` SHA1 gate
- Long soaks manual under `logs/long_runs/`

## CLI catalog

Parked from AGENTS. Session commands stay in `snes/harvest/AGENTS.md`.
`HEADLESS=1` on live probes; glance is `harvest.clock_glance`; no MP4.

```bash
./run_bot.sh play --autoplay --state latest

# Live power-on with bot (window + [ ] speed): title → D1 handoff → multi-day
uv run python -m harvest.runtime.harvest_bot play --autoplay --power-on --end-of-spring
# --no-d1-handoff skips town talks/shed/sleep after power-on

# Boot / power-on (clean diary → Spring D1 07:00 town)
uv run python -m harvest.scripts.boot_probe --state Y1_Inside_House
HEADLESS=1 uv run python -m harvest.scripts.boot_probe --power-on \
  --out recordings/power_on_boot_probe.json

# D1 town recon (docs/town_day1_recon.md)
uv run python -m harvest.scripts.town_day1_recon checklist
HEADLESS=1 uv run python -m harvest.scripts.town_day1_recon auto \
  --state Y1_Spring_D1_AnnEve --out recordings/town_day1_rest_auto.json

# Multi-day soak (M3)
HEADLESS=1 uv run python -m harvest.scripts.run_to_day2 \
  --state Y1_Inside_House --end-of-spring \
  --out recordings/run_spring_month.json \
  --save-end-state Y1_Summer_D1_Morning

# Power-on continuous (rr-5in): D1 handoff auto + multi-day
HEADLESS=1 uv run python -m harvest.scripts.run_to_day2 --power-on --until-day 2 \
  --out recordings/power_on_d1_handoff_d2.json
# Composed D1 handoff from the power-on town gate (six talks → truck → shed → D2)
HEADLESS=1 uv run python -m harvest.scripts.town_day1_recon auto \
  --state Y1_Spring_D1_Town_Gate \
  --save-end-state Y1_D2_Morning_After_D1 \
  --out recordings/town_day1_town_gate_composed.json
# Do not start D2 work from Y1_D2_Morning_After_D1 — grape return-to-bin
# seals at the house fence (rr-oqri).
HEADLESS=1 uv run python -m harvest.scripts.run_to_day2 --power-on --end-of-spring \
  --out recordings/power_on_spring_to_summer.json

# Mountain berry variants
HEADLESS=1 uv run python -m harvest.scripts.mountain_berry_probe \
  --state Y1_Inside_House --screenshot recordings/mountain_grape_stand.png
HEADLESS=1 uv run python -m harvest.scripts.mountain_berry_probe \
  --state Y1_Inside_House --pick --screenshot recordings/mountain_grape_kept.png
HEADLESS=1 uv run python -m harvest.scripts.mountain_berry_probe \
  --state Y1_Inside_House --ship --until-lunch \
  --out recordings/mountain_segments_clock.json

# Post-shop hoe+seed collect + 3x3 plant (8 around (13,28); rr-m7mk)
HEADLESS=1 uv run python -m harvest.scripts.d2_plant_probe \
  --state Y1_After_Buy_Potato --out recordings/d2_plant_probe.json
HEADLESS=1 uv run python -m harvest.scripts.d2_plant_probe \
  --state Y1_After_Buy_Potato --hoe-only --out recordings/d2_hoe_ring.json
HEADLESS=1 uv run python -m harvest.scripts.d2_plant_probe \
  --state Y1_After_Buy_Potato --water --out recordings/d2_plant_water.json

# Leftover smash: 10 bushes pick+toss → dump fences in ponds → 10 stones in ponds → hammer 4 large → axe 2 stumps
HEADLESS=1 uv run python -m harvest.scripts.d2_leftover_probe \
  --state Y1_After_Buy_Potato --out recordings/d2_leftover_smash.json
HEADLESS=1 uv run python -m harvest.scripts.d2_leftover_probe --dump

# Stamina object from a pin (`player.stamina` is current/max/tool_hits)
uv run python -m harvest.runtime.harvest_bot world --state Y1_Inside_House --compact
# Drain + outdoor spa until current == max, then return (rr-pzw)
HEADLESS=1 uv run python -m harvest.scripts.hot_spring_probe \
  --state Y1_D2_Night_Farm \
  --min-stamina full --target-stamina 70 --return-to-farm \
  --out recordings/hot_spring_full.json
uv run python -m harvest.scripts.hot_spring_probe \
  --state Y1_D2_Night_Farm \
  --min-stamina full --target-stamina 70 --return-to-farm --watch

# Harvest + ship + post-5pm wallet credit (rr-53g)
HEADLESS=1 uv run python -m harvest.scripts.harvest_ship_money_probe \
  --state Y1_Day09_Harvest_Mode_Start \
  --out recordings/harvest_ship_5pm_money.json

# Gate A multi-day successor: harvest phases + money>$100 (rr-y8n)
HEADLESS=1 uv run python -m harvest.scripts.run_to_day2 \
  --state Y1_Day09_Harvest_Mode_Start --days 1 \
  --out recordings/run_spring_gate_a_day09.json

# Record task (F5) / tests
uv run python -m harvest.runtime.harvest_bot play --state latest --record <name> --no-day-plan
uv run python -m unittest tests.test_day_plan_sequences tests.test_task_progress -v

# Editor
./kickoff.sh
PYTHONPATH=.. uv run --project .. python -m retro_harness.editor_launcher harvest -- --state latest
```
