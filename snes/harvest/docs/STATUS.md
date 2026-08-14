# Status — Harvest Moon (SNES)

## Program gate

| Field | Value |
|-------|-------|
| Current maturity | **M3** (calendar multi-day); Gate A economy closed; Gate B continuous open |
| Best verified result | Gate A Day09 **$1260→$3180** Clean; ship probe re-verified same delta; power-on continuous **21 ovn / money $400** (terminal return_home D23 pre-fix); empty-can 3/3 fixture GREEN |
| Last verification | 2026-08-10 (Gate B soak v6: **21 ovn** past prior D23 0x08 hang; sleep outdoor-evening wait; tip **CROP_WATER refill + NAV_CROP freeze**) |
| Runtime class | Bronze |
| Intervention class | Clean |
| Gate board | [MILESTONES.md](MILESTONES.md) · structure debt [CODE_QUALITY_REVIEW.md](CODE_QUALITY_REVIEW.md) |

| Field | Value |
|-------|-------|
| Status | **Gate A closed**; empty-can mostly closed; **ship verify closed** (`rr-9xyy`); **ExitToFarm 0x08 residual closed** (`rr-uru1` — sticky dismiss + soak past D23); Gate B tip = **CROP_WATER refill + income** (`rr-5in`) |
| Integration | `HarvestMoon-Snes` |
| ROM | `roms/Harvest Moon.sfc` via `retro_setup` (SHA1 gate) |
| Start contract | Clean power-on → new diary → Spring D1 07:00 town gate; multi-day via `--power-on` auto D1 handoff |
| Completion contract | Campaign (multi-year farm / marriage / ending) — TBD |
| Evidence | `recordings/rr_9xyy_ship_money_day09.json`; `recordings/power_on_spring_to_summer.json`; `recordings/run_spring_gate_a_day09.json`; `recordings/empty_can_refill_probe.json` |

## Done

- M1/M2 instrumentation + day planner + multi-day shell
- Sleep always finds house; morning settle after final overnight
- **2026-08-01 clean power-on bootstrap** (headless):
  ```bash
  HEADLESS=1 uv run python -m harvest.scripts.boot_probe --power-on \
    --out recordings/power_on_boot_probe.json
  ```
  - Power-on → title → `START` → fresh diary → deterministic player name
    `AAAA` → opening dialogue, using only controller input
  - End: Spring **D1** 07:00, town `0x04`, `(712,424)`, input unlocked
  - `initial_state_loads=0`, `mid_run_state_loads=0`, `ram_writes=0`
  - This fixes the stale “Day 1” fixture ambiguity. It does **not** make the
    existing D2→Summer result a power-on replay.
- **2026-07-28 M3 overnight**: Spring D2 → D4
- **2026-07-28 full spring calendar** (headless):
  ```bash
  HEADLESS=1 uv run python -m harvest.scripts.run_to_day2 \
    --state Y1_Inside_House --end-of-spring \
    --out recordings/run_spring_month.json \
    --save-end-state Y1_Summer_D1_Morning
  ```
  - Start Spring **D2** 06:08 → end Summer **D1** 06:00 house `(136,120)`
  - `days_completed=29`, `day_failures=[]`, `morning_ready=true`, `mid_run_state_load=false`
  - ~207k frames / ~10 min wall
  - Phase success counts (every overnight completed its scheduled work or optional partial clear):
    - `EXIT_TO_FARM` 29, `CLEAR_FIELD` 29, `BUY_SEEDS` 3, `ENSURE_WATERING_CAN` 20,
      `ENSURE_CROP_SEEDS` 22, `NAV_CROP` 22, `CROP_WATER` 22
  - Money stayed **$100** (spent seeds, **no harvest income**) — crop plant path was effectively a no-op
- **2026-08-01 D1 town handoff (human rest + auto replay)**: six social events;
  completion mask `0x3F` at `d1_town_event_mask` (`0x11F74`).
  **Automated** from gate: Ann|Eve → `0x03`. **Full rest** from
  `Y1_Spring_D1_AnnEve` via `tasks/town_day1_rest.json` (11 134f) → peak mask
  `0x3F` → truck → house sleep → D2 (`auto` success 2026-08-01). Free shed
  **grass seeds** stand `(96,118)` + **watering can** `(96,168)` verified into
  carry from `house_size=0` house; soft-optional after rest because AnnEve
  fixtures are `house_size=2` (breaks farm exit). Tooling:
  `harvest.scripts.town_day1_recon` / `scripts/record_town_day1_recon.sh`.
  Details: [town_day1_recon.md](town_day1_recon.md).

## Architecture cleanup (2026-08-03)

| Item | Status |
|------|--------|
| A5 contract preflight in day-plan probe | **Done** — `preflight_phase_contract` / `tool_tags_from_ram`; probe emits `contract_preflight` + planned summary; soft notes only |
| `run_to_day2 --save-end-state` gzip | **Fixed** — raw s9xsnp writes broke stable-retro load (`BadGzipFile`) |
| Empty-can west-pocket staging | **Partial** — stages via `(12,29)` before fence clear; fence toss still stalls after lifting 1 post |
| 6-day growth soak from watered fixture | **Keep-alive OK** — D2→D8 Clean; daily `CROP_WATER` real deltas (can 17→2); crops grow `0x55→0x5F` then mature `0x60` at D8. House end-state map is **not** farm metatiles — earlier “crops gone” was a false negative. Exit-to-farm scan: **3 mature** at (13,25)/(12–13,26). Journal now records `watered=N` / `no_work`; outdoor plan waters **before** `CLEAR_FIELD` when dry crops exist (rr-3v9). |

## Crop / domain gap (plant fixtures in; water/ship loop open)

Spring calendar still had **no harvest income** ($100 floor). Root causes and fixes:

| Issue | Status |
|-------|--------|
| Virgin soil `CROP_WATER` no-op (`no plots detected`) | Fixed: planner → hoe → plant |
| Shop seeds stock>0 but bag not in carry | **ROM-verified**: shed shelf pick at (190,118) → tool `0x07` |
| Seed equip restored watering can (swapped seeds away) | Fixed: leave seeds+hoe in carry |
| Only 2 carry slots | Day plan plant pass (hoe+seeds) then can+water pass |
| Plant establish | **ROM-verified 2026-08-01** from `Y1_After_Buy_Potato`: seeds+hoe → near-player fallback till → `planted=1`, dry `0x54` tiles, stock 1→0 |
| Same-day water after plant | **ROM OK with charged can** (Dry→3×`0x55`); day-plan order unit-locked; **empty-can natural fill still open** |
| Grow → harvest → ship → money > $100 | **Gate A CLOSED (rr-y8n)** via multi-day Day09 successor: harvest+establish+money>$100; Day09 5pm farm wait wired into calendar loop. Full `Y1_Inside_House`→Summer still limited by empty-can water (parent rr-20w) |

ROM smoke (2026-08-01 plant):
```text
EnsureCropSeeds → tools 0x07/0x02 (potato+hoe)
NAV_CROP → preferred field (248,472)
CROP_ESTABLISH → fallback till near player (not remote 35,27 / 19,48)
planted=1, dry 0x54 tiles, pot stock=0
```

Test crop fixtures (for growth / ship work):

| State | Contents |
|-------|----------|
| `Y1_Test_Crops_Planted_Dry` | Spring D2 ~13:00, **3 dry** potato `0x54` at (12–13,25–26) |
| `Y1_Test_Crops_Planted_Watered` | Same plot **watered** `0x55` (can was LiveRamEditor-filled; natural refill still open) |
| `Y1_Test_Crops_DayPlus6` | Spring **D8** morning house after 6d soak; **farm exit** shows **3 mature** potato `0x60` (do not trust house metatile buffer) |
| `potato_plant_end` | Larger west-field reference: 8 wet `0x55` |
| `Y1_Day09_Harvest_Mode_Start` | Later harvest/ship work (mature tiles); **shipping income posts at 5pm** |

## Next acceptance

1. ~~Close **power-on → full D1** without the AnnEve fixture (rr-bhr).~~
   **CLOSED 2026-08-09 night:** pure talks peak `0x3F`; truck leave + sleep →
   D2 bed; outdoor dog intro pure-completes (name `AAAA` on tilemap `0x5F`,
   `$099F=3`) → `event_flags_1f68=0x00B1` + free-move `0x4000`; shed grass+can
   into carry. Evidence: `recordings/gate_b_anneve_full_shed.json` (peak
   `0x3F`, D2, grass+can, `mid_run_state_loads=0`, `ram_writes=0`);
   `recordings/gate_b_dog_intro_shed.json` (rest_end → intro → shed).
   **Root cause:** house→farm with `0x0011` fires `CODE_83CEAE`; free-move
   clears until dog name entry finishes (not permanent softlock). `house_size`
   not causal. Task: `CompleteOutdoorMorningIntroTask` before shed in
   `_shed_starter_tools`. Details: [town_day1_recon.md](town_day1_recon.md).
2. **Natural empty-can refill** to a CheckToolSuccess-valid tile (`F0`/`F9`–`FD`).
   **Mapped 2026-08-01**: main pond **F0** ~(31–34,31–33); human stand
   `(32,34)` face up (`go_to_water_source_end`); north lip `(33,30)` face down
   ROM-fills 0→20. Non-fill: F1/F8 north stream, F2 shipping ditch, F7 north
   pool. **y=31 fence wall (x=11–29)** cuts west plant pocket off from F0 —
   clearing ≥1 fence opens full BFS. Refill selection now preferred-only
   (never F8), main-pond band first; blocked path starts fence-open subtask.
   Landmark `pond_edge` corrected to `(32,34)` (was shipping F2).
   **2026-08-09 night (rr-jwju / rr-3q27):** unit-locked order preferred-edge
   **before** fence-open; main-pond select uses true reachability (not partial
   hop); nearest pathable corridor stand; fence pond-nav hop + local-drop when
   pond BFS dies; refill mid-nav soft repath; return_home hard timeout 5500f;
   post-gap multi-hop densify + carry-drop; **CROP_WATER `refill_bounds` y_min
   14→10** so north **F9** ~(26,12) is in-bounds (was excluded → always fence).
   **2026-08-10 night (rr-3q27):** ROM recon closed false F9 path — north **F9 is
   sealed** from west plant pocket by y=13–14 fence bar (full BFS never reaches
   F9 stands; manhattan hops to ~(21,23) were false positives). Multihop
   preferred edges now require hop *nearly arrives* (end within 3) so sealed
   F9/FA no longer starve fence-open.
   **2026-08-10 later (rr-3q27):** **Natural empty-can fill GREEN** on
   `Y1_Test_Crops_Planted_Dry` Clean — `can_peak=20`, `refill_count=1`,
   `watered=2/3` (`recordings/empty_can_refill_probe.json`, ~8k frames).
   ROM path: corridor_only fence open → **east→south** wall cross (empty gap
   charge soft-blocks on (13,31)) → **west→south-lip** from (28,32) soft-block
   band → F0 stand `(32,34)` face up fill → water-return north charge.
   **2026-08-10 tip close (rr-3q27):** **3/3 water GREEN** after residual
   crop-walk recovery — dry fixture `can_peak=20`, `refill=1`, `watered=3`,
   `dry_end=[]` (~9k frames). Root cause residual (12,26): reorder thrash +
   mid-plot wet tiles not walkable during follow. Fixes: reorder cap 3/step;
   residual crop-stand on wet neighbor with full 3x3 `extra_walkable`; no
   temp_blocked when ≤2 from stand north of wall. **return_home** off-stand
   re-nav capped (force enter / fail clean). Short Clean multi-day
   `Y1_Inside_House --days 3` → 3 overnights, `mid_run_state_loads=0`,
   `ram_writes=0` (`recordings/inside_house_3day_clean.json`); day-plan soft
   fails ENSURE_CAN / CROP_WATER on D3–D4 still residual under **rr-20w**.
   Parent **rr-20w** stays open for full Spring income.
   **2026-08-10 (rr-5go9 CLOSED):** Power-on continuous **CROP_WATER refill densify
   GREEN** after fence. Root: east→south stuck at (29,30) (RIGHT soft-edge) +
   soft-band long UP re-entered gap + east-pond densify thrash. Fixes:
   east-only/south-only corridor charge split; gap-south fallback after 3×
   (29,30) thrash; gap-safe soft/south lip (cap UP at low x); east_pond y≥32
   only; past-fence pure-south bail. Evidence:
   - Dry fixture: `can_peak=20`, `refill=1`, `watered=3`
     (`recordings/empty_can_refill_probe.json`, ~12k f)
   - Power-on `--end-of-spring`: **CROP_WATER ×3 success** D9–D11
     `watered=6` each, D9 `refills=1 can=20`; crop_survival **wet=2 dry=4**
     (not stuck dry=6); overnights=10 money=$160 mid_run=0 Clean
     (`recordings/power_on_spring_to_summer.json`). Prior terminal
     `return_home multi_nav timeout` D12 — **cleared 2026-08-10** (below).
   **2026-08-10 (rr-5in return_home GREEN / Gate B still PARTIAL):** D12
   house approach after water/CLEAR no longer hard-fails. Root: densify used
   pond column x=512 (lateral-align through water from ~(854,527)); mid-yard
   (118,486) sat outside force-enter + escape gates. Fixes in `home.py`:
   east free lane **x≥576** (east of pond); far-east pre-escape west+north;
   west free side keeps near-player x; force enter y≤80; mid-yard re-nav;
   south_of_fence / door_far escape. Evidence
   `recordings/power_on_spring_to_summer.json`:
   - **14 overnights** to Spring **D16**, money=$160, Clean mid_run=0
   - D12: Pre-escape far-east → south escape → sleep → D13 (not multi_nav die)
   - D13–D15 return_home all succeed; terminal **`reason=budget`** (not house nav)
   - Residual was **rr-qc9r** (closed below).
   **2026-08-10 (rr-qc9r CLOSED):** Late-spring **CROP_WATER thrash GREEN**.
   Root: soft/south_far lip charges oscillated (25,34)↔(29,32) via trailing UP;
   densify preferred pure-north (29,35)→(29,34) and direct 7-tile F0; near-F0
   re-queued long RIGHT charges that overshot to (36,36); exhausted
   `_south_lip_charges` left later tiles densify-only. Fixes in
   `crop_planter.py`: pure-east south lip (no UP on y=34); soft LEFT brief;
   densify short east hops require east gain; near-F0 multihop/act skip
   re-charge; soft-reset charges on new refill; thrash arm skips near-F0.
   Evidence Clean mid_run=0:
   - Dry fixture: `can_peak=20`, `refill=1`, `watered=3`
     (`recordings/empty_can_refill_probe.json`, ~10.7k f)
   - Power-on `--end-of-spring`: **CROP_WATER success** D9 `watered=6 refills=1`,
     D11+D13 `watered=6` each (no D13–15 dry=6 thrash); **HARVEST_ROUTE**
     shipped≥5 → money **$160→$400**; **21 overnights** to Spring **D23**;
     terminal `return_home timeout` (exit_to_farm) — not water/budget.
     (`recordings/power_on_spring_to_summer.json`, ~300k f)
   - Parent **rr-5in** residual: Summer D1 not reached (return_home D23) +
     harvest ship timeouts; Gate B full still open.
   **2026-08-10 (rr-ws8h CLOSED unit):** `return_home` no longer hard-fails
   `timeout phase=exit_to_farm` while already on house tilemap. Fix:
   house-arrival short-circuit every `step` (+ timeout defense) and approach
   geometry extracted to `planner/tasks/home_approach.py`. Unit-locked mid-
   phase exit_to_farm + timeout-on-house SUCCESS. Full power-on soak residual
   stays under **rr-5in**; ship verify **closed** (**rr-9xyy**); ExitToFarm
   dialogue residual **rr-uru1**.
3. Same-day water after plant: day-plan order
   `CROP_ESTABLISH` → `ENSURE_WATERING_CAN` → `CROP_WATER` is unit-locked.
   **ROM natural empty-can fill + 3/3 dry water OK** on dry fixture
   (`can_peak=20`, `watered=3`, `dry_end=[]`). **Power-on continuous refill
   + water GREEN** D9–D13 late spring (rr-5go9 + rr-qc9r).
4. ~~Multi-day growth from `Y1_Test_Crops_Planted_Watered`~~ — **done** (mature `0x60` at D8; journal water deltas).
5. ~~Harvest + ship + post-5pm money assert (rr-53g)~~ — **CLOSED 2026-08-09 night** Clean:
   ```bash
   HEADLESS=1 uv run python -m harvest.scripts.harvest_ship_money_probe \
     --state Y1_Day09_Harvest_Mode_Start \
     --out recordings/harvest_ship_5pm_money.json
   ```
   - `HarvestTask` **shipped_count=24** / harvested=24 (bin drop; `shipping_money` 0→1920 same-day)
   - Farm **5pm ShippingScene** (hour=17, stay tilemap < 4); wallet still flat pre-sleep
   - Overnight settle: wallet **$1260 → $3180** (+1920); `shipping_money`→0
   - Checkpoints: `Y1_Harvest_Ship_Pre5pm` / `Y1_Harvest_Ship_Post5pm` / `Y1_Harvest_Ship_PostSleep`
   - Evidence: `recordings/harvest_ship_5pm_money.json` journal (`money_rose_after_5pm_window`)
   - ROM: wallet `AddMoney` is overnight after 5pm scene — not instant at bin or cutscene
   - Helpers: `harvest.core.shipping_credit`; day-plan phase journal records `shipped_count`
6b. **Power-on continuous Spring→Summer with income (rr-5in)** — **PARTIAL 2026-08-09 night**:
   ```bash
   HEADLESS=1 uv run python -m harvest.scripts.run_to_day2 --power-on --until-day 2 \
     --out recordings/power_on_d1_handoff_d2.json
   # Full claim (still RED residual):
   HEADLESS=1 uv run python -m harvest.scripts.run_to_day2 --power-on --end-of-spring \
     --out recordings/power_on_spring_to_summer.json
   ```
   - **GREEN:** power-on → TownDay1Handoff (peak talks + truck + outdoor dog intro + shed
     grass+can, `house_size=0`) → D2 farm; `mid_run_state_loads=0`, money $300
   - **Wired:** `run_to_day2 --power-on` auto-runs D1 handoff before multi-day
   - **RED pins (prior):** (1) sleep miss at bed (70,86) D7 after 5 overnights
     (`rr-m0wq`); (2) `ENSURE_CROP_SEEDS` multi_nav 1-waypoint hang S0D4 ~11:02.
     Residual: `recordings/rr_5in_residual.json`
   - **2026-08-10 `rr-6byj` CLOSED (Clean):** hang was 2-slot **carry thrash** —
     shelf A replaces **selected**; hoe grab then seed grab (or reverse) kept
     re-entering `near_shed_to_shed` 1-wp multi_nav forever. Fix: X-swap so
     keep-tool is backpack before shelf when both slots filled; skip swap when
     backpack empty; `max_shed_trips` fail-clean; ENSURE_CROP_SEEDS/CAN optional.
     Evidence: `recordings/rr_6byj_ensure_crop_seeds_probe.json` —
     `Y1_After_Buy_Potato` → hoe+seeds ready, `multi_nav_starts=2`, `trips=2`,
     SUCCESS ~2248f, `mid_run_state_loads=0`, `ram_writes=0`.
   - **2026-08-10 `rr-m0wq` CLOSED (Clean):** D7 bed miss was face-left + A
     (walks into mattress) and B-after-A (cancels Yes on sleep confirm).
     `GoToSleepTask` now face-up only, B only before first A of each attempt,
     A-only confirm/dismiss, toss held item once, re-nav if off-stand mid-verify.
     Evidence: `recordings/rr_m0wq_sleep_days6.json` — `Y1_Inside_House` D2→D8,
     6/6 overnights first-try (incl. D7→D8), `mid_run_state_loads=0`,
     `ram_writes=0`, Clean.
   - Parent residual: `rr-20w` / `rr-5in` full power-on→Summer income still open
  - **2026-08-10 night (rr-o00y CLOSED):** Empty-can refill after sparse plant
    **GREEN** on power-on continuous. Path: fence open south at (18,35) →
    south_far lip (y≈32 east corridor) → F0 fill `can=20`, watered=2/2 D5;
    D6–D7 water holds. Dry fixture still `can_peak=20` watered=3.
    Evidence: `recordings/power_on_spring_to_summer.log` REFILL OK +
    `recordings/rr_o00y_fill_green.json`.
  - **2026-08-10 night (rr-6g7g CLOSED):** return_home hands-clear after power-on
    water days. Root: CLEAR_FIELD left held stone/weed (`0x0D`/`0x09`); in-place
    field toss re-picked or failed. Fix: always nav to open drop south of house,
    multi-face stationary A-drop (fence_flow proven); door push prefers walk over
    blind B-hold. Evidence: `recordings/power_on_spring_to_summer.json` —
    **7 overnights**, no `could not clear hands`, crops wet=2/alive, Clean
    mid_run_loads=0. Parent **rr-5in** residual: dies D9
    `nav_house_front failed: multi_nav timeout` (money $100, Summer not reached).
  - **2026-08-10 night (rr-5in PARTIAL):** sparse plant water + empty-can fill
    GREEN (rr-o00y); hands-clear GREEN (rr-6g7g). Still short of Summer D1
    money>100. Residual: house approach multi_nav after water days + first
    potato ship window.
  - **2026-08-10 night (rr-5in house approach GREEN / still PARTIAL Gate B):**
    Root of D9 `nav_house_front multi_nav timeout`: CLEAR ends south of the
    **y=31 fence wall** (x=11–29). Mid-wall densify + SW rock pocket had no
    BFS path. Fix: densified approach east (x≥480) or west (x≤160) of fence
    (or open gap when wall confirmed); CLEAR exit-staging + SW pre-escape;
    MultiMapNav early softlock fail. Evidence:
    `recordings/power_on_spring_to_summer.json` — **11 overnights** to Spring
    D13, **money=$160** (`money_gt_100=true`), Clean mid_run_loads=0; dies
    `reason=budget` (not house nav). Residual **rr-5go9**: CROP_WATER refill
    thrash fails (dry=6 watered=0 east-crawl densify stuck) so crops never
    mature/ship and frame budget ends before Summer D1.
6. ~~End-of-spring / continuous soak with **money > 100** + harvest phases (rr-y8n / Gate A)~~ — **CLOSED 2026-08-09 night** Clean multi-day successor:
   ```bash
   HEADLESS=1 uv run python -m harvest.scripts.run_to_day2 \
     --state Y1_Day09_Harvest_Mode_Start --days 1 \
     --out recordings/run_spring_gate_a_day09.json
   ```
   - `mid_run_state_load=false`, `gate_a_economy_ok=true`
   - `HARVEST_ROUTE` shipped=24 / harvested=24; `CROP_ESTABLISH` planted=6
   - Wallet **$1260 → $3180** overnight (NightReset `AddMoney`; farm work often already past 5pm)
   - Calendar loop: after day plan, if `shipping_money>0` and hour&lt;17 on farm → `FarmShippingWaitTask` (Day09 path) before return/sleep
   - Journal summary: `harvest_phases_present`, `crop_establish_nonzero`, `total_shipped`, `final_money`
   - Full `--end-of-spring` from `Y1_Inside_House` still flaky (empty-can `CROP_WATER` fail; return_home hang observed D5) — parent rr-20w / Gate B
7. Optional: `HOT_SPRING_STAMINA` — **ROM natural-entry verified 2026-07-31**:
   farm drain → `farm_to_spa` → upper pond B+A bath (50→110+) → reverse
   `mountain_to_farm` → farm. Corridor debris-free (`mountain_spa_validate`).
   ```bash
   HEADLESS=1 uv run python -m harvest.scripts.mountain_spa_validate
   HEADLESS=1 uv run python -m harvest.scripts.hot_spring_probe \
     --state latest_backup_sunday_go_to_mountain_20260427_152011 \
     --min-stamina 100 --target-stamina 30 --return-to-farm
   ```

```bash
HEADLESS=1 uv run python -m harvest.scripts.run_to_day2 \
  --state Y1_Inside_House --end-of-spring \
  --out recordings/run_spring_month.json
```

## Traps

- First mountain grape is a **ground spawn** at the x=20 stand `~(326,409)` tile `(20,25)` (west stump/rock to the east). Not the carpenter 2×2 plants. Walk the tape corridor (land → east gap, never A on Gotz `0x025B` ~(31,37) → west loop → stand). Return **jumps** the x=20 grape cliff onto mid terrace `~(328,568)` then east/south dirt — do not reverse the west loop (second cliff still blocks due-south). A on the grape pixel sets `held=0x03` and opens Eat / Don't eat — **Down then A to keep**. Mash-A eats it. Mountain dialogue with `held=0` is Gotz.
  **Pickup GREEN 2026-08-12** (`rr-14xx`): `Y1_Inside_House` → stand 1650f, keep grape **1913f / 31.88s**, `held=0x03` lock=1, no talk. Shot `recordings/mountain_grape_kept.png`.
  **Spring D2 GREEN 2026-08-13** (`rr-nn3x`): natural `Y1_Inside_House` → mountain pick/keep → reverse corridor → farm F2 bin north stand `(8,28)` face down; `shipping_money` **0→150** at 3658f. A later auto D2 CrossMap `BUY_SEEDS` claimed return with wallet **$300→$450** (grape credit only) — that was a false shop success (no `0x1C`, potato stayed 0).
  **Seed shop GREEN 2026-08-13** (`rr-uos8`): `BuySeedsTask` nav to `shop_door` plaza `(602,274)` face up → `0x1C` → clerk `(182,342)` Buy/Don't buy `0x033D` → potato **0→1**, money **300→100**, back on farm `0x00` at 2542f. Evidence: `recordings/buy_seeds_d2_probe.json`. CrossMap origin-return without those deltas is now a miss. Outdoor plan puts `CLEAR_FIELD` after `BUY_SEEDS` (not ROM-verified on a full day).
- Viewport BFS; sleep bed pixel + face up; doors reject held items
- Multi-day owns return/sleep (`include_end_day=False` on day task)
- Scene wake coordinates: house `y < 100` until settle ~(136,120)
- Seed bags: inventory **count** can be >0 while tool id `0x07` is not in carry pair — bags sit on the shed shelf after shop buy; X only swaps the 2 carried slots
- Only two carry slots: plant day uses hoe+seeds first, then can for water after the bag is spent
- Crop planner full-farm centers east of the x≈32 fence (e.g. 35,27) are often **unreachable** from the early-spring west pocket — establish uses near-player fallback till
- Viewport hop nav must end at the hoe stand; remote centers skip all hoe tiles as `no path`
- Empty watering can: ToolUsed early-outs at 0; fill is `ToolAnimationWateringCan` when `CheckToolSuccess` returns 2 (property `F0`/`F9`–`FD` → can=`0x14`). F1/F8/F2/F7 do **not** fill — never select them. Main F0 pond is primary; y=31 fence wall blocks west pocket until cleared.
- **Shipping bin money is not instant** — bin drop bumps `shipping_money` only; farm **5pm** runs ShippingScene dialogue; wallet credit is **overnight** `AddMoney` (rr-53g verified $1260→$3180). Stay on farm (tilemap < 4) at hour 17.
- `CLEAR_FIELD` morning budget ~3500f is intentional so seed shop is not starved
- ROM SHA1 must match `rom.sha`
- Stamina: tool use drains for real (`INFINITE_STAMINA` off by default). **Noon lunch** is a fixed +20 at 12:00 (decomp `HaveLunch`). Mid-route “+20 on mountain” is that pulse, not spa.
- Hot spring = **upper outdoor pond on mountain `0x10`** (not camp tent pond, not cave `0x29`). Mountain tilemap stays **0x10 all seasons** (palette only). Path: west mid y~470 → full west climb to y=361 → east mid → ridge x≈433 → lip **tile(38,12)** ~(619,201); water **0xF7** at **(39,12)**. Soak = **B+A+direction** into F7 (`player_action=3`); A-alone with watering can does not enter. Post-soak re-cross west before return. Spa corridor **debris-free** (83 off-path stumps/rocks ignored). Routes: `farm_to_spa` / `fish_spot_to_outdoor_spa` / reverse `mountain_to_farm` with nearest-waypoint slice. Mid-route +20 is **noon lunch**.

## Success metrics (track on each soak)

| Metric | Current | Target |
|--------|---------|--------|
| Continuous days without mid-run load | 29 (spring calendar) | Full season + natural summer entry |
| Money growth | Multi-day Gate A **$1260→$3180** Clean (rr-y8n); probe rr-53g same window | Parent: full Inside_House→Summer income |
| Plant / water / harvest counts | Gate A: plant=6 + harvest/ship 24/24 Day09 multi-day; empty-can water still flaky on virgin spring | Non-zero plant/water/harvest on full spring |
| Intervention class | Clean | Keep Clean |
| Runtime class | Bronze | Bronze until route stable; then Silver workstream |
| Frames to money > $100 | Day09 ship soak ~30k frames (harvest+5pm+sleep) | Measure on continuous spring loop |
| M-gate | M3 | M4 natural-entry summer; M5 domain depth |

Gate board: [MILESTONES.md](MILESTONES.md). Structure debt:
[CODE_QUALITY_REVIEW.md](CODE_QUALITY_REVIEW.md). Planning-stack direction:
[PLANNING_STACK.md](PLANNING_STACK.md). Layer ownership:
[bot_architecture_plan.md](bot_architecture_plan.md). Future work: [plan.md](plan.md).

Architecture note (2026-08-01): production `TaskContract`s are declared on
crop establish/water, harvest, coop, ensure tools, exit/sleep, and hot spring;
`evaluate_task_contract()` soft-checks maps/tools/RAM field names. Skill
boundary factories cover coop feed/ship, farm shipping bin, and talk presses.
Domain monofiles remain the production path until skill extraction + crop
income close-loop land.

## Key states

| State | Role |
|-------|------|
| `Y1_Inside_House` | Spring D2 morning house — spring soak start |
| power-on (no state) | Verified Spring D1 07:00 town gate bootstrap; natural D1→D2 still open |
| `Y1_Summer_D1_Morning` | Written end of spring calendar soak (verify before reuse) |
| `Y1_After_Buy_Potato` | Post seed purchase (stock=1, carry often empty) |
| `Y1_Test_Crops_Planted_Dry` | 3 dry potato `0x54` — plant-path fixture |
| `Y1_Test_Crops_Planted_Watered` | Same plot watered `0x55` — growth fixture |
| `Y1_After_Till_Plant` | Reference tilled/planted field |
| `Y1_After_Sleep` | Spring D3-ish morning |
| `Y1_Day09_Harvest_Mode_Start` | Mature potatoes harvest/ship fixture (rr-53g source) |
| `Y1_Harvest_Ship_Pre5pm` | Post-bin-drop, pre-hour-17 (shipping_money up, wallet flat) |
| `Y1_Harvest_Ship_Post5pm` | After farm ShippingScene at 17:00 (wallet still flat) |
| `Y1_Harvest_Ship_PostSleep` | Next morning after wallet `AddMoney` settle |
