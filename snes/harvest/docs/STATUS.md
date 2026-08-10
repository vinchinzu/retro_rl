# Status — Harvest Moon (SNES)

## Program gate

| Field | Value |
|-------|-------|
| Current maturity | **M3** (calendar multi-day); crop economy still short of M4 domain |
| Best verified result | Gate A multi-day Day09: harvest 24 + plant 6 + wallet **$1260→$3180** Clean; power-on→D2 handoff+shed Clean (rr-5in partial) |
| Last verification | 2026-08-09 (rr-5in power-on D2; rr-y8n Gate A) |
| Runtime class | Bronze |
| Intervention class | Clean |

| Field | Value |
|-------|-------|
| Status | **Gate A economy closed**; power-on→D2 continuous Clean; full power-on→Summer income still open |
| Integration | `HarvestMoon-Snes` |
| ROM | `roms/Harvest Moon.sfc` via `retro_setup` (SHA1 gate) |
| Start contract | Clean power-on → new diary → Spring D1 07:00 town gate; multi-day via `--power-on` auto D1 handoff |
| Completion contract | Campaign (multi-year farm / marriage / ending) — TBD |
| Evidence | `recordings/power_on_d1_handoff_d2.json` (rr-5in D2); `recordings/run_spring_gate_a_day09.json` (Gate A); `recordings/rr_5in_residual.json` |

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
   Residual: return_home hang ~D5; water tile reliability (2/3); full
   Inside_House multi-day (rr-20w). Parent **rr-20w** stays open.
3. Same-day water after plant: day-plan order
   `CROP_ESTABLISH` → `ENSURE_WATERING_CAN` → `CROP_WATER` is unit-locked.
   **ROM natural empty-can fill OK** on dry fixture (`can_peak=20`); water
   after fill partial (2/3 tiles) — return nav residual.
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
   - **RED pins:** (1) first full-spring attempt sleep miss at bed (70,86) D7 after 5
     overnights; (2) re-run hung on `ENSURE_CROP_SEEDS` multi_nav 1-waypoint S0D4 ~11:02
     after seed spend ($100). Residual: `recordings/rr_5in_residual.json`
   - Child beads: `rr-6byj` (ENSURE_CROP_SEEDS hang), `rr-m0wq` (sleep D7); empty-can
     natural refill still under `rr-3q27` / parent `rr-20w`
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

Planning-stack direction (skill composition, contracts, advisor apply gate):
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
