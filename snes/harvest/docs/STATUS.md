# Status — Harvest Moon (SNES)

## Program gate

| Field | Value |
|-------|-------|
| Current maturity | **M3** (calendar multi-day); crop economy still short of M4 domain |
| Best verified result | Continuous ROM spring month from `Y1_Inside_House` (Spring D2 06:08) → Summer D1 06:00 house, **29 overnights**, no mid-run state load |
| Last verification | 2026-08-07 (crop keep-alive D2→D8 mature); power-on bootstrap 2026-08-01 |
| Runtime class | Bronze |
| Intervention class | Clean |

| Field | Value |
|-------|-------|
| Status | **Spring calendar soak verified**; plant/harvest income not yet closed |
| Integration | `HarvestMoon-Snes` |
| ROM | `roms/Harvest Moon.sfc` via `retro_setup` (SHA1 gate) |
| Start contract | Clean power-on → new diary → Spring D1 07:00 town gate; M3 soak remains a separate D2 fixture run |
| Completion contract | Campaign (multi-year farm / marriage / ending) — TBD |
| Evidence | `recordings/run_spring_month.json`; `logs/long_runs/run_spring_month_*.log`; end state `Y1_Summer_D1_Morning.state` |

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
| Grow → harvest → ship → money > $100 | **Multi-day keep-alive + mature potatoes verified** (see above); harvest/ship + 5pm wallet assert still open |

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

1. Close **power-on → full D1** without the AnnEve fixture (rr-bhr).
   **2026-08-09:** composed pure talks peak `0x3F`; Gate B truck is leave-only
   rest slice + `GoToSleep` (morning settle). D2 bed free-move OK indoors.
   **Open:** house→farm clears free-move (`game_state & 0x4000`;
   `event_flags_1f68` truck `0x0011` vs Y1 `0x00B1`) → door soft-lock
   `~(133,425)` → `0x5F` (`farm_control_lost`). Not remodel/`house_size`.
   `Y1_Inside_House` shed still ROM-OK. Details: [town_day1_recon.md](town_day1_recon.md).
2. **Natural empty-can refill** to a CheckToolSuccess-valid tile (`F0`/`F9`–`FD`).
   **Mapped 2026-08-01**: main pond **F0** ~(31–34,31–33); human stand
   `(32,34)` face up (`go_to_water_source_end`); north lip `(33,30)` face down
   ROM-fills 0→20. Non-fill: F1/F8 north stream, F2 shipping ditch, F7 north
   pool. **y=31 fence wall (x=11–29)** cuts west plant pocket off from F0 —
   clearing ≥1 fence opens full BFS. Refill selection now preferred-only
   (never F8), main-pond band first; blocked path starts fence-open subtask.
   Landmark `pond_edge` corrected to `(32,34)` (was shipping F2).
3. Same-day water after plant: day-plan order
   `CROP_ESTABLISH` → `ENSURE_WATERING_CAN` → `CROP_WATER` is unit-locked.
   **ROM with charged can OK** (Dry fixture + can=20 → 3 wet `0x55`); still
   needs natural fill (item 2) for empty-can start without RAM poke.
4. ~~Multi-day growth from `Y1_Test_Crops_Planted_Watered`~~ — **done** (mature `0x60` at D8; journal water deltas).
5. Harvest + ship route from mature keep-alive plot; **bin drop no longer requires instant money** (code fix);
   assert **wallet money rises after 5pm**. Save pre-5pm and post-5pm checkpoints.
6. From `Y1_Inside_House`, multi-day soak with **money > 100** after first potato harvest window.
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
- **Shipping bin money is not instant** — bin drop clears carry immediately; wallet/shipping credit at **5pm** (HarvestTask counts drop without money delta)
- `CLEAR_FIELD` morning budget ~3500f is intentional so seed shop is not starved
- ROM SHA1 must match `rom.sha`
- Stamina: tool use drains for real (`INFINITE_STAMINA` off by default). **Noon lunch** is a fixed +20 at 12:00 (decomp `HaveLunch`). Mid-route “+20 on mountain” is that pulse, not spa.
- Hot spring = **upper outdoor pond on mountain `0x10`** (not camp tent pond, not cave `0x29`). Mountain tilemap stays **0x10 all seasons** (palette only). Path: west mid y~470 → full west climb to y=361 → east mid → ridge x≈433 → lip **tile(38,12)** ~(619,201); water **0xF7** at **(39,12)**. Soak = **B+A+direction** into F7 (`player_action=3`); A-alone with watering can does not enter. Post-soak re-cross west before return. Spa corridor **debris-free** (83 off-path stumps/rocks ignored). Routes: `farm_to_spa` / `fish_spot_to_outdoor_spa` / reverse `mountain_to_farm` with nearest-waypoint slice. Mid-route +20 is **noon lunch**.

## Success metrics (track on each soak)

| Metric | Current | Target |
|--------|---------|--------|
| Continuous days without mid-run load | 29 (spring calendar) | Full season + natural summer entry |
| Money growth | Floor **$100** (no harvest income) | Money **> $100** after first potato harvest (~D+6) |
| Plant / water / harvest counts | Plant path ROM-ok; water partial; harvest open | Non-zero planted/watered/harvested in phase journal |
| Intervention class | Clean | Keep Clean |
| Runtime class | Bronze | Bronze until route stable; then Silver workstream |
| Frames to money > $100 | n/a | Measure on first closed crop loop |
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
| `Y1_Day09_Harvest_Mode_Start` | Harvest/ship later; money from bin at **5pm** |
