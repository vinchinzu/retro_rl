# Status — Harvest Moon (SNES)

## Program gate

| Field | Value |
|-------|-------|
| Current maturity | **M3** (calendar multi-day); crop economy still short of M4 domain |
| Best verified result | Continuous ROM spring month from `Y1_Inside_House` (Spring D2 06:08) → Summer D1 06:00 house, **29 overnights**, no mid-run state load |
| Last verification | 2026-08-01 (power-on bootstrap); calendar soak 2026-07-28 |
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

## Crop / domain gap (plant fixtures in; water/ship loop open)

Spring calendar still had **no harvest income** ($100 floor). Root causes and fixes:

| Issue | Status |
|-------|--------|
| Virgin soil `CROP_WATER` no-op (`no plots detected`) | Fixed: planner → hoe → plant |
| Shop seeds stock>0 but bag not in carry | **ROM-verified**: shed shelf pick at (190,118) → tool `0x07` |
| Seed equip restored watering can (swapped seeds away) | Fixed: leave seeds+hoe in carry |
| Only 2 carry slots | Day plan plant pass (hoe+seeds) then can+water pass |
| Plant establish | **ROM-verified 2026-08-01** from `Y1_After_Buy_Potato`: seeds+hoe → near-player fallback till → `planted=1`, dry `0x54` tiles, stock 1→0 |
| Same-day water after plant | Partial: water works when can has charge; **empty-can pond refill still flaky** (PreCheckToolSuccess path) |
| Grow → harvest → ship → money > $100 | **Not yet** multi-day verified; shipping bin money settles at **5pm**, not on drop |

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
| `potato_plant_end` | Larger west-field reference: 8 wet `0x55` |
| `Y1_Day09_Harvest_Mode_Start` | Later harvest/ship work (mature tiles); **shipping income posts at 5pm** |

## Next acceptance

1. Close the **natural D1** handoff: town gate `(712,424)` → farm → sleep →
   D2, then rerun the month without a state load. Current `town_to_farm`
   assumes the old `(756,422)` gate and times out from this real opening.
2. Natural empty-can refill at north stream / pond (not south-only bounds; not shipping F2).
3. Same-day water after plant without RAM can poke: `CROP_ESTABLISH` → `ENSURE_WATERING_CAN` → `CROP_WATER`.
4. Multi-day growth from `Y1_Test_Crops_Planted_Watered` (~6 days to potato harvest).
5. Harvest + ship route; assert **money rises after 5pm** (not immediately on bin drop). Save pre-5pm and post-5pm checkpoints for shipping tests.
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
- Empty watering can: ToolUsed early-outs at 0; refill is ToolAnimation + PreCheckToolSuccess==2 — natural refill still flaky
- **Shipping bin money is not instant** — shipped goods credit at **5pm** (test with pre/post-5pm saves)
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
[PLANNING_STACK.md](PLANNING_STACK.md). Future work: [plan.md](plan.md).

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
