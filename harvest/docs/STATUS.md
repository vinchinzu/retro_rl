# Status — Harvest Moon (SNES)

## Program gate

| Field | Value |
|-------|-------|
| Current maturity | **M3** (calendar multi-day); crop economy still short of M4 domain |
| Best verified result | Continuous ROM spring month from `Y1_Inside_House` (Spring D2 06:08) → Summer D1 06:00 house, **29 overnights**, no mid-run state load |
| Last verification | 2026-07-28 |
| Runtime class | Bronze |
| Intervention class | Clean |

| Field | Value |
|-------|-------|
| Status | **Spring calendar soak verified**; plant/harvest income not yet closed |
| Integration | `HarvestMoon-Snes` |
| ROM | `roms/Harvest Moon.sfc` via `retro_setup` (SHA1 gate) |
| Start contract | Named morning state (`Y1_Inside_House`) |
| Completion contract | Campaign (multi-year farm / marriage / ending) — TBD |
| Evidence | `recordings/run_spring_month.json`; `logs/long_runs/run_spring_month_*.log`; end state `Y1_Summer_D1_Morning.state` |

## Done

- M1/M2 instrumentation + day planner + multi-day shell
- Sleep always finds house; morning settle after final overnight
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

## Crop / domain gap (open → plant path closed)

Spring calendar still had **no harvest income** ($100 floor). Root causes and fixes:

| Issue | Status |
|-------|--------|
| Virgin soil `CROP_WATER` no-op (`no plots detected`) | Fixed: planner → hoe → plant |
| Shop seeds stock>0 but bag not in carry | **ROM-verified**: shed shelf pick at (190,118) → tool `0x07` |
| Seed equip restored watering can (swapped seeds away) | Fixed: leave seeds+hoe in carry |
| Only 2 carry slots | Day plan plant pass (hoe+seeds) then can+water pass |
| Plant establish | **ROM-verified** from `Y1_After_Buy_Potato`: `planted=1`, tiles `0x54` dry potato, stock 1→0 |
| Same-day water after plant | Partial: needs can re-fetch from field (shed nav from deep field still flaky) |
| Grow → harvest → ship → money > $100 | **Not yet** multi-day verified |

ROM smoke (2026-07-28):
```text
EnsureCropSeeds → tools 0x07/0x02 (potato+hoe)
CropWaterTask plant → planted=1, field shows 0x54 crops, pot stock=0
Water without can → SUCCESS partial (plant kept); second pass needs ENSURE_WATERING_CAN
```

## Next acceptance

1. Same-day water after plant: `CROP_ESTABLISH` (hoe+seed) → `ENSURE_WATERING_CAN` → `CROP_WATER` (water-only).
2. From `Y1_Inside_House`, multi-day soak with **money > 100** after first potato harvest window (~day+6).
3. Phase journal: non-zero `planted_count` / `watered_count`, then `HARVEST_ROUTE`.
4. Re-run `--end-of-spring` without mid-run state load.
5. Optional: `HOT_SPRING_STAMINA` — **ROM natural-entry verified 2026-07-31**:
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
- `CLEAR_FIELD` morning budget ~3500f is intentional so seed shop is not starved
- ROM SHA1 must match `rom.sha`
- Stamina: tool use drains for real (`INFINITE_STAMINA` off by default). **Noon lunch** is a fixed +20 at 12:00 (decomp `HaveLunch`). Mid-route “+20 on mountain” is that pulse, not spa.
- Hot spring = **upper outdoor pond on mountain `0x10`** (not camp tent pond, not cave `0x29`). Mountain tilemap stays **0x10 all seasons** (palette only). Path: west mid y~470 → full west climb to y=361 → east mid → ridge x≈433 → lip **tile(38,12)** ~(619,201); water **0xF7** at **(39,12)**. Soak = **B+A+direction** into F7 (`player_action=3`); A-alone with watering can does not enter. Post-soak re-cross west before return. Spa corridor **debris-free** (83 off-path stumps/rocks ignored). Routes: `farm_to_spa` / `fish_spot_to_outdoor_spa` / reverse `mountain_to_farm` with nearest-waypoint slice. Mid-route +20 is **noon lunch**.

## Key states

| State | Role |
|-------|------|
| `Y1_Inside_House` | Spring D2 morning house — spring soak start |
| `Y1_Summer_D1_Morning` | Written end of spring calendar soak (verify before reuse) |
| `Y1_After_Buy_Potato` | Post seed purchase (stock=1, carry often empty) |
| `Y1_After_Till_Plant` | Reference tilled/planted field |
| `Y1_After_Sleep` | Spring D3-ish morning |
