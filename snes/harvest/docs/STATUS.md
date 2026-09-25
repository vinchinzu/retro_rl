# Status: Harvest Moon (SNES)

## Program gate

| Field | Value |
|-------|-------|
| Current maturity | M3 |
| Best verified result | Clean power-on D2 spine ends Spring D2 17:00 (grape phase, potato buy, wallet $300 to $100). Gate A fixture $1260 to $3180. Power-on spring soak: 21 overnights, $400, stopped on D23 short of Summer. |
| Last verification | 2026-08-23 |
| Runtime class | Bronze |
| Intervention class | Clean |

Gate B is open. The open check is power-on through Summer D1 with harvest
income and no mid-run load. A calendar shell from `Y1_Inside_House`
(2026-07-28) reached Summer D1 with money still $100 and no harvest income.
That shell is not Gate B.

## Verified

- Integration `HarvestMoon-Snes`. ROM `roms/Harvest Moon.sfc` through
  `retro_setup` (SHA1 gate).
- Clean power-on bootstrap (2026-08-01): title, new diary, name `AAAA`,
  Spring D1 07:00, town `0x04` at `(712,424)`.
  `recordings/power_on_boot_probe.json`. Loads and RAM writes are 0. This
  does not turn later fixture soaks into power-on replays.
- Power-on D2 spine (2026-08-23): `--power-on --stop-after-d2-shipping`
  ends Spring D2 17:00 on the farm. Grape phase succeeds. Potato buy and
  establish run. ShippingScene is dismissed. Wallet $300 to $100 (seed
  spend; grape credit posts overnight). Two dry potato tiles. Clean.
  `recordings/power_on_d2_spine_clear_final.json`. Gate B stays open. Do
  not start this from `Y1_D2_Morning_After_D1`.
- Gate A is a fixture, not power-on. From `Y1_Day09_Harvest_Mode_Start`,
  one day ships 24 and the wallet goes $1260 to $3180 overnight.
  `recordings/run_spring_gate_a_day09.json`. Clean. It closes the fixture
  economy check only.
- Power-on `--end-of-spring` (2026-08-10): 21 overnights to Spring D23,
  wallet $400, Clean, `mid_run_state_loads=0`. Terminal reason is
  `return_home` timeout. Summer D1 is not reached.
  `recordings/power_on_spring_to_summer.json`.

## Not this gate

- Any run that starts from `Y1_D3_Morning` or another save-state pin,
  including a spring that reaches Summer from that pin. Those notes stay in
  the living residual. They are not written here.
- A shop menu, or a cross-map return to the origin, without a wallet or
  stock change.
- Empty-can fill on `Y1_Test_Crops_Planted_Dry`. That is a fixture. It does
  not close Gate B.

## Check

```bash
HEADLESS=1 uv run python -m harvest.scripts.boot_probe --power-on \
  --out recordings/power_on_boot_probe.json

HEADLESS=1 uv run python -m harvest.scripts.run_to_day2 --power-on \
  --stop-after-d2-shipping \
  --out recordings/power_on_d2_spine_clear_final.json

HEADLESS=1 uv run python -m harvest.scripts.run_to_day2 --power-on --end-of-spring \
  --out recordings/power_on_spring_to_summer.json
```

Future work: [plan.md](plan.md). `bd ready -l harvest -l spine`.
