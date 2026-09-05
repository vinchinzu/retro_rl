## Residual — rr-20w.2.3 D2 field clearing

**Status:** GREEN. Clean power-on `--stop-after-d2-clear` complete.
`recordings/power_on_d2_farm_clear.json`. Leave pin
`Y1_D2_PowerOn_FarmClear`. No `--video`. No STATUS.

### Verified this session

- **Clean power-on D2 LIVE GREEN.** `--stop-after-d2-clear`:
  **393223f / 371309 planner / 1059.6s wall**. Start D1 07:00 town $300.
  End D2 **18:01** farm `(41,58)` stam 52 **$100**. Debris
  weeds/fences/stones/small_rocks/large_rocks/stumps **0**. Planted **8**,
  wet **8**. `shipped_before_17=True` (grape bin `shipping_money=150`).
  `outcome=complete`, two consecutive settled observations. Clean:
  `ram_writes=0`, `mid_run_state_loads=0`, `initial_state_loads=0`.
  Glance `d2_leftover_spec("all", done=True)` misses none. Pin
  `Y1_D2_PowerOn_FarmClear`.
- **Shop from grape-bin LIVE GREEN.** Pin `Y1_After_Ship_Berry`. BUY_SEEDS
  potato **0→1**, money **300→100**. Report:
  `recordings/d2_shop_from_grape_bin.json`.
- **Power-on shop LIVE GREEN.** Pin `Y1_D2_PowerOn_AfterShop` (14:02,
  pinch `(7,26)`, $100, seeds 1).
- **Upper-NE / bin-row / ditch-lip / NW paddock spa prefixes GREEN** in
  `farm_gate` (`farm_wp`). Isolated leftover rocks **30→0** and stumps
  **36→0** on the after-shop lineage cannot claim shipping (18:09 pin,
  leftover journal has no grape deposit).
- **Composer fold.** Deleted unused `d2_leftover_phases` /
  `leftover_section_phases`. Live leftover order is `next_d2_spec`.
  `d2_work.py` ~1010 LOC.

### Agent-stopping failures (do not rediscover by soaking)

| Failure | Symptom | Landed? | Next probe |
|---|---|---|---|
| Silent 576k / no JSON | killed run, ptrace blocked | sidecar + interrupt report | inspect `.progress.json` |
| Clock hour reset 24k goal stall | hour tick looked like progress | hour removed from goal key | — |
| Day-plan SUCCESS idle | shop skipped, optional CLEAR_FIELD failed | planner continues `D2FarmClearTactic` | — |
| D2 plan used quota `CLEAR_FIELD` | 13/777 lift then skip | D2 daytime phase is `D2_FARM_CLEAR` | — |
| `CLEAR_PLOT` forever | 0 seeds, never leftover | one `CLEAR_PLOT` then smash if `potato_seeds==0` | — |
| `partial_clear remaining=1` | bushes 506→1 then required abort | `leftover_chain_decision` continues | — |
| Shed map = empty farm | `0x26` scanned as wipe | `farm_map_loaded` requires tilemap `0x00` | — |
| Unobs skips child | `ENSURE_HAMMER` froze in shed 24k | step child even off-farm | — |
| **Spa `route_mountain` pixel_stuck** | stam 4, rocks 43, `(41,11)` | **GREEN 4315f 4→100** | — |
| **Shop `NAV_FARM_EXIT` timeout** | south lane; then `(7,27)` lunch | **POWER-ON GREEN 0→1 / $300→$100** | — |
| **HaveLunch 12:05 pin** | three 10k reds, d-pad ignored | halted that command | do not relaunch `Y1_D2_GrapeBin_PowerOn` |
| **Spa through stump `(38,9)`** | leftover rocks spa timeout at `(38,8)` | **GREEN 4498f 4→100** | do not first-hop x=39 above y=9 |
| **Spa LEFT on bin y=28** | `soft_solid pin` `(29,28)` into stump `(18,28)` | **GREEN 4166f 4→100** | do not LEFT y=28 east of ditch |
| **Spa from `(13,23)` house/A8** | `_FARM_TO_PATH` house hop; y=24 LEFT vs A8 | **GREEN 4047f 4→100** | y=23 x=10-14 north to y=22 then house column |
| **Spa from NW paddock `(8,16)`** | AfterShop clear 18:07 rocks 46 | outer x=4 B-run; power-on complete | — |

### Exact next action

Bead contract is met. Do **not** STATUS-promote Gate B. Next spine from
`bd ready -l harvest -l spine` (likely `rr-20w.2.4` or water-refill
`rr-3ae8`). Claim one.

### Non-claims

- No STATUS promotion
- Did not start from `Y1_D2_Morning_After_D1`
- Did not record a BFS-closable walk
- Did not treat CrossMap origin-return as shop success
- Did not fourth-red the 12:05 HaveLunch grape-return pin
- Did not leftover smash from `Y1_D2_AfterShop_AfterStumps`
- Did not claim shipping from the 18:09 leftover pin
