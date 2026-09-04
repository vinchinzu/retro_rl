## Residual — rr-20w.2.3 D2 field clearing

**Status:** IN PROGRESS. Sidecar works. Power-on leftover smash is not
video-ready. Do not launch another 17-minute `--power-on` until the two
hops below are green from a pin. No `--video`. No STATUS.

### Verified this session

- Motion watchdog: planted 6-hit axe is not a 360f nav stall.
  Leftover SE stumps 5→0 in 3382f (`Y1_D2_Leftover_Checkpoint`) and 3899f
  (`Y1_D2_Wood_SpaReturn`). Reports:
  `recordings/d2_stumps_se_after_watchdog_fix.json`,
  `recordings/d2_wood_spa_return_stumps_after_watchdog_fix.json`.
- `run_to_day2` now writes an atomic sidecar (`<out>.progress.json`),
  SIGTERM/SIGINT terminal JSON, and optional `--checkpoint-on-progress`
  pins under `recordings/d2_progress_checkpoints/`.
- Latest power-on (`recordings/power_on_d2_farm_clear.json`, 322165f /
  1036s, Clean): grape shipped; shop skipped; leftover reached
  **weeds/fences/stones 0**, **rocks 51→43**, **stumps 38**, stam 4;
  then **spa `route_mountain` pixel_stuck (661,185) replans=4**.
  Checkpoint: `recordings/d2_progress_checkpoints/latest.state`
  (and `progress_f322000.state`).

### Agent-stopping failures (do not rediscover by soaking)

Each of these aborted a continuous power-on. Fixes already in the dirty
tree are marked landed. Remaining reds need a **pin hop**, not another
full D1→D2 soak.

| Failure | Symptom | Landed? | Next probe |
|---|---|---|---|
| Silent 576k / no JSON | killed run, ptrace blocked | sidecar + interrupt report | inspect `.progress.json` |
| Clock hour reset 24k goal stall | hour tick looked like progress | hour removed from goal key | — |
| Day-plan SUCCESS idle | shop skipped, optional CLEAR_FIELD failed, `include_end_day=False` spun | planner continues `D2FarmClearTactic` | — |
| D2 plan used quota `CLEAR_FIELD` | 13/777 lift then skip | D2 daytime phase is `D2_FARM_CLEAR` | — |
| `CLEAR_PLOT` forever | 0 seeds, pocket still dirty, never leftover | after one `CLEAR_PLOT`, smash if `potato_seeds==0` | — |
| `partial_clear remaining=1` | bushes 506→1 then required abort | `leftover_chain_decision` continues | last-weed still can 24k-stall |
| Shed map = empty farm | `0x26` scanned as wipe; `CLEAR_ROCKS nw` no-op | `farm_map_loaded` requires tilemap `0x00` | — |
| Unobs skips child | `ENSURE_HAMMER` froze in shed 24k | step child even off-farm | — |
| **Shop `NAV_FARM_EXIT` timeout** | after grape, 12-wp south lane from north of y=31 fence; 10k timeout; BUY_SEEDS deferred; money stays $300 | south-lane threshold y≥27 (still red) | **`harvest-route` + `harvest-shop` from bin/shed-south pin** |
| **Spa `route_mountain` pixel_stuck** | stam 4, rocks 43, stumps 38, pos (661,185) / tile (41,11) | child now steps; route still red | **`harvest-route` from `latest.state`** — do not 300k `--power-on` |

### Exact next action

Do **not** relaunch `--power-on`. Green one hop:

1. Spa: from `recordings/d2_progress_checkpoints/latest.state` (or
   `Y1_D2_Wood_SpaReturn` if that pin is cleaner), farm→spa with stam 4
   at SE/NE farm. `route_mountain` pixel_stuck at (661,185) is the live
   miss. Use `harvest-route`.
2. Only after spa-from-rocks is green: shop `NAV_FARM_EXIT` after grape
   bin (not `Y1_D2_Morning_After_D1`). Use `harvest-shop` + `harvest-route`.

Then one power-on with sidecar. Do not STATUS.

### Non-claims

- No STATUS promotion
- No natural power-on Day 2 farm-clear (shop + 8 wet potatoes + last 43
  rocks / 38 stumps still open)
- No D2 movie / `--video`
