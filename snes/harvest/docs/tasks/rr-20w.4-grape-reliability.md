## rr-20w.4 — grape reliability: never fence, never NPC, never late

**Status:** unit GREEN (1320 pass, +13 new); ROM D3→D13 ×2 from
`Y1_D3_Morning`, Clean. **Do not STATUS.** This closes defect §1 of
[rr-20w-run13-defects.md](rr-20w-run13-defects.md); §2/§3/§4 are untouched.

Evidence: `logs/spring_d3_30/run14_grapefix.log`,
`logs/spring_d3_30/run15_grapefix.log`, vs `run12_short.log` (same 10 days,
old code) and `run13_full_spring.log`.

### The mechanism behind all three pins

run13 named three byte-identical `soft_solid pin` sites and asked whether
`2204e37b` had fixed a site rather than a mechanism. It had. The mechanism is
**a bounce the stall guards cannot see**:

`Navigator.stasis` resets on *tile* movement, and `_pixel_stuck` resets on
*pixel* movement. A close-range walk whose primary direction is blocked falls
back to a `secondary` cardinal — and when that secondary axis is already on
target, the step is purely sideways. The farmer crosses a tile boundary, both
guards reset, and the walk steps back. Forever.

Confirmed on live tiles at the return pin. Path `0x0C`, farmer at `(8,6)`,
target the crossroads `(132,128)` = `(8,8)`:

```
   5  A0 A0 A1 *D0 A0 A0 A0 A0  A0  A1 *F0
   6  A0 A0 A1  A0 A0 A0 A0 A0  @@  A0  A0
   7  A0 A0 A1  A1 A1 A1 A0 *FF *FF  A0  A1     <- (7,7),(8,7) sealed
   8  A0 A0 A0  A0 A0 A0 A0 A0  A0  A0  A0
```

`dirs_toward(dx=-1, dy=27)` = primary `down` (into `0xFF`), secondary `left`
(dx error: **1 px**). So: left, right, left, right, 300 frames, phase lost.
BFS, given the same RAM, routes `(9,6) (9,7) (9,8) (8,8)` on the first try.

### Fixes

**Route — remove the pin site.** `_PATH_MOUNTAIN_GATE_TO_FARM` descends the
gate on the lane the *inbound* climb already proves live
(`(137,19)→(137,52)→(137,92)→(150,110)→(148,128)`): column 8 for rows 1–5,
step east to column 9 at row 6, east on row 7. The grape return no longer
aims at the crossroads at all. Path return leg **400f → 280f**.

`_PATH_TO_FARM` is deleted so nothing can concatenate it after a mountain
exit. Plaza/town return is `_PATH_PLAZA_TO_FARM` (`path_to_farm`).
`_SPA_TO_FARM` / `mountain_to_farm` / grape-downhill / `ExitToFarm` /
`ReturnHome` use `mountain_exit_then_farm` or `path_return_to_farm`. A
named-route scan fails if any 0x10 south-exit then aims at `(132,128)`.

**Nav — three general guards** (`multi_nav.py`, policy in `nav_corridor.py`):

- `CloseRangeLatch` — distance-to-target is the only thing the bounce cannot
  fake. 30 frames of *moving* without closing hands the waypoint to BFS.
  Frozen-in-place is excluded: that is the post-transition tile-load wait,
  already owned by `_pixel_stuck`.
- `close_range_action` drops a secondary cardinal whose axis is already
  inside the arrival radius — the bounce generator itself.
- A `soft_solid` pin now **recovers** (block the cell, resync entities,
  replan; on the second, skip the waypoint — never on mountain `0x10`, where
  skipping a corridor hop walks into Gotz) up to `PIN_RECOVERY_LIMIT`, then
  fails closed as before with a `recoveries=` count.

**Moving NPCs — yield, don't shove and don't reroute.** Entity tiles were
only re-read on a BFS replan, and the close-range walk never replans. They now
refresh every 4 frames, and when a live sprite holds the next tile the farmer
*waits* (≤90f, stall guards suspended) instead of charging it or leaving a
proven corridor. Mountain ring-padding now covers the whole map, not just
`x>=28`, and is dropped whenever it would seal every exit from the farmer's
own tile.

**Never fence:** `RUN_DIR_STALL_FRAMES` 90 → 45. A `force_run` hop that makes
no progress is a wall; 1.5 s of holding B into it was 1 s too long.

**Task — a pin costs a replan, not the day.** `MountainGrapeShipTask` retried
only `return_to_bin`, only on `0x10`. `_retry_leg` re-arms either leg from the
live pose (a fresh route slice, so a different path) up to `max_leg_retries`.

**Task — never fetch a grape that cannot be banked.** The bin stops crediting
around the 17:00 ShippingScene: the grape leaves the farmer's hands,
`shipping_money` does not move, and it is on the ground where no drop retry
can reach it. Four occurrences (run13 ×1, run14 ×2, run15 ×1) — **every one
at 16:00–18:01, none before.** An outbound leg still walking at
`hard_return_hour` now aborts and walks home rather than spend another 3 000
frames on a grape it cannot bank; `_finish_or_walk_home` with nothing shipped
reports FAILURE, not best-effort SUCCESS. run15 D12 is that guard firing
(`no loop can bank a grape past 16:00` at 18:01, where run14 spent the whole
trip and lost the grape on the ground).

### Measured

| | pre | post |
|---|---|---|
| `Y1_Inside_House` house→bin, 1 grape | 3100f | **2988f** |
| …path return leg | 400f | **280f** |
| …stalled frames | 1055 | **898** |
| D3→D13 `soft_solid` / `pixel_stuck` phase losses | 1 (run12) · 9 in 19d (run13) | **0** (run14, run15) |
| D3→D13 grapes banked | 8 / 7 phases (run12) | **9 / 8 phases** (run14, run15 — identical, deterministic) |

run14 and run15 are byte-identical in berry outcomes except D12, which is the
new deadline guard replacing a lost grape. Both Clean
(`ram_writes=0`, `mid_run_state_loads=0`).

### Remaining, in order

1. **Second-grape pick miss** — `pick: mountain berry unverified after 3
   picks held=0x00 pos=(326,409)`, twice in run14. The farmer is standing
   *exactly* on `GRAPE_STAND_PX` and three A presses yield nothing, so the
   ground grape did not respawn on that map re-entry. Costs 150 G and
   degrades correctly (best-effort `1/2 shipped`). The fix is a forage
   *search* over nearby spawns, not a nav fix — new work.
2. **A slow loop that starts in time and returns late** still loses its grape
   the same way — run14/run15 D8, started ~11:30, still on the mountain at
   16:00, `560->560`. The deadline guard cannot help once the grape is in
   hand; either the return has to get faster still, or a grape held past the
   bin's closing time needs somewhere else to go.
3. run13 §2/§3/§4 (`select_carry_0x07`, phase-order starvation, the silent
   `_reorder_remaining_water_steps` drop) are untouched.

### Non-claims

No STATUS. No spring recalibration — `grape_success_rate` should be re-measured
from a full D3→D30 run, not from these 10 days. `--power-on` not re-run.
