# Return To Ship Plan

Current goal: do not start the Ceres escape-to-ship grind until the route and
tooling are honest enough to make the work tractable.

## Current Truth

- `NONE -> Start` is reproducible with `boot-probe --macro-name none_to_start`.
- The room chain to Ridley is now deterministic:
  - `Start -> 0xDFD7 Magnet Stairs`
  - `0xDFD7 -> 0xE021 Dead Scientist`
  - `0xE021 -> 0xE06B Flat Room`
  - `0xE06B -> 0xE0B5 Ridley Room`
- The Ridley room is not a short-horizon trigger problem.
  - Ridley becomes active only after a long in-room wait window.
  - Short probes that score after a few hundred frames are invalid here.
- The countdown trigger itself is still unsolved.
  - waiting alone is not enough
  - naive floor shots and naive move-shoot probes did not start the timer

## Why Pause Before Return To Ship

- The current route work proved room-to-room navigation.
- The remaining problem is now event logic, not pathfinding.
- Return-to-ship routing before solving the Ridley retreat/countdown trigger
  would be slop: every downstream segment depends on a clean countdown start.

## Improvements Needed

- Add long-horizon event-aware probing.
  - `boot-probe` needs periodic snapshots and arbitrary frame save support.
  - We need to save live actor states such as `RidleyAppeared`, not just room
    entries.
- Expose more useful RAM for boss-event debugging.
  - projectile positions
  - more enemy slots
  - room-script / event-state bytes if available
  - hurt / invulnerability / contact state if available
- Tighten the Ceres macro publishing story.
  - publish named room macros, not just one-off shell history
  - keep deterministic entry states for `DeadScientist`, `FlatRoom`, and
    `Ridley`
- Keep the editor dependency narrow.
  - use the upstream `kennycason` editor for JSON export
  - if we patch it, prefer export/debug commands for room state / script data
    over chasing prettier PNG rendering

## Tools Needed

- `boot-probe` upgrades:
  - `--save-frame-at`
  - `--save-frame-every`
  - `--log-info-keys boss_hp,boss_x,boss_y,...`
- A room-event search helper for active states.
  - long waits in the prefix
  - macro suffix search after the wait
  - scoring on timer start / event state, not only room position
- Optional editor CLI extension:
  - dump room-state headers
  - dump enemy sets / enemy scripts
  - dump scroll / setup ASM pointers for Ceres rooms
- A simple proof-packager:
  - turn selected screenshots into labeled GIFs
  - keep the proof artifacts in one folder per claim

## Execution Plan

1. Publish the deterministic `Start -> CeresRidley` chain cleanly.
2. Add the missing long-horizon probe features.
3. Lock a reproducible `RidleyAppeared` or equivalent active-state save.
4. Solve `Ridley retreat -> countdown start` from that active state.
5. Re-run the same trigger from the chained live route, not only from the
   active-state save.
6. Only after countdown start is honest, break down the actual return-to-ship
   route into segments.

## Exit Criteria Before Escape Work

- A named macro or chained function reaches Ridley reproducibly from `Start`.
- Ridley appearance timing is documented and backed by saved proof frames.
- Countdown start is reproducible from a live route, not just a convenient
  in-room save.
- The next unverified gap is genuinely `countdown room routing`, not another
  observability hole.
