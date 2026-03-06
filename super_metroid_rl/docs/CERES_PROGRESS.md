# Ceres Progress

Current goal: collapse the remaining segmented Ceres route into a single honest
`Start -> Landing Site` macro. The room-to-room path is now deterministic; the
only unsolved pieces are the uninterrupted fresh Ridley handoff and the
continuous lower-`DF45` countdown climb without checkpoint reload.

## Verified So Far

Room path from SMEDIT/nav data:

1. `0xDF45` Ceres Elevator
2. `0xDF8D` Falling Tile Room
3. `0xDFD7` Magnet Stairs Room
4. `0xE021` Dead Scientist Room
5. `0xE06B` Ceres Flat Room
6. `0xE0B5` Ceres Ridley Room

Published stock bootstrap:

- `NONE -> Start` is now reproducible via `boot-probe --macro-name none_to_start`

Verified Ceres local route:

- `Start -> Magnet Stairs` is deterministic and reaches settled `0xDFD7`
- `Start -> fresh Ridley ground` is now deterministic and reaches settled
  `0xE0B5` with `health=99`
- `0xDFD7 -> 0xE021` is deterministic via the discovered stair descent:
  - top shelf checkpoint: `CeresMagnetMidShelf`
  - lower-left floor checkpoint: `CeresMagnetLowerLeft`
  - right door transition: `RIGHT+B:96:0` from `CeresMagnetLowerLeft`
- `0xE021 -> 0xE06B` is deterministic from `CeresDeadScientist` with:

```text
RIGHT+B:216:0
WAIT:0:30
```

- `0xE06B -> 0xE0B5` is deterministic from `CeresFlatRoom` with:

```text
RIGHT+B:240:0
WAIT:0:80
```

- `Ceres Ridley` room behavior is now verified on a long horizon:
  - Ridley does not appear immediately on room entry
  - from a live `CeresFlatRoom -> CeresRidley` transition, the actor begins
    visibly moving after roughly 870 active-room frames
  - proof frames were captured at `frame_1200.png`, `frame_1800.png`, and
    `frame_2400.png` under `debug_screens/proof/ceres_ridley_wait_window/`
- The countdown trigger is now verified from live RAM:
  - from `CeresRidleyAppeared`, passive damage eventually drops Samus to
    `health=27`
  - at that threshold, `timer_type` flips and the self-destruct countdown
    starts
  - proof frames were captured under
    `debug_screens/proof/ceres_countdown_visual_probe/`
- Fresh Ridley ground is now characterized precisely:
  - from `CeresRidleyGround`, a pure no-input wait reaches the stable
    `27 HP` checkpoint at frame `2321`
  - that live `27 HP` state is not equivalent to `CeresRidleyPreTrigger`
  - `CeresRidleyPreTrigger` already has the countdown state machine primed:
    `timer_type=3` on the first frame and a live timer at frame `17`
  - the saved `CeresRidleyGroundWait2321` checkpoint needs an additional
    `WAIT:0:540` before the published escape route matches the pre-trigger
    phase and reaches `DF45`
- The solved countdown escape now reaches the elevator room deterministically:
  - `CeresRidleyGroundWait2321 -> DF45` is now published as
    `ceres_ridley_ground_27hp_to_elevator_room`
  - `CeresRidleyPreTrigger -> DF45` is published as
    `ceres_pretrigger_to_elevator_room`
  - `CeresRidleyAppeared -> DF45` is published as
    `ceres_ridley_appeared_to_elevator_room`
  - both finish in `0xDF45` with `game_state=8`
  - the key countdown sub-anchors are now published:
    - `CeresRidleyPreTrigger`
    - `CeresEscapeMagnetCountdown`
    - `CeresEscapeFallingTileCountdown`
    - `CeresEscapeElevatorCountdown`
    - `CeresEscapeElevatorLowerLedge`
    - `CeresEscapeElevatorMidWall`
- The concrete local route pieces solved so far are:
  - `CeresRidleyPreTrigger -> DF45` room chain:
    `E0B5 -> E06B -> E021 -> DFD7 -> DF8D -> DF45`
  - `E0B5` exit:

```text
LEFT+A:40:0
LEFT:1000:0
```

  - `DFD7` upper-door climb from `CeresEscapeMagnetCountdown`:

```text
LEFT:80:0
A:16:0
RIGHT+A:124:0
LEFT+A:60:0
LEFT:320:0
```

  - `DF8D -> DF45` from `CeresEscapeFallingTileCountdown`:

```text
LEFT+A:40:0
LEFT:380:0
```
- `CeresEscapeElevatorCountdown -> CeresEscapeElevatorLowerLedge` is now
  deterministic with:

```text
LEFT+A:70:0
```

- `CeresEscapeElevatorLowerLedge -> Landing Site` is now deterministic with:

```text
LEFT+A:94:0
RIGHT+A:80:0
LEFT+A:80:0
RIGHT+A:80:0
RIGHT+A:100:0
LEFT+A:70:0
RIGHT+A:90:0
LEFT+A:50:0
```

  - this route reaches `game_state=32` in `DF45`, then rides the elevator
    handoff/cutscene into `0x91F8`
- Macro from `Start`:

```text
RIGHT+A:24:0
RIGHT:120:0
LEFT:120:0
RIGHT+B:240:60
RIGHT:24:0
RIGHT+B:24:0
RIGHT+B+A:24:0
RIGHT+A:24:0
RIGHT:24:0
RIGHT:24:0
RIGHT:24:0
RIGHT:24:0
RIGHT+B:24:12
RIGHT:24:0
```

Replay command:

```bash
./.venv/bin/python -m super_metroid_rl boot-probe \
  --from-state Start \
  --nav "RIGHT+A:24:0 RIGHT:120:0 LEFT:120:0 RIGHT+B:240:60 RIGHT:24:0 RIGHT+B:24:0 RIGHT+B+A:24:0 RIGHT+A:24:0 RIGHT:24:0 RIGHT:24:0 RIGHT:24:0 RIGHT:24:0 RIGHT+B:24:12 RIGHT:24:0" \
  --settle 140 \
  --expected-room 0xDFD7 \
  --expected-game-state 8
```

- Full published `Start -> fresh Ridley ground` macro:

```bash
./.venv/bin/python -m super_metroid_rl boot-probe \
  --from-state Start \
  --macro-name ceres_start_to_ridley_ground
```

## Notes

- Do not trust the rendered Ceres PNGs as the sole source of truth. Use:
  - SMEDIT exported room JSON / collision grids
  - nav-room / nav-info door geometry
  - live RAM (`room_id`, `samus_x`, `samus_y`, `door_transition`, timers)
- Editor follow-up:
  - active checkout is now `kennycason/super_metroid_editor` at `f9d9f67`
  - the previous `vinchinzu` fork was preserved at
    `super_metroid_editor_vinchinzu_backup_20260306`
  - the upstream CLI exports the JSON layout we actually need
    (`nav_graph.json`, `rooms/*.json`)
  - the old fork's PNG rendering commands are not the active dependency;
    routing work should stay grounded in exported JSON plus live RAM
- The crucial first-room insight was: jump up-right off the start lip, then
  descend left before traversing right to the bottom door.
- The crucial Ridley-room insight was: the room has a long idle window. Short
  local searches miss the event entirely unless they budget for ~15-30 seconds
  of live room time before evaluating appearance / attack behavior.
- The crucial escape-room insight was: treat countdown rooms as separate local
  platforming problems with published anchors. That is what unlocked
  `DFD7 -> DF45` honestly instead of trying to solve the whole escape at once.
- The crucial new Ridley-room insight was: `27 HP` is necessary but not
  sufficient. The real handoff depends on hidden countdown-phase state, which is
  why `CeresRidleyGroundWait2321` still needs an additional `WAIT:0:540`.
- The crucial new elevator insight was: the solved climb is honest from the
  published `CeresEscapeElevatorLowerLedge` anchor, but the same visible pose is
  not yet enough to guarantee a continuous countdown climb from the lower-room
  live state.

## Next Breakdown

1. Close the uninterrupted fresh Ridley handoff:
   - reproduce `CeresRidleyGround -> CeresRidleyGroundWait2321 -> live countdown`
     without relying on the saved checkpoint
   - publish a single fresh-ground countdown macro
2. Close the uninterrupted lower-`DF45` climb:
   - continue the bottom-state search from `CeresEscapeElevatorCountdown`
   - publish a direct `DF45 bottom -> Landing Site` macro
3. Only after those two gaps are honest, publish the full
   `Start -> Landing Site` clear.
