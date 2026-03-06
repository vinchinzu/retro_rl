# Super Metroid RL

Super Metroid currently rides on the shared `platformer_common` runtime for
recording, verification, hill climbing, chaining, and playback, plus a local
navigation stack for room graphs and waypoint generation.

The active Bronze target is pragmatic and explicit:

1. Publish honest route anchors (`Start`, `ZebesStart`, room-entry states).
2. Make the `ZebesStart -> Bomb Torizo` route reproducible with room-aware
   tooling, recordings, and chained action sequences.
3. Keep publishing earlier stock boot paths from `NONE` as named macros instead
   of hiding them in ad hoc scripts.

## Current Active Stack

`python -m super_metroid_rl` routes in three directions:

- `nav-*` commands use `super_metroid_rl/navigation/` directly.
- `doctor` and `boot-probe` use `super_metroid_rl/bronze_tools.py`.
- `play`, `verify`, `hillclimb`, `watch`, `chain`, `trace-map`, `selftest`,
  and related commands delegate to `platformer_common.runner`.

Core files:

- `navigation/`: room graph, collision loader, waypoint generation, trace maps
- `platformer_common/levels/super_metroid.py`: the published segment configs
- `record_tasker.py`: manual `.bk2` recording helper
- `state_manager.py`: state naming / management helper
- `maps/`: area composites and local reference PNGs
- `custom_integrations/SuperMetroid-Snes/`: the published states and data.json

Frozen experiments live under `legacy/`.
Historical runbooks and one-off orchestration now live under `docs/archive/`
and `scripts/archive/` instead of the active top level.

## Bronze Anchors

The repo currently relies on these published states:

- `Start`: Ceres Elevator Room (`0xDF45`)
- `ZebesStart`: Landing Site (`0x91F8`)
- `BossTorizo`: Bomb Torizo Room (`0x9804`)

The main active route is the 12-segment `ZebesStart -> Bomb Torizo` path
defined in `platformer_common/levels/super_metroid.py` and
`super_metroid_rl/navigation/route.py`.

Published stock bootstrap:

- `none_to_start`: `NONE -> Start` (controllable Ceres Elevator Room)
- `ceres_start_to_ridley_ground`: `Start -> fresh Ridley ground`
- `ceres_ridley_ground_to_27hp_wait_state`: fresh Ridley ground -> stable
  `27 HP` checkpoint
- `ceres_ridley_ground_27hp_to_elevator_room`: `27 HP` checkpoint -> lower
  `DF45`
- `ceres_elevator_countdown_to_lowerledge`: lower `DF45` countdown setup ->
  stable lower ledge
- `ceres_lowerledge_to_landing_site`: lower ledge -> elevator handoff ->
  Landing Site

That bootstrap lives in `super_metroid_rl/bronze_tools.py` and can be replayed
with `boot-probe`.

## Bronze Prerequisites

The shared root README expects Bronze work to have real states, real maps, and
honest route tooling. For Super Metroid that means:

- key anchor states load and land in the expected rooms
- all segment start states exist
- area maps exist for trace rendering
- nav data exists in SMEDIT export layout (`nav_graph.json` + `rooms/*.json`)

Check that with:

```bash
uv run python -m super_metroid_rl doctor
```

If the navigation export is missing, build it with the editor CLI:

```bash
cd super_metroid_rl/super_metroid_editor
./gradlew -q :cli:runCli -Pargs="--rom ../roms/rom.sfc export -o /tmp/sm_export"
./gradlew -q :cli:runCli -Pargs="--rom ../roms/rom.sfc render-area crateria -o ../maps/crateria.png --items --enemies --labels"
```

Important: `super_metroid_rl/refs/sm-json-data/` is useful reference data, but
the active nav loader does not consume it directly. The active nav commands
expect the SMEDIT export layout.

## Working Commands

List levels and routes:

```bash
uv run python -m super_metroid_rl list-levels
uv run python -m super_metroid_rl list-routes
```

Record / replay / optimize a segment:

```bash
uv run python -m super_metroid_rl -l sm_landing_site play
uv run python -m super_metroid_rl -l sm_landing_site verify --actions recording.json
uv run python -m super_metroid_rl -l sm_landing_site hillclimb --seed recording.json
uv run python -m super_metroid_rl -l sm_landing_site watch --actions hillclimb_best_final.json
```

Navigation tooling:

```bash
uv run python -m super_metroid_rl nav-path --from 0x91F8 --to 0x9804 --abilities morph_ball missile
uv run python -m super_metroid_rl nav-room --room 0x92FD --entry left --exit 0x96BA
uv run python -m super_metroid_rl nav-waypoints --segment parlor_descent
```

Trace a run on the area maps:

```bash
uv run python -m super_metroid_rl -l sm_parlor_descent watch --actions best.json
uv run python -m super_metroid_rl -l sm_parlor_descent trace-map --actions best.json --area crateria -o parlor_trace.png
```

Replay the published stock bootstrap from `NONE`:

```bash
uv run python -m super_metroid_rl boot-probe \
  --from-state NONE \
  --macro-name none_to_start
```

The named macro defaults to the expected `Start` room/gameplay check, so a
successful run finishes at room `0xDF45` with `game_state=8`.

Replay the solved Ceres countdown escape from the published countdown anchors:

```bash
uv run python -m super_metroid_rl boot-probe \
  --from-state Start \
  --macro-name ceres_start_to_ridley_ground

uv run python -m super_metroid_rl boot-probe \
  --from-state CeresRidleyGround \
  --macro-name ceres_ridley_ground_to_27hp_wait_state \
  --save-name CeresRidleyGroundWait2321

uv run python -m super_metroid_rl boot-probe \
  --from-state CeresRidleyGroundWait2321 \
  --macro-name ceres_ridley_ground_27hp_to_elevator_room

uv run python -m super_metroid_rl boot-probe \
  --from-state CeresRidleyPreTrigger \
  --macro-name ceres_pretrigger_to_elevator_room

uv run python -m super_metroid_rl boot-probe \
  --from-state CeresRidleyAppeared \
  --macro-name ceres_ridley_appeared_to_elevator_room

uv run python -m super_metroid_rl boot-probe \
  --from-state CeresEscapeElevatorCountdown \
  --macro-name ceres_elevator_countdown_to_lowerledge

uv run python -m super_metroid_rl boot-probe \
  --from-state CeresEscapeElevatorLowerLedge \
  --macro-name ceres_lowerledge_to_landing_site
```

`ceres_pretrigger_to_elevator_room`,
`ceres_ridley_appeared_to_elevator_room`, and
`ceres_ridley_ground_27hp_to_elevator_room` all finish in lower `DF45` with
`game_state=8`. `ceres_lowerledge_to_landing_site` is now the published finish
for the solved shaft climb and elevator handoff.

## Current Risks

- The published stock bootstrap currently stops at `Start` in Ceres. The
  follow-on `Start -> ZebesStart` route is still separate work from the active
  `ZebesStart -> Bomb Torizo` chain.
- Ceres is now reproducible as a segmented flow, but the uninterrupted
  `Start -> Landing Site` clear is still not a single published macro.
  The remaining honest gaps are the fresh Ridley-room wait/handoff and the
  continuous lower-`DF45` countdown climb without checkpoint reload.
- The generic platformer selftest death probe is not valid for the published SM
  route anchors, so SM now skips that check until a real deterministic death
  probe is published.
- Nav commands need SMEDIT export data. If `/tmp/sm_export` is missing, nav
  tests will skip and nav-path/nav-room will fail.
