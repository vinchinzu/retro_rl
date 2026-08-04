# Autoplay And Speedrun Plan

## Bronze Target

Make one stable-retro SMW state reproducible end to end:

```bash
uv run python -m SMW -l smw_yoshi_island_1 play
uv run python -m SMW -l smw_yoshi_island_1 verify --actions <recording.json>
uv run python -m SMW -l smw_yoshi_island_1 hillclimb --seed <recording.json>
uv run python -m SMW -l smw_yoshi_island_1 watch --actions <best.json>
```

The repo currently has no local SMW ROM, so emulator-backed verification is
blocked until `SMW/roms/smw.sfc` or a stable-retro imported ROM is present.

## Phase 1 - Autoplay Foundation

- [x] Create `SMW/` workspace.
- [x] Add custom stable-retro integration metadata and expanded RAM fields.
- [x] Register initial stable-retro level states in `retro_harness.platformer`.
- [ ] Add local ROM symlink and verify SHA1.
- [ ] Smoke-test `uv run python -m SMW -l smw_yoshi_island_1 play`.
- [ ] Record one clean human completion.
- [ ] Verify replay headlessly and save trace JSON.
- [ ] Hillclimb from the recording.
- [ ] Promote best action file into a route manifest.

## Phase 2 - Segment Contract

Each segment needs:

- start state name
- expected `GameMode == 0x14`
- expected translevel/submap when known
- progress axis and direction
- completion condition
- death/stall condition
- trace fingerprint fields
- best known recording/action path

Minimum trace fields:

- frame
- raw buttons and action index
- `camera_x`, `camera_y`
- `player_x`, `player_y`
- `player_x_speed`, `player_y_speed`
- `player_animation`
- `player_blocked_dir`
- `lives`
- `game_mode`
- `translevel`
- `powerup`
- `p_meter`
- `on_ground`

## Phase 3 - Chained 11-Exit Route

The first speedrun route is the normal 11-exit route, not credits warp/ACE.
Draft manifest: [../routes/11_exit_seed.json](../routes/11_exit_seed.json).

Initial route work:

1. Use built-in states where they exist.
2. Manually create missing route anchors with the interactive recorder.
3. Verify each segment standalone.
4. Generate chained states from successful segment endings.
5. Score candidates in both standalone and chained contexts.
6. Save the selected route under `optimizer/`.

Missing route anchors expected early:

- Donut Secret 1
- Donut Secret House
- Star World 1 through Star World 5
- Front Door / Bowser

## Phase 4 - Speedrun Tightening

Once the route completes:

- Add per-segment timing reports.
- Add start/end state fingerprints to reject stale recordings.
- Run per-segment hillclimbs with raw button mutation.
- Add route alternatives for no-cape/no-starworld practice.
- Record boss-room and keyhole-specific subsegments.

## Phase 5 - Canonical Reuse

Promote shared mechanics back into `retro_harness.platformer` only after SMW proves
the shape:

- RAM-trace schema
- route manifests with standalone/chained selections
- arbitrary RAM-value death/completion hooks
- state fingerprint validation
- route report format
