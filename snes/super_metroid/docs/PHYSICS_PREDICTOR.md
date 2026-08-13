# Physics Predictor — Super Metroid Trajectory Prediction

**Status:** Scaffolded interface + stub implementation (2026-08-13)

Protocol for predicting Super Metroid trajectories from start state + input
sequence without requiring full emulation for every candidate. Designed to
integrate with external physics engines (e.g., `sm_rev` MiniStep predictor)
or stub implementations for testing.

## Purpose

Enable route planning and trajectory optimization by querying candidate
motion paths offline:

1. **Route planning:** evaluate alternative input sequences for hops without
   emulator rollout cost
2. **SMEDIT integration:** live trajectory preview in route panel
3. **Hop optimization:** genetic/hill-climbing search over input tapes
4. **TAS development:** fast candidate filtering before emulator validation

### Important: Predictor vs Emulator Ground Truth

**StubPredictor and MiniStep are for search speed only, not ground truth.**

- **Planning phase**: Use predictor for fast candidate evaluation and filtering
- **Validation phase**: Run winning inputs on real emulator (stable-retro / SMEDIT snes9x)
- **Conflict resolution**: If predictor and emulator disagree, **emulator wins**
- **Room-clear claims**: Require emulator validation path (skip/xfail without ROM in CI is fine)

This separation allows fast offline planning while ensuring production trajectories are emulator-verified.

**Workflow:**
1. Search with predictor (filter 1000s of candidates quickly)
2. Validate top candidates on emulator (ground truth)
3. Only claim room-clear after emulator success

### Roles and Certification

**MiniStep is a wind tunnel** for trajectory search — fast, approximate physics for filtering candidates.

**Legality check is SuperMetroidEnv** (stable-retro emulator) — ground truth for room-clear claims.

**BK2 and LSMV are tape formats** (`tas/bk2.py`, `tas/lsmv.py`) used for TAS movie import/replay — 
they are NOT MiniStep-certified and should not be assumed to match MiniStep predictions without 
emulator validation. Use them for movie replay and annotation, not as physics predictor input.

## Architecture

```
┌─────────────────────────┐
│  Route Planner          │
│  (map_planning.py)      │
└───────────┬─────────────┘
            │ predict(start, inputs)
            v
┌─────────────────────────┐
│  PhysicsPredictor       │  Protocol (physics_sim.py)
│  - predict()            │
│  - name()               │
└───────────┬─────────────┘
            │
    ┌───────┴───────┐
    v               v
┌────────────┐  ┌──────────────┐
│ StubPredictor  SmRevClient   │
│ (testing)  │  │ (sm_rev bin) │
└────────────┘  └──────────────┘
```

## Data Structures

### SimState

Minimal Super Metroid state for physics prediction (subset of `SuperMetroidState`):

- Position: `samus_x`, `samus_y`, `samus_x_sub`, `samus_y_sub`
- Velocity: `velocity_x`, `velocity_y` + subpixel
- Momentum: `momentum_x`, `momentum_x_sub`
- Physics state: `pose`, `facing`, `movement_type`
- Speed booster: `speed_counter`, `speed_flag`, `shinespark_timer`

### FrameInput

Controller input for a single frame:

- `buttons`: SNES button mask (matches `retro_harness.controls`)

### TrajectoryFrame

Predicted Samus state at one frame:

- Frame number
- Full kinematics (position, velocity, momentum, pose, etc.)

### Trajectory

Complete prediction result:

- `start`: Initial `SimState`
- `frames`: Sequence of `TrajectoryFrame`
- `predictor`: Backend identity for provenance

## Predictor Backends

### StubPredictor (testing)

Deterministic fake physics for offline tests:

- **Intentionally simplified:** Not accurate Super Metroid physics
- **Physics model:** Direct pixel-per-frame movement (`x += vx`, `y += vy`)
  - B jump sets `vy = -5`, then `y += vy` (e.g., 200 - 5 = 195)
  - LEFT/RIGHT sets `vx = ±2`, then `x += vx`
  - Simple ground collision at y=200
- **Not MiniStep:** Golden fixtures reflect StubPredictor behavior, not MiniStep/sm_rev
- **Use for:** Protocol contract tests, CLI smoke tests, route planning unit
  tests without ROM

```python
from super_metroid.physics_sim import StubPredictor, SimState, FrameInput

predictor = StubPredictor(name="test")
start = SimState(frame=0, room_id=0x91F8, samus_x=100, samus_y=200, ...)
inputs = [FrameInput(buttons=0x40) for _ in range(60)]  # Hold LEFT
trajectory = predictor.predict(start, inputs)
# Frame 1: x=98 (100-2), y=200 (grounded)
```

### SmRevClient (external)

Client for `sm_rev_predict` MiniStep-based physics predictor:

- **Transport:** Subprocess stdin/stdout JSON (current implementation)
- Calls external `sm_rev_predict` binary via subprocess
- Gracefully raises RuntimeError if binary not available (CI stays green)
- Environment: `SM_REV_PATH` env var or `sm_rev_predict` in PATH
- **Search speed only**: NOT ground truth - winners require emulator validation

```python
from super_metroid.physics_sim import SmRevClient

# Use sm_rev_predict from PATH or SM_REV_PATH
predictor = SmRevClient()
trajectory = predictor.predict(start, inputs)

# Or specify path explicitly
predictor = SmRevClient(binary_path="/path/to/sm_rev_predict")
trajectory = predictor.predict(start, inputs)
```

**sm_rev wire protocol (subprocess stdin/stdout):**

Currently speaks `Trajectory.to_dict()` internal format over subprocess:

```bash
# Request: SimState + inputs
echo '{
  "start": {"frame": 0, "room_id": 37368, "samus_x": 100, ...},
  "inputs": [{"buttons": 128}, {"buttons": 129}]
}' | sm_rev predict

# Response: Trajectory.to_dict() format
{
  "predictor": "sm_rev@...",
  "start": {"frame": 0, "room_id": 37368, ...},
  "frames": [{"frame": 1, "room_id": 37368, "samus_x": 102, ...}],
  "inputs": [{"buttons": 128}, {"buttons": 129}]
}
```

**Future transport:** sm_rev may migrate to HTTP API while keeping JSON
keys stable. Golden fixtures in `tests/fixtures/predict_request_response.json`
define the contract; either transport (subprocess or HTTP) can reuse these
fixtures.

**sm_rev integration:** The `vinchinzu/sm_rev` sibling repository provides
a MiniStep-based physics kernel. Once available, `SmRevClient` will call
its `predict` CLI for frame-by-frame trajectory prediction.

## CLI Tool

`scripts/tools/predict_trajectory.py` demonstrates trajectory prediction:

```bash
# Use stub predictor with synthetic inputs (smedit-tas-1 format by default)
uv run python snes/super_metroid/scripts/tools/predict_trajectory.py \
  --frames 60 --predictor stub

# Write smedit-tas-1 format for route panel
uv run python snes/super_metroid/scripts/tools/predict_trajectory.py \
  --frames 120 --predictor stub --output trajectory.json

# Internal format for debugging
uv run python snes/super_metroid/scripts/tools/predict_trajectory.py \
  --frames 60 --predictor stub --format internal

# Print summary
uv run python snes/super_metroid/scripts/tools/predict_trajectory.py \
  --frames 60 --predictor stub --summary

# Use sm_rev backend (when available)
SM_REV_PATH=/path/to/sm_rev uv run python \
  snes/super_metroid/scripts/tools/predict_trajectory.py \
  --frames 60 --predictor sm_rev
```

**Output formats:**
- `smedit-tas-1` (default): SMEDIT TasMovie format for route panel consumption
- `internal`: Raw Trajectory.to_dict() with snake_case fields for debugging

## Output Formats

### smedit-tas-1 (SMEDIT TasMovie)

Route panel native format for trajectory preview and route planning:

```json
{
  "format": "smedit-tas-1",
  "meta": {
    "gameName": "SuperMetroid-Snes",
    "startState": "LandingSite",
    "romSha1": null
  },
  "buttonOrder": ["B", "Y", "Select", "Start", "Up", "Down", "Left", "Right", "A", "X", "L", "R"],
  "frames": ["............", ".......r....", "b......r...."],
  "trace": [
    {"frame": 0, "x": 100, "y": 200, "roomId": 37368},
    {"frame": 2, "x": 104, "y": 200, "subX": 0, "subY": 0, "pose": 0, "roomId": 37368}
  ]
}
```

**Schema:**
- `format`: Always `"smedit-tas-1"`
- `meta`: Game name, start state name, ROM SHA1 (null in stub/CI)
- `buttonOrder`: 12-button order (matches retro_harness.controls)
- `frames`: 12-char mnemonics (`.` = released, letter = pressed) — native format encoded from button masks
- `trace`: Sparse frames with x/y required, other fields optional
  - **roomId**: Room header pointer (WRAM $079B), NOT door ID. Door transitions show as roomId changing.
  - **enemies**: Optional list for SMEDIT overlay and damage-boost routing. Omitted when empty.
    Point structure: `{id/slot, x, y, hp?}` with extra fields OK.
    Used for Ceres→Morph→Bomb damage-boost routing once sm_rev fills it.
  - **subX/subY/pose**: Optional kinematics fields
  - Extra fields ignored by route planner (forward compatibility)

**Consumed by:** SMEDIT route panel for trajectory overlay and interactive editing

### internal (Debug)

Raw `Trajectory.to_dict()` with snake_case fields for debugging and test fixtures.

## Integration Points

### Current Consumers

**SMEDIT route panel:** Consumes `smedit-tas-1` format for trajectory preview

### Planned Consumers

1. **map_planning.py:** Trajectory-aware route planning
   - Evaluate hop candidates by predicted landing position
   - Filter infeasible input sequences before emulator validation

2. **SMEDIT route panel:** Enhanced live trajectory preview
   - Visualize predicted path overlaid on room geometry
   - Adjust input tape interactively with instant feedback

3. **Hop optimization:** Genetic/hill-climbing search
   - Generate candidate input tapes
   - Score by predicted trajectory (distance to goal, time, risk)
   - Keep best candidates for emulator validation

4. **TAS development:** Fast candidate filtering
   - Test many input variations offline
   - Promote promising candidates to full emulator validation

## sm_rev Integration

The `vinchinzu/sm_rev` sibling repository provides a MiniStep-based Super
Metroid physics kernel for accurate frame-by-frame prediction.

### Wire Protocol

**Current transport:** Subprocess stdin/stdout JSON using `Trajectory.to_dict()` format

```bash
# Request: {"start": SimState, "inputs": [FrameInput, ...]}
echo '{"start": {...}, "inputs": [...]}' | sm_rev predict

# Response: Trajectory.to_dict() format
{
  "predictor": "sm_rev@...",
  "start": {...},
  "frames": [{...}, ...],
  "inputs": [{...}, ...]
}
```

**Golden fixtures:** `tests/fixtures/predict_request_response.json` defines
the wire contract. JSON keys are stable; transport may migrate from subprocess
to HTTP without breaking contract.

**Future transport:** sm_rev may add HTTP API (`SM_REV_URL`) while keeping
JSON schema unchanged. SmRevClient will detect available transport and use
the fastest option.

### Integration Status

- **Protocol defined:** `SmRevClient` ready to call external binary
- **Transport:** Subprocess stdin/stdout (HTTP may be added)
- **sm_rev availability:** Not yet published; design allows graceful skip
- **Environment:** `SM_REV_PATH` env var or `sm_rev` in PATH
- **Fallback:** Tests use `StubPredictor` by default (no ROM required)

### When sm_rev is Available

1. Set `SM_REV_PATH` to sm_rev binary location
2. Use `--predictor sm_rev` in CLI
3. Or: `load_predictor("sm_rev")` in code

## Testing

Pure offline tests with `StubPredictor` — no ROM required:

```bash
uv run pytest snes/super_metroid/tests/test_physics_sim.py -v
```

Tests validate:

- Protocol contract
- Data structure serialization (JSON round-trip)
- StubPredictor determinism
- SmRevClient graceful unavailable handling

## Design Notes

### Why Not Full Emulator?

Full emulation is expensive for trajectory search:

- Each candidate input tape requires full rollout
- Many candidates needed for optimization (100s–1000s)
- Predictor can be orders of magnitude faster for simple queries

Trade-off: predictor is less accurate than emulator but much faster.
Use predictor for candidate filtering, emulator for validation.

### Why Separate from Emulator?

1. **Speed:** Specialized physics kernel can skip non-physics state
2. **Portability:** Can run on different platforms / no ROM required for some
   use cases
3. **Isolation:** Planning code doesn't depend on emulator setup
4. **Testing:** Stub predictor allows offline tests without ROM

### Accuracy vs Speed

- **StubPredictor:** Fast, deterministic, not accurate — for tests only
- **SmRevClient:** Accurate Super Metroid physics, fast kernel — for
  planning
- **Full Emulator:** Gold standard, but expensive — for validation

## Future Work

1. **sm_rev integration:** Connect to real MiniStep predictor when available
2. **State loading:** Parse emulator save states for `SimState` extraction
3. **Input tape loading:** Support existing hop/tape formats from
   `human_tape` module
4. **Route planning consumer:** Wire into `map_planning.py` for trajectory-
   aware hop evaluation
5. **SMEDIT panel:** Live trajectory preview in route editor
6. **Optimization harness:** Genetic/hill-climbing search over input tapes

## Related

- **sm_rev:** `vinchinzu/sm_rev` sibling repository (MiniStep kernel)
- **door_kinematics.py:** Existing kinematics framework for door transitions
- **takeoff.py:** Existing takeoff windows for in-room hops
- **map_planning.py:** Route planning over editor graphs
- **human_tape:** Input tape recording/replay infrastructure
