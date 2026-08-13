# First Ceres Room Fixture

## Overview

This PR adds the first Ceres room hop/tape fixture: **Ceres Elevator (0xDF45) → Falling Tile Room (0xDF8D)**. This is the first room of the overnight bar: Ceres → Morph → Bomb, searched with the hop planning layer and validated on the real emulator.

## What's Included

### 1. Fixture Module (`first_room_fixture.py`)

Located at: `snes/super_metroid/routes/kpdr/ceres/first_room_fixture.py`

Provides:
- `CeresFirstRoomFixture`: Data structure for searched hop/tape
- `search_ceres_first_room()`: Search function using `TrajectoryEvaluator`
- `validate_ceres_first_room()`: Emulator validation path

### 2. Tests (`test_ceres_first_room.py`)

Located at: `snes/super_metroid/tests/test_ceres_first_room.py`

Test coverage:
- **Offline search tests** (no ROM): Use `StubPredictor` for search speed
- **Emulator validation tests** (ROM required): Skip without ROM using `@pytest.mark.skipif(not ROM_AVAILABLE)`
- **Fixture data structure tests**: Verify `room_clear` property never claims success without emulator validation

### 3. Integration with Ceres Package

Updated `snes/super_metroid/routes/kpdr/ceres/__init__.py` to export:
- `CeresFirstRoomFixture`
- `search_ceres_first_room`
- `validate_ceres_first_room`

## Policy

### Search vs. Emulator Ground Truth

- **Search layer**: Uses `PhysicsPredictor` (default `StubPredictor`, or `sm_rev_predict` if present) for speed
- **Emulator validation**: Uses `validate_trajectory_on_emulator` (stable-retro / SMEDIT snes9x) for ground truth
- **Room-clear claims**: **Never** claim room-clear from Mini/stub alone. Emulator wins.

### Testing Without ROM

Tests are designed to pass without ROM:
- Offline search tests use `StubPredictor` (no ROM required)
- Emulator validation tests use `@pytest.mark.skipif(not ROM_AVAILABLE)` and skip gracefully

Run offline tests:
```bash
uv run pytest snes/super_metroid/tests/test_ceres_first_room.py -v
```

## Architecture

### Search Flow

```
1. search_ceres_first_room(predictor=StubPredictor)
   ├─> TrajectoryEvaluator
   ├─> evaluate_takeoff_trajectory()
   └─> CeresFirstRoomFixture (emulator_validated=False)

2. validate_ceres_first_room(fixture, start_state_path)
   ├─> validate_trajectory_on_emulator()
   └─> CeresFirstRoomFixture (emulator_validated=True, emulator_success=True/False)
```

### Key Properties

- `CeresFirstRoomFixture.predictor_feasible`: Heuristic from predictor
- `CeresFirstRoomFixture.emulator_validated`: True if run on real emulator
- `CeresFirstRoomFixture.room_clear`: **Ground truth** — only True when `emulator_validated AND emulator_success`

## Future Work

1. **Start State**: Add Ceres Elevator start state to `custom_integrations/SuperMetroid-Snes/`
2. **Search Improvements**: Replace greedy search with A* or beam search
3. **Input Variants**: Try A/B jumps, not just straight RIGHT movement
4. **Export Format**: Add smedit-tas-1 tape export for SMEDIT integration
5. **Remaining Ceres Rooms**: Extend to Falling → Magnet → Scientist → Flat → Ridley

## Status

- ✅ Fixture module created
- ✅ Tests written (offline tests pass)
- ✅ Package integration complete
- ⏳ ROM validation tests (skip without ROM)
- ⏳ Start state capture needed
- ⏳ Real search needs improvement (current: greedy RIGHT-only)

## Room IDs

```python
ROOM_CERES_ELEVATOR = 0xDF45  # Starting room
ROOM_CERES_FALLING = 0xDF8D   # First Ceres room (target)
```

## Notes

- This PR focuses on **infrastructure** for the first room, not a final solved trajectory
- Search is intentionally simplified (greedy RIGHT) to demonstrate the workflow
- Emulator validation requires ROM + start state (not included in this PR)
- No room-clear claim without emulator validation
