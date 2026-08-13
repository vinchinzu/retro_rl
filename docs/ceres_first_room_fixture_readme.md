# First Ceres Room Fixture

## Overview

This PR adds the first Ceres room hop/tape fixture: **Ceres Elevator (0xDF45) → Falling Tile Room (0xDF8D)**. This is the first room of the overnight bar: Ceres → Morph → Bomb, using the **real existing tape** from `routes.kpdr.ceres.outbound`, validated on the real emulator.

## What's Included

### 1. Fixture Module (`first_room_fixture.py`)

Located at: `snes/super_metroid/routes/kpdr/ceres/first_room_fixture.py`

Provides:
- `CeresFirstRoomFixture`: Data structure for tape with emulator validation results
- `get_ceres_first_room_tape()`: Extract real tape from existing outbound route
- `validate_ceres_first_room()`: Emulator validation path (uses `SM_CERES_ELEV_STATE` env var)
- `button_names_to_mask()`: Convert button names to button mask for FrameInput

### 2. Tests (`test_ceres_first_room.py`)

Located at: `snes/super_metroid/tests/test_ceres_first_room.py`

Test coverage:
- **Button conversion tests** (offline): Verify button name → mask conversion
- **Tape extraction tests** (offline, no ROM): Verify tape comes from real source
- **Emulator validation tests** (ROM + start state required): Skip gracefully without prerequisites
- **Fixture data structure tests**: Verify `room_clear` property never claims success without emulator validation

### 3. Integration with Ceres Package

Updated `snes/super_metroid/routes/kpdr/ceres/__init__.py` to export:
- `CeresFirstRoomFixture`
- `get_ceres_first_room_tape`
- `validate_ceres_first_room`

## Tape Source

### Real Tape (NOT Greedy Search)

The tape is extracted from the existing product route in `snes/super_metroid/routes/kpdr/ceres/outbound.py`:
- Function: `_ceres_outbound_to_scientist_spans()`
- Covers: Elevator → Falling → Magnet → Scientist
- This fixture: Takes documented prefix for Elevator → Falling only

ActionSpan sequence (first room):
```python
(("RIGHT", "A"), 24),    # Jump start
(("RIGHT",), 120),        # Run right
(("LEFT",), 120),         # Turn left
(("RIGHT", "B"), 60),     # Dash right (simplified, no arm-pump)
# Total: 324 frames
```

**NOT invented**:
- ❌ No greedy RIGHT search
- ❌ No invented physics
- ✅ Real tape from existing route
- ✅ Converted ActionSpan → FrameInput

## Policy

### Search vs. Emulator Ground Truth

- **Tape source**: Real product route from `outbound.py`
- **Emulator validation**: Uses `validate_trajectory_on_emulator` (stable-retro / SMEDIT snes9x) for ground truth
- **Room-clear claims**: **Never** claim room-clear without emulator validation
- **Mini/stub**: Not used (real tape, not search)

### Start State: Environment Variable

**Do NOT commit .state or ROM blobs to repo.**

Emulator validation uses env var `SM_CERES_ELEV_STATE`:
```bash
export SM_CERES_ELEV_STATE="/path/to/ceres_elevator.state"
uv run pytest snes/super_metroid/tests/test_ceres_first_room.py -v
```

Tests skip gracefully if:
- `ROM_AVAILABLE` is False
- `SM_CERES_ELEV_STATE` not set
- Start state file does not exist

### Testing Without ROM

Offline tests pass without ROM:
```bash
# Offline tests (button conversion, tape extraction)
uv run pytest snes/super_metroid/tests/test_ceres_first_room.py::TestButtonConversion -v
uv run pytest snes/super_metroid/tests/test_ceres_first_room.py::TestCeresFirstRoomTape -v

# All tests (validation skips without ROM)
uv run pytest snes/super_metroid/tests/test_ceres_first_room.py -v
```

## Architecture

### Tape Flow

```
1. get_ceres_first_room_tape()
   ├─> Extract from outbound._ceres_outbound_to_scientist_spans()
   ├─> Convert ActionSpan → FrameInput via button_names_to_mask()
   └─> CeresFirstRoomFixture (emulator_validated=False)

2. validate_ceres_first_room(fixture)
   ├─> Read SM_CERES_ELEV_STATE env var
   ├─> validate_trajectory_on_emulator()
   └─> CeresFirstRoomFixture (emulator_validated=True, emulator_success=True/False)
```

### Key Properties

- `CeresFirstRoomFixture.tape_source`: Documents source (not "search" or "greedy")
- `CeresFirstRoomFixture.emulator_validated`: True if run on real emulator
- `CeresFirstRoomFixture.room_clear`: **Ground truth** — only True when `emulator_validated AND emulator_success`

## Room IDs

```python
ROOM_CERES_ELEVATOR = 0xDF45  # Starting room
ROOM_CERES_FALLING = 0xDF8D   # First Ceres room (target)
```

## Status

- ✅ Fixture module created (real tape, not greedy)
- ✅ Tests written (offline tests pass)
- ✅ Package integration complete
- ✅ Tape source documented (`outbound.py`)
- ✅ Env var `SM_CERES_ELEV_STATE` for start state
- ⏳ ROM validation tests (skip without ROM)
- ⏳ Start state capture needed (not in this PR, use env var)

## Future Work

1. **Start State Capture**: Document how to capture Ceres Elevator start state (not committed to repo)
2. **Full Tape**: Expand to include arm-pump expansion (currently simplified)
3. **Export Format**: Add smedit-tas-1 tape export for SMEDIT integration
4. **Remaining Ceres Rooms**: Extend to Falling → Magnet → Scientist → Flat → Ridley

## Notes

- This PR uses **real existing tape** from `outbound.py`, not invented physics
- No greedy search or StubPredictor search
- Emulator validation requires ROM + start state (via `SM_CERES_ELEV_STATE`)
- No room-clear claim without emulator validation
- No .state or ROM blobs committed to repo
