# First Ceres Room Fixture

## Overview

This PR adds the first Ceres room hop/tape fixture: **Ceres Elevator (0xDF45) → Falling Tile Room (0xDF8D)**. Uses the **EXACT prefix** from `routes.kpdr.ceres.outbound._ceres_outbound_to_scientist_spans` including full arm-pump expansion.

## Tape Source: EXACT Product Prefix

### Raw Spans (from outbound.py lines 35-39)

```python
(("RIGHT", "A"), 24, False),
(("RIGHT",), 120, False),
(("LEFT",), 120, False),
(("RIGHT", "B"), 240, True),  # Expands via _arm_pump_dash_spans
((), 60, False),
```

### Expansion Details

**RIGHT+B 240 frames with arm-pump (period=2):**
- `_arm_pump_dash_spans("RIGHT", 240, reason)` expands to:
- Frames 0-1: RIGHT+B+L (0x481)
- Frames 2-3: RIGHT+B+R (0x881)
- Frames 4-5: RIGHT+B+L (0x481)
- ... (120 spans of 2 frames each)

**Total: 564 frames**
- RIGHT+A: 24 frames
- RIGHT: 120 frames
- LEFT: 120 frames
- RIGHT+B (arm-pump): 240 frames
- Idle: 60 frames

## What's Included

### 1. Fixture Module (`first_room_fixture.py`)

- `get_ceres_first_room_tape()`: Extract EXACT tape with full arm-pump
- `validate_ceres_first_room()`: Emulator validation (uses `SM_CERES_ELEV_STATE`)
- `button_names_to_mask()`: Convert button names to FrameInput button mask

### 2. Tests (`test_ceres_first_room.py`)

Test coverage:
- **Button conversion**: Verify button name → mask (0x481, 0x881, etc.)
- **EXACT frame count**: Assert 564 frames (not shortened)
- **Arm-pump expansion**: Verify L↔R pattern at period=2 for 240 frames
- **Span boundaries**: First 24 RIGHT+A, last 60 idle
- **Emulator validation**: Skip gracefully without ROM + start state

Run tests:
```bash
# All offline tests (no ROM)
uv run pytest snes/super_metroid/tests/test_ceres_first_room.py::TestButtonConversion -v
uv run pytest snes/super_metroid/tests/test_ceres_first_room.py::TestCeresFirstRoomTape -v

# With ROM + start state
export SM_CERES_ELEV_STATE="/path/to/ceres_elevator.state"
uv run pytest snes/super_metroid/tests/test_ceres_first_room.py -v
```

## Policy

### Real Tape (NOT Search)

✅ **EXACT prefix from `outbound.py`** (not invented)
✅ **Full arm-pump expansion** via `_arm_pump_dash_spans` (not shortened)
✅ **564 frames** matching product spans
❌ **NOT greedy search or StubPredictor**
❌ **NOT shortened to 60 frames**

### Start State: Environment Variable

**Do NOT commit .state or ROM:**
- Use env var: `SM_CERES_ELEV_STATE="/path/to/ceres_elevator.state"`
- Tests skip gracefully without ROM_AVAILABLE or start state
- Emulator validation only when both present

### Room Clear Policy

- `room_clear` only True when `emulator_validated AND emulator_success`
- Never claim room-clear without emulator validation
- Predictor not used (real tape, not search)

## Architecture

### Button Masks

```python
# D-pad + face buttons
RIGHT = 0x80  # bit 7
LEFT = 0x40   # bit 6
A = 0x100     # bit 8
B = 0x01      # bit 0

# Shoulders (arm-pump)
L = 0x400     # bit 10
R = 0x800     # bit 11

# Combinations
RIGHT+A = 0x180
RIGHT+B+L = 0x481  # Arm-pump L
RIGHT+B+R = 0x881  # Arm-pump R
```

### Tape Flow

```
1. get_ceres_first_room_tape()
   ├─> EXACT raw spans from outbound.py
   ├─> Expand RIGHT+B 240 with arm-pump (L↔R, period=2)
   ├─> Convert to FrameInput via button_names_to_mask()
   └─> CeresFirstRoomFixture (564 frames, emulator_validated=False)

2. validate_ceres_first_room(fixture)
   ├─> Read SM_CERES_ELEV_STATE env var
   ├─> validate_trajectory_on_emulator()
   └─> CeresFirstRoomFixture (emulator_validated=True)
```

## Room IDs

```python
ROOM_CERES_ELEVATOR = 0xDF45  # Starting room
ROOM_CERES_FALLING = 0xDF8D   # First Ceres room (target)
```

## Status

- ✅ EXACT tape from `outbound.py` (564 frames)
- ✅ Full arm-pump expansion (NOT shortened)
- ✅ Tests verify frame count and arm-pump pattern
- ✅ Env var `SM_CERES_ELEV_STATE` for start state
- ✅ No ROM/.state blobs committed
- ✅ Never claim room_clear without emulator
- ⏳ Start state capture (not in this PR)

## Comparison: Product vs This Fixture

| Aspect | Product Spans | This Fixture |
|--------|--------------|--------------|
| Source | `_ceres_outbound_to_scientist_spans()` | EXACT prefix (lines 35-39) |
| RIGHT+B frames | 240 with arm-pump | 240 with arm-pump ✅ |
| Arm-pump expansion | Via `_arm_pump_dash_spans` | Same expansion ✅ |
| Total frames | 564 (prefix) | 564 ✅ |
| Frame count | EXACT | EXACT ✅ |

## Notes

- This is the **EXACT product tape prefix**, not a shortened sketch
- RIGHT+B 240 frames fully expanded via `_arm_pump_dash_spans` (period=2)
- Frame count: **564 frames** (verified in tests)
- No greedy search, no invented physics, no shortened spans
- Emulator validation uses `SM_CERES_ELEV_STATE` (not committed)
