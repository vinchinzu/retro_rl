# Physics Predictor Integration for Hop/Takeoff Planning

This document describes how hop and takeoff planning uses `PhysicsPredictor` for trajectory-based evaluation.

## Overview

The physics predictor enables route planning to evaluate hop candidates **without full emulation**. Planners query predicted trajectories to assess feasibility, optimize input sequences, and validate kinematics constraints.

### Important: Predictor vs Emulator Ground Truth

**StubPredictor and MiniStep are for search speed only, not ground truth.**

- **Planning phase**: Use predictor for fast candidate evaluation
- **Validation phase**: Run winning inputs on real emulator (stable-retro / SMEDIT snes9x)
- **Conflict resolution**: If predictor and emulator disagree, **emulator wins**
- **Room-clear claims**: Require emulator validation path (skip/xfail without ROM in CI is fine)

This separation allows fast offline planning while ensuring production trajectories are emulator-verified.

## Architecture

```
Planning Layer (takeoff.py, door_kinematics.py, map_planning.py)
          ↓
   hop_planning.py (TrajectoryEvaluator)
          ↓
   physics_sim.py (PhysicsPredictor protocol)
          ↓
   StubPredictor (offline) | SmRevClient (production)
```

## Core Components

### `hop_planning.py`

Main integration module providing:

- **`TrajectoryEvaluator`**: Wraps a `PhysicsPredictor` and provides planning-level APIs
- **`HopCandidate`**: Data structure combining hop intent + predicted trajectory
- **Convenience functions**: `evaluate_hop_trajectory()`, `evaluate_takeoff_trajectory()`

### Key APIs

```python
from super_metroid.hop_planning import TrajectoryEvaluator, evaluate_hop_trajectory
from super_metroid.physics_sim import StubPredictor, FrameInput
from super_metroid.takeoff import TakeoffWindow

# Create evaluator (defaults to StubPredictor for tests)
evaluator = TrajectoryEvaluator()

# Or use custom predictor
from super_metroid.physics_sim import SmRevClient
evaluator = TrajectoryEvaluator(SmRevClient())

# Evaluate a hop candidate
takeoff = TakeoffWindow((100, 120), "RIGHT")
start_state = SimState.from_sm_state(current_state)
inputs = [FrameInput(buttons=0x80) for _ in range(20)]  # RIGHT for 20 frames

candidate = evaluator.evaluate_hop(
    takeoff,
    start_state,
    inputs,
    target_x_range=(140, 160),  # Optional feasibility check
    target_y_range=(180, 200),
)

if candidate.feasible:
    print(f"Hop lands at ({candidate.final_x}, {candidate.final_y})")
    print(f"Trajectory: {len(candidate.trajectory.frames)} frames")
else:
    print(f"Infeasible: {candidate.reason}")
```

### Door Transition Evaluation

```python
from super_metroid.door_kinematics import DoorKinematics

# Predict trajectory from door leave/entry kinematics
door_kin = DoorKinematics.from_state(state)
inputs = [FrameInput(buttons=0x80) for _ in range(30)]

trajectory = evaluator.evaluate_door_transition(door_kin, inputs)

# Check if trajectory reaches target
final_frame = trajectory.frames[-1]
if target_x_lo <= final_frame.samus_x <= target_x_hi:
    print("Door transition feasible")
```

## Testing Strategy

### Offline Tests (No ROM)

All tests use `StubPredictor` by default:

```bash
# Run hop_planning tests
uv run pytest snes/super_metroid/tests/test_hop_planning.py -v

# Run physics_sim tests
uv run pytest snes/super_metroid/tests/test_physics_sim.py -v
```

**Test coverage proves:**
1. Planning calls `PhysicsPredictor.predict()`
2. Trajectory results affect feasibility assessment
3. Custom predictors can be injected
4. Door kinematics → trajectory conversion works

**These tests validate the planning layer only**, not physics accuracy.

### Emulator Validation (Ground Truth)

Room-clear claims require emulator validation:

```python
# 1. Plan with predictor (fast search)
evaluator = TrajectoryEvaluator(StubPredictor())
candidate = evaluator.evaluate_hop(takeoff, start, inputs)

if candidate.feasible:
    # 2. Validate on real emulator (ground truth)
    result = run_on_emulator(candidate.inputs)
    if result.success:
        print("Room clear: emulator-verified")
    else:
        print("Predictor was wrong, discard candidate")
```

**Emulator validation tests** can use `pytest.mark.skipif(not ROM_AVAILABLE)` or `xfail` to pass in CI without ROM.

### Production Predictors

When `SM_REV_PATH` is available:

```python
from super_metroid.physics_sim import load_predictor

# Auto-detect backend (still needs emulator validation)
predictor = load_predictor("sm_rev")  # Uses SM_REV_PATH env var
evaluator = TrajectoryEvaluator(predictor)
```

## Integration Points

### 1. Takeoff Window Evaluation

`takeoff.py` defines `TakeoffWindow` and `PlatformHop`. Planning can now:

```python
from super_metroid.hop_planning import evaluate_takeoff_trajectory

# Evaluate platform hop from position
candidate = evaluate_takeoff_trajectory(
    takeoff_window,
    start_x=100,
    start_y=200,
    inputs=button_sequence,
)
```

### 2. Door Kinematics

`door_kinematics.py` provides `DoorKinematics` snapshots. Planning can predict post-door trajectories:

```python
evaluator = TrajectoryEvaluator()
trajectory = evaluator.evaluate_door_transition(door_leave_kin, inputs)
```

### 3. Map Planning

`map_planning.py` can use trajectory evaluation for route feasibility:

```python
# Future: Query trajectory for edge cost estimation
# edge_cost = estimate_edge_cost(evaluator, edge, inventory)
```

## Predictor Backends

### StubPredictor (Default)

- **Purpose**: Fast search, offline testing, CI
- **Physics**: Simplified (linear motion, fake gravity)
- **Deterministic**: Yes (same inputs → same trajectory)
- **ROM required**: No
- **Ground truth**: ❌ No - for search speed only

```python
from super_metroid.physics_sim import StubPredictor

predictor = StubPredictor(name="test")
```

**Use for**: Candidate filtering, unit tests, CI without ROM

**Do not use for**: Final validation, room-clear claims

### SmRevClient (Production)

- **Purpose**: Better physics approximation for search
- **Physics**: MiniStep-based (closer to SM, but still approximate)
- **Backend**: External `sm_rev` binary (subprocess)
- **ROM required**: Yes (sm_rev needs ROM)
- **Ground truth**: ❌ No - still needs emulator validation

```python
from super_metroid.physics_sim import SmRevClient

predictor = SmRevClient()  # Uses SM_REV_PATH env var
# or
predictor = SmRevClient(binary_path="/path/to/sm_rev")
```

**Use for**: Better search heuristics than StubPredictor

**Do not use for**: Final validation without emulator confirmation

### Emulator (Ground Truth)

- **Purpose**: Final validation, room-clear claims
- **Physics**: Real Super Metroid (stable-retro / SMEDIT snes9x)
- **Deterministic**: Yes (real game physics)
- **ROM required**: Yes
- **Ground truth**: ✅ Yes - authoritative

**Use for**: All production claims, room-clear validation

**Conflict resolution**: When predictor and emulator disagree, emulator wins.

## Design Principles

1. **Protocol-based**: `PhysicsPredictor` is an ABC; new backends just implement `predict()`
2. **Test-first**: All planning tests work offline with `StubPredictor`
3. **Layered**: Planning imports `hop_planning`, not `physics_sim` directly
4. **Trajectory as data**: Results are `Trajectory` dataclasses (serializable, inspectable)
5. **Search vs validation split**: Predictor for fast search, emulator for ground truth
6. **Emulator authority**: When predictor and emulator disagree, emulator wins

## Future Extensions

- **Batch prediction**: Evaluate multiple candidates in parallel
- **Trajectory caching**: Memoize common input sequences
- **Cost functions**: Score trajectories by frame count, energy usage, etc.
- **Constraint satisfaction**: Find input sequences satisfying multi-constraint hops

## Related Files

- `snes/super_metroid/hop_planning.py` - This integration layer
- `snes/super_metroid/physics_sim.py` - Predictor protocol + backends
- `snes/super_metroid/takeoff.py` - Takeoff windows, platform hops
- `snes/super_metroid/door_kinematics.py` - Door transition kinematics
- `snes/super_metroid/tests/test_hop_planning.py` - Integration tests
- `snes/super_metroid/tests/test_physics_sim.py` - Predictor tests

## Example: Evaluating Multiple Hop Variants

```python
from super_metroid.hop_planning import TrajectoryEvaluator
from super_metroid.physics_sim import FrameInput

evaluator = TrajectoryEvaluator()

# Try different input sequences for same takeoff
candidates = []
for num_frames in [15, 20, 25, 30]:
    inputs = [FrameInput(buttons=0x80) for _ in range(num_frames)]
    candidate = evaluator.evaluate_hop(
        takeoff_window,
        start_state,
        inputs,
        target_x_range=(target_x, target_x + 16),
        target_y_range=(target_y, target_y + 16),
    )
    candidates.append(candidate)

# Pick shortest feasible
feasible = [c for c in candidates if c.feasible]
if feasible:
    best = min(feasible, key=lambda c: c.frames)
    print(f"Best hop: {best.frames} frames to ({best.final_x}, {best.final_y})")
```

## CI / Test Requirements

### Unit Tests (Always Run)

- **Offline tests only**: No ROM, no sm_rev binary
- **Fast**: StubPredictor is deterministic and instant
- **Coverage**: Proves predictor is called and results are used
- **No mocks**: Use real `StubPredictor` (builder pattern)

### Emulator Validation Tests (ROM Required)

- **Mark with**: `@pytest.mark.skipif(not ROM_AVAILABLE)` or `@pytest.mark.xfail`
- **Purpose**: Validate winning candidates on real emulator
- **Ground truth**: Emulator result is authoritative
- **CI**: Can skip/xfail without ROM

Example:

```python
import pytest
from super_metroid.emulator import ROM_AVAILABLE, run_on_emulator

@pytest.mark.skipif(not ROM_AVAILABLE, reason="ROM not available")
def test_ceres_morph_room_clear():
    """Validate Ceres→Morph trajectory on real emulator."""
    candidate = plan_ceres_to_morph()  # Uses predictor
    result = run_on_emulator(candidate.inputs)  # Ground truth
    assert result.room_id == MORPH_ROOM_ID
```

## Questions / Support

See `snes/super_metroid/AGENTS.md` for workflow and repo layout.
Physics predictor landed on PR #1 (`cursor/sm-physics-predictor-a0fa`).
Planning integration is PR #2 (`cursor/sm-planning-predictor-1a41`).
