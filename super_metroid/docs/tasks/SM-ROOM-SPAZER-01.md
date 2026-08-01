# TASK SM-ROOM-SPAZER-01: Spazer Room — exit/collect residual after scaffold

## Recipe step
room practice residual (dual-track)

## Model
Luna

## Wave type
implement

## Own files only
- `policies/room_clears/room_a447_from_a408_to_a408.json` only
- optional note: `docs/tasks/SM-ROOM-SPAZER-01-note.md`

## Context
- SCAFFOLD-SPAZER **RED**: pin `room=0xA447 pose=138 x=85 y=187 door_transition=0`
- One knob on collect/return approach.
- Practice only.

## Do
1. One knob; promote only if green isolate.
2. Residual with pin if red.

## Verify
```bash
uv run python super_metroid/scripts/room/run_problem.py run room_a447_from_a408_to_a408
```
