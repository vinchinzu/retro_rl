# TASK SM-ROOM-METAL: Practice — Metal Pirates Room (harder easy)

## Recipe step
room practice (dual-track; combat-ish room)

## Model
Luna

## Wave type
implement

## Own files only
- problem-local: `room_b62b_from_b482_to_b5d5`
- optional note

## Context
- Queue rank ~56: Metal Pirates `0xB62B`, teleport yes.
- Harder than pure traversal rooms — practice track only.

## Do
1. Scaffold/teleport/run; promote only if green isolate.
2. Residual honest pin if combat stalls.

## Verify
```bash
uv run python super_metroid/scripts/room/run_problem.py teleport room_b62b_from_b482_to_b5d5
uv run python super_metroid/scripts/room/run_problem.py run room_b62b_from_b482_to_b5d5
```
