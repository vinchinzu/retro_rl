# TASK SM-ROOM-SCAFFOLD-HOPPER: Scaffold Hopper Energy Tank Room

## Recipe step
room practice scaffold (dual-track)

## Model
Luna

## Wave type
implement

## Own files only
- problem-local: `room_a15b_from_a130_to_a130` policy under `policies/room_clears/`
- optional note: `docs/tasks/SM-ROOM-SCAFFOLD-HOPPER-note.md`

## Context
- BOOT-01 green: teleport fixture ready `room_a15b_from_a130.state` → `0xA15B`
- Practice only.

## Do
1. Scaffold + run isolate; promote only if green.
2. Residual pin if red.

## Verify
```bash
uv run python super_metroid/scripts/room/run_problem.py scaffold room_a15b_from_a130_to_a130
uv run python super_metroid/scripts/room/run_problem.py teleport room_a15b_from_a130_to_a130
uv run python super_metroid/scripts/room/run_problem.py run room_a15b_from_a130_to_a130
```
