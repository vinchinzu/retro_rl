# TASK SM-ROOM-SCAFFOLD-SPAZER: Scaffold Spazer Room (post BOOT-01)

## Recipe step
room practice scaffold (dual-track)

## Model
Luna

## Wave type
implement

## Own files only
- problem-local: `room_a447_from_a408_to_a408` policy under `policies/room_clears/`
- optional note: `docs/tasks/SM-ROOM-SCAFFOLD-SPAZER-note.md`

## Context
- BOOT-01 green: teleport fixture ready `room_a447_from_a408.state` → `0xA447`
- Practice only; not continuous Spazer tip.

## Do
1. Scaffold + run isolate; promote only if green.
2. Residual pin if red.

## Verify
```bash
uv run python super_metroid/scripts/room/run_problem.py scaffold room_a447_from_a408_to_a408
uv run python super_metroid/scripts/room/run_problem.py teleport room_a447_from_a408_to_a408
uv run python super_metroid/scripts/room/run_problem.py run room_a447_from_a408_to_a408
```
