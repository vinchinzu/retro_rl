# TASK SM-ROOM-SCAFFOLD-BILLY: Scaffold Billy Mays' Room

## Recipe step
room practice scaffold (dual-track)

## Model
Luna

## Wave type
implement

## Own files only
- problem-local: `room_a1d8_from_a1ad_to_a1ad` policy under `policies/room_clears/`
- optional note: `docs/tasks/SM-ROOM-SCAFFOLD-BILLY-note.md`

## Context
- BOOT-01 green: teleport fixture ready `room_a1d8_from_a1ad.state` → `0xA1D8`
- Practice only.

## Do
1. Scaffold + run isolate; promote only if green.
2. Residual pin if red.

## Verify
```bash
uv run python super_metroid/scripts/room/run_problem.py scaffold room_a1d8_from_a1ad_to_a1ad
uv run python super_metroid/scripts/room/run_problem.py teleport room_a1d8_from_a1ad_to_a1ad
uv run python super_metroid/scripts/room/run_problem.py run room_a1d8_from_a1ad_to_a1ad
```
