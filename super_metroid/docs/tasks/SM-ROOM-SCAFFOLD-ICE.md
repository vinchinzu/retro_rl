# TASK SM-ROOM-SCAFFOLD-ICE: Scaffold Ice Beam Room (post BOOT-01)

## Recipe step
room practice scaffold (dual-track)

## Model
Luna

## Wave type
implement

## Own files only
- problem-local: `room_a890_from_a8b9_to_a8b9` policy under `policies/room_clears/`
- optional note: `docs/tasks/SM-ROOM-SCAFFOLD-ICE-note.md`

## Context
- BOOT-01 green: teleport fixture ready `room_a890_from_a8b9.state` → `0xA890`
- Item room practice; not continuous Ice collection evidence.

## Do
1. Scaffold + run isolate; promote only if green.
2. Residual pin if red.

## Verify
```bash
uv run python super_metroid/scripts/room/run_problem.py scaffold room_a890_from_a8b9_to_a8b9
uv run python super_metroid/scripts/room/run_problem.py teleport room_a890_from_a8b9_to_a8b9
uv run python super_metroid/scripts/room/run_problem.py run room_a890_from_a8b9_to_a8b9
```
