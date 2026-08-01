# TASK SM-ROOM-EASY-03: Practice — Grapple Tutorial Room 2

## Recipe step
room practice (dual-track)

## Model
Luna

## Wave type
implement

## Own files only
- problem-local: `room_abd2_from_ab64_to_ac00`
- optional note

## Context
- Queue open easy rank ~55: Grapple Tutorial 2 `0xABD2`, teleport yes.
- Ice tutorial is separate (SM-ROOM-ICE-TUT). Not continuous.

## Do
1. Scaffold / teleport / run.
2. Promote only on isolated green.
3. Residual next open easy (Metal Pirates `room_b62b_...` if still open).

## Verify
```bash
uv run python super_metroid/scripts/room/run_problem.py teleport room_abd2_from_ab64_to_ac00
uv run python super_metroid/scripts/room/run_problem.py run room_abd2_from_ab64_to_ac00
```
