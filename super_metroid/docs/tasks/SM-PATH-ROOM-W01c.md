# TASK SM-PATH-ROOM-W01c: Path-room clear – Speed Booster Hall

## Recipe step
primitive promote | docs (room practice)

## Model
Flash

## Wave type
implement

## Own files only
- `policies/room_clears/room_acf0_from_ad1b_to_b07a.json` (create or edit)
- entry fixture for **this problem only** if needed
- optional residual: `docs/tasks/SM-PATH-ROOM-W01c-residual.md`

## Context (minimal)
- Dual-track only — never continuous evidence
- Problem: `room_acf0_from_ad1b_to_b07a` · room `0xACF0` Speed Booster Hall
- PATH_ROOM_BOARD open; completion-path priority

## Read first
- `docs/routes/ROOM_WORK_QUEUE.md`
- `docs/research/PATH_ROOM_BOARD.md`
- `scripts/room/run_problem.py`

## Do
1. Scaffold or load Speed Booster Hall problem.
2. Verified room-clear policy from doorway-natural entry.
3. Residual with next card ID if needed.

## Do not
- Touch continuous spine or STATUS
- Edit sibling path-room policies

## Acceptance
- [ ] Isolated run green **or** honest residual with pin
- [ ] Residual next card ID + one change

## Verify commands
```bash
uv run python super_metroid/scripts/room/run_problem.py teleport room_acf0_from_ad1b_to_b07a
uv run python super_metroid/scripts/room/run_problem.py run room_acf0_from_ad1b_to_b07a
```

## Done when
Residual filed. Dual-track non-claim required.
