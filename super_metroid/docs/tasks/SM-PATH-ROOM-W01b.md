# TASK SM-PATH-ROOM-W01b: Path-room clear – Bubble Mountain

## Recipe step
primitive promote | docs (room practice)

## Model
Flash

## Wave type
implement

## Own files only
- `policies/room_clears/room_acb3_from_b07a_to_aedf.json` (create or edit)
- entry fixture for **this problem only** if needed
- optional residual: `docs/tasks/SM-PATH-ROOM-W01b-residual.md`

## Context (minimal)
- Dual-track only — never continuous evidence
- Problem: `room_acb3_from_b07a_to_aedf` · room `0xACB3` Bubble Mountain
- PATH_ROOM_BOARD open; completion-path priority
- Doorway-natural entry for practice (not continuous spine)

## Read first
- `docs/routes/ROOM_WORK_QUEUE.md`
- `docs/research/PATH_ROOM_BOARD.md`
- `scripts/room/run_problem.py`
- `docs/tasks/SM-PATH-ROOM-W01a.md` (parallel sibling — do not edit its files)

## Do
1. Scaffold or load Bubble Mountain room problem.
2. Produce a verified room-clear policy from natural doorway entry.
3. Residual with next card ID if more geometry needed.

## Do not
- Touch kpdr continuous controllers or `continuous.py`
- Claim continuous / STATUS
- Edit W01a/c/d policies

## Acceptance
- [ ] `run_problem.py run` green **or** honest residual with pin
- [ ] Residual with next card ID + one change

## Verify commands
```bash
uv run python super_metroid/scripts/room/run_problem.py teleport room_acb3_from_b07a_to_aedf
uv run python super_metroid/scripts/room/run_problem.py run room_acb3_from_b07a_to_aedf
```

## Done when
Residual filed. Dual-track non-claim required.
