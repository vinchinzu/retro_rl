# TASK SM-PATH-ROOM-W01d: Path-room clear – Single Chamber

## Recipe step
primitive promote | docs (room practice)

## Model
Flash

## Wave type
implement

## Own files only
- `policies/room_clears/room_ad5e_from_b656_to_ae07.json` (create or edit)
- entry fixture for **this problem only** if needed
- optional residual: `docs/tasks/SM-PATH-ROOM-W01d-residual.md`

## Context (minimal)
- Dual-track only — never continuous evidence
- Problem: `room_ad5e_from_b656_to_ae07` · room `0xAD5E` Single Chamber
- PATH_ROOM_BOARD open; completion-path / Wave approach geometry
- Tough tier — honest RED residual is acceptable

## Read first
- `docs/routes/ROOM_WORK_QUEUE.md`
- `docs/research/PATH_ROOM_BOARD.md`
- `scripts/room/run_problem.py`

## Do
1. Scaffold or load Single Chamber problem.
2. Iterate isolated run; promote only on green.
3. Residual with next card ID (same-room R1 or next path room).

## Do not
- Touch continuous spine or STATUS
- Forge progression/capacity for green claims

## Acceptance
- [ ] Isolated run green **or** honest residual with pin
- [ ] Residual next card ID + one change

## Verify commands
```bash
uv run python super_metroid/scripts/room/run_problem.py teleport room_ad5e_from_b656_to_ae07
uv run python super_metroid/scripts/room/run_problem.py run room_ad5e_from_b656_to_ae07
```

## Done when
Residual filed. Dual-track non-claim required.
