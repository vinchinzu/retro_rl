# TASK SM-PATH-ROOM-W01a: Path-room clear – Frog Speedway

## Recipe step
primitive promote | docs (room practice)

## Model
Flash

## Wave type
implement

## Own files only
- `policies/room_clears/room_b106_from_af72_to_b167.json` (create or edit)
- entry fixture under `custom_integrations/SuperMetroid-Snes/` for **this
  problem only** if bootstrap needed
- optional residual: `docs/tasks/SM-PATH-ROOM-W01a-residual.md`

## Context (minimal)
- Dual-track only — **never** continuous evidence
- Problem: `room_b106_from_af72_to_b167` · room `0xB106` Frog Speedway
- PATH_ROOM_BOARD: open path rooms; prioritize completion-path geometry
- Use doorway-natural entry (door-warp settle then play) for practice only

## Read first
- `docs/routes/ROOM_WORK_QUEUE.md`
- `docs/research/PATH_ROOM_BOARD.md` (Frog Speedway row)
- `scripts/room/run_problem.py`
- `docs/tasks/SM-ROOM-SEG-01.md` (card style)

## Do
1. Bootstrap / teleport fixture if missing for this problem only.
2. Scaffold policy if missing; iterate isolated `run` until green or honest pin.
3. Promote only on green isolated run (`--promote`).
4. Residual with next room card ID if more geometry needed (`SM-PATH-ROOM-W01a-R1`
   or next path-room card).

## Do not
- Touch any kpdr continuous controllers or `continuous.py`
- Claim continuous / STATUS
- Edit another problem’s policy

## Acceptance
- [ ] `run_problem.py run` green for this room **or** honest residual with pin
- [ ] Residual with next card ID + one change
- [ ] Dual-track non-claim

## Verify commands
```bash
uv run python super_metroid/scripts/room/run_problem.py teleport room_b106_from_af72_to_b167
uv run python super_metroid/scripts/room/run_problem.py run room_b106_from_af72_to_b167
# promote only if green:
# uv run python super_metroid/scripts/room/run_problem.py run room_b106_from_af72_to_b167 --promote
```

## Done when
Residual filed. Planner owns queue refresh; never continuous compose.
