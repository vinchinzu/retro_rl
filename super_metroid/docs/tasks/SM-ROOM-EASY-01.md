# TASK SM-ROOM-EASY-01: Practice track — Blue Brinstar Boulder Room

## Recipe step
room practice (dual-track — **never** continuous evidence)

## Model
Luna

## Wave type
implement

## Own files only
- policy / problem artifacts for **only**:
  `room_a1ad_from_9f64_to_a1d8` under `policies/room_clears/` and any
  problem-local notes
- optional: `docs/tasks/SM-ROOM-EASY-01-note.md`

Do **not** edit continuous, STATUS, kpdr spine controllers, progression.

## Context
- ROOM_WORK_QUEUE top open easy: Blue Brinstar Boulder `0xA1AD`
  problem `room_a1ad_from_9f64_to_a1d8` (teleport yes).
- Dual-track: practice greens ≠ KPDR continuous integrity.

## Read first
- `docs/routes/ROOM_WORK_QUEUE.md`
- `scripts/room/run_problem.py` (scaffold / teleport / run / promote)
- `rooms/` practice helpers as needed

## Do
1. Scaffold if missing; bootstrap/teleport if needed.
2. Iterate policy until isolated run green **or** residual with pin.
3. Promote only on green isolated run (practice promote, not STATUS).
4. Residual: next easy problem rank.

## Do not
- Claim continuous
- Edit spine controllers

## Acceptance
- [ ] Isolated run green **or** honest residual
- [ ] Note dual-track non-claim

## Verify
```bash
uv run python super_metroid/scripts/room/run_problem.py teleport room_a1ad_from_9f64_to_a1d8
uv run python super_metroid/scripts/room/run_problem.py run room_a1ad_from_9f64_to_a1d8
# promote only if green:
# uv run python super_metroid/scripts/room/run_problem.py run room_a1ad_from_9f64_to_a1d8 --promote
```
