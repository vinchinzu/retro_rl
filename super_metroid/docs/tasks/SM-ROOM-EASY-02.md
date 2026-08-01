# TASK SM-ROOM-EASY-02: Practice track — Crab Hole

## Recipe step
room practice (dual-track — never continuous evidence)

## Model
Luna

## Wave type
implement

## Own files only
- problem-local only: `room_d21c_from_d3b6_to_d08a`
- optional note `docs/tasks/SM-ROOM-EASY-02-note.md`

No continuous / STATUS / kpdr / progression.

## Context
- Queue rank ~53: Crab Hole `0xD21C`, teleport ready.
- Maridia practice; not continuous Maridia entry.

## Read first
- `docs/routes/ROOM_WORK_QUEUE.md`
- `scripts/room/run_problem.py`

## Do
1. Scaffold / teleport / run.
2. Promote only if isolated green.
3. Residual next open easy.

## Acceptance
- [ ] Green isolate or residual
- [ ] Dual-track non-claim

## Verify
```bash
uv run python super_metroid/scripts/room/run_problem.py teleport room_d21c_from_d3b6_to_d08a
uv run python super_metroid/scripts/room/run_problem.py run room_d21c_from_d3b6_to_d08a
```
