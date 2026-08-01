# TASK SM-PINK-PB-HARD: Pink PB maze pure residual (parked KPDR epic)

## Recipe step
1 pure controller (hard parked geometry — continuous never required)

## Model
Luna

## Wave type
implement

## Own files only
- `routes/kpdr/pink_pb_maze.py`
- optional residual: `docs/tasks/SM-PINK-PB-HARD-residual.md`

## Context
- Pink PB `0x9E11` is **parked** on project KPDR (Alpha PB preferred).
- Still valuable dual-track / optional backfill pure work.
- Existing play helpers in `pink_pb_maze.py` — harden mid-maze or collect hop
  with one knob; pure green **nice**, residual acceptable.
- Dev states: `dev_b1_pb_door_entered.state` / related b1 states if present.

## Read first
- `routes/kpdr/pink_pb_maze.py`
- `routes/kpdr/pb_door.py` (hint only)
- `docs/routes/ROUTE_KPDR.md` (parked note)

## Do
1. Pick **one** failure mode (mid-maze stall, wall, morph bomb collect) and
   apply one bounded fix.
2. If source state available, run pure probe; else residual “needs capture”.
3. Do not attach to continuous spine or STATUS.

## Acceptance
- [ ] Code change + residual
- [ ] Tests that import module still pass if any; else controller_common green
- [ ] Explicit parked / not continuous

## Verify
```bash
uv run pytest super_metroid/tests/test_post_spore_controller.py -q
# pure only if you have a named source:
# uv run python super_metroid/scripts/probe/kpdr.py pure <segment> --source <state>
```
