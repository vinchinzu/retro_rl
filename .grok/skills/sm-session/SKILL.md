---
name: sm-session
description: >
  Super Metroid session loop: census first, machine search for red rooms,
  progress is rooms on the master tape. Use when working in
  snes/super_metroid, starting a hop, composing a tip, or running /sm-session.
---

# SM session

Planner owns `docs/STATUS.md`. Language: `snes/super_metroid/CONTEXT.md`.

## Loop

Rules and commands: `snes/super_metroid/AGENTS.md`. Open work:
`snes/super_metroid/docs/plan.md`.

1. Run the census row for the room
   (`scripts/tools/tape_census.py --rows N`). `as_recorded` or `retimed`
   means there is nothing to tune.
2. A red room gets a search or a rollout fight against the ROM: save,
   try candidates, restore. Do not edit one constant and re-run by hand.
3. A room counts when it clears from the previous room's real leave and
   its buttons are on the master tape.
4. Soft max ~1000 LOC: merge into the owner or delete
   (`CODING_STANDARDS.md`).
5. User says watch / headed: open a window first.
   `uv run python snes/super_metroid/scripts/probe/kpdr.py pure <hop> --source <pin> --headed`

## Skills

| Job | Skill |
|-----|-------|
| Movement hop | `sm-pure-hop` |
| Same-pin bench / wiki fight | `sm-room-policy` |
| Clean / no-assist fight | `sm-no-assist-boss` |
| TipSpec / SpineHop / `--to` | `sm-compose` |
| 4×4 room demo reel | `sm-room-grid` |

## Non-claims (every residual)

Did not STATUS-promote. Did not change `DEFAULT_CONTINUOUS_TIP`. Did not
overwrite `recordings/<tip>.json` on a red run. Did not forge progression RAM.
Did not treat a pin run as power-on Gravity.
