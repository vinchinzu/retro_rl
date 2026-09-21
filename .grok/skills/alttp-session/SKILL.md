---
name: alttp-session
description: >
  ALttP sitting gates: one spine bead, one living residual, RAM glance leave.
  Use when working in snes/alttp, starting a Sanctuary hop, room_engine edge,
  or running /alttp-session.
---

# ALttP session

Planner owns `docs/STATUS.md`. State-load greens are not continuous claims.
Size: [CODING_STANDARDS.md](../../../CODING_STANDARDS.md) (~1000 LOC, merge or delete).

## Loop

1. `bd ready -l alttp -l spine` — claim **exactly one**.
   Done: that bead is `in_progress`.
2. Overwrite the living residual named in [`snes/alttp/AGENTS.md`](../../../snes/alttp/AGENTS.md)
   for this bead under `snes/alttp/docs/tasks/`.
   Done: that file holds this sitting's leftover.
3. Drive from `docs/routes/ROOM_WORK_QUEUE.md` + `escape_graph` blockers.
   Geometry in `maps/room_XX.json`. Play via `scripts/room_engine.py`.
   Leave proof: RAM glance (room hex, module/submodule, x/y band, sword,
   `$F3CC` follower, keys) — no MP4. Save-state pins are not leave proof.
   Done: glance matches that leftover.

Natural-entry: a hop is route-ready only from the real predecessor continuous
state. Ladder: `planned` → `isolated` → `natural_entry` → `continuous`.

## Skills

| Job | Skill |
|-----|-------|
| Occupancy / RAM-claim halt | `predict-path` |

## Non-claims (every residual)

Did not STATUS-promote. Did not treat a save-state pin as power-on. Did not
write `$F3CC` / sword / keys. Did not copy approach xy into Python. Did not
overwrite `recordings/verified_tip_run.json` on a red.
