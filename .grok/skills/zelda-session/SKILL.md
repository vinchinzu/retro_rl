---
name: zelda-session
description: >
  Zelda I sitting gates: one spine bead, one living residual, halt after 3
  reds, glance leave. Use when working in nes/zelda_i, starting a Survival
  hop, composing the spine, or running /zelda-session.
---

# Zelda session

Survival greens are not Clean STATUS; planner owns `docs/STATUS.md`.
Size: [CODING_STANDARDS.md](../../../CODING_STANDARDS.md) (~1000 LOC, merge or delete).

## Loop

1. `bd ready -l zelda_i -l spine` — claim **exactly one**.
   Done: that bead is `in_progress`.
2. Overwrite the living residual named in [`nes/zelda_i/AGENTS.md`](../../../nes/zelda_i/AGENTS.md).
   Done: that file holds this sitting's leftover.
3. Halt after 3 serial reds on the same checkbox.
   Done: residual BLOCKED; sitting stopped.
4. Compose from the predecessor hop or dungeon enter. Leave proof:
   `zelda_i.screen_glance` (room hex, mode, x/y band, TF bits, earned
   keys/bombs, hearts lo==hi) + `--no-video`. Mid-dungeon save-state pins
   are not leave proof.
   Done: glance matches that leftover.

Occupancy: [predict-path](../predict-path/SKILL.md) — miss → block that cell → replan; no path → stand.

## Skills

| Job | Skill |
|-----|-------|
| Survival route work | `zelda-assisted-route` |
| Occupancy / RAM-claim halt | `predict-path` |
