---
name: zelda-session
description: >
  Zelda I session gates: one spine bead, one knob, one living residual,
  Survival vs Clean dual-track, halt after 3 reds, no STATUS from a pin.
  Use when working in nes/zelda_i, starting a Survival hop, composing the
  spine, or running /zelda-session.
---

# Zelda session

One bead. One change. Assisted greens are not Clean STATUS.
Planner owns `docs/STATUS.md`.

## Loop

1. `bd ready -l zelda_i -l spine` — claim **exactly one**.
2. Dual-track: Survival (`--infinite-life` / health refill) vs Clean.
   Assisted greens are not Clean STATUS. Planner owns `docs/STATUS.md`.
3. Overwrite ONE living residual: `nes/zelda_i/docs/tasks/rr-tne2-residual.md`.
   Delete closed-hop residuals instead of stacking. Do not mint `_vN` /
   window dumps. Overwrite one report JSON.
4. Halt after 3 serial reds on the SAME checkbox → BLOCKED residual, stop.
5. File ≥500 lines → split before a knob. File ≥800 → refuse the knob and
   split first. Do not boil already-split `level4_*`.
6. Do not edit `STATUS.md` or promote a pin/practice green as Clean M5.
7. Glance leave with `zelda_i.screen_glance` (room hex, mode, x/y band, TF
   bits, earned keys/bombs, hearts lo==hi). No MP4. `--no-video` on spine
   CLIs.
8. Occupancy/predict halt is **not** duplicated here. See
   [predict-path](../predict-path/SKILL.md): occupancy miss → block that
   cell → replan; no path → stand; do not probe a path OccupancyWalker can
   close.

## Skills

| Job | Skill |
|-----|-------|
| Survival route work | `zelda-assisted-route` |
| Occupancy / RAM-claim halt | `predict-path` |

## Non-claims (every residual)

Did not STATUS-promote. Did not overwrite Clean M5. Did not poke
doors/keys/undiscovered items. Did not grant Map/Whistle.
