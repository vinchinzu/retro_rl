---
name: sm-session
description: >
  Super Metroid session gates: one spine bead, residual owns pin/checkbox/CLI,
  one run per check. Never halt. Use when working in
  snes/super_metroid, starting a hop, composing a tip, or running /sm-session.
---

# SM session

Planner owns `docs/STATUS.md`. Language: `snes/super_metroid/CONTEXT.md`.
Never halt. After a red, keep the controller and change one knob.

## Loop

1. Run once from the pin the residual names. Proof is leftover RAM on that
   run. Run a second time only to compare against TAS or another skill
   (`settled_gs8` hop clock), never to prove determinism.
2. Overwrite the **living** residual
3. Soft max ~1000 LOC: merge into the **Composer** or delete. No sibling
   extract (`CODING_STANDARDS.md`). Gut sittings use `/gut-package`.
4. User says watch / headed / autopilot: open a window **first**.
   `--headed` is `retro_harness.headed`.
   `uv run python snes/super_metroid/scripts/probe/kpdr.py pure <hop> --source <pin> --headed`
   `./play <pin> --headed --assist-full`. The residual's dedicated probe
   is one run.

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
