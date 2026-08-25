---
name: sm-session
description: >
  Super Metroid session gates: one spine bead, one knob, one living residual,
  dual-track, halt after 3 red windows, no STATUS from a pin. Use when working
  in snes/super_metroid, starting a hop, composing a tip, or running
  /sm-session.
---

# SM session

One bead. One change. Practice greens are not continuous evidence.
Planner owns `docs/STATUS.md`.

## Loop

1. `bd ready -l super_metroid -l spine` — claim **exactly one**.
2. Dual from the named predecessor pin. Same pin both rows.
3. Overwrite `scratch/<hop>_dual.json`. Do not mint `_vN` or `_window_*`.
4. Overwrite the **living** residual (`docs/tasks/<open-tip>-residual.md`).
   Delete closed-hop residuals instead of stacking them.
5. File ≥500 lines → split before the knob. File ≥800 → stop and split.
6. Three red windows / PARTIALs on the same checkbox → BLOCKED residual,
   stop. Phase ladder: `snes/super_metroid/docs/tasks/HARD_ROOM_SPLITS.md`.
7. Do not edit `STATUS.md` or `DEFAULT_CONTINUOUS_TIP`.
8. Next hop must still clear from the new leave pin before calling the hop
   wired. Glance-check the leave with `super_metroid.hop_glance` (room, gs=8,
   pose class, xy band, boss bit) — not an MP4.

## Skills

| Job | Skill |
|-----|-------|
| Movement hop | `sm-pure-hop` |
| Same-pin bench / wiki fight | `sm-room-policy` |
| Clean / no-assist fight | `sm-no-assist-boss` |
| TipSpec / SpineHop / `--to` | `sm-compose` |

## Non-claims (every residual)

Did not STATUS-promote. Did not change `DEFAULT_CONTINUOUS_TIP`. Did not
overwrite `recordings/<tip>.json` on a red run. Did not forge progression RAM.
