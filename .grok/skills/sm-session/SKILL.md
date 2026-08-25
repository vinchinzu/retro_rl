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
Planner owns `docs/STATUS.md`. Language: `snes/super_metroid/CONTEXT.md`.

Living tip is **Phantoon**. Next rung is **Gravity** (`rr-kw8t`). Ice is
prefix CI. Tapes are tools. Survival energy+ammo only.

## Loop

1. `bd ready -l super_metroid -l spine` — claim **exactly one**. Immediate:
   `rr-kw8t` Gravity on the Phantoon tip.
2. Dual from the named predecessor pin. Same pin both rows. Gravity starts
   from `scratch/post_phantoon_leave.state` (`0xCC6F` ~(1240,139) p10 gs=8).
3. Overwrite `scratch/<hop>_dual.json` only. Do not mint `_vN` or `_window_*`.
   **Do not revert or delete the hop body because a dual was RED.** The
   controller stays. Next takeoff is an add, not a restore-to-empty.
4. Overwrite the **living** residual
   (`snes/super_metroid/docs/tasks/rr-kw8t-residual.md`).
   Delete closed-hop residuals instead of stacking them.
5. File ≥500 lines → split before the knob. File ≥800 → stop and split.
6. Three red windows / PARTIALs on the same checkbox → BLOCKED residual,
   **stop dualing that checkbox**. Halt is not revert. Phase ladder:
   `snes/super_metroid/docs/tasks/HARD_ROOM_SPLITS.md`.
7. User says watch / headed / autopilot: open a window **first**.
   `--headed` is `retro_harness.headed` (any hop, not a custom pygame loop).
   `uv run python snes/super_metroid/scripts/probe/kpdr.py pure <hop> --source <pin> --headed`
   `./play <pin> --headed --assist-full`. Do not dual headless before the window.
8. Do not edit `STATUS.md` or `DEFAULT_CONTINUOUS_TIP`.
9. Leave must **Sync** to the next room (doorway pause / a few frames ok).
   If it will not join, **both rooms are one change**. Re-pin the next hop.
   Glance-check the leave with `super_metroid.hop_glance` (room, gs=8,
   pose class, xy band, boss bit) — not an MP4.

Prefix slop Chip is parallel only under Sync. It is not a second tip.

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
Did not treat a pin dual as power-on Gravity.
