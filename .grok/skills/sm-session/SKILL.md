---
name: sm-session
description: >
  Super Metroid session gates: one spine bead, residual owns pin/checkbox/CLI,
  dual-track, Gravity continues from residual. Use when working in
  snes/super_metroid, starting a hop, composing a tip, or running /sm-session.
---

# SM session

One bead. One change. Practice greens are not continuous evidence.
Planner owns `docs/STATUS.md`. Language: `snes/super_metroid/CONTEXT.md`.

Living tip is **Phantoon**. Next rung is **Gravity** (`rr-kw8t`). Ice is
prefix CI. Tapes are tools. Survival energy+ammo only.

Pin, checkbox, and probe CLI:
`snes/super_metroid/docs/tasks/rr-kw8t-residual.md`.

## Loop

1. `bd ready -l super_metroid -l spine` — claim **exactly one**. Immediate:
   `rr-kw8t` Gravity on the Phantoon tip. Empty ready while that bead
   is in_progress: continue the residual.
2. Dual from the pin the residual names. Same pin both rows.
3. Overwrite `scratch/<hop>_dual.json` only. After a red: keep the
   controller. After **three of the same miss class**: dump a phase pin at
   the last held seat, or **replace** the takeoff (one trajectory).
4. Overwrite the **living** residual
   (`snes/super_metroid/docs/tasks/rr-kw8t-residual.md`).
   Delete closed-hop residuals instead of stacking them.
5. Soft max ~1000 LOC: merge into the **Composer** or delete. No sibling
   extract (`CODING_STANDARDS.md`). Gut sittings use `/gut-package`.
6. Gravity epic (`rr-1xc2`) **continues** from the residual. Three reds do
   not restart the hop or dest-dual Attic. Three of the same miss class →
   new trajectory or dump pin
   (`snes/super_metroid/docs/tasks/HARD_ROOM_SPLITS.md`). Stop repeating
   the same dual.
7. User says watch / headed / autopilot: open a window **first**.
   `--headed` is `retro_harness.headed`.
   `uv run python snes/super_metroid/scripts/probe/kpdr.py pure <hop> --source <pin> --headed`
   `./play <pin> --headed --assist-full`. Dual CLI is the residual's
   dedicated probe.
8. Do not edit `STATUS.md` or `DEFAULT_CONTINUOUS_TIP`.
9. Leave must **Sync** to the next room (doorway pause / a few frames ok).
   If it will not join, **both rooms are one change**. Re-pin the next hop.
   Glance a phase checkbox against the **phase** LeaveSpec (`hop_glance`);
   dest-room spec only on the hop's leave checkbox. Not an MP4.

Prefix slop Chip is parallel only under Sync. It is not a second tip.

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
Did not treat a pin dual as power-on Gravity.
