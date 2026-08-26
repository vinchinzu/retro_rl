---
name: harvest-session
description: >
  Harvest Moon session gates: one active spine bead, one living residual,
  diagnostic pivot after 3 serial reds, no STATUS from a pin. Do not record a
  walk BFS can close.
  Use when working in snes/harvest, starting a D2 hop, CLEAR_PLOT, grape,
  shop, pocket clear, or running /harvest-session.
---

# Harvest session

One active bead at a time. Carry it to a checkable completion. Planner owns
`docs/STATUS.md`. Do not STATUS-promote Gate B.

## Loop

1. `bd ready -l harvest -l spine` — claim **exactly one**. Immediate:
   `rr-20w.2.3` D2 CLEAR_PLOT (P0). Water-refill `rr-3ae8` is on this filter
   too — still claim one.
2. Overwrite living residual `snes/harvest/docs/tasks/rr-20w.2.3-residual.md`.
   Delete closed residuals instead of stacking them.
3. Overwrite one JSON report. Do not mint `_vN` or `_window_*`.
4. File ≥500 → split before the knob. File ≥800 → refuse the knob.
5. Three serial reds on the same checkbox → stop repeating that live command.
   Preserve the last report, build a tighter replay/unit harness, rank and
   instrument hypotheses, then continue. Mark BLOCKED only for a genuine
   external blocker after exhausting in-scope alternatives; red count alone
   is not a blocker.
6. Do not edit `STATUS.md`. No STATUS from a pin.
7. Glance with `harvest.clock_glance` (tilemap, hour/minute vs ClockTimeline,
   wallet/shipping delta, crop/plot flags). `HEADLESS=1`. No MP4.

Natural entry is power-on. Work-entry after grape+shop:
`Y1_After_Buy_Potato` / `Y1_D2_PostShipper_WorkStart`. **Do not** start D2
from `Y1_D2_Morning_After_D1` — grape return-to-bin seals at the house fence
(rr-oqri).

## Skills (do not skip)

| Job | Skill |
|-----|-------|
| Movement hop / BFS / cliff | `harvest-route` |
| Pick / talk / keep-menu | `harvest-interact` |
| Seed/tool shop door+wallet | `harvest-shop` |

**ENFORCE:** corridor hops use `harvest-route` — do not record a walk BFS can
close. Interact = scan tape, don't record. Shop = reject CrossMap
returned-to-origin without shop/wallet change.

## Non-claims (every residual)

Did not STATUS-promote. Did not start from `Y1_D2_Morning_After_D1`. Did not
record a BFS-closable walk. Did not treat CrossMap origin-return as shop
success.
