---
name: harvest-route
description: >
  Optimize one Harvest Moon corridor hop: mountain, path, town, or farm.
  Bench the current autobot from a live pin, implement a RAM/MultNav/cliff
  policy, then bench again. Cut corner-hug stasis. Do not record a walk that
  BFS can already close. Use when the user says "faster hop", "cliff jump",
  "segment time", "corner collision", "minimize frames", mountain grape
  entrance/exit, or runs /harvest-route.
---

# Harvest route (one hop, bench, record last)

Picks, talks, and keep-menus are [harvest-interact](../harvest-interact/SKILL.md).
This skill is movement only.

Time is money. Every hop is frames at **60 fps** via
`harvest.tasks.mountain_berry.format_segment_time`. Print **before / after / Δ**.
Negative Δ is faster. Same enter pin both rows.

## This turn

1. **Name the hop.** From-tilemap → to-tilemap (or stand). Mountain D2 splits
   are `mountain_entry_to_grape` and `grape_to_mountain_exit` — not house→bin.
2. **Bench BEFORE** the product body from the live predecessor pin
   (`Y1_Inside_House` for D2 mountain). Save JSON under `recordings/`.
   ```bash
   HEADLESS=1 uv run python -m harvest.scripts.mountain_berry_probe \
     --state Y1_Inside_House --ship \
     --out recordings/mountain_segments_before.json
   ```
3. **Implement without a new tape.** Prefer named `SEGMENTS` / `ROUTES`,
   `force_run` downhill off a cliff lip (BFS treats the lip as solid), or a
   shorter waypoint list. Do not reverse a long inbound loop when a downhill
   drop lands on a recorded terrace. You cannot jump *up* a cliff.
4. **Cut corners.** `task_recorder analyze` `stasis_windows` ≥45f while holding
   a direction is a wall/corner hug. Move the last hop onto the **open face**
   of the door or exit, not the corner tile. `force_run` only on a clear axis.
   Distilling a tape: drop any hop whose tile appears in those windows.
5. **Bench AFTER** from the same pin. Wire only if success **and** faster.
   Domain-close the hop (held forage, shipping delta, or destination tilemap).
   Do not STATUS-promote a pin bench as a full-day tip.

## Record last

Record a corridor only after live BFS from the land tile has no gap (see
[INTERACT.md](../../../snes/harvest/docs/INTERACT.md)). Keep the tape short.
Shop doors are [harvest-shop](../harvest-shop/SKILL.md).
