---
name: sm-compose
description: >
  Wire a Super Metroid continuous tip from already-green hops: TipSpec,
  SpineHop, catalog, next-hop leave check. Scratch reports only. Use when
  the user says "compose", "wire the tip", "--to phantoon", "catalog hop",
  "do not STATUS", or runs /sm-compose.
---

# SM compose

Hops must already be dual-green from natural pins. This card is wiring +
power-on/pin compose, not a new controller.

## This turn

1. Claim the compose bead (`spine` + `compose`).
2. Append `SpineHop` / `TipSegment` CLI fields. `run_to` is `TipSpec` only —
   no new `start_to_*.py`, no second hop runner.
3. Parent tip stays the previous `--to`. Do not append these hops onto an
   earlier tip that should still end at its published room.
4. Compose from the named pin, then power-on if that is the bead.
   Overwrite `scratch/<tip>_dual.json`. Never overwrite
   `recordings/<tip>.json` on a red run.
5. Glance the final dict: room, gs=8, items/beams, boss bit if a fight.
   `--no-video`.
6. Next hop from the leave pin still clears, or keep the old body.

Do not edit `STATUS.md` or `DEFAULT_CONTINUOUS_TIP`. Planner STATUS is a
follow-on bead.
