---
name: sm-pure-hop
description: >
  Super Metroid pure movement hop from a natural predecessor pin: dual-green
  leave, named leave pin, package layout. Use when the user says "pure hop",
  "next room", "leave pin", "Red Tower climb", "hard room split", or runs
  /sm-pure-hop.
---

# SM pure hop

One door-to-door (or named in-room phase). Natural predecessor pin, not a
door-warp. Session gates: `sm-session`.

## This turn

1. **Name the hop.** From-room hex → to-room hex + items.
   `skill_bank.make_hop_key`.
2. **Pin in** is the previous hop's leave, already dual-green.
3. **Implement** in `routes/kpdr/<pkg>/` (package from day 1 if ≥2 hops).
   Room-prefixed geometry. RLE as JSON under `routes/kpdr/data/`.
   In-room jumps: `takeoff.TakeoffWindow` / `PlatformHop` — hop `side` is
   D-pad `LEFT`/`RIGHT`.
4. **Unit-test** seat / action / wrong-room without the emulator.
5. **Dual** from the same pin. Overwrite `scratch/<hop>_dual.json`.
6. **Glance** `hop_glance.grade_report` against the leave spec (room, gs=8,
   pose class, xy band). A human still is enough; do not record an MP4.
7. **Pin out** written once. Do not clobber an older named pin.

Wire only after dual-green **and** the next hop still enters from pin out.
Do not STATUS-promote.

## Hard in-room climbs

3 serial PARTIALs of the same miss class → dump a named phase pin from the
natural path and iterate from that handoff, or replace the trajectory
(`snes/super_metroid/docs/tasks/HARD_ROOM_SPLITS.md`). Intermediate dumps
are not hop GREEN. Glance the phase seat with `hop_glance` against the
**phase** spec (not the dest-room leave). Pin out the held seat once
(`scratch/post_<hop>_<phase>.state`).
