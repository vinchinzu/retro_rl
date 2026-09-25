# Plan

## Goal

Get a Heat clear (Item-1), then Air past screen 4, then natural entry from
power-on. Maturity stays M3 until a segment starts from the real predecessor
instead of a stage pin.

## Next

1. Heat sections E, F, and G from the camera-9 entry, then the boss door.
   Weapons and items were 0 at camera 9
   (`recordings/heat_s8_cam9/heat_segment.json`).
2. Heat boss clear and an Item-1 pin (`$009B` bit `$01`, Atomic Fire
   `$009A` bit `$01`).
3. Air with Item-1 past progress 984, to camera >= 5.
4. Natural entry from power-on through screen 2, without loading `Level1`.

## Do not repeat

Goblin-solid grids, hold-B only, feet alignment alone, screen-align alone,
the appear-flag fall, a zero-mask global solid, screen-7 low-alcove RIGHT,
and jumping up into an already solid upper Yoku. Empty-cloud stand under
fceumm stayed red. An external FCEUX stick pin is optional and is not a
Clean result. Addresses are in `docs/ram_map.md`.
