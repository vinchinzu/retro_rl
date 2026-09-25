# Plan: Final Fight

Verified segments are in `docs/STATUS.md`. RAM is in `docs/ram_map.md`.

## Next

1. Kill the Area 1 HP 250 thug from `Stage3_Area1_hp50_L1_cam2560` with
   controller input only. No heal poke and no `--force-enemy-hp`.
2. From that kill, reach and defeat Boss 3 without writing `game_status`.
3. Replace the clear-area bridges after Damnd and Sodom with the natural
   clear-round (`0x0CD2` / status `0x0A`).
4. Only then chain a title-to-credits run with no mid-run save.

`edge_combat.area1_andore_action` is the current throw recipe (close to about
dx 32, then `UP+Y`). Continuous `LEFT+Y` whiffs. Do not treat a dev Boss 3
state as the milestone.

## Policy shape

The segment tree already lives in `final_fight/policy.py` via
`retro_harness.combat.build_segment_tree`: align, fight the nearest living
actor, otherwise walk right. New work is a branch for a measured stall or
boss, not a second tree.
