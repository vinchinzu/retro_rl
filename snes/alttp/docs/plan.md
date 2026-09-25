# ALTTP plan

Future work only. Verified facts stay in `docs/STATUS.md`. Do not claim a new continuous tip. Room work stays room-based.

## Next

1. Keep the continuous tip at room `0x50`. `run_opening_spine.py --through room_01`, `room_72`, and `zelda` stay fail-closed until a clean power-on composition leaves that room. Do not promote `room_50_east_to_0x01` from a state load.
2. Open the Zelda cell from a real predecessor, then escort (lamp and sewers) until `$F3CC == 1` on a walked rescue. `CastleZeldaFollower` already has `$F3CC == 1` as loaded. That pin is not the rescue.
3. The 2026-09-20 state-load on `0x81` `west_to_0x80` did not spend a small key (gold jail door, `$F366` and `$F367` still 0). Those big-key bytes were not tried. That red is not a graph hop and not continuous. Detail is `docs/tasks/residual.md`.
4. Dungeon-start saves (`EasternPalaceEntry` and the other `*Entry` pins) stay development only. Do not fold them into the Sanctuary graph or into STATUS.
5. Keep the graph capability-coarse. Promote a hop only from the predecessor the ladder names. The `0x55` key and shutter path stays alternate practice, not the Sanctuary plan.
6. Leave `gauntlet/` and `romhack/` empty until the opening path is clean.

## Non-goals

- Full dungeon ladder or Ganon credits.
- GT arena training.
- YAZE editor embedding.
- Re-committing `refs/z3-json-data/`, or deleting the committed `z3-json-data/` dump.
