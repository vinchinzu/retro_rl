# ALTTP plan

Future work only. Verified facts stay in `docs/STATUS.md`. Do not claim a new continuous tip. Room work stays room-based.

## Next

See [PORT_PLAN.md](PORT_PLAN.md) for the dungeon ladder and credits.

1. Keep the Clean continuous tip at room `0x50`. The `zelda` and `sanctuary`
   controller routes are development Survival evidence until a no-assist,
   no-load composition earns C1. Do not promote `room_50_east_to_0x01` from a
   state load.
2. The real predecessor now reaches Zelda through `0x81` → `0x71` → `0x70`
   → `0x80`, then walks the rescue and sewer route to Sanctuary. Reproduce
   the disclosed result with `audit_power_on.py --through sanctuary` and its
   tape replay.
3. Convert that route to Clean/non-assisted evidence before changing the
   continuous tip or closing C1. Detail and exact measurements are in
   `docs/tasks/residual.md`.
4. Dungeon-start saves (`EasternPalaceEntry` and the other `*Entry` pins) stay development only. Do not fold them into the Sanctuary graph or into STATUS.
5. Keep the graph capability-coarse. Promote a hop only from the predecessor the ladder names. The `0x55` key and shutter path stays alternate practice, not the Sanctuary plan.
6. Leave `gauntlet/` and `romhack/` empty until the opening path is clean.

## Non-goals

- GT arena training.
- YAZE editor embedding.
- Re-committing `refs/z3-json-data/`, or deleting the committed `z3-json-data/` dump.
