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
3. Continue epic `rr-eog9` at `rr-eog9.4`, from replay-derived
   `Play_eog9_3` on loaded/standing screen `0x1E`, `(3328, 2047)`:
   - Use `screen_1e_sahasrahla.json` for the hut approach and doorway. Fold
     outdoor point navigation into the existing walker/Composer; do not add
     a separate hut runner. Verify dialogue and each resource pickup in RAM.
   - Earn roadside resources under `rr-yx26` if the hut needs bombs. The
     crossing leaves rupees/bombs at zero; grass cutting alone is no payout.
   - Then verify the hedge/Armos approach (`rr-eog9.5`) and lobby transition
     (`rr-eog9.6`) from that real predecessor, without loading `EasternPalaceEntry`.
   - Compose `--through eastern_entry` into the existing spine/audit CLIs
     (`rr-eog9.7`), replay from power-on, and finish whole-route tests (`.8`).
   - Preserve the proved southern detour through `0x25 → 0x2D → 0x2E → 0x26`
     and the outdoor stairs into `0x1E`. The direct east exits from the
     `0x1D` north lip are inaccessible; a screen-graph edge is not a tile path.
   - Reproduce crossing evidence in
     `recordings/eog9_crossing_20261001/power_on_replay.json` before changing
     its policy. This is Survival development evidence, not a Clean promote.
4. Convert the opening route to Clean/non-assisted evidence before changing the
   continuous tip or closing C1. Detail and exact measurements are in
   `docs/tasks/residual.md`.
5. Dungeon-start saves (`EasternPalaceEntry` and the other `*Entry` pins) stay development only. Do not fold them into the Sanctuary graph or into STATUS.
6. Keep the graph capability-coarse. Promote a hop only from the predecessor the ladder names. The `0x55` key and shutter path stays alternate practice, not the Sanctuary plan.
7. Leave `gauntlet/` and `romhack/` empty until the opening path is clean.

## Non-goals

- GT arena training.
- YAZE editor embedding.
- Re-committing `refs/z3-json-data/`, or deleting the committed `z3-json-data/` dump.
