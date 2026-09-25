# ALTTP status

Planner owns this file. A save-state pin is not a continuous clear.

## Program gate

| Field | Value |
|-------|-------|
| Current maturity | M1 |
| Best verified result | Clean power-on through castle grounds, secret hole, uncle fighter sword, courtyard, main hall, and room `0x50` |
| Last verification | 2026-08-02 |
| Runtime class | Bronze |
| Intervention class | Clean |
| Continuous tip | Room `0x50` (NW chamber). Not extended on this date. |
| Machine report | `recordings/verified_tip_run.json` |

## Verified facts

Graph ladder is planned, then isolated, then natural_entry, then continuous. State-load greens are not continuous.

Continuous from one clean power-on environment: `castle_to_sword`, `sword_to_secret_entrance_clear`, `pocket_to_main_hall`, and `castle_dungeon_prefix` (`0x61` to `0x60` to `0x50`). `full_tip.run_to_verified_tip` is that chain. The 2026-08-01 composition is still the tip. Do not overwrite the report on a red.

`room_50_east_to_0x01` is natural_entry (2026-08-02), not continuous. The only forward exit from `0x50` is east to `0x01`. South returns to `0x60`. No B1 stairs in `0x50`.

Fighter sword is `$F359 >= 1`. Zelda tagalong is `$F3CC == 1` and is not set on this tip. Sanctuary is not verified.

ROM sha1 `6d4f10a8b10e10dbe624cb23cf03b88bb8252973`. Boot state `YazeSlot000`. Control ready is module `0x07` or `0x09` with submodule `0`.

z3-json-data: committed JSON dump `z3-json-data/`; gitignored pin checkout `refs/z3-json-data/` at `1eb7a785…`. Loader prefers the refs checkout when it exists, then the dump. Do not commit the refs checkout. Do not delete the dump.
