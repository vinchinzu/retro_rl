# ALTTP status

Planner owns this file. A save-state pin is not a continuous clear.

## Program gate

| Field | Value |
|-------|-------|
| Current maturity | M1 |
| Best verified result | Clean power-on through room `0x50`; Survival-assisted controller plus zero-load replay through Zelda rescue and Sanctuary |
| Last verification | 2026-10-01 |
| Runtime class | Bronze |
| Intervention class | Clean |
| Continuous tip | Room `0x50` (NW chamber). Not extended on this date. |
| Machine report | `recordings/verified_tip_run.json`; assisted evidence in `recordings/sanctuary_20261001/` |

## Verified facts

Graph ladder is planned, then isolated, then natural_entry, then continuous. State-load greens are not continuous.

Continuous from one clean power-on environment: `castle_to_sword`, `sword_to_secret_entrance_clear`, `pocket_to_main_hall`, and `castle_dungeon_prefix` (`0x61` to `0x60` to `0x50`). `full_tip.run_to_verified_tip` is that chain. The 2026-08-01 composition is still the tip. Do not overwrite the report on a red.

`room_50_east_to_0x01` is natural_entry (2026-08-02), not continuous. The only forward exit from `0x50` is east to `0x01`. South returns to `0x60`. No B1 stairs in `0x50`.

On 2026-10-01, the disclosed Survival controller reached room `0x81`, opened
the cell through `0x71` → `0x70` → `0x80`, rescued Zelda, and walked her to
Sanctuary. The committed tape replay reached room `0x12` at `(1272,568)` with
control ready, `$F3C5 == 2`, `$F3C6 == 0x15`, and `$F3CC == 0`. It had seven
HP refills and no progression or capacity writes. This is route evidence under
the assisted contract; it does not extend the Clean continuous tip.

Fighter sword is `$F359 >= 1`. The Clean tip still has no Zelda follower.
The Survival route verified the follower at the cell (`$F3CC == 1`) and its
release on Sanctuary (`$F3CC == 0`), with progress `$F3C5 == 2` and flags
`$F3C6 & 0x04` set.

ROM sha1 `6d4f10a8b10e10dbe624cb23cf03b88bb8252973`. Boot state `YazeSlot000`. Control ready is module `0x07` or `0x09` with submodule `0`.

z3-json-data: committed JSON dump `z3-json-data/`; gitignored pin checkout `refs/z3-json-data/` at `1eb7a785…`. Loader prefers the refs checkout when it exists, then the dump. Do not commit the refs checkout. Do not delete the dump.
