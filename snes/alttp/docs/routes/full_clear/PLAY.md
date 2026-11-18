# Play

Not a finished autoplay. Rows 1–4 are the only true rows. Nothing after them is playable.

## Runnable prefix

The Clean tape `recordings/clean_escape_20261002` ends in room `0x12`. `Play_eog9_7` is room `0xB9` at `(4856, 6104)`, sword 1, lamp 1, HP 28/32, follower 0, keys 0. Rows 1–4 of `ORDER.md` are the only true rows: Link's Uncle (`L1Sword`), Zelda's Chest (`Lamp`), Sewers - Dark Cross (`KeyH2`), and Sanctuary (`SanctuaryHeart`, requires `KeyH2`).

## Frontier

See [the living residual](../../tasks/residual.md) for the single current
handoff. `alttp.dungeon.nav:reach_cannonball_hall_clear` is the development
controller from `Play_eog9_7`. It reaches the clear area on B9's lower floor;
the upper-floor exit into `0xA9` is still unproved. This is not a spine row.

## Logic order

227 census rows run through Ganon's Triforce. That node has an empty requires list. The pin does not encode seven crystals on the pyramid. No row after 4 is playable.

## Credits

AP-197 is not implemented and there is no credits tape. The plan ends at the Triforce, then that gap.

## Contradictions

`audit_dark.md`, `audit_early.md`, `audit_eastern.md`, `audit_finale.md`, `audit_late.md`, and `audit_light.md` each read `NONE`. Those lanes check row status, the crystal gate, the credits tape, and `north_002`. They did not catch the 202-cell entry list. That list is withdrawn in `ORDER.md` and on `maps/room_b9.json`.
