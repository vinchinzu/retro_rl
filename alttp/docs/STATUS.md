# ALTTP — Status

## Program gate

| Field | Value |
|-------|-------|
| Current maturity | M1 |
| Best verified result | Title → castle grounds → secret hole → uncle fighter sword; post-sword south chamber of room `0x55` |
| Last verification | 2026-07-29 (headless sword segment + south-chamber approach) |
| Runtime class | Bronze |
| Intervention class | Clean |

| Item | State |
|------|--------|
| Integration `Zelda3-Snes` | done |
| ROM sha1 `6d4f10a8b10e10dbe624cb23cf03b88bb8252973` | done |
| Boot state `YazeSlot000` | present |
| Title / file-select RAM | `module` `$10` (`0x01`/`0x02`) |
| Control ready | `module` in `{0x07,0x09}` and `submodule==0` |
| Castle grounds screen | light-world `$8A == 0x1B` |
| Secret passage room | indoors `$A0` base `0x55` |
| Fighter sword | equip `$F359 >= 1` |
| Follower / Zelda | `$F3CC` tagalong (`1` = Zelda) — **not yet** set on measured path |
| Dungeon keys | `$F36F` (`0xFF` = blank HUD sentinel in room `0x55` so far) |
| Dev saves | `HyruleCastleGrounds.state`, `FirstAction.state`, `FighterSword.state` |
| z3-json-data (optional local refs) | local pin `1eb7a785…` via `scripts/setup_z3_json_data.py`; see `docs/Z3_JSON_DATA.md` |
| Opening-route catalog (z3-backed) | `python -m alttp.opening_route_catalog` validates Link's House→castle rooms/nodes/connections; emit `recordings/opening_route_catalog.json` |

## Current milestone

### Title → fighter sword (opening segment)

Scripted path from `YazeSlot000` / castle-grounds predecessor:

1. Wait for title (`module==0x01`), inject blank SRAM.
2. START into file select; create slot-1 name; load.
3. Wake / exit Link's House with the proven button script.
4. Overworld screen BFS north/west to screen `0x1B`.
5. Walk to secret-hole approach (~world 2430,1704; Yaze entrance `0x7D` @ 2432,1696).
6. Bush-lift + hole-drop into room `0x55` (`SECRET_HOLE_ENTRY_SCRIPT`).
7. Approach uncle and mash dialogue until `$F359 >= 1`.

Acceptance: fighter sword equip RAM ≥ 1 (preferably from `--natural` chain).

### Next milestone — sword → Zelda

After fighter sword in room `0x55`:

1. Dismiss hold-up-item pose (`$5D == 21`) before combat/nav.
2. Navigate multi-screen room `0x55` (south chamber via LEFT×100 + DOWN×250).
3. Clear soldiers, obtain small key / open shutter path out of `0x55`.
4. Reach Zelda’s cell; set follower `$F3CC == 1`.
5. Escort to Sanctuary (room `0x12` / OW screen `0x13` — confirm on ROM).

Acceptance: `follower_indicator == 1` (preferably natural chain). **Not yet verified.**

### Verified facts (2026-07-29)

- Approach from `HyruleCastleGrounds` reaches `near_secret_hole` at ~`(2430,1704)`.
- Proven entry: face up, `A`×4, wait 20, `UP`×56 (min measured UP walk after A/wait: 40).
- Uncle dialogue in room `0x55` yields fighter sword without progression writes.
- Post-sword hold-up-item (`$5D==21`) needs ~95 frames LEFT to dismiss.
- South chamber of `0x55` at ~`(2680,2925)`; DOWN×280+ traps in stair pocket.
- South floor stairs exit to OW near Yaze `0x32` (secret cellar door), not deeper dungeon.
- Main south gate remains soldier-blocked (text `0x0E`) until sword.
- **Blocker:** no measured room transition out of `0x55` toward Zelda cell yet.
