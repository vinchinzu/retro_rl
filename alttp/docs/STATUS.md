# ALTTP — Status

## Program gate

| Field | Value |
|-------|-------|
| Current maturity | M1 |
| Best verified result | Title → castle grounds → secret hole → uncle fighter sword → south combat chamber → stairs exit outdoors (secret entrance clear) |
| Last verification | 2026-07-30 (headless `sword_to_zelda` stairs exit from `FighterSword`) |
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
| Controller primitives | `alttp.primitives` is the live stack (`run_script` / `settle_control` / `move_*` / `fight_nearby`); segments use `alttp.route_report` |
| Escape capability graph | `alttp.escape_graph` — RAM nodes/legs grounds→Sanctuary; continuous through south chamber; rest **planned** |
| Save-state work queue | `alttp.work_queue` — 60 `Zelda3-Snes` states ranked for Sanctuary; `docs/routes/ROOM_WORK_QUEUE.md` |

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

### Next milestone — courtyard → castle → Zelda → Sanctuary

Secret entrance is **cleared** (dev path from `FighterSword`). After stairs exit:

1. Escape the outdoor hedge pocket at ~`(2248,1755)` on screen `0x1B`
   (UP re-enters stairs; walkable box is tight — courtyard path unfinished).
2. Enter main castle door; clear early B1 / key-shutter legs
   (work-queue: `CastleB1Key` / `Shutter*`).
3. Reach Zelda’s cell; set follower `$F3CC == 1`.
4. Escort via mantle + sewers (needs Lamp from house) to Sanctuary
   (`alttp.escape_graph` plan; room `0x12` / OW screen `0x13` — confirm on ROM).

Acceptance: `follower_indicator == 1`, then `in_sanctuary` (preferably natural chain).
**Not yet verified.** Drive probes from `docs/routes/ROOM_WORK_QUEUE.md`.

### Verified facts (2026-07-30)

- Approach from `HyruleCastleGrounds` reaches `near_secret_hole` at ~`(2430,1704)`.
- Proven entry: face up, `A`×4, wait 20, `UP`×56 (min measured UP walk after A/wait: 40).
- Uncle dialogue in secret entrance yields fighter sword without progression writes.
- Post-sword hold-up-item (`$5D==21`) needs ~95 frames LEFT to dismiss.
- South combat chamber (guards) at ~`(2680,2925)` via LEFT×100 + DOWN×250.
- **Secret-entrance clear:** align stairs ~`(2672,2916)` then DOWN → outdoors
  screen `0x1B` ~`(2248,1755)` (`left_secret_entrance`; `sword_to_zelda` phase
  `secret_entrance_exited`). Screenshots: `recordings/probe_secret_exit/clear/`.
- Off-center deep south (~y≥2960) soft-locks indoors without transitioning.
- Outdoor landing is a tight hedge pocket; UP re-enters stairs. Path to main
  castle door from that pocket is **not** measured yet.
- Main south gate remains soldier-blocked (text `0x0E`) until sword.
- **Blocker:** courtyard pocket → main castle door → Zelda cell (not secret entrance).
