# ALTTP — Status

## Program gate

| Field | Value |
|-------|-------|
| Current maturity | M1 |
| Best verified result | Title → castle grounds → secret hole → uncle fighter sword → south combat chamber → stairs exit outdoors → bush-cut courtyard → main door → room `0x61` |
| Last verification | 2026-07-30 (headless `pocket_to_main_hall` from `FighterSword` via stairs exit) |
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
| Main hall room | indoors `$A0` base `0x61` — **entered** from courtyard main door |
| Dungeon keys | `$F36F` (`0xFF` = blank HUD sentinel in room `0x55` so far) |
| Dev saves | `HyruleCastleGrounds` = grounds spawn controllable (not hole approach); `FighterSword` = room 0x55 post-uncle (dev only) — see anchors/STATE_SEMANTICS |
| z3-json-data (optional local refs) | local pin `1eb7a785…` via `scripts/setup_z3_json_data.py`; see `docs/Z3_JSON_DATA.md` |
| Opening-route catalog (z3-backed) | `python -m alttp.opening_route_catalog` validates Link's House→castle rooms/nodes/connections; emit `recordings/opening_route_catalog.json` |
| Controller primitives | `alttp.primitives` is the live stack (`run_script` / `settle_control` / `move_*` / `fight_nearby`); segments use `alttp.route_report` |
| Package layout | Core at `alttp/` root; continuous trunk in `alttp/opening_route/`; `gauntlet/` + `romhack/` shells; see `docs/ARCHITECTURE.md` |
| Escape capability graph | `alttp.opening_route.escape_graph` — continuous through **main hall 0x61**; Zelda/Sanctuary planned |
| Segment contract | `alttp.opening_route.segment` — `castle_to_sword`, `sword_to_secret_entrance_clear`, `pocket_to_main_hall` |
| Multi-truth anchors | `alttp.opening_route.anchors` + `docs/TRIGGER_HANDOFF.md` |
| Session façade | `alttp.session.AlttpSession` (selective snapshot / caps / segment play) |
| Save-state work queue | `alttp.opening_route.work_queue` — 60 `Zelda3-Snes` states ranked for Sanctuary; `docs/routes/ROOM_WORK_QUEUE.md` |

## Continuous spine (graph)

```text
castle_grounds → room_55_uncle → room_55_sword → room_55_south
  → courtyard_secret_pocket → main_hall (0x61)   [continuous tip]
  → zelda_cell → mantle → sewers → sanctuary   [planned]
```

Alternate internal key/shutter path from `room_55_south` remains on the graph
for work-queue practice but is not the default Sanctuary plan.

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

### Next milestone — main hall → Zelda cell → escort → Sanctuary

Courtyard pocket → main door → room `0x61` is **measured** (dev path from
`FighterSword` via stairs exit). Remaining:

1. Clear early B1 / key-shutter legs from main hall
   (work-queue: `CastleB1Key` / `Shutter*`).
2. Reach Zelda’s cell; set follower `$F3CC == 1`.
3. Escort via mantle + sewers (needs Lamp from house) to Sanctuary
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
- Outdoor landing is a tight hedge pocket; UP re-enters stairs. **Escape needs
  bush-cutting** (walk-only stays ~48×64). Measured path: cut S/W → gardens →
  hard south to y≈2024 → west to x≈2040 → north to door approach ~(2040,1790)
  → UP → room `0x61` (`pocket_to_main_hall`; screenshots under
  `recordings/probe_courtyard_door/south_door/`).
- Main south gate remains soldier-blocked (text `0x0E`) until sword.
- **Blocker:** main hall B1 → Zelda cell (`$F3CC==1`) → escort → Sanctuary.
