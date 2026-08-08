# ALTTP — Status

## Program gate

| Field | Value |
|-------|-------|
| Current maturity | M1 |
| Best verified result | Clean power-on → castle grounds → secret hole → uncle fighter sword → south combat chamber → stairs exit outdoors → courtyard → main hall → room `0x50` |
| Last verification | 2026-08-02 (tip exit 0x50→0x01 natural_entry + exhaustive probe; continuous tip still 0x50 from 2026-08-01 clean run) |
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
| Opening-route catalog (z3-backed) | `python -m alttp.opening_route.catalog` validates Link's House→castle rooms/nodes/connections; emit `recordings/opening_route_catalog.json` |
| Controller primitives | `alttp.primitives` is the live stack (`run_script` / `settle_control` / `move_*` / `fight_nearby`); segments use `alttp.route_report` |
| Package layout | Core at `alttp/` root; continuous trunk in `alttp/opening_route/`; `gauntlet/` + `romhack/` shells; see `docs/ARCHITECTURE.md` |
| Escape capability graph | `alttp.opening_route.escape_graph` — continuous through **NW chamber 0x50**; **0x50→0x01** natural_entry; Zelda/Sanctuary planned |
| Segment contract | continuous: `castle_to_sword`, `sword_to_secret_entrance_clear`, `pocket_to_main_hall`, `castle_dungeon_prefix`; `full_tip.run_to_verified_tip` composes them from power-on to `room_50`; Zelda path planned (not a live segment); escort planned |
| Room engine | `maps/room_XX.json` + `opening_route.room_engine` + `scripts/room_engine.py` (`docs/ROOM_ENGINE.md`) |
| Graph west exit | `room_61` → `room_60` **continuous** (`main_hall_west_to_0x60`) |
| Dungeon prefix | `room_61` → `room_60` → `room_50` **continuous** (`castle_dungeon_prefix`, clean power-on) |
| Room maps | `maps/room_{55,60,61,62,50,01,51,52,70,71,72,80,81,82}.json` |
| Tip resolution | `anchors.resolve_continuous_tip_node` (session.continuous_tip_node) |
| Work queue focus | Tip `room_50` east→`0x01` (natural_entry); find B1 stairs after 0x01 chain; key/0x55 alternate |
| Multi-truth anchors | `alttp.opening_route.anchors` + `docs/TRIGGER_HANDOFF.md` |
| Session façade | `alttp.session.AlttpSession` (selective snapshot / caps / segment play) |
| Save-state work queue | `work_queue` + `work_queue_data.yaml` — discover `.state`, merge curated status/notes, rank via path tags; export `docs/routes/ROOM_WORK_QUEUE.md` |

## Continuous spine (graph)

```text
castle_grounds → room_55_uncle → room_55_sword → room_55_south
  → courtyard_secret_pocket → main_hall (0x61) → main_west (0x60) → nw (0x50)
                                                              [continuous tip]
  → north connector (0x01)                                    [natural_entry]
  → … B1 stairs? … → zelda_cell → mantle → sewers → sanctuary  [planned]
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

### Next milestone — after 0x01 → B1 stairs → Zelda cell → escort → Sanctuary

Courtyard pocket → main door → room `0x61` is **measured**. Main-hall room
itself is now scripted (dev `CastleMain`):

1. **Done (continuous):** compose clear 0x61 hostiles → side corridor → west
   door → room `0x60` → north door → room `0x50`
   (`castle_dungeon_prefix`; each door remains map-authoritative through
   `room_engine`).
2. **Done (natural_entry, 2026-08-02):** room `0x50` east → `0x01` is the only
   physical forward exit from the continuous tip (exhaustive probe). Graph edge
   `room_50_east_to_0x01`.
3. **Open:** after `0x01` → B1 stairs → Zelda cell; set follower `$F3CC == 1`.
   Exploration chain `0x01→0x52→0x62` measured; stairs not yet isolated.
4. Escort via mantle + sewers (Lamp) to Sanctuary.

Acceptance: `follower_indicator == 1`, then `in_sanctuary` (preferably natural chain).
**Not yet verified.** Drive probes from `docs/routes/ROOM_WORK_QUEUE.md`.

### Verified facts (through 2026-08-01)

- Approach from `HyruleCastleGrounds` reaches `near_secret_hole` at ~`(2430,1704)`.
- Proven entry: face up, `A`×4, wait 20, `UP`×56 (min measured UP walk after A/wait: 40).
- Uncle dialogue in secret entrance yields fighter sword without progression writes.
- Post-sword hold-up-item (`$5D==21`) needs ~95 frames LEFT to dismiss.
- South combat chamber (guards) at ~`(2680,2925)` via LEFT×100 + DOWN×250.
- **Secret-entrance clear:** align stairs ~`(2672,2916)` then DOWN → outdoors
  screen `0x1B` ~`(2248,1755)` (`left_secret_entrance`; `secret_entrance_clear`
  phase `secret_entrance_exited`). Screenshots: `recordings/probe_secret_exit/clear/`.
- Off-center deep south (~y≥2960) soft-locks indoors without transitioning.
- Outdoor landing is a tight hedge pocket; UP re-enters stairs. **Escape needs
  bush-cutting** (walk-only stays ~48×64). Measured path: cut S/W → gardens →
  hard south to y≈2024 → west to x≈2040 → north to door approach ~(2040,1790)
  → UP → room `0x61` (`pocket_to_main_hall`; screenshots under
  `recordings/probe_courtyard_door/south_door/`).
- Main south gate remains soldier-blocked (text `0x0E`) until sword.
- **Main hall 0x61 (2026-07-31):** entry ~(760,3520); hostiles on carpet;
  corridor y≈3320; west → `0x60`, east → `0x62`, south → courtyard. Tools:
  `room_sense` sprite boxes + edge detect. Segment west exit 3/3 from
  `CastleMain` (initial isolated evidence; promoted by the 2026-08-01
  clean-prefix proof below).
- **Main west 0x60 (2026-07-31):** landing ~(511,3320); path west/north shaft
  → UP → room `0x50` ~(376,3088). Map `maps/room_60.json`; graph
  `room_60_north_to_0x50` was initially isolated from `CastleRoom60` and is
  now part of the clean prefix below.
- **Verified-tip composition (2026-08-01):**
  `scripts/run_to_verified_tip.py` kept one power-on environment through
  `castle_to_sword` → `secret_entrance_clear` → `pocket_to_main_hall`, reached
  room `0x61` in 8,578 frames. The same environment then ran
  `castle_dungeon_prefix` to room `0x50` in 2,571 additional frames, proving
  its natural entry. `run_to_verified_tip.py` now composes all four segments;
  rerun result is the canonical artifact `recordings/verified_tip_run.json`.
- **Room-0x50 east → 0x01 (2026-08-02):** exhaustive probe after clear (grid,
  cardinal rays, west-wall holds) found **only** east→`0x01` and south→`0x60`.
  No B1 stairs in 0x50. Edge promoted to graph `natural_entry`
  (`room_50_east_to_0x01`). Natural evidence:
  `recordings/natural_room_50_east.json` + chain from `CastleRoom60` north.
- **Post-0x01 exploration (2026-08-02):** natural chain
  `0x50→0x01→0x52` (clear guards) `→0x62` measured
  (`recordings/probe_room_50/discover_chain.json`). Dense scan: no stairs in
  0x01/0x52/0x62. `maps/room_70.json` seeded from `CastleB2Landing` (B1 side).
- **Maps:** Sanctuary-path room JSON under `maps/` for 0x55/60/61/62/50/01/51/
  52 and B1 0x70/71/72/80/81/82 (doors partial on B1).
- **Blocker:** after room `0x01` → B1 stairs → Zelda cell (`$F3CC==1`) → escort
  → Sanctuary.
