# ALTTP — Plan

## Next

1. **After room 0x50 → Zelda cell** — `castle_dungeon_prefix` is now a
   clean power-on suffix through `0x50`. Isolate the next physical B1 door
   with `room_engine.py run` (not mega-segments), then add it as a typed
   `DungeonRoomEdge`. Goal: `$F3CC==1`.
2. **Implement `escort_to_sanctuary`** after follower is set (Lamp + sewers).
3. Promote graph edges only with natural-entry evidence
   (`planned` → `isolated` → `natural_entry` → `continuous`). Keep graph
   capability-coarse; expand nodes only when a hop is measured or acquires a cap.
4. Keep continuous segments green on state-load; treat `FighterSword` /
   `CastleMain` as development only. Prefer `secret_entrance_clear` module name.
5. Drive probes from work queue **frontier / zelda / b1** groups first; key/shutter
   and pure `exit_0x55` are **alternate** practice only.
6. Extend the clean power-on `run_to_verified_tip` composition only after each
   following edge passes natural entry; eventual goal is title → Zelda →
   Sanctuary.
7. Defer arena / romhack / asset-editor (`gauntlet/`, `romhack/`) until the
   opening continuous path is clean.

## Done (recent)

- **All Sanctuary-path room maps:** `maps/room_{55,60,61,62,50,01,51,52,71,72,80,81,82}.json`
  + continuous `room_60→room_50` (`north_to_0x50`) in the clean power-on
  prefix. z3-json-data workspace
  fallback + US/JP vanilla note.
- **Main hall room 0x61 + room engine:** JSON map authority, typed doors,
  generic clear/path/door push, graph continuous `room_61→room_60`, compact CLI
  `scripts/room_engine.py` (SM-style low context).
- **Courtyard pocket → main door → room 0x61:** bush-cut route, door approach
  ~(2040,1790), UP trigger; graph edge `pocket_to_main_hall` continuous;
  segment + anchors + `scripts/pocket_to_main_hall.py` + probe tooling.
- **Package split:** `opening_route/` continuous trunk; `gauntlet/` + `romhack/`
  ownership shells; core RAM/primitives/startup at root; compat shims.
- **Segment contract** + multi-truth **anchors** + `AlttpSession` façade.
- **Escape graph** continuous through NW chamber `0x50`; outdoor primary Sanctuary
  plan; key path kept as alternate.
- Docs: `ARCHITECTURE.md`, `TRIGGER_HANDOFF.md` (hole + stairs + main door).
- Controller consolidation: live segments use `alttp.primitives` + shared
  `alttp.route_report`.
- Escape capability graph grounds → Sanctuary with sword/lamp/zelda gates.
- Sanctuary-path save-state work queue + `ROOM_WORK_QUEUE.md`.
- Title → Hyrule Castle grounds (screen `0x1B`) scripted + headless verified.
- z3-backed opening-route catalog/validation CLI.
- Castle grounds → secret hole (`0x7D` / room `0x55`) → uncle fighter sword.
- Secret-entrance clear: south chamber stairs → outdoor pocket `0x1B`.

## Non-goals (for now)

- Full dungeon ladder / Ganon credits
- GT arena RL training
- YAZE editor embedding
