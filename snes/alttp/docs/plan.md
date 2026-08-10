# ALTTP — Plan

## Program role (solver flagship)

ALTTP is **substrate B** of the solver flagship triangle (Super Metroid +
ALTTP + SMZ3). Vanilla skills and capability-graph edges feed the shared
item-logic planner; SMZ3 is the seed-abstract proof. See
`docs/SOLVER_ARCHITECTURE.md`. Prefer graph edges with `requires` / natural-entry
promotion over one-off full-game routes.

## Next

1. **After room 0x01 → B1 stairs → Zelda cell** — tip exit `0x50→0x01` is
   natural_entry. Find F1→B1 stairs (not in 0x50/0x01/0x52 dense scans). Use
   `room_engine.py run` + B1 reverse from `CastleB2Landing`/`room_70`. Goal:
   `$F3CC==1`.
2. Promote `0x50→0x01` to continuous only after clean power-on composition
   includes it; optionally extend `run_to_verified_tip` past `room_50`.
3. **Implement `escort_to_sanctuary`** after follower is set (Lamp + sewers).
4. Promote graph edges only with natural-entry evidence
   (`planned` → `isolated` → `natural_entry` → `continuous`). Keep graph
   capability-coarse; expand nodes only when a hop is measured or acquires a cap.
5. Keep continuous segments green on state-load; treat `FighterSword` /
   `CastleMain` as development only. Prefer `secret_entrance_clear` module name.
6. Drive probes from work queue **frontier / zelda / b1** groups first; key/shutter
   and pure `exit_0x55` are **alternate** practice only.
7. Defer arena / romhack / asset-editor (`gauntlet/`, `romhack/`) until the
   opening continuous path is clean.

## Done (recent)

- **Outdoor hops → map authority (2026-08-09, rr-m32):** secret-entrance clear
  is thin glue over `room_engine` + `maps/room_55.json` door
  `stairs_to_courtyard` (no open-loop LEFT×100+DOWN×250). Courtyard geometry
  lives in `maps/screen_1b_courtyard.json`; `pocket_to_main_hall` loads
  approach/push from the map (bush-cut locomotion remains). Live verified
  FighterSword → outdoors → room `0x61`. Remaining: castle-grounds → hole
  still uses measured open-loop approach + bush-lift candidates.
- **Tip exit after 0x50 (2026-08-02):** exhaustive probe — only forward exit is
  east→`0x01` (natural_entry graph edge). Exploration chain
  `0x01→0x52→0x62`; `maps/room_70.json` seed; no B1 stairs isolated yet.
- **All Sanctuary-path room maps:** `maps/room_{55,60,61,62,50,01,51,52,70,71,72,80,81,82}.json`
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
