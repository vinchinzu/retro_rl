# ALTTP — Plan

## Next

1. **Main hall / early B1 → Zelda cell** (`$F3CC==1`) — active continuous
   blocker. Work-queue: `CastleB1Key` / `Shutter*` first.
2. Escort (Lamp + sewers) → Sanctuary after follower is set.
3. Promote graph edges only with natural-entry evidence
   (`planned` → `isolated` → `natural_entry` → `continuous`).
4. Keep `castle_to_sword` / `sword_to_zelda` / `pocket_to_main_hall` green on
   state-load; treat `FighterSword.state` / `CastleMain.state` as development only.
5. Drive next room probes from `alttp.opening_route.work_queue` /
   `docs/routes/ROOM_WORK_QUEUE.md` (continuous-spine blockers first).
6. Optional: full natural-chain title → main hall; re-emit catalog with live
   observations.
7. Defer arena / romhack / asset-editor (`gauntlet/`, `romhack/`) until the
   opening continuous path is clean.

## Done (recent)

- **Courtyard pocket → main door → room 0x61:** bush-cut route, door approach
  ~(2040,1790), UP trigger; graph edge `pocket_to_main_hall` continuous;
  segment + anchors + `scripts/pocket_to_main_hall.py` + probe tooling.
- **Package split:** `opening_route/` continuous trunk; `gauntlet/` + `romhack/`
  ownership shells; core RAM/primitives/startup at root; compat shims.
- **Segment contract** + multi-truth **anchors** + `AlttpSession` façade.
- **Escape graph** continuous through main hall; outdoor primary Sanctuary
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
