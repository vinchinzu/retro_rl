# ALTTP — Plan

## Next

1. Finish `sword_to_zelda`: exit multi-screen room `0x55` (key/shutter),
   reach Zelda cell, set `$F3CC==1`, escort to Sanctuary.
2. Keep `castle_to_sword.py` green on state-load and `--natural`; treat
   `FighterSword.state` as a development checkpoint only.
3. Drive next room probes from `alttp.work_queue` /
   `docs/routes/ROOM_WORK_QUEUE.md` (0x55 exit / key / shutter first).
4. Optional: re-emit catalog with live boot/sword/zelda observations.
5. Add overworld map assets only when tile-accurate nav is needed.
6. Defer arena / romhack / asset-editor until more opening-route segments are
   named and acceptance-tested.

## Done (recent)

- Controller consolidation: live segments use `alttp.primitives` + shared
  `alttp.route_report` (no dual settle/macro stacks).
- Offline `test_primitives.py`; deleted pure-theater `opening_overworld_route_plan`.
- Escape capability graph (`alttp.escape_graph`) grounds → Sanctuary with
  sword/key/lamp/zelda gates; continuous through 0x55 south chamber only.
- Sanctuary-path save-state work queue (`alttp.work_queue`,
  `scripts/export_work_queue.py` → `docs/routes/ROOM_WORK_QUEUE.md`).
- Title → Hyrule Castle grounds (screen `0x1B`) scripted + headless verified.
- z3-backed opening-route catalog/validation CLI
  (`alttp.opening_route_catalog`) for Link's House → castle grounds.
- Castle grounds → secret hole (`0x7D` / room `0x55`) → uncle fighter sword
  (`alttp.castle_to_sword`, proven `SECRET_HOLE_ENTRY_SCRIPT`).
- Post-sword scaffolding: hold-up dismiss, south-chamber approach, follower/key
  RAM (`alttp.sword_to_zelda`, `$F3CC` / `$F36F`).

## Non-goals (for now)

- Full dungeon ladder / Ganon credits
- GT arena RL training
- YAZE editor embedding
