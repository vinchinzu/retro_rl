# Plan — Mega Man 2 (NES)

## Goal

Advance from M3 (Air Man late-stage isolated) toward a verified continuous clear.

## Next milestones

1. **Air Man deep late-stage** — clear past screen 4 toward boss door.
2. **M4 natural-entry** — screen-2+ from power-on without warping to `Level1`.
3. **Boss segment** — isolated Air Man boss from a door-entry state.
4. **Stage chain** — stage select → clear → weapons → next master.

## Bottleneck

**Post-s4: rider kill Clean; empty cloud stand still RED.** Island solid prog
937–984. Pulse-B kills `0x3D`; feet_dy=0 @ dx≤2 still freefall. Solid decode:
`aobject_tsa` timer, flag 192 facing — not solid bits. No camera ≥5 yet.

Geometry (verified 2026-08-09/10 + fpd6 + rr-54ui night):

- Tile solids end prog 984; Air Tikki is **0x40** damage enemy (not landable)
- LL **0x3D/0x3E** spawns ~prog 961; kill rider then stand (object-solid)
- Pulse B (period 3–8) required; hold-B under-fires
- Kill with dy≳20 freezes gap (co-sink); kill near Y-meet still no `ft=1`
- Placement ROM: lsmmega/mm2 `airman_wily2_objects_set.asm` idx5 mapset4 x=C0 y=20

## Suggested next experiments

1. **Body AI solid-arm (primary, rr-54ui)** — disasm `objects_kaminari_goro`
   after child `0x3D` dies; TAS pin when feet stick. Already ruled out:
   `aobject_tsa` as solid type, flag 192 as solid bit, feet_dy=0 alone.
   Evidence: `docs/CLOUD_LAND_RED_PIN.md`, `scripts/cloud_solid_decode.py`.
2. Screen-align (player/body same scr, cam≥4) then relative-descent land.
3. Chain mapset 5–6 LLs → camera ≥5 → boss door; freeze AirScreen2→5 (3/3).
4. Do **not** re-sweep goblin-solid, “LL absent”, hold-B only, or pure feet_dy grids.

## Notes

- Platform: NES (fceumm via stable-retro custom integration).
- Shared ROM root: `roms/Nintendo/NES/`.
- Mid-stage recipe is frame-indexed from Level1 / AirLanded; re-probe if Level1 save shifts.
- Late-stage (`start=screen2`) is frame-indexed from AirScreen2.
- `AirScreen3` / `AirScreen4` clear states are mid-air; do not treat as grounded starts.
- A button needs a rising edge after load (hold-from-frame-1 does not jump).
- Probe screenshots: `recordings/air_fan_probe/`, `air_boost/`, `air_bird/`, `air_spawn/`.
