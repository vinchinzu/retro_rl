# Plan — Mega Man 2 (NES)

## Goal

Advance from M3 (Air Man late-stage isolated) toward a verified continuous clear.

## Next milestones

1. **Air Man deep late-stage** — clear past screen 4 toward boss door.
2. **M4 natural-entry** — screen-2+ from power-on without warping to `Level1`.
3. **Boss segment** — isolated Air Man boss from a door-entry state.
4. **Stage chain** — stage select → clear → weapons → next master.

## Bottleneck

**Post-s4: rider kill Clean; empty cloud stand still RED (engine residual).**
Island solid prog 937–984. Pulse-B kills `0x3D`. Body AI has **no solid-arm**
on rider death; appearing_block `$10` never set; force-place at cloud top still
freefall under fceumm. No camera ≥5 yet.

Geometry (verified 2026-08-09/10 + fpd6 + rr-54ui night):

- Tile solids end prog 984; Air Tikki is **0x40** damage enemy (not landable)
- LL **0x3D/0x3E** spawns ~prog 961; kill rider then stand (object-solid expected)
- Pulse B (period 3–8) required; hold-B under-fires
- Kill with dy≳20 freezes gap (co-sink); kill near Y-meet still no `ft=1`
- Placement ROM: lsmmega/mm2 `airman_wily2_objects_set.asm` idx5 mapset4 x=C0 y=20
- Cloud top ≈ by−16 (OAM); kill window cam=3 vs body scr=4

## Suggested next experiments

1. **Human/TAS frame pin (primary residual)** — when feet stick on empty cloud,
   dump sy/by/`$2C`/body tsa/flag/cam. Diff vs freefall dumps in
   `docs/CLOUD_LAND_RED_PIN.md` + `scripts/cloud_screen_align.py`.
2. **Alternate path past s4** without cloud ride (if any Clean route exists).
3. Chain mapset 5–6 LLs → camera ≥5 only after stand freezes a state.
4. Do **not** re-sweep goblin-solid, “LL absent”, hold-B only, feet_dy grids,
   screen-align-only, or solid pokes (fall_top/appear/flag08) already negative.

## Notes

- Platform: NES (fceumm via stable-retro custom integration).
- Shared ROM root: `roms/Nintendo/NES/`.
- Mid-stage recipe is frame-indexed from Level1 / AirLanded; re-probe if Level1 save shifts.
- Late-stage (`start=screen2`) is frame-indexed from AirScreen2.
- `AirScreen3` / `AirScreen4` clear states are mid-air; do not treat as grounded starts.
- A button needs a rising edge after load (hold-from-frame-1 does not jump).
- Probe screenshots: `recordings/air_fan_probe/`, `air_boost/`, `air_bird/`, `air_spawn/`.
