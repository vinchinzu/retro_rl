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
937–984. LL spawns mapset4; pulse-B kills `0x3D` in air; Y-meet after kill
reaches dx≈5–10 but freefall continues. No camera ≥5 yet.

Geometry (verified 2026-08-09/10 + fpd6 + rr-54ui night):

- Tile solids end prog 984; Air Tikki is **0x40** damage enemy (not landable)
- LL **0x3D/0x3E** spawns ~prog 961; kill rider then stand (object-solid)
- Pulse B (period 3–8) required; hold-B under-fires
- Kill with dy≳20 freezes gap (co-sink); kill near Y-meet still no `ft=1`
- Placement ROM: lsmmega/mm2 `airman_wily2_objects_set.asm` idx5 mapset4 x=C0 y=20

## Suggested next experiments

1. **Object-solid decode (primary, rr-54ui)** — `aobject_tsa=$4E0`, post-kill
   flag/type 118, feet-on-top geometry vs sy==by. Evidence:
   `recordings/air_post4_cloud/RED_PIN.md`. Probe: `scripts/cloud_land_probe.py`.
2. Chain mapset 5–6 LLs → camera ≥5 → boss door; freeze AirScreen2→5 (3/3).
3. Do **not** re-sweep goblin-solid, “LL absent”, or hold-B only.

## Notes

- Platform: NES (fceumm via stable-retro custom integration).
- Shared ROM root: `roms/Nintendo/NES/`.
- Mid-stage recipe is frame-indexed from Level1 / AirLanded; re-probe if Level1 save shifts.
- Late-stage (`start=screen2`) is frame-indexed from AirScreen2.
- `AirScreen3` / `AirScreen4` clear states are mid-air; do not treat as grounded starts.
- A button needs a rising edge after load (hold-from-frame-1 does not jump).
- Probe screenshots: `recordings/air_fan_probe/`, `air_boost/`, `air_bird/`, `air_spawn/`.
