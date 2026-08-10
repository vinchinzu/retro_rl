# Plan — Mega Man 2 (NES)

## Goal

Advance from M3 (Air Man late-stage isolated) toward a verified continuous clear.

## Next milestones

1. **Heat boss door + Item-1** (rr-809 residual) — climb from `HeatScreen7`
   alcove → boss clear → pin Item-1 / Atomic Fire.
2. **Air with Item-1 past s4** — **rr-810** (blocked until Item-1).
3. **M4 natural-entry** — screen-2+ from power-on without warping to `Level1`.
4. **Boss segment** — isolated Air Man boss from a door-entry state.
5. **Stage chain** — stage select → clear → weapons → next master.

## Bottleneck

**Post-s4 Air cloud solid RED; Heat dual-green through pre-boss cam ≥7
(rr-809 PARTIAL); boss climb / Item-1 still open.**

- Pulse-B kills `0x3D`; empty `0x3E` never arms appear; gap ~296px unjumpable
- Heat screens 1–7 **GREEN** (s5Ground → cam7); s7 alcove low ceiling / wall lock
  prog 1792; no boss_hp yet
- FCEUX human stick pin protocol documented (external)

## Suggested next experiments

1. HeatScreen7 climb past low ceiling / micro-ledge sy124 → boss door.
2. Heat boss clear + Item-1 unlock pin (`$009B\|$01`) — finish **rr-809**.
3. Air Fan + Item-1 deploy past prog 984 → camera ≥5 — **rr-810**.
4. Optional: FCEUX empty-cloud RAM pin (`docs/HEAT_ITEM1_PATH.md`).
5. Do **not** re-sweep goblin-solid, “LL absent”, hold-B only, feet_dy grids,
   screen-align-only, fall_top/appear/flag08, or zero-mask global solid.
## Notes

- Platform: NES (fceumm via stable-retro custom integration).
- Shared ROM root: `roms/Nintendo/NES/`.
- Mid-stage recipe is frame-indexed from Level1 / AirLanded; re-probe if Level1 save shifts.
- Late-stage (`start=screen2`) is frame-indexed from AirScreen2.
- `AirScreen3` / `AirScreen4` clear states are mid-air; do not treat as grounded starts.
- A button needs a rising edge after load (hold-from-frame-1 does not jump).
- Probe screenshots: `recordings/air_fan_probe/`, `air_boost/`, `air_bird/`, `air_spawn/`.
