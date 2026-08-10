# Plan — Mega Man 2 (NES)

## Goal

Advance from M3 (Air Man late-stage isolated) toward a verified continuous clear.

## Next milestones

1. **Heat→Air Item-1 chain** (rr-f3nr residual) — Heat mid/boss → Item-1 → Air
   past s4 with platforms (`docs/HEAT_ITEM1_PATH.md`).
2. **Air Man deep late-stage** — cam ≥5 via Item-1 or cloud stand (if FCEUX pin).
3. **M4 natural-entry** — screen-2+ from power-on without warping to `Level1`.
4. **Boss segment** — isolated Air Man boss from a door-entry state.
5. **Stage chain** — stage select → clear → weapons → next master.

## Bottleneck

**Post-s4 Air cloud solid RED; Heat mid dual-green through s5 (rr-808 PARTIAL);
pre-boss / Item-1 still open.**

- Pulse-B kills `0x3D`; empty `0x3E` never arms appear; gap ~296px unjumpable
- Heat screens 1–5 **GREEN**; death band ~prog 1500 on s5; boss / Item-1 open
- FCEUX human stick pin protocol documented (external)

## Suggested next experiments

1. Heat late / pre-boss from `HeatScreen5` (ladder/vertical; past prog 1500).
2. Heat boss clear + Item-1 unlock pin (`$009B\|$01`) — **rr-809**.
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
