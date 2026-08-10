# Plan — Mega Man 2 (NES)

## Goal

Advance from M3 (Air Man late-stage isolated) toward a verified continuous clear.

## Next milestones

1. **Air Man deep late-stage** — clear past screen 4 toward boss door.
2. **M4 natural-entry** — screen-2+ from power-on without warping to `Level1`.
3. **Boss segment** — isolated Air Man boss from a door-entry state.
4. **Stage chain** — stage select → clear → weapons → next master.

## Bottleneck

**Post-s4: rider kill Clean; empty cloud stand still RED (rr-54ui PARTIAL).**
No camera ≥5. Residual child **rr-f3nr**.

- Pulse-B kills `0x3D`; body `0x3E` stays; appear `$10` never armed by body AI
- Sole `LDA #$90` flag arm in PRG = Heat appearing_block AI (`14_23`)
- Zero-mask appear force → global solid (fceumm path works); localized masks fail
- **No Air-first Clean alt past s4**: Item-1 needs Heat (`weapons=$00` on Fan);
  jump envelope cannot cover ~296px gap after prog 984
- Human path = cloud ride ×5; TAS path = Item-1 skip (Heat-first)

## Suggested next experiments

1. **rr-f3nr: FCEUX/human stick pin** — dump sy/by/`$2C`/body fl/tsa/xs/ys/cam
   on a real empty-cloud stand vs freefall (`docs/CLOUD_LAND_RED_PIN.md`).
2. **rr-f3nr: Heat→Air Item-1 Clean segment** as alternate past s4 milestone.
3. Chain mapset 5–6 LLs → camera ≥5 only after stand freezes a state.
4. Do **not** re-sweep goblin-solid, “LL absent”, hold-B only, feet_dy grids,
   screen-align-only, fall_top/appear/flag08, or zero-mask global solid.
## Notes

- Platform: NES (fceumm via stable-retro custom integration).
- Shared ROM root: `roms/Nintendo/NES/`.
- Mid-stage recipe is frame-indexed from Level1 / AirLanded; re-probe if Level1 save shifts.
- Late-stage (`start=screen2`) is frame-indexed from AirScreen2.
- `AirScreen3` / `AirScreen4` clear states are mid-air; do not treat as grounded starts.
- A button needs a rising edge after load (hold-from-frame-1 does not jump).
- Probe screenshots: `recordings/air_fan_probe/`, `air_boost/`, `air_bird/`, `air_spawn/`.
