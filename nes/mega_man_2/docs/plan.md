# Plan — Mega Man 2 (NES)

## Goal

Advance from M3 (Air Man late-stage isolated) toward a verified continuous clear.

## Next milestones

1. **Air Man deep late-stage** — clear past screen 4 toward boss door.
2. **M4 natural-entry** — screen-2+ from power-on without warping to `Level1`.
3. **Boss segment** — isolated Air Man boss from a door-entry state.
4. **Stage chain** — stage select → clear → weapons → next master.

## Bottleneck

**Post-s4 gap: solid island prog 937–984 (`AirFanPlatform`); ~296px open pit to
screen5 (prog 1280). One Mega Man jump only reaches ~1065–1071.**

Screen2 late recipe clears camera screen 4 mid-air but never lands past the
island. Pure RIGHT ~prog 1065–1071; shoot+Pipi boost ~1086 min_sy~23; still pit
on screen 4. No camera ≥5.

Geometry (verified 2026-08-09/10):

- Solids are **tiles** (`tile_feet`/`tile_center`); type36 pink head is a
  **damage enemy** (periodic teleport-hit), not a landable platform
- “Standing on goblin” at AirScreen2 is a y=52 **tile** platform under/near the
  sprite; hops onto type36 never set feet=1 elevated
- `AirLeftPlatform` = short left ledge prog 902–905; leftward returns prior chain
- Ladder bar never `tile_feet==2`; no wind; camera_y=0 through death
- Type35 eggs/birds; freefall past 984 has **zero** tile hits
- No Lightning Lord / new object types in 400–600f edge camp

## Suggested next experiments

1. **Lightning Lord cloud spawn** — alternate earlier route timing (before full
   y=84 descent); log object types while holding camera toward pit.
2. Map-match: which stage segment is prog~950 (Pipi C vs Matasaburo E vs drop D).
3. Only then freeze AirScreen2→target 5 (3/3) and boss door.

## Notes

- Platform: NES (fceumm via stable-retro custom integration).
- Shared ROM root: `roms/Nintendo/NES/`.
- Mid-stage recipe is frame-indexed from Level1 / AirLanded; re-probe if Level1 save shifts.
- Late-stage (`start=screen2`) is frame-indexed from AirScreen2.
- `AirScreen3` / `AirScreen4` clear states are mid-air; do not treat as grounded starts.
- A button needs a rising edge after load (hold-from-frame-1 does not jump).
- Probe screenshots: `recordings/air_fan_probe/`, `air_boost/`, `air_bird/`, `air_spawn/`.
