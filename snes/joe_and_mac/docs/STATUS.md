# Status — Joe & Mac


## Program gate

| Field | Value |
|-------|-------|
| Current maturity | M2 |
| Best verified result | Controllable Stage 1 checkpoint |
| Last verification | 2026-07-22 |
| Runtime class | Bronze |
| Intervention class | Clean |

| Field | Value |
|-------|-------|
| Status | **boot verified** |
| Integration | `JoeAndMac-Snes` |
| ROM zip | `Joe & Mac - Caveman Ninjas.zip` |

## Done

- Directory layout and integration stubs (`data.json` / `metadata.json` /
  `scenario.json`)
- `scripts/setup_rom.py` wiring via `retro_harness.env`
- Plan notes for control style and first segment milestone
- Deterministic reset/title/map-to-Stage-1 script (2,820 frames)
- Controllable `Stage1.state` checkpoint
- Confirmed gameplay-active (`0x0081`), actor-state (`0x0082`), and horizontal
  progress (`0x006C`) fields
- Reset-to-stage screenshot: [boot_stage1.png](../recordings/boot_stage1.png)
- Movement RAM probe and focused unit tests

## Not done

- Segment policies / behavior tree
- Continuous multi-segment or full-game runs

## Next

Clear the first traversable segment from `Stage1.state`: move right, jump gaps,
and attack the nearest threat.
