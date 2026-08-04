# Status — Pilotwings


## Program gate

| Field | Value |
|-------|-------|
| Current maturity | M2 |
| Best verified result | Airborne Lesson 1 checkpoint |
| Last verification | 2026-07-22 |
| Runtime class | Bronze |
| Intervention class | Clean |

| Field | Value |
|-------|-------|
| Status | **boot verified** |
| Integration | `Pilotwings-Snes` |
| ROM zip | `Pilotwings.zip` |

## Done

- Directory layout and integration stubs (`data.json` / `metadata.json` /
  `scenario.json`)
- `scripts/setup_rom.py` wiring via `retro_harness.env`
- Plan notes for control style and first segment milestone
- Deterministic reset-to-Lesson-1 light-plane script (1,920 frames)
- Airborne, unpaused `Lesson1Plane.state` checkpoint
- Confirmed altitude (`0x0058`), pitch control (`0x005D`), and heading
  (`0x0060`) fields
- Reset-to-flight screenshot: [boot_lesson1_plane.png](../recordings/boot_lesson1_plane.png)
- Directional RAM probe and focused unit tests

## Not done

- Segment policies / behavior tree
- Continuous multi-segment or full-game runs

## Next

Complete the Lesson 1 light-plane objective from `Lesson1Plane.state`.
