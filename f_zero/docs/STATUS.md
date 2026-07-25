# Status — F-Zero


## Program gate

| Field | Value |
|-------|-------|
| Current maturity | M2 |
| Best verified result | Mute City race-start checkpoint |
| Last verification | 2026-07-22 |
| Runtime class | Bronze |
| Intervention class | Clean |

| Field | Value |
|-------|-------|
| Status | **boot verified; Mute City race-start checkpoint** |
| Integration | `FZero-Snes` |
| ROM zip | `F-Zero.zip` |

## Done

- Directory layout and integration stubs (`data.json` / `metadata.json` /
  `scenario.json`)
- `scripts/setup_rom.py` wiring via `snes_oneshot.rom_setup`
- Plan notes for control style and first segment milestone
- Deterministic reset/menu sequence selects Grand Prix, Blue Falcon, beginner
  Knight League, and Mute City I
- `scripts/boot_probe.py` reaches the race countdown at frame **1080** and
  saves `custom_integrations/FZero-Snes/MuteCity.state`
- Boot screenshot: [`recordings/boot_mute_city.png`](../recordings/boot_mute_city.png)
- Controlled probes confirm raw speed word `0x0002`, race/track state
  `0x0046/0x0047`, and lateral words `0x007F` / `0x00A6`
- From the same warmed replay point, straight acceleration changes raw speed
  `2452→3196`; LEFT/RIGHT change lateral `344→211/465`

## Not done

- Lap, rank, energy, heading, track progress, and collision-state RAM
- Segment policies / behavior tree
- Continuous multi-segment or full-game runs

## Next

From `MuteCity.state`, record an initial centerline trajectory and complete one
lap without crashing out.

## Verify

```bash
uv run --frozen python f_zero/scripts/boot_probe.py
uv run --frozen python f_zero/scripts/ram_probe.py
uv run --frozen pytest f_zero/tests -q
```
