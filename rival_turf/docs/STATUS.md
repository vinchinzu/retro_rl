# Status — Rival Turf!


## Program gate

| Field | Value |
|-------|-------|
| Current maturity | M2 |
| Best verified result | Fight-ready Stage 1 checkpoint |
| Last verification | 2026-07-22 |
| Runtime class | Bronze |
| Intervention class | Clean |

| Field | Value |
|-------|-------|
| Status | **boot verified; fight-ready Stage 1 checkpoint** |
| Integration | `RivalTurf-Snes` |
| ROM zip | `Rival Turf!.zip` |

## Done

- Directory layout and integration stubs (`data.json` / `metadata.json` /
  `scenario.json`)
- `scripts/setup_rom.py` wiring via `snes_oneshot.rom_setup`
- Plan notes for control style and first segment milestone
- Deterministic reset/menu script selects one-player + Jack Flak and reaches
  Stage 1 at frame 2000
- `scripts/boot_probe.py` walks into the opening two-enemy lock and saves
  `custom_integrations/RivalTurf-Snes/Stage1.state` at frame **2360**
- Boot screenshot: [`recordings/boot_stage1.png`](../recordings/boot_stage1.png)
- Controlled RAM probes confirm player X `0x0202`, player Y `0x0205`, player
  active `0x0200`, and active run state `0x00AB == 1`
- `scripts/ram_probe.py` reproduces RIGHT/LEFT/UP/DOWN coordinate deltas

## Not done

- Enemy slots / health, camera/progress, stage, and lock/clear flags
- Segment policies / behavior tree
- Continuous multi-segment or full-game runs

## Next

From `Stage1.state`, clear the opening two-enemy street lock and detect the
camera/progress advance.

## Verify

```bash
uv run --frozen python rival_turf/scripts/boot_probe.py
uv run --frozen python rival_turf/scripts/ram_probe.py
uv run --frozen pytest rival_turf/tests -q
```
