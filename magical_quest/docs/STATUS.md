# Status — The Magical Quest Starring Mickey Mouse

| Field | Value |
|-------|-------|
| Ladder rank | 7 |
| Tier | 3 |
| Status | **boot verified; controllable Stage 1 checkpoint** |
| Integration | `MagicalQuest-Snes` |
| ROM zip | `Magical Quest starring Mickey Mouse, The.zip` |

## Done

- Directory layout and integration stubs (`data.json` / `metadata.json` /
  `scenario.json`)
- `scripts/setup_rom.py` wiring via `snes_oneshot.rom_setup`
- Plan notes for control style and first segment milestone
- Deterministic reset/menu sequence selects one-player/default Mickey and
  reaches Stage 1 at frame **2400**
- `scripts/boot_probe.py` saves
  `custom_integrations/MagicalQuest-Snes/Stage1.state`
- Boot screenshot: [`recordings/boot_stage1.png`](../recordings/boot_stage1.png)
- Controlled probes confirm player/world X `0x0024`, horizontal progress
  `0x002A`, and gameplay-active `0x02C0 == 1`
- From the same replay point, RIGHT/LEFT change X `360→370/352` and progress
  `2696→2720/2664`

## Not done

- Player Y/velocity/grounded, health/lives, room/stage, enemies, and doors
- Segment policies / behavior tree
- Continuous multi-segment or full-game runs

## Next

From `Stage1.state`, clear the first room/segment to the next door or
checkpoint.

## Verify

```bash
uv run --frozen python magical_quest/scripts/boot_probe.py
uv run --frozen python magical_quest/scripts/ram_probe.py
uv run --frozen pytest magical_quest/tests -q
```
