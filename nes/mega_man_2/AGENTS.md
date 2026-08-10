# Agent Instructions — mega_man_2

Scripted NES completion agent for **Mega Man 2** (platforming track; maturity M3).

## Identity

| Field | Value |
|-------|-------|
| Status | Air Man screen-4 clear from AirScreen2 (M3); LL kill OK; cloud solid residual (disasm) |
| Integration | `MegaMan2-Nes` |
| Shared ROM zip | `roms/Nintendo/NES/Mega Man II.zip` |
| Local ROM | `mega_man_2/roms/` (via `scripts/setup_rom.py`) |

## Commands

```bash
uv run python nes/mega_man_2/scripts/setup_rom.py
uv run python nes/mega_man_2/scripts/boot_probe.py
uv run python nes/mega_man_2/scripts/run_air_segment.py --trials 3
uv run python nes/mega_man_2/scripts/run_air_segment.py --state AirLanded --trials 3
uv run python nes/mega_man_2/scripts/run_air_segment.py --state AirScreen2 --target-screen 3 --trials 3
uv run python nes/mega_man_2/scripts/run_air_segment.py --state AirScreen2 --target-screen 4 --trials 3
uv run python nes/mega_man_2/scripts/run_air_segment.py --state AirScreen2 --target-screen 5 --trials 3
uv run python nes/mega_man_2/scripts/run_air_segment.py --target-screen 1  # legacy
uv run pytest nes/mega_man_2/tests -q
```

## Next milestone

Past screen 4 from `AirFanPlatform` (prog 937–984) toward boss door.
**LL spawns** mapset4 (`0x3D`/`0x3E` ~prog 961; `docs/LL_SPAWN_DECODE.md`).
**Rider kill Clean** (pulse B on `0x3D`). Residual: empty cloud **object-solid
never arms** (body never `LDA #$90`; sole appear-arm is Heat appearing_block AI).
Zero-mask force proves fceumm solid path works; no Air-first Clean skip (Item-1
needs Heat; gap ~296px). Docs: `CLOUD_LAND_RED_PIN.md`. Child **rr-f3nr**:
FCEUX/human stick RAM pin or Heat→Air Item-1 segment.

## Norms

- Prefer nearest local docs (`docs/STATUS.md`, `docs/plan.md`) over root notes.
- Keep RAM maps, save states, and policies inside this game directory.
- NES actions use `retro_harness.nes` (9-button fceumm layout).
- Line length 88; type hints; `uv run pytest` for tests.
- `AirScreen1` alone is mid-air over a pit — use `AirLanded` for grounded scr1.
- `AirScreen2` uses `AirManPolicy(start="screen2")` (not level1/landed recipes).
- `AirScreen3` / `AirScreen4` are mid-air clear snaps — use `AirFanPlatform` for
  grounded post-s3 iteration (solid **prog 937–984**). Pink head type36 =
  damage enemy (not landable); platforms are tiles.
- `AirLeftPlatform` = short left ledge (prog~902–905). Ladder bar ≠ feet=2.
- Jump needs A rising edge after load; continuous A from frame 1 does not jump.
- Do not save type36-overlap or left-ledge hops as past-island checkpoints.
- LL watch: `$0400` types **0x3D/0x3E** (not 35/36). Goblin is **0x40**.
- Kill LL rider with **pulsed B** (period 3–8); hold-B under-fires. Body `0x3E`
  stays after `0x3D` dies (type 6 death anim ~12f). Stand may not set `tile_feet==1`.
- Empty-cloud residual: not X, not feet_dy=0, not screen-align alone. Body AI
  never arms appear (`LDA #$90` only in appear-block AI). Zero-mask force =
  global solid (path OK). No Air-first Clean alt (Item-1 needs Heat). Next =
  rr-f3nr FCEUX stick pin or Heat→Air Item-1.
