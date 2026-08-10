# Agent Instructions — mega_man_2

Scripted NES completion agent for **Mega Man 2** (platforming track; maturity M3).

## Identity

| Field | Value |
|-------|-------|
| Status | Air s4 clear; post-s4 cloud RED; Heat1+HeatScreen1 dual-green (Item-1 chain) |
| Integration | `MegaMan2-Nes` |
| Shared ROM zip | `roms/Nintendo/NES/Mega Man II.zip` |
| Local ROM | `mega_man_2/roms/` (via `scripts/setup_rom.py`) |

## Commands

```bash
uv run python nes/mega_man_2/scripts/setup_rom.py
uv run python nes/mega_man_2/scripts/boot_probe.py
uv run python nes/mega_man_2/scripts/boot_heat_probe.py
uv run python nes/mega_man_2/scripts/run_air_segment.py --trials 3
uv run python nes/mega_man_2/scripts/run_air_segment.py --state AirLanded --trials 3
uv run python nes/mega_man_2/scripts/run_air_segment.py --state AirScreen2 --target-screen 3 --trials 3
uv run python nes/mega_man_2/scripts/run_air_segment.py --state AirScreen2 --target-screen 4 --trials 3
uv run python nes/mega_man_2/scripts/run_heat_segment.py --trials 3
uv run pytest nes/mega_man_2/tests -q
```

## Next milestone

**Heat→Air Item-1 chain** (rr-f3nr PARTIAL residual): Heat mid/boss → Item-1 unlock
→ Air past s4 with platforms. Dual-green done: `Heat1` entry + `HeatScreen1`.
Doc: `docs/HEAT_ITEM1_PATH.md`. Cloud solid still RED; do not re-grid.

Air post-s4 context: LL spawns mapset4 (`0x3D`/`0x3E`); rider kill Clean; empty
cloud object-solid never arms. Gap ~296px. FCEUX stick pin protocol in
HEAT_ITEM1_PATH (external).

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
  global solid (path OK). No Air-first Clean alt (Item-1 needs Heat).
- Stage select `$002A`: Wily=0, Air=2, Heat=8. Password→select at Wily;
  `LEFT`→Heat, `UP`→Air. Items `$009B` bit `$01` = Item-1 (Heat clear).
- Heat boot: `boot_to_heat_man_script` / `boot_heat_probe.py` → `Heat1`.
  Early Heat: `HeatManPolicy` / `run_heat_segment.py`.
