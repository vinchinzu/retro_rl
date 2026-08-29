# Agent Instructions — tmnt_ii

Scripted NES completion agent for **Teenage Mutant Ninja Turtles II: The Arcade Game** (linear_combat track; maturity **M3**).

## Identity

| Field | Value |
|-------|-------|
| Status | first Stage 1 wave clear (M3) |
| Integration | `TeenageMutantNinjaTurtlesII-Nes` |
| Shared ROM zip | `roms/Nintendo/NES/Teenage Mutant Ninja Turtles II - The Arcade Game.zip` |
| Local ROM | `tmnt_ii/roms/` (via `scripts/setup_rom.py`) |
| Clear checkpoint | `Stage1_Clear` (score≥5 from `Level1`) |

## Commands

```bash
uv run python tmnt_ii/scripts/setup_rom.py
uv run python tmnt_ii/scripts/boot_probe.py
SDL_VIDEODRIVER=dummy SDL_AUDIODRIVER=dummy \
  uv run python tmnt_ii/scripts/run_stage1_segment.py
SDL_VIDEODRIVER=dummy SDL_AUDIODRIVER=dummy \
  uv run python tmnt_ii/scripts/run_stage1_segment.py --from-boot --trials 3
uv run pytest tmnt_ii/tests -q
```

## Traps

- After ~4 kills, feet pin the **right edge** — keep walking RIGHT and you
  stall. **Face LEFT + B** to finish the lock (score 4→5).
- Fire floor at bottom — do not hold DOWN.
- Score is at `0x03F0` (PTS); health `0x0568`; lives `0x004D`.

## Next milestone

Extend past score 5 / unlock next packs (`--target 8` from boot leftover).

## Norms

- Prefer nearest local docs (`docs/STATUS.md`, `docs/plan.md`) over root notes.
- Keep RAM maps, save states, and policies inside this game directory.
- NES actions use `retro_harness.nes` (9-button fceumm layout).
- Line length 88; type hints; `uv run pytest` for tests.
