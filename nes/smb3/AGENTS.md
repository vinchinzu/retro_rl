# Agent Instructions — smb3

Scripted NES completion agent for **Super Mario Bros. 3** (platforming track; maturity M3 segment).

## Identity

| Field | Value |
|-------|-------|
| Status | World 1-1 clear verified |
| Integration | `SuperMarioBros3-Nes` |
| Shared ROM zip | `roms/Nintendo/NES/Super Mario Bros. 3.zip` |
| Local ROM | `smb3/roms/` (via `scripts/setup_rom.py`) |
| 1-1 policy | `smb3/policies/level1_1.json` |
| 1-2 policy | `smb3/policies/level1_2.json` |

## Commands

```bash
uv run python nes/smb3/scripts/setup_rom.py
uv run python nes/smb3/scripts/boot_probe.py
uv run python nes/smb3/scripts/run_level1.py
uv run python nes/smb3/scripts/run_level1.py --from-state Level1_1
uv run python nes/smb3/scripts/run_level1.py --level 1-2
uv run python nes/smb3/scripts/run_level1.py --level 1-2 --from-state Level1_2
uv run pytest nes/smb3/tests -q
```

## Traps

- Boot map pose is **not** on the enterable 1-1 node: need RIGHT then UP, then A.
- AfterLevel1 is not immediately controllable: wait Map_Operation `$0D` (~114f).
- 1-1 → 1-2 is **two** RIGHT hops (T-junction tile, then 1-2 panel `$04`).
- 1-1/1-2 policies are frame-synced to natural entry; desyncs if map walk or
  level-load settle frames change (re-hillclimb from natural entry).
- NES actions use `retro_harness.nes` (9-button fceumm layout): B=run, A=jump.
- Progress uses `x_page (0x75) * 256 + hpos (0x90)` only while `x_page < 0x18`.

## Next milestone

World 1-3 natural-entry from AfterLevel2.

## Norms

- Prefer nearest local docs (`docs/STATUS.md`, `docs/plan.md`) over root notes.
- Keep RAM maps, save states, and policies inside this game directory.
- Line length 88; type hints; `uv run pytest` for tests.
