# Agent Instructions — alttp_rando

**ALTTP Randomizer** (single-game). Simpler solver ground than SMZ3.
Reuse `alttp` skills — do **not** fork that tree.

Docs: `docs/STATUS.md`, `docs/plan.md`, `docs/RANDOMIZER.md`.

## ROM trap (critical)

Use **Japanese 1.0 only**:

| ROM | Path | Role |
|-----|------|------|
| JP 1.0 | `roms/zelda3_jp.sfc` | **this package** + SMZ3 (xxh32 `0x8AC8FD15`) |
| USA | `roms/zelda3.sfc` | `alttp/` only — **never** wire as primary here |

Internal title must be `ZELDANODENSETSU`, not `THE LEGEND OF ZELDA`.

## Commands

```bash
# Wire JP ROM into ALTTPRando-Snes
uv run python -m alttp_rando.scripts.setup_rom

# Headless FirstPlay.state (first controllable frame)
SDL_VIDEODRIVER=dummy uv run python -m alttp_rando.scripts.make_boot
SDL_VIDEODRIVER=dummy uv run python -m alttp_rando.scripts.make_boot --force

# Play + record from FirstPlay (auto-boot if missing)
./play
# or:
uv run python -m alttp_rando.scripts.play
uv run python -m alttp_rando.scripts.play --no-record
uv run python -m alttp_rando.scripts.play --rebuild-boot
uv run python -m alttp_rando.scripts.play --vanilla   # USA alttp skills

uv run pytest snes/alttp_rando/tests -q
uv run python -c "from alttp_rando.seed import ensure_test_seed; ensure_test_seed()"
```

F5 in `./play` saves into `custom_integrations/ALTTPRando-Snes/`.
Recordings → `recordings/` (MP4 + JSON).

## Traps

- Do not symlink USA `zelda3.sfc` into this integration.
- `FirstPlay` is post-intro control (Link's House), not name select.
- Seed ROM / ALTTPR patch integration still open; demo uses JP vanilla.
- Logic edges are planned until natural-entry skills bind them.

## Immediate goal

M1 boot ✓ (JP FirstPlay). Next: bind house→uncle edge to vanilla skill +
fixture/patched seed playable boot.
