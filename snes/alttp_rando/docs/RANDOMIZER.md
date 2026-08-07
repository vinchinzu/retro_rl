# ALTTP Randomizer tooling

## ROM requirement — Japanese 1.0

This package uses **ALttP Japanese 1.0 only** (same dump as SMZ3 / samus.link):

| Dump | Path | xxHash32 (seed SMZ3) | Package |
|------|------|----------------------|---------|
| **JP 1.0** | `roms/zelda3_jp.sfc` | `0x8AC8FD15` | **alttp_rando**, smz3 |
| USA | `roms/zelda3.sfc` | (different) | `alttp/` only |

Internal title: `ZELDANODENSETSU` (not `THE LEGEND OF ZELDA`).

```bash
uv run python -m alttp_rando.scripts.setup_rom
```

`setup_rom` refuses to wire the USA dump as primary.

## Community generators (to wire)

| Tool | Notes |
|------|--------|
| [ALTTPR](https://alttpr.com/) | Standard web generator / race seeds |
| SahasrahBot / local patchers | Offline builds later |

Generator-agnostic seed packages under `seeds/<name>/` with `meta.json`,
optional spoiler/locations, later patch or full `.sfc`.

## Fixture / demo seeds

```bash
uv run python -c "from alttp_rando.seed import ensure_test_seed; print(ensure_test_seed().directory)"
# JP vanilla FirstPlay demo meta:
# snes/alttp_rando/seeds/demo_seed/
```

Until a patched rando ROM is wired, play uses JP vanilla + `FirstPlay.state`.

## Boot + play

```bash
SDL_VIDEODRIVER=dummy uv run python -m alttp_rando.scripts.make_boot
./play                          # record MP4+JSON, F5 → integration
uv run python -m alttp_rando.scripts.play --no-record
uv run python -m alttp_rando.scripts.play --rebuild-boot
```

`FirstPlay` is the first controllable frame after intro (Link's House), not
name select. Automated boot may run name entry once; the saved state skips it.
