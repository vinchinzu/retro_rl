# SMZ3 randomizer tooling

## What SMZ3 is

[SMZ3](https://samus.link/) combines Super Metroid and A Link to the Past into
one SNES ROM with portals between worlds and a shared randomized item pool.
Community sites:

- Web randomizer: https://samus.link/
- Upstream source: https://github.com/tewtal/SMZ3Randomizer
- Cas' fork (tracker / QoL): https://github.com/TheTrackerCouncil/SMZ3Randomizer

This project targets the classic **samus.link / tewtal V11** pipeline so seed
patches from the public API apply with the matching base IPS.

## Seed generation (online)

```bash
uv pip install pyz3r   # if needed
uv run python smz3/scripts/generate_seed.py --test
```

Uses `pyz3r.sm` → `POST https://samus.link/api/randomizers/smz3/generate`.

Default test settings:

| Key | Value |
|-----|-------|
| seed | 1337 |
| smlogic | normal |
| goal | defeatboth |
| swordlocation | uncle |
| morphlocation | original |
| gamemode | normal |
| players | 1 |

Output package (`smz3/seeds/<name>/`):

| File | Contents |
|------|----------|
| `meta.json` | hash, URL, settings, version |
| `spoiler.json` | sphere / location spoilers |
| `locations.json` | raw location→item ids |
| `seed_patch.bin` / `.b64` | world seed patch |
| `smz3.sfc` | 6 MiB playable combo ROM (local only) |

## Combo ROM build (local)

Pipeline (ported from the web client `prepareRom`):

1. **Merge** unheadered SM (3 MiB) + Z3 (1 MiB) → 6 MiB ExHi-style image
2. **Apply** base IPS `zsm.ips` (gzip in `refs/zsm.ips.gz`)
3. **Apply** seed patch records: `(u32 LE dest, u16 LE length, data)*`

```bash
uv run python smz3/scripts/setup_roms.py
uv run python smz3/scripts/setup_base_ips.py
uv run python smz3/scripts/generate_seed.py --test
uv run python smz3/scripts/smoke_rom.py smz3/seeds/test_seed/smz3.sfc
```

ROMs are never committed. Base IPS is third-party; keep under `refs/`
(gitignored).

## Offline CLI (optional, later)

tewtal `Randomizer.CLI` verb:

```text
smz3 --single --rom --seed <n> --ips <zsm.ips> ...
```

Requires .NET, base IPS from the combo asm project, and named vanilla ROM
files. Prefer the online + Python ROM builder for foundation work.

## Integration note

stable-retro needs a per-game integration directory. Point
`custom_integrations/SMZ3-Snes/rom.sfc` at a built seed ROM (or copy bytes)
before emulator sessions. Low WRAM SM fields in `data.json` match vanilla and
are verified on the combo image (see `docs/ram_map.md`). Combo
`SRAM_CURRENT_GAME` at `$A1:73FE` is not yet readable via `get_ram()`.

## Boot (M1)

```bash
uv run python smz3/scripts/wire_integration_rom.py
SDL_VIDEODRIVER=dummy uv run python smz3/scripts/probe_boot.py --save-png
# or
SDL_VIDEODRIVER=dummy uv run python smz3/scripts/smoke_rom.py --boot --controllable
```

Combo always resets into Super Metroid. Fresh file → Landing Site (`0x91F8`).
