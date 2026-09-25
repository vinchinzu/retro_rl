# ALTTP randomizer tooling

Commands and traps: `AGENTS.md`. This file is the ROM and seed pointer only.

## ROM

Japanese 1.0 only, same dump family as SMZ3.

| Dump | Path | xxHash32 | Package |
|------|------|----------|---------|
| JP 1.0 | `roms/zelda3_jp.sfc` | `0x8AC8FD15` | this package, smz3 |
| USA | `roms/zelda3.sfc` | different | `alttp/` only |

Internal title must be `ZELDANODENSETSU`, not `THE LEGEND OF ZELDA`.
`setup_rom` refuses to wire the USA dump as primary.

## Generators still to wire

| Tool | Notes |
|------|--------|
| [ALTTPR](https://alttpr.com/) | Web generator and race seeds |
| SahasrahBot or a local patcher | Offline builds later |

Seed packages live under `seeds/<name>/` with `meta.json`, optional spoiler and locations, and later a patch or full `.sfc`.

Fixture packages `demo_seed`, `fixture_1337`, `fixture_1338`, and `fixture_1339` are JP vanilla metadata. They are not shuffled ROMs. `FirstPlay.state` is the first controllable frame in Link's House, not name select.
