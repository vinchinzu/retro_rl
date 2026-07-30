# Agent Instructions — smz3

Super Metroid + A Link to the Past **combined randomizer** (SMZ3) workspace.
Long-term: roll a seed, race two bots on it, capture video. Near-term: seed
tooling, combo ROM build, room-timeout stop rule, and reuse of vanilla
`alttp/` + `super_metroid/` packages — do **not** fork those trees.

## Norms

- Combined seed ROMs and vanilla source ROMs stay gitignored.
- Seed packages live under `seeds/` (metadata may be committed later; ROM/patch
  blobs stay local / gitignored).
- Base IPS: `refs/zsm.ips.gz` from tewtal SMZ3Randomizer (setup script).
- Room timeout: **3×** standard room time → game over (`room_timeout.py`).
- Headless probes: `SDL_VIDEODRIVER=dummy`.
- Prefer importing `super_metroid.*` and `alttp.*` over copy-paste.

## Immediate goal

1. ~~Generate a test seed + playable combo ROM.~~
2. ~~Wire stable-retro integration (`SMZ3-Snes`) and boot smoke.~~
3. ~~Detect active world (SM vs Z3) in combo RAM (WRAM heuristic).~~
4. ~~Drive early rooms (Landing → Parlor) + 3× room timeout.~~
5. Map portal → settled Z3; longer segments + video.
6. Dual-bot race harness + video (scaffold in `race.py`).

Vanilla primitives for both games should continue to mature in their own
folders; this tree is the randomizer / race layer on top.

## Commands

```bash
# Vanilla ROM symlinks + base IPS
uv run python smz3/scripts/setup_roms.py
uv run python smz3/scripts/setup_base_ips.py

# Pinned test seed (1337, uncle sword, original morph) + combo ROM
uv run python smz3/scripts/generate_seed.py --test
uv run python smz3/scripts/smoke_rom.py smz3/seeds/test_seed/smz3.sfc

# Point stable-retro integration at the seed ROM
uv run python smz3/scripts/wire_integration_rom.py
SDL_VIDEODRIVER=dummy uv run python smz3/scripts/smoke_rom.py --boot
# M1: power-on → first controllable SM frame
SDL_VIDEODRIVER=dummy uv run python smz3/scripts/smoke_rom.py --boot --controllable
SDL_VIDEODRIVER=dummy uv run python smz3/scripts/probe_boot.py --save-png
# M2: Landing Site → Parlor + room timeout
SDL_VIDEODRIVER=dummy uv run python smz3/scripts/probe_early_rooms.py --save-png
uv run python smz3/scripts/probe_early_rooms.py --list-portals

# Arbitrary seed
uv run python smz3/scripts/generate_seed.py --seed 42 --name seed_42

# Unit tests (no network / no ROM required for most)
uv run --frozen pytest smz3/tests -q
```

Online seed generation needs `pyz3r` (`uv sync --extra smz3`).

## Layout

| Path | Role |
|------|------|
| `seed.py` / `rom_builder.py` | samus.link seed + combo ROM build |
| `boot.py` | Power-on → first SM controllable frame |
| `early_route.py` | Landing Site → Parlor segment + timeout |
| `portals.py` | Fixed SM↔Z3 portal catalog |
| `ram.py` | Dual SM/Z3 WRAM snapshot |
| `room_timeout.py` | 3× baseline dwell → game over |
| `world.py` | SM/Z3 detect + package dispatch |
| `race.py` | Dual-bot race plan scaffold |
| `seeds/` | Seed packages (meta, spoiler, patch, rom) |
| `refs/` | Base IPS + optional randomizer clone |
| `custom_integrations/SMZ3-Snes/` | stable-retro integration stubs |

Status: `docs/STATUS.md`. Plan: `docs/plan.md`. RAM: `docs/ram_map.md`.
Early rooms: `docs/EARLY_ROOMS.md`. Randomizer: `docs/RANDOMIZER.md`.
