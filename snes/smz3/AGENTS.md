# Agent Instructions — smz3

SM + ALttP **combined randomizer**. Reuse vanilla `alttp/` + `super_metroid/`
— do **not** fork those trees. Docs: `docs/STATUS.md`, `docs/plan.md`,
`docs/EARLY_ROOMS.md`, `docs/RANDOMIZER.md`, `docs/ram_map.md`.

## Commands

```bash
uv run python smz3/scripts/setup_roms.py
uv run python smz3/scripts/setup_base_ips.py
uv run python smz3/scripts/generate_seed.py --test
uv run python smz3/scripts/wire_integration_rom.py

SDL_VIDEODRIVER=dummy uv run python smz3/scripts/smoke_rom.py --boot --controllable
SDL_VIDEODRIVER=dummy uv run python smz3/scripts/probe_early_rooms.py --save-png
SDL_VIDEODRIVER=dummy uv run python smz3/scripts/probe_portal.py --through-portal --save-png --save-state
SDL_VIDEODRIVER=dummy uv run python smz3/scripts/probe_outdoor.py --video --save-png
SDL_VIDEODRIVER=dummy uv run python smz3/scripts/probe_house.py --video --save-png

uv run python smz3/scripts/play_portal.py --state PortalSettled
uv run --frozen pytest smz3/tests -q
# Online seeds: uv sync --extra smz3
```

## Traps

- ALttP must be **Japanese 1.0** (`roms/zelda3_jp.sfc`, xxh32 `0x8AC8FD15`).
  USA `zelda3.sfc` is for `alttp/` only — wrong dump → portal hang module `$0F`.
- SM ROM: `roms/SuperMetroid.sfc` (`0xCADB4883`). Base IPS: `refs/zsm.ips.gz`.
- Room timeout: **3×** baseline dwell → game over (`room_timeout.py`).
- Interactive play: focus **pygame window** for ESC/Q (terminal ESC ignored).
- Seeds/ROMs gitignored; packages under `seeds/`.

## Immediate goal

Dual-bot race harness + video (`race.py` scaffold). Early quest path through
house chest is verified; mature primitives stay in vanilla game folders.
