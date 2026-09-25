# Agent instructions: alttp_rando

Single-game ALTTP randomizer. Reuse `alttp` skills. Do not fork that tree.
Docs: `docs/STATUS.md`, `docs/plan.md`, `docs/RANDOMIZER.md`.

## ROM trap

Japanese 1.0 only: `roms/zelda3_jp.sfc` (xxh32 `0x8AC8FD15`, internal title
`ZELDANODENSETSU`). USA `roms/zelda3.sfc` belongs to `alttp/` only. Do not
symlink it into this integration.

## Commands

```bash
uv run python -m alttp_rando.scripts.setup_rom
SDL_VIDEODRIVER=dummy uv run python -m alttp_rando.scripts.make_boot
./play
uv run python -m alttp_rando.scripts.play --no-record
uv run python -m alttp_rando.scripts.play --vanilla

SDL_VIDEODRIVER=dummy uv run python -m alttp_rando.scripts.run_house_to_uncle
SDL_VIDEODRIVER=dummy uv run python -m alttp_rando.scripts.run_house_to_uncle_session
uv run python -m alttp_rando.scripts.run_opening_tip_campaign --mode dry --publish-docs

uv run pytest snes/alttp_rando/tests -q
```

`./play` runs from this package directory. F5 saves into
`custom_integrations/ALTTPRando-Snes/`.
`make_boot --force` rebuilds `FirstPlay.state`.

## Layout

| Path | Role |
|------|------|
| `scripts/setup_rom.py` | Wire the JP ROM. Refuses the USA dump. |
| `scripts/make_boot.py` | Headless `FirstPlay.state` |
| `seeds/` | Fixture packages (`demo_seed`, `fixture_1337` to `fixture_1339`). Not shuffled ROMs. |
| `recordings/` | JSON evidence. An MP4 from `./play` is not leave proof. |

## Traps

- `FirstPlay` is Link's House after the intro, not name select.
- `house_to_uncle` is natural_entry on vanilla JP FirstPlay. It is not a shuffled-seed clear.
- The opening S/T dry-run is substrate=vanilla and seed_source=fixture. It is not shuffled robustness.
- A patched ALTTPR ROM is still open. Demo seeds are JP vanilla fixtures.
- Leave proof is the JSON report (room, module, sword), not an MP4.
- Ladder matches vanilla `alttp`: planned, then isolated, then natural_entry, then continuous. A state load is not continuous.
- `run_house_to_uncle_session` replans on vanilla FirstPlay. It is not shuffled S/T.
- Early-graph edges past `house_to_uncle` stay planned until a skill binds them.
