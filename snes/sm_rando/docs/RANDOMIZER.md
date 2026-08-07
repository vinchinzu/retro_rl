# Super Metroid Randomizer tooling

## Community generators (to wire)

| Tool | Notes |
|------|--------|
| [VARIA Randomizer](https://randommetroidsolver.pythonanywhere.com/) | Popular logic presets; local CLI possible |
| Super Metroid Randomizer (SMR) | Classic item rando lineage |

This package stays **generator-agnostic**: seed packages under `seeds/<name>/`
with `meta.json`, optional `spoiler.json` / `locations.json`, and later a
patch or full `.sfc`.

## Fixture / demo seeds

```bash
uv run python -c "from sm_rando.seed import ensure_test_seed; print(ensure_test_seed().directory)"
uv run python -c "from sm_rando.seed import ensure_demo_seed; print(ensure_demo_seed().directory)"
```

- `seeds/test_seed/` — deterministic offline placement list for logic-graph practice.
- `seeds/demo_seed/` — playable demo meta: vanilla SM FirstPlay (Ceres), not shuffled.

Neither is a real shuffled ROM until generator wiring lands.

## Vanilla ROM

Same dump as `super_metroid` / `smz3`: `roms/SuperMetroid.sfc`
(SHA1 `da957f0d63d14cb441d215462904c4fa8519c613`, xxh32 `0xCADB4883`).

```bash
uv run python -m sm_rando.scripts.setup_rom
SDL_VIDEODRIVER=dummy uv run python -m sm_rando.scripts.make_boot
./play   # FirstPlay + record
```
