# Agent Instructions — sm_rando

**Super Metroid Randomizer** (single-game). Simpler solver ground than SMZ3.
Reuse `super_metroid` skills — do **not** fork that tree.

Docs: `docs/STATUS.md`, `docs/plan.md`, `docs/RANDOMIZER.md`.

## Commands

```bash
# Wire vanilla Super Metroid ROM into SMRando-Snes
uv run python -m sm_rando.scripts.setup_rom

# Power-on → FirstPlay.state (Ceres elevator, game_state==8)
SDL_VIDEODRIVER=dummy uv run python -m sm_rando.scripts.make_boot

# Interactive play + record (default) into recordings/play_*.mp4
./play
# or: uv run python -m sm_rando.scripts.play
# flags: --no-record  --rebuild-boot  --max-frames N  --vanilla

uv run pytest snes/sm_rando/tests -q
SDL_VIDEODRIVER=dummy uv run python -m sm_rando.scripts.run_vertical_slice
SDL_VIDEODRIVER=dummy uv run python -m sm_rando.scripts.run_morph_policy
# Multi-seed early tip S/T (ship→morph) via SeedCampaignRunner
SDL_VIDEODRIVER=dummy uv run python -m sm_rando.scripts.run_early_tip_campaign --mode dry
# Live (real emulator tip per seed; needs ROM): --mode live
SDL_VIDEODRIVER=dummy uv run python -m sm_rando.scripts.harvest_entry_corpus
SDL_VIDEODRIVER=dummy uv run python -m sm_rando.scripts.evaluate_entry_corpus
uv run python -c "from sm_rando.seed import ensure_demo_seed; print(ensure_demo_seed().directory)"
```

## Play controls

- arrows = D-pad · Z=B · X=A · A=Y · S=X · TAB=turbo · [/]=speed
- F5 = save state under `custom_integrations/SMRando-Snes/`
- Default start: `FirstPlay` (no title/name menus)

## Traps

- Until a real rando generator is wired, the integration ROM is **vanilla SM**.
- First controllable frame is **Ceres elevator** (`0xDF45`), not Landing Site.
- Logic graph edges are **planned** until pure skills bind them.
- The vertical-slice runner uses the vanilla ROM substrate and writes an
  audited manifest to `recordings/vertical_slice.run.json`; it is not patched-
  randomizer evidence.
- SMZ3 is harder; prove multi-seed patterns here first when possible.

## Immediate goal

Multi-seed early tip S/T dry-run is published (fixture seeds, vanilla substrate).
Next: patched generator ROMs + live multi-seed morph tip; do not claim shuffled
seed-robustness until those land.
